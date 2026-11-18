//! Perihel — local tool-calling agent harness for llama.cpp.
//!
//! llama.cpp has no native tool-call API, so tool calls are constrained with a
//! GBNF grammar (embedded in the binary). The model emits one JSON tool call
//! per turn; deterministic code executes it and feeds the observation back.
//! There is no fixed round limit: Ctrl+C stops at any time.

use clap::Parser;
use serde_json::{json, Value};
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const GRAMMAR: &str = include_str!("../tools.gbnf");

const DEFAULT_SYSTEM: &str = r#"You are Perihel, a careful local coding agent.
Reply with exactly ONE JSON object and nothing else:
{"tool": "TOOL_NAME", "args": {...}}
Tools:
- {"tool":"fs.list","args":{"path":"."}}  list a directory
- {"tool":"fs.read","args":{"path":"a.txt"}}  read a text file
- {"tool":"fs.write","args":{"path":"a.txt","content":"..."}}  write a file (use \n escapes)
- {"tool":"fs.mkdir","args":{"path":"new/dir"}}  create a directory
- {"tool":"fs.copy","args":{"src":"a.bin","dst":"dir/a.bin"}}  copy a file
- {"tool":"run.cmd","args":{"command":"python","args":["-m","pytest","-q"]}}  run an allowlisted command
- {"tool":"run.test","args":{"path":"tests/test_x.py"}}  run tests on one file
- {"tool":"reply.text","args":{"text":"..."}}  finish the turn
Rules: one tool per reply; paths are relative to the workspace, no absolute paths or `..`;
inspect before writing, verify after writing; never claim success you did not verify."#;

const TOOLS: &[&str] = &[
    "fs.list", "fs.read", "fs.write", "fs.mkdir", "fs.copy",
    "run.cmd", "run.test", "reply.text",
];

#[derive(Parser, Debug)]
#[command(name = "perihel", version = "0.1.0",
          about = "Local tool-calling agent harness (llama.cpp + GBNF)")]
struct Cli {
    #[arg(long, default_value = "http://127.0.0.1:8090")]
    base_url: String,
    #[arg(long, default_value = "local-model")]
    model: String,
    #[arg(long, default_value = ".")]
    workspace: PathBuf,
    #[arg(long)]
    grammar: Option<PathBuf>,
    #[arg(long)]
    no_grammar: bool,
    #[arg(short, long)]
    prompt: Option<String>,
    /// Maximum tool rounds per turn. 0 = unlimited (default). Ctrl+C stops.
    #[arg(long, default_value_t = 0)]
    max_rounds: usize,
    /// Plain line-oriented CLI instead of the full-screen harness UI.
    #[arg(long)]
    cli: bool,
}

// ------------------------------------------------------------------ client

struct Client {
    base: String,
    model: String,
    grammar: Option<String>,
    messages: Vec<Value>,
    agent: ureq::Agent,
}

impl Client {
    fn new(base: &str, model: &str, grammar: Option<String>) -> Self {
        let mut messages = Vec::new();
        messages.push(json!({"role": "system", "content": DEFAULT_SYSTEM}));
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(900)))
            .build();
        let agent = ureq::Agent::new_with_config(config);
        Self { base: base.trim_end_matches('/').to_string(), model: model.to_string(), grammar, messages, agent }
    }

    fn reset(&mut self) {
        self.messages.clear();
        self.messages.push(json!({"role": "system", "content": DEFAULT_SYSTEM}));
    }

    fn chat(&mut self, stream: bool) -> Result<String, String> {
        let mut body = json!({
            "model": self.model,
            "messages": self.messages,
            "temperature": 0.4,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 4096,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": false},
        });
        if let Some(g) = &self.grammar {
            body["grammar"] = json!(g);
        }
        let url = format!("{}/v1/chat/completions", self.base);
        let resp = self.agent
            .post(&url)
            .send_json(body)
            .map_err(|e| format!("llama-server unreachable at {}: {}", self.base, e))?;

        if !stream {
            let text: Value = resp.into_body().read_json().map_err(|e| format!("bad JSON: {e}"))?;
            return extract_content(&text);
        }

        // SSE: read `data: {...}` lines, stream delta content to stdout.
        let reader = resp.into_body().into_reader();
        let mut full = String::new();
        let mut last_progress = Instant::now();
        for line in io::BufReader::new(reader).lines() {
            let line = line.map_err(|e| format!("stream read error: {e}"))?;
            if !line.starts_with("data:") {
                continue;
            }
            let data = line[5..].trim();
            if data == "[DONE]" {
                break;
            }
            let piece: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let delta = piece["choices"][0]["delta"]["content"].as_str().unwrap_or("");
            if !delta.is_empty() {
                full.push_str(delta);
                print!("{delta}");
                let _ = io::stdout().flush();
                last_progress = Instant::now();
            }
            let _ = last_progress;
        }
        println!();
        Ok(full)
    }
}

fn extract_content(payload: &Value) -> Result<String, String> {
    let choices = payload.get("choices").and_then(|v| v.as_array())
        .ok_or_else(|| format!("no choices: {}", payload))?;
    let content = choices.first()
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    if content.is_empty() {
        Err("empty completion (reasoning may have consumed the budget)".into())
    } else {
        Ok(content)
    }
}

// ------------------------------------------------------------------ tools

fn safe_path(root: &Path, value: &str) -> Result<PathBuf, String> {
    if value.trim().is_empty() {
        return Err("path must be non-empty".into());
    }
    let raw = Path::new(value);
    if raw.is_absolute() || raw.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return Err(format!("path refused (must stay inside workspace): {value:?}"));
    }
    let resolved = root.join(raw);
    let ok = resolved.starts_with(root) || resolved == root;
    if !ok {
        return Err(format!("path refused (outside workspace): {value:?}"));
    }
    Ok(resolved)
}

fn arg_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key).and_then(|v| v.as_str())
        .ok_or_else(|| format!("missing string argument: {key}"))
}

fn allow_command(argv: &[String]) -> Result<(), String> {
    let prog = Path::new(&argv[0]).file_name().and_then(|s| s.to_str()).unwrap_or("");
    let rest = &argv[1..];
    let allowed = match prog {
        "python" | "python3" => {
            (rest.len() >= 2 && rest[0] == "-m" && rest[1] == "pytest")
                || (rest.len() == 1 && rest[0].ends_with(".py"))
        }
        "node" => (rest.len() == 1 && rest[0].ends_with(".js"))
            || (rest.len() >= 2 && rest[0] == "-e"),
        "npm" => rest.first().map(|s| s == "test").unwrap_or(false),
        "npx" => rest.len() >= 2 && rest[0] == "tsc" && rest[1] == "--noEmit",
        "git" => matches!(rest.first().map(|s| s.as_str()), Some("status" | "diff" | "log"))
            || (rest.len() == 3 && rest[0] == "branch" && rest[1] == "--show-current"),
        "rg" | "pytest" => true,
        _ => false,
    };
    if allowed { Ok(()) } else { Err(format!("command/arguments refused: {argv:?}")) }
}

fn run_process(argv: &[String], cwd: &Path) -> Result<(bool, String), String> {
    allow_command(argv)?;
    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run {}: {e}", argv[0]))?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = child.wait_with_output().map_err(|e| e.to_string())?;
                let mut text = String::from_utf8_lossy(&out.stdout).to_string();
                text.push_str(&String::from_utf8_lossy(&out.stderr));
                if text.len() > 6000 {
                    text = text[text.len() - 6000..].to_string();
                }
                return Ok((status.success(), text));
            }
            Ok(None) => {
                if start.elapsed() > Duration::from_secs(60) {
                    let _ = child.kill();
                    return Err("command timed out after 60s".into());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(e.to_string()),
        }
    }
}

struct ToolResult { ok: bool, output: String, refused: Option<String> }

impl ToolResult {
    fn message(&self) -> String {
        if let Some(r) = &self.refused {
            format!("TOOL RESULT: REFUSED — {r}\nPick a different tool or fix the arguments.")
        } else {
            format!("TOOL RESULT ({}):\n{}", if self.ok { "OK" } else { "FAILED" },
                    &self.output[..self.output.len().min(6000)])
        }
    }
}

fn execute(call: &Value, root: &Path) -> ToolResult {
    let name = call.get("tool").and_then(|v| v.as_str()).unwrap_or("");
    let args = call.get("args").unwrap_or(&Value::Null);
    let wrap = |r: Result<ToolResult, String>| match r {
        Ok(t) => t,
        Err(e) => ToolResult { ok: false, output: String::new(), refused: Some(e) },
    };

    match name {
        "fs.list" => wrap((|| {
            let p = safe_path(root, arg_str(args, "path")?)?;
            let mut lines = Vec::new();
            for e in fs::read_dir(&p).map_err(|e| e.to_string())? {
                let e = e.map_err(|e| e.to_string())?;
                let rel = e.path().strip_prefix(root).unwrap_or(&e.path()).display().to_string();
                if e.path().is_dir() {
                    lines.push(format!("d  {rel}"));
                } else {
                    let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                    lines.push(format!("f  {size:>12}  {rel}"));
                }
            }
            lines.sort();
            Ok(ToolResult { ok: true, output: lines.into_iter().take(300).collect::<Vec<_>>().join("\n"), refused: None })
        })()),
        "fs.read" => wrap((|| {
            let p = safe_path(root, arg_str(args, "path")?)?;
            let mut text = fs::read_to_string(&p).map_err(|e| e.to_string())?;
            if text.len() > 12000 {
                text.truncate(12000);
                text.push_str("\n... [truncated]");
            }
            Ok(ToolResult { ok: true, output: text, refused: None })
        })()),
        "fs.write" => wrap((|| {
            let p = safe_path(root, arg_str(args, "path")?)?;
            let content = arg_str(args, "content")?;
            if let Some(parent) = p.parent() { fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
            fs::write(&p, content).map_err(|e| e.to_string())?;
            Ok(ToolResult { ok: true, output: format!("wrote {} chars -> {}", content.len(), p.display()), refused: None })
        })()),
        "fs.mkdir" => wrap((|| {
            let p = safe_path(root, arg_str(args, "path")?)?;
            fs::create_dir_all(&p).map_err(|e| e.to_string())?;
            Ok(ToolResult { ok: true, output: format!("directory ready -> {}", p.display()), refused: None })
        })()),
        "fs.copy" => wrap((|| {
            let src = safe_path(root, arg_str(args, "src")?)?;
            let dst = safe_path(root, arg_str(args, "dst")?)?;
            if src == dst { return Err("src and dst are the same file".into()); }
            if let Some(parent) = dst.parent() { fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
            let n = fs::copy(&src, &dst).map_err(|e| e.to_string())?;
            Ok(ToolResult { ok: true, output: format!("copied {n} bytes -> {}", dst.display()), refused: None })
        })()),
        "run.cmd" => wrap((|| {
            let prog = arg_str(args, "command")?.to_string();
            let argv: Vec<String> = args.get("args").and_then(|v| v.as_array())
                .ok_or("args must be a list of strings")?
                .iter().map(|v| v.as_str().map(str::to_string).ok_or("bad arg"))
                .collect::<Result<Vec<_>, _>>()?;
            let mut full = vec![prog];
            full.extend(argv);
            let (ok, out) = run_process(&full, root)?;
            Ok(ToolResult { ok, output: out, refused: None })
        })()),
        "run.test" => wrap((|| {
            let p = safe_path(root, arg_str(args, "path")?)?;
            let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy().to_string();
            let argv = vec!["python".to_string(), "-m".into(), "pytest".into(), "-q".into(), rel];
            let (ok, out) = run_process(&argv, root)?;
            Ok(ToolResult { ok, output: out, refused: None })
        })()),
        _ => ToolResult { ok: false, output: String::new(), refused: Some(format!("unknown tool: {name}")) },
    }
}

fn parse_call(text: &str) -> Option<Value> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start { return None; }
    let obj: Value = serde_json::from_str(&text[start..=end]).ok()?;
    let tool = obj.get("tool")?.as_str()?;
    if !TOOLS.contains(&tool) { return None; }
    if !obj.get("args")?.is_object() { return None; }
    Some(obj)
}

// ------------------------------------------------------------------ agent

struct App {
    client: Client,
    root: PathBuf,
    max_rounds: usize,
    tui: bool,
    round: usize,
    history: Vec<(String, String)>,
}

impl App {
    fn add(&mut self, kind: &str, text: &str) {
        self.history.push((kind.to_string(), text.to_string()));
        if self.tui { self.draw(""); }
    }

    fn run_turn(&mut self, user: &str) {
        self.client.messages.push(json!({"role":"user","content":user}));
        self.add("user", user);
        let mut round = 0;
        loop {
            round += 1;
            self.round = round;
            if self.tui { self.draw(""); }
            else { println!("[round {}] thinking...", round); }
            let raw = match self.client.chat(!self.tui) {
                Ok(r) => r,
                Err(e) => { self.add("error", &e); return; }
            };
            let call = match parse_call(&raw) {
                Some(c) => c,
                None => { self.add("assistant", raw.trim()); return; }
            };
            self.client.messages.push(json!({"role":"assistant","content":raw}));
            let name = call["tool"].as_str().unwrap_or("").to_string();
            if name == "reply.text" {
                let text = call["args"]["text"].as_str().unwrap_or("").to_string();
                self.add("assistant", &text);
                return;
            }
            let result = execute(&call, &self.root);
            let preview = result.refused.clone().unwrap_or_else(|| result.output.lines().next().unwrap_or("").to_string());
            if !self.tui {
                println!("[tool] {name} -> {}", &preview[..preview.len().min(160)]);
            }
            self.add("tool", &format!("{name} -> {}", &preview[..preview.len().min(160)]));
            self.client.messages.push(json!({"role":"user","content":result.message()}));
            if self.max_rounds > 0 && round >= self.max_rounds {
                self.add("error", "round limit reached (--max-rounds)");
                return;
            }
        }
    }

    // Minimal full-screen harness-style rendering with ANSI escapes.
    fn draw(&self, input: &str) {
        let (cols, rows) = term_size();
        let body_h = rows.saturating_sub(5).max(4);
        let mut out = String::new();
        out.push_str("\x1b[H\x1b[2J");
        out.push_str(&format!("\x1b[48;5;24m\x1b[97m PERIHEL  \x1b[2m local agent harness \x1b[0m  model:{}  grammar:{}  tools:{}\x1b[0m\n",
            &self.client.model, if self.client.grammar.is_some() {"ON"} else {"OFF"}, TOOLS.len()));
        out.push_str(&format!("{}\n", "-".repeat(cols)));
        let mut lines: Vec<String> = Vec::new();
        for (kind, text) in &self.history {
            let prefix = match kind.as_str() {
                "user" => "\x1b[32m>\x1b[0m ",
                "assistant" => "\x1b[36m*\x1b[0m ",
                "tool" => "\x1b[33m>\x1b[0m ",
                "error" => "\x1b[31m!\x1b[0m ",
                _ => "\x1b[34mi\x1b[0m ",
            };
            for (i, seg) in wrap(text, cols.saturating_sub(3)).iter().enumerate() {
                lines.push(format!("{}{}", if i == 0 { prefix } else { "  " }, seg));
            }
        }
        let start = lines.len().saturating_sub(body_h);
        for line in &lines[start..] {
            out.push_str(line);
            out.push_str("\n");
        }
        for _ in 0..body_h.saturating_sub(lines.len().saturating_sub(start)) { out.push('\n'); }
        out.push_str(&format!("{}\n", "-".repeat(cols)));
        out.push_str(&format!("\x1b[48;5;236m Enter send | /help | /model <name> | /quit | Ctrl+C stop | round {}/{}\x1b[0m\n",
            self.round, if self.max_rounds == 0 { "inf".into() } else { self.max_rounds.to_string() }));
        out.push_str("\x1b[32m>\x1b[0m ");
        out.push_str(input);
        print!("{out}");
        let _ = io::stdout().flush();
    }
}

fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(20);
    let mut out = Vec::new();
    for line in text.lines() {
        if line.len() <= width { out.push(line.to_string()); continue; }
        let mut cur = String::new();
        for word in line.split_whitespace() {
            if !cur.is_empty() && cur.len() + word.len() + 1 > width {
                out.push(cur.clone());
                cur.clear();
            }
            if !cur.is_empty() { cur.push(' '); }
            cur.push_str(word);
        }
        if !cur.is_empty() { out.push(cur); }
    }
    if out.is_empty() { out.push(String::new()); }
    out
}

fn term_size() -> (usize, usize) {
    // Windows Terminal reports size through the console; fall back for redirects.
    let cols = std::env::var("COLUMNS").ok().and_then(|s| s.parse().ok()).unwrap_or(120);
    let rows = std::env::var("LINES").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    (cols, rows)
}

fn banner(app: &App) {
    if app.tui { return; }
    println!("perihel 0.1.0 | model={} | rounds={} | grammar={} | workspace={}",
        app.client.model,
        if app.max_rounds == 0 { "unlimited".into() } else { app.max_rounds.to_string() },
        if app.client.grammar.is_some() { "ON" } else { "OFF" },
        app.root.display());
    println!("Type /help for commands, Ctrl+C to stop.");
}

fn main() {
    let cli = Cli::parse();
    let canonical = fs::canonicalize(&cli.workspace).unwrap_or_else(|_| cli.workspace.clone());
    // Windows canonicalize() returns \\?\C:\... ; strip it for readable output.
    let root = PathBuf::from(canonical.to_string_lossy().trim_start_matches(r"\\?\"));
    let grammar = if cli.no_grammar { None } else {
        Some(match &cli.grammar {
            Some(p) => fs::read_to_string(p).unwrap_or_else(|e| {
                eprintln!("cannot read grammar {}: {e}", p.display());
                GRAMMAR.to_string()
            }),
            None => GRAMMAR.to_string(),
        })
    };
    let client = Client::new(&cli.base_url, &cli.model, grammar);
    let mut app = App {
        client,
        root,
        max_rounds: cli.max_rounds,
        tui: !cli.cli,
        round: 0,
        history: Vec::new(),
    };

    if app.tui {
        print!("\x1b[?1049h\x1b[?25l");
        let _ = io::stdout().flush();
    }

    banner(&app);

    if let Some(prompt) = &cli.prompt {
        app.run_turn(prompt);
    } else {
        loop {
            let mut line = String::new();
            if app.tui { app.draw(""); }
            match io::stdin().read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {}
                Err(e) => { eprintln!("input error: {e}"); break; }
            }
            let cmd = line.trim();
            if cmd.is_empty() { continue; }
            if cmd.starts_with('/') {
                let (verb, rest) = cmd[1..].split_once(' ').unwrap_or((&cmd[1..], ""));
                match verb {
                    "quit" | "exit" => break,
                    "help" => println!("/help /quit /reset /model <name> /rounds <n> /tools /workspace"),
                    "reset" => { app.client.reset(); app.history.clear(); println!("conversation reset"); }
                    "model" => {
                        if !rest.is_empty() { app.client.model = rest.trim().to_string(); }
                        println!("model = {}", app.client.model);
                    }
                    "rounds" => {
                        if let Ok(n) = rest.trim().parse::<usize>() { app.max_rounds = n; }
                        println!("max rounds = {}", app.max_rounds);
                    }
                    "tools" => println!("{}", TOOLS.join(", ")),
                    "workspace" => println!("{}", app.root.display()),
                    _ => println!("unknown command: {verb}"),
                }
                continue;
            }
            app.run_turn(cmd);
        }
    }

    if app.tui {
        print!("\x1b[?25h\x1b[?1049l");
        let _ = io::stdout().flush();
    }
}

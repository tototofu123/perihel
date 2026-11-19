//! Perihel — local tool-calling agent harness for llama.cpp.
//!
//! llama.cpp has no native tool-call API, so tool calls are constrained with a
//! GBNF grammar (embedded in the binary). The model emits one JSON tool call
//! per turn; deterministic code executes it and feeds the observation back.
//! There is no fixed round limit: Ctrl+C stops at any time.

use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::enable_raw_mode;
use serde_json::{json, Value};
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

mod markdown;

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
- {"tool":"web.search","args":{"query":"latest news"}}  keyless web search (Bing RSS)
- {"tool":"web.fetch","args":{"url":"https://example.com"}}  fetch a page as readable text
- {"tool":"reply.text","args":{"text":"..."}}  finish the turn
Rules: one tool per reply; paths are relative to the workspace, no absolute paths or `..`;
inspect before writing, verify after writing; never claim success you did not verify."#;

const TOOLS: &[&str] = &[
    "fs.list", "fs.read", "fs.write", "fs.mkdir", "fs.copy",
    "run.cmd", "run.test", "web.search", "web.fetch", "reply.text",
];

const COMMANDS: &[(&str, &str)] = &[
    ("/help", "show this help"),
    ("/model [name]", "pick a model (no arg opens picker)"),
    ("/session", "browse and restore past sessions"),
    ("/thinking on|off", "toggle streamed reasoning"),
    ("/rounds <n>", "set max tool rounds (0 = unlimited)"),
    ("/tools", "list available tools"),
    ("/workspace", "show the sandbox root"),
    ("/reset", "clear the conversation"),
    ("/quit", "exit perihel"),
];

/// Commands matching what the user has typed so far. An empty query matches all.
fn command_matches(input: &str) -> Vec<(&'static str, &'static str)> {
    let q = input.trim_start();
    if !q.starts_with('/') { return Vec::new(); }
    COMMANDS.iter().filter(|(name, _)| name.starts_with(q)).take(6).copied().collect()
}

/// The bare command word, without the argument hint (`/model [name]` -> `/model`).
fn command_word(name: &str) -> &str {
    name.split(' ').next().unwrap_or(name)
}

#[derive(Parser, Debug)]
#[command(name = "perihel", version = "0.1.0",
          about = "Local tool-calling agent harness (llama.cpp + GBNF)")]
struct Cli {
    #[arg(long, default_value = "http://127.0.0.1:8090")]
    base_url: String,
    /// Model name. If omitted, the first available local model is selected.
    #[arg(long)]
    model: Option<String>,
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
    /// Full-screen harness UI. This is the default; the flag is accepted for
    /// backwards compatibility with the README and desktop shortcut.
    #[arg(long, hide = true)]
    tui: bool,
    /// Show the model's streamed reasoning (off by default; slow).
    #[arg(long)]
    thinking: bool,
}

// ------------------------------------------------------------------ client

struct Client {
    base: String,
    model: String,
    grammar: Option<String>,
    messages: Vec<Value>,
    agent: ureq::Agent,
    thinking: bool,
    last_stats: String,
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.len() <= max { return s.to_string(); }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) { end -= 1; }
    format!("{}\n... [truncated]", &s[..end])
}

fn stats_line(tokens: u64, chars: usize, elapsed: Duration) -> String {
    let secs = elapsed.as_secs_f64().max(0.001);
    let measured = if tokens > 0 { tokens } else { (chars / 4) as u64 };
    format!("[stats] {} tokens in {:.1}s = {:.1} tok/s{}",
        measured, secs, measured as f64 / secs,
        if tokens > 0 { "" } else { " (estimated)" })
}

fn print_stats(tokens: u64, chars: usize, elapsed: Duration) {
    println!("{}", stats_line(tokens, chars, elapsed));
}

impl Client {
    fn new(base: &str, model: &str, grammar: Option<String>, thinking: bool) -> Self {
        let mut messages = Vec::new();
        messages.push(json!({"role": "system", "content": DEFAULT_SYSTEM}));
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(900)))
            .build();
        let agent = ureq::Agent::new_with_config(config);
        Self { base: base.trim_end_matches('/').to_string(), model: model.to_string(), grammar, messages, agent, thinking, last_stats: String::new() }
    }

    fn reset(&mut self) {
        self.messages.clear();
        self.messages.push(json!({"role": "system", "content": DEFAULT_SYSTEM}));
    }

    /// Models available from llama.cpp's /v1/models plus local GGUF files.
    fn available_models(&self, root: &Path) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        if let Ok(resp) = self.agent.get(&format!("{}/v1/models", self.base)).call() {
            if let Ok(v) = resp.into_body().read_json::<Value>() {
                if let Some(arr) = v.get("data").and_then(|d| d.as_array()) {
                    for m in arr {
                        if let Some(id) = m.get("id").and_then(|x| x.as_str()) {
                            // llama.cpp reports the full GGUF path; show the name.
                            let short = id.rsplit(['\\', '/']).next().unwrap_or(id);
                            out.push(short.to_string());
                        }
                    }
                }
            }
        }
        let dirs = vec![root.to_path_buf(), root.join("models"), PathBuf::from("models")];
        for dir in dirs {
            if let Ok(entries) = fs::read_dir(dir) {
                for e in entries.flatten() {
                    let p = e.path();
                    if p.extension().and_then(|s| s.to_str()) == Some("gguf") {
                        let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
                        if !name.to_lowercase().contains("mmproj") { out.push(name); }
                    }
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }

    fn chat(&mut self, stream: bool) -> Result<String, String> {
        let started = Instant::now();
        let mut body = json!({
            "model": self.model,
            "messages": self.messages,
            "temperature": 0.4,
            "top_p": 0.95,
            "top_k": 20,
            // Cap reasoning: this model can loop for 30k tokens before answering.
            "max_tokens": if self.thinking { 8000 } else { 4096 },
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
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
            let msg = &text["choices"][0]["message"];
            let mut content = msg["content"].as_str().unwrap_or("").to_string();
            // Thinking models can put the answer in reasoning_content.
            if content.is_empty() {
                content = msg["reasoning_content"].as_str().unwrap_or("").to_string();
            }
            if content.is_empty() && self.thinking {
                // The reasoning loop ate the budget: retry once with it off.
                self.thinking = false;
                let retry = self.chat(stream);
                self.thinking = true;
                return retry;
            }
            if content.is_empty() {
                return Err("empty completion (reasoning may have consumed the budget)".into());
            }
            let toks = text["usage"]["completion_tokens"].as_u64().unwrap_or(0);
            self.last_stats = stats_line(toks, content.len(), started.elapsed());
            return Ok(content);
        }

        // SSE: stream reasoning and content deltas as they arrive (typewriter).
        let reader = resp.into_body().into_reader();
        let mut full = String::new();
        let mut reasoning = String::new();
        let mut tokens = 0u64;
        let mut thinking_header = false;
        for line in io::BufReader::new(reader).lines() {
            let line = line.map_err(|e| format!("stream read error: {e}"))?;
            if !line.starts_with("data:") { continue; }
            let data = line[5..].trim();
            if data == "[DONE]" { break; }
            let piece: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(t) = piece["usage"]["completion_tokens"].as_u64() { tokens = t; }
            let delta = &piece["choices"][0]["delta"];
            // The chat template puts thinking in reasoning_content.
            let reasoning_delta = delta["reasoning_content"].as_str().unwrap_or("");
            if !reasoning_delta.is_empty() {
                if !thinking_header {
                    print!("\n\x1b[2m[thinking] ");
                    thinking_header = true;
                }
                reasoning.push_str(reasoning_delta);
                print!("\x1b[2m{reasoning_delta}\x1b[0m");
                let _ = io::stdout().flush();
            }
            let content = delta["content"].as_str().unwrap_or("");
            if !content.is_empty() {
                if thinking_header {
                    print!("\x1b[0m\n");
                    thinking_header = false;
                }
                full.push_str(content);
                print!("{content}");
                let _ = io::stdout().flush();
            }
        }
        if thinking_header { print!("\x1b[0m"); }
        println!();
        // Some llama.cpp builds route the grammar-constrained tool call into
        // reasoning_content instead of content.
        let chars = if full.is_empty() { reasoning.len() } else { full.len() + reasoning.len() };
        if full.is_empty() { full = reasoning.clone(); }
        self.last_stats = stats_line(tokens, chars, started.elapsed());
        print_stats(tokens, chars, started.elapsed());
        Ok(full)
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

// ------------------------------------------------------------------ sessions

fn sessions_dir() -> PathBuf {
    if let Ok(p) = std::env::var("PERIHEL_SESSIONS_DIR") { return PathBuf::from(p); }
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".perihel").join("sessions")
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SessionFile {
    id: String,
    title: String,
    saved_at: String,
    messages: Vec<Value>,
}

fn save_session(client: &Client) -> Option<(PathBuf, SessionFile)> {
    if client.messages.len() <= 1 { return None; }
    let dir = sessions_dir();
    fs::create_dir_all(&dir).ok()?;
    let first_user = client.messages.iter()
        .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("user"))
        .and_then(|m| m.get("content").and_then(|v| v.as_str()))
        .unwrap_or("session");
    let title = first_user.chars().take(48).collect::<String>()
        + if first_user.chars().count() > 48 { "..." } else { "" };
    let id = client.messages.iter()
        .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("user"))
        .and_then(|m| m.get("id").and_then(|v| v.as_str()))
        .unwrap_or("current").to_string();
    let file = SessionFile {
        id: id.clone(), title,
        saved_at: chrono_stamp(),
        messages: client.messages.clone(),
    };
    let path = dir.join(format!("{}.json", sanitize_name(&id)));
    fs::write(&path, serde_json::to_string_pretty(&file).ok()?).ok()?;
    Some((path, file))
}

fn chrono_stamp() -> String {
    // Seconds since epoch is enough for a readable "saved at" label.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    format!("{secs}")
}

fn sanitize_name(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .take(60).collect()
}

fn list_sessions() -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(sessions_dir()) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") { continue; }
            if let Ok(text) = fs::read_to_string(&p) {
                if let Ok(s) = serde_json::from_str::<SessionFile>(&text) {
                    out.push((p, s.title));
                }
            }
        }
    }
    out.sort_by(|a, b| b.0.cmp(&a.0));
    out.truncate(30);
    out
}

fn load_session(client: &mut Client, path: &Path) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let file: SessionFile = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    client.messages = file.messages;
    Ok(())
}

// ------------------------------------------------------------------ web tools

#[derive(Default)]
struct WebRate { last: Option<Instant>, times: VecDeque<Instant> }
static WEB_RATE: OnceLock<Mutex<WebRate>> = OnceLock::new();

fn web_slot() -> Result<(), String> {
    let mut state = WEB_RATE.get_or_init(|| Mutex::new(WebRate::default()))
        .lock().map_err(|e| e.to_string())?;
    let now = Instant::now();
    state.times.retain(|t| now.duration_since(*t) < Duration::from_secs(60));
    if let Some(last) = state.last {
        if now.duration_since(last) < Duration::from_secs(5) {
            return Err("web rate limit: wait 5s between calls".into());
        }
    }
    if state.times.len() >= 10 {
        return Err("web rate limit: 10 calls per minute reached".into());
    }
    state.last = Some(now);
    state.times.push_back(now);
    Ok(())
}

fn html_unescape(s: &str) -> String {
    s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        .replace("&quot;", "\"").replace("&#39;", "'").replace("&nbsp;", " ")
}

fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.as_bytes() {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(*b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn http_get(url: &str, timeout_s: u64) -> Result<String, String> {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err("only http(s) URLs are allowed".into());
    }
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(timeout_s)))
        .build();
    let agent = ureq::Agent::new_with_config(config);
    let resp = agent.get(url)
        .header("User-Agent", "Mozilla/5.0 (compatible; perihel-local/0.1)")
        .call()
        .map_err(|e| format!("request failed: {e}"))?;
    let mut body = resp.into_body();
    let bytes = body.read_to_vec().map_err(|e| format!("read failed: {e}"))?;
    let limited = if bytes.len() > 2 * 1024 * 1024 { &bytes[..2 * 1024 * 1024] } else { &bytes };
    Ok(String::from_utf8_lossy(limited).to_string())
}

fn web_search(args: &Value) -> ToolResult {
    let query = match arg_str(args, "query") { Ok(q) => q, Err(e) => return ToolResult { ok: false, output: String::new(), refused: Some(e) } };
    if let Err(e) = web_slot() {
        return ToolResult { ok: false, output: String::new(), refused: Some(e) };
    }
    let url = format!("https://www.bing.com/search?format=rss&q={}", url_encode(query));
    let body = match http_get(&url, 15) {
        Ok(b) => b,
        Err(e) => return ToolResult { ok: false, output: e, refused: None },
    };
    let mut out = Vec::new();
    for item in body.split("<item>").skip(1) {
        let item = item.split("</item>").next().unwrap_or(item);
        let tag = |name: &str| -> String {
            let open = format!("<{name}>");
            let close = format!("</{name}>");
            item.split_once(&open)
                .and_then(|(_, rest)| rest.split_once(&close))
                .map(|(v, _)| html_unescape(v.trim().trim_start_matches("<![CDATA[").trim_end_matches("]]>").trim()))
                .unwrap_or_default()
        };
        let title = tag("title");
        let link = tag("link");
        let desc = tag("description");
        if !title.is_empty() {
            out.push(format!("- {title}\n  {link}\n  {}", truncate_chars(&desc, 300)));
        }
        if out.len() >= 8 { break; }
    }
    let text = if out.is_empty() { "no results (Bing RSS returned no items)".into() } else { out.join("\n") };
    ToolResult { ok: true, output: text, refused: None }
}

fn web_fetch(args: &Value) -> ToolResult {
    let url = match arg_str(args, "url") { Ok(u) => u, Err(e) => return ToolResult { ok: false, output: String::new(), refused: Some(e) } };
    if let Err(e) = web_slot() {
        return ToolResult { ok: false, output: String::new(), refused: Some(e) };
    }
    let html = match http_get(url, 15) {
        Ok(h) => h,
        Err(e) => return ToolResult { ok: false, output: e, refused: None },
    };
    let mut text = String::new();
    let mut skip = 0usize;
    let mut rest = html.as_str();
    while let Some(pos) = rest.find('<') {
        text.push_str(&html_unescape(&rest[..pos]));
        let after = &rest[pos..];
        if let Some(end) = after.find('>') {
            let tag = after[1..end].trim().to_ascii_lowercase();
            if tag.starts_with("script") || tag.starts_with("style") || tag.starts_with("nav")
                || tag.starts_with("header") || tag.starts_with("footer") || tag.starts_with("svg") {
                if !tag.starts_with('/') { skip += 1; }
            } else if tag.starts_with('/') { skip = skip.saturating_sub(1); }
            else if ["p", "h1", "h2", "h3", "li", "br", "div", "section", "article"].iter().any(|t| tag.starts_with(t)) {
                text.push('\n');
            }
            rest = &after[end + 1..];
        } else { break; }
    }
    text.push_str(rest);
    let collapsed: Vec<&str> = text.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
    let out = truncate_chars(&collapsed.join("\n"), 6000);
    let _ = skip;
    ToolResult { ok: true, output: out, refused: None }
}

struct ToolResult { ok: bool, output: String, refused: Option<String> }

impl ToolResult {
    fn message(&self) -> String {
        if let Some(r) = &self.refused {
            format!("TOOL RESULT: REFUSED — {r}\nPick a different tool or fix the arguments.")
        } else {
            format!("TOOL RESULT ({}):\n{}", if self.ok { "OK" } else { "FAILED" },
                    &truncate_chars(&self.output, 6000))
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
        "web.search" => wrap((|| {
            let r = web_search(args);
            Ok(r)
        })()),
        "web.fetch" => wrap((|| {
            let r = web_fetch(args);
            Ok(r)
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
    picking_model: bool,
    model_list: Vec<String>,
    model_cursor: usize,
    picking_session: bool,
    session_list: Vec<(PathBuf, String)>,
    session_cursor: usize,
    session_id: String,
    busy: bool,
    /// Lines scrolled up from the bottom of the conversation.
    scroll: usize,
    /// Highlighted row in the slash-command menu.
    suggest_cursor: usize,
}

impl App {
    fn add(&mut self, kind: &str, text: &str) {
        self.history.push((kind.to_string(), text.to_string()));
        self.scroll = 0; // snap back to the newest line
        if self.tui { self.draw(""); }
    }

    fn save_current_session(&mut self) {
        for m in &mut self.client.messages {
            if m.get("role").and_then(|v| v.as_str()) == Some("user") && m.get("id").is_none() {
                m["id"] = json!(self.session_id);
            }
        }
        if let Some((path, _)) = save_session(&self.client) {
            if !self.tui { println!("[session] saved {}", path.display()); }
        }
    }

    /// Print a message that survives the full-screen redraw in TUI mode.
    fn inform(&mut self, text: &str) {
        if self.tui { self.add("system", text); }
        else { println!("{text}"); }
    }

    fn run_turn(&mut self, user: &str) {
        self.client.messages.push(json!({"role":"user","content":user}));
        self.add("user", user);
        let mut round = 0;
        loop {
            round += 1;
            self.round = round;
            if self.tui { self.draw(""); } else { println!("[round {}] thinking...", round); }
            self.busy = true;
            if self.tui { self.draw(""); }
            let raw = match self.client.chat(!self.tui) {
                Ok(r) => r,
                Err(e) => { self.busy = false; self.add("error", &e); return; }
            };
            self.busy = false;
            let call = match parse_call(&raw) {
                Some(c) => c,
                None => {
                    if !self.tui { println!("[assistant] {}", raw.trim()); }
                    self.add("assistant", raw.trim());
                    return;
                }
            };
            self.client.messages.push(json!({"role":"assistant","content":raw}));
            let name = call["tool"].as_str().unwrap_or("").to_string();
            if name == "reply.text" {
                let text = call["args"]["text"].as_str().unwrap_or("").to_string();
                if !self.tui { println!("[assistant] {text}"); }
                self.add("assistant", &text);
                return;
            }
            let result = execute(&call, &self.root);
            let preview = result.refused.clone().unwrap_or_else(|| result.output.lines().next().unwrap_or("").to_string());
            if !self.tui {
                println!("[tool] {name} -> {}", truncate_chars(&preview, 160));
            }
            self.add("tool", &format!("{name} -> {}", truncate_chars(&preview, 160)));
            self.client.messages.push(json!({"role":"user","content":result.message()}));
            if self.max_rounds > 0 && round >= self.max_rounds {
                self.add("error", "round limit reached (--max-rounds)");
                return;
            }
        }
    }

    // Minimal full-screen harness-style rendering with ANSI escapes.
    fn draw_picker(&self) {
        let (cols, _) = term_size();
        let mut out = String::from("\x1b[H\x1b[2J");
        out.push_str(&format!("\x1b[48;5;24m\x1b[97m SELECT MODEL \x1b[0m  {}\x1b[0m\n", self.client.model));
        out.push_str(&format!("{}\n", "-".repeat(cols.saturating_sub(1))));
        out.push_str("  arrows / j k : move   number : jump   Enter : select   q : cancel\n\n");
        for (i, m) in self.model_list.iter().enumerate() {
            let ptr = if i == self.model_cursor { "\x1b[32m>\x1b[0m" } else { " " };
            let name = if m == &self.client.model {
                format!("{m}  \x1b[2m(current)\x1b[0m")
            } else { m.clone() };
            out.push_str(&format!("  {ptr} [{}] {}\n", i, name));
        }
        out.push_str(&format!("\n{}\n", "-".repeat(cols)));
        out.push_str("\x1b[2K\x1b[32m>\x1b[0m ");
        print!("{out}");
        let _ = io::stdout().flush();
    }

    fn draw_session_picker(&self) {
        let (cols, _) = term_size();
        let mut out = String::from("\x1b[H\x1b[2J");
        out.push_str("\x1b[48;5;24m\x1b[97m SESSIONS \x1b[0m  recent conversations\x1b[0m\n");
        out.push_str(&format!("{}\n", "-".repeat(cols.saturating_sub(1))));
        out.push_str("  arrows / j k : move   number : jump   Enter : restore   q : cancel\n\n");
        if self.session_list.is_empty() {
            out.push_str("  (no saved sessions yet - chat a little first)\n");
        }
        for (i, (_, title)) in self.session_list.iter().enumerate() {
            let ptr = if i == self.session_cursor { "\x1b[32m>\x1b[0m" } else { " " };
            out.push_str(&format!("  {ptr} [{}] {}\n", i, title));
        }
        out.push_str(&format!("\n{}\n", "-".repeat(cols)));
        out.push_str("\x1b[2K\x1b[32m>\x1b[0m ");
        print!("{out}");
        let _ = io::stdout().flush();
    }

    fn draw(&self, input: &str) {
        self.draw_cursor(input, input.len());
    }

    fn draw_cursor(&self, input: &str, cursor: usize) {
        let (cols, rows) = term_size();
        let out = self.frame_with_size(input, cursor, cols, rows);
        print!("{out}");
        let _ = io::stdout().flush();
    }

    /// Build one full screen frame. Kept separate from drawing so layout can be
    /// unit-tested at a fixed size.
    fn frame_with_size(&self, input: &str, cursor: usize, cols: usize, rows: usize) -> String {
        let q = input.trim_start();
        let suggestions = command_matches(q);
        let sel = self.suggest_cursor.min(suggestions.len().saturating_sub(1));
        let extra = if suggestions.is_empty() { 0 } else { suggestions.len() + 2 };
        // Reserve header, separators, status bar, input line and suggestions.
        let body_h = rows.saturating_sub(6 + extra).max(1);
        let mut out = String::new();
        // Autowrap off while drawing: every row is pre-wrapped, and a line that
        // ends exactly at the edge must not push the rest of the frame down.
        out.push_str("\x1b[?7l\x1b[0m\x1b[2J\x1b[H");
        let header = format!(" PERIHEL  local agent harness  model:{}  grammar:{}  tools:{}",
            &self.client.model, if self.client.grammar.is_some() {"ON"} else {"OFF"}, TOOLS.len());
        out.push_str(&format!("\x1b[48;5;24m\x1b[97m{}\x1b[0m\n", clip_plain(&header, cols.saturating_sub(1))));
        out.push_str(&format!("{}\n", "-".repeat(cols.saturating_sub(1))));
        let mut lines: Vec<String> = Vec::new();
        for (kind, text) in &self.history {
            let prefix = match kind.as_str() {
                "user" => "\x1b[32m>\x1b[0m ",
                "assistant" => "\x1b[36m*\x1b[0m ",
                "tool" => "\x1b[33m>\x1b[0m ",
                "error" => "\x1b[31m!\x1b[0m ",
                _ => "\x1b[34mi\x1b[0m ",
            };
            let body: Vec<String> = if kind == "assistant" {
                markdown::render(text, cols.saturating_sub(3))
            } else {
                wrap(text, cols.saturating_sub(3))
            };
            for (i, seg) in body.iter().enumerate() {
                lines.push(format!("{}{}", if i == 0 { prefix } else { "  " }, seg));
            }
        }
        if self.busy {
            lines.push("\x1b[33m*\x1b[0m generating... (Ctrl+C stops)".to_string());
        }
        let max_scroll = lines.len().saturating_sub(body_h);
        let scroll = self.scroll.min(max_scroll);
        let end = lines.len() - scroll;
        let start = end.saturating_sub(body_h);
        for line in &lines[start..end] {
            out.push_str(line);
            out.push_str("\x1b[0K\n"); // erase to end of line, keeping the text
        }
        for _ in 0..body_h.saturating_sub(end - start) {
            out.push_str("\x1b[2K\n");
        }
        out.push_str(&format!("{}\n", "-".repeat(cols.saturating_sub(1))));
        if !suggestions.is_empty() {
            out.push_str(&format!("{}\n", clip_ansi("\x1b[1m  commands\x1b[0m\x1b[2m   up/down select, tab complete, enter run\x1b[0m", cols)));
            for (i, (name, desc)) in suggestions.iter().enumerate() {
                let ptr = if i == sel { "\x1b[32m>\x1b[0m" } else { " " };
                let label = if i == sel {
                    format!("\x1b[1;36m{name}\x1b[0m")
                } else {
                    format!("\x1b[36m{name}\x1b[0m")
                };
                let row = format!("  {ptr} {label}  \x1b[2m{desc}\x1b[0m");
                out.push_str(&format!("{}\x1b[0K\n", clip_ansi(&row, cols)));
            }
            out.push_str(&format!("{}\x1b[0K\n", clip_ansi("\x1b[2m  Tab completes, Enter runs\x1b[0m", cols)));
        }
        let cap = if self.max_rounds == 0 { "inf".into() } else { self.max_rounds.to_string() };
        // Keep the status bar to one row: drop optional pieces on narrow terminals.
        let mut bar = if scroll > 0 {
            format!(" SCROLLED +{scroll} lines  PgDn: latest  PgUp: older")
        } else {
            String::from(" Enter send | /model | /session | /thinking | /quit | Ctrl+C")
        };
        let round = format!(" | round {}/{}", self.round, cap);
        if disp_width_in(&bar) + disp_width_in(&round) + 2 <= cols { bar.push_str(&round); }
        if !self.client.last_stats.is_empty() {
            let stats = format!(" | {}", self.client.last_stats);
            if disp_width_in(&bar) + disp_width_in(&stats) + 2 <= cols { bar.push_str(&stats); }
        }
        out.push_str(&format!("\x1b[48;5;236m\x1b[97m{}\x1b[0m\n", clip_plain(&bar, cols.saturating_sub(1))));
        out.push_str("\x1b[2K\x1b[32m>\x1b[0m "); // clear the input line first
        // Horizontal scroll so a long line never wraps into the status bar.
        let c = cursor.min(input.len());
        let chars: Vec<char> = input.chars().collect();
        let cidx = input[..c].chars().count();
        let avail = cols.saturating_sub(5).max(8);
        let start = if cidx >= avail { cidx + 1 - avail } else { 0 };
        let end = (start + avail).min(chars.len());
        if start > 0 { out.push_str("\x1b[2m<\x1b[0m"); }
        for (i, ch) in chars[start..end].iter().enumerate() {
            let idx = start + i;
            if idx == cidx {
                out.push_str(&format!("\x1b[7m{ch}\x1b[0m"));
            } else {
                out.push(*ch);
            }
        }
        if cidx >= end { out.push_str("\x1b[7m \x1b[0m"); }
        out.push_str("\x1b[?7h");
        out
    }
}

fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(16).saturating_sub(2);
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() { out.push(String::new()); continue; }
        let mut cur = String::new();
        let mut cur_w = 0usize;
        for word in line.split_whitespace() {
            let word_w: usize = word.chars().map(char_width).sum();
            if word_w > width {
                if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
                let mut piece = String::new();
                let mut piece_w = 0usize;
                for c in word.chars() {
                    let cw = char_width(c);
                    if piece_w + cw > width {
                        out.push(std::mem::take(&mut piece));
                        piece_w = 0;
                    }
                    piece.push(c);
                    piece_w += cw;
                }
                cur = piece;
                cur_w = piece_w;
                continue;
            }
            let extra = if cur.is_empty() { 0 } else { 1 };
            if cur_w + extra + word_w > width {
                out.push(std::mem::take(&mut cur));
                cur.push_str(word);
                cur_w = word_w;
            } else {
                if !cur.is_empty() { cur.push(' '); }
                cur.push_str(word);
                cur_w += extra + word_w;
            }
        }
        if !cur.is_empty() { out.push(cur); }
    }
    if out.is_empty() { out.push(String::new()); }
    out
}

#[derive(Debug)]
enum Key {
    Char(char, bool), Enter, Left(bool), Right(bool), Up, Down, Backspace, BackspaceWord,
    Delete, Home, End, Esc, PageUp, PageDown, Other,
}

/// Read one key in raw mode. Raw mode is disabled again before returning, so
/// Ctrl+C during a generation terminates the process normally.
fn read_key() -> Key {
    if enable_raw_mode().is_err() {
        let mut line = String::new();
        return match io::stdin().read_line(&mut line) {
            Ok(0) => Key::Esc,
            Ok(_) => Key::Char(line.chars().next().unwrap_or('\n'), false),
            Err(_) => Key::Esc,
        };
    }
    loop {
        if let Ok(Event::Key(k)) = event::read() {
            if k.kind != KeyEventKind::Press { continue; }
            let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
            let word = ctrl || k.modifiers.contains(KeyModifiers::ALT);
            let key = match k.code {
                KeyCode::Enter => Key::Enter,
                KeyCode::Left => Key::Left(word),
                KeyCode::Right => Key::Right(word),
                KeyCode::Up => Key::Up,
                KeyCode::Down => Key::Down,
                KeyCode::Backspace if ctrl => Key::BackspaceWord,
                KeyCode::Backspace => Key::Backspace,
                KeyCode::Delete => Key::Delete,
                KeyCode::Home => Key::Home,
                KeyCode::End => Key::End,
                KeyCode::Esc => Key::Esc,
                KeyCode::PageUp => Key::PageUp,
                KeyCode::PageDown => Key::PageDown,
                KeyCode::Char(c) => Key::Char(c, ctrl),
                _ => Key::Other,
            };
            let _ = crossterm::terminal::disable_raw_mode();
            return key;
        }
    }
}

fn prev_boundary(s: &str, pos: usize) -> usize {
    s[..pos].char_indices().next_back().map(|(i, _)| i).unwrap_or(0)
}

fn next_boundary(s: &str, pos: usize) -> usize {
    if pos >= s.len() { return s.len(); }
    s[pos..].char_indices().nth(1).map(|(i, _)| pos + i).unwrap_or(s.len())
}

fn is_space_at(s: &str, start: usize, end: usize) -> bool {
    s[start..end].chars().all(char::is_whitespace)
}

/// Start of the word before `pos`, skipping the run of whitespace first.
fn prev_word(s: &str, pos: usize) -> usize {
    let mut i = pos.min(s.len());
    while i > 0 {
        let p = prev_boundary(s, i);
        if is_space_at(s, p, i) { i = p } else { break }
    }
    while i > 0 {
        let p = prev_boundary(s, i);
        if is_space_at(s, p, i) { break }
        i = p;
    }
    i
}

/// End of the word at or after `pos`, skipping leading whitespace.
fn next_word(s: &str, pos: usize) -> usize {
    let mut i = pos.min(s.len());
    while i < s.len() {
        let n = next_boundary(s, i);
        if is_space_at(s, i, n) { i = n } else { break }
    }
    while i < s.len() {
        let n = next_boundary(s, i);
        if is_space_at(s, i, n) { break }
        i = n;
    }
    i
}

/// Prompts typed during this session, oldest first. Up/Down walk it when the
/// slash menu is not showing, like a shell.
fn prompt_history() -> &'static Mutex<Vec<String>> {
    static HISTORY: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    HISTORY.get_or_init(|| Mutex::new(Vec::new()))
}

fn remember_prompt(line: &str) {
    let line = line.trim();
    if line.is_empty() { return; }
    let mut hist = prompt_history().lock().unwrap();
    if hist.last().map(|last| last == line).unwrap_or(false) { return; }
    hist.push(line.to_string());
    if hist.len() > 200 { hist.remove(0); }
}

fn term_size() -> (usize, usize) {
    // Ask the real console first. COLUMNS/LINES are not set on Windows, which
    // used to make us lay out for 120 columns inside a narrow window: lines
    // soft-wrapped, every row count was wrong, and text got clipped.
    if let Ok((c, r)) = crossterm::terminal::size() {
        if c >= 20 && r >= 8 { return (c as usize, r as usize); }
    }
    let cols = std::env::var("COLUMNS").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
    let rows = std::env::var("LINES").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    (cols, rows)
}

/// Display columns used by a char (CJK and fullwidth forms take two).
pub(crate) fn char_width(c: char) -> usize {
    match c as u32 {
        0x1100..=0x115F | 0x2E80..=0x303E | 0x3041..=0x33FF | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF | 0xA000..=0xA4CF | 0xAC00..=0xD7A3 | 0xF900..=0xFAFF
        | 0xFE30..=0xFE6F | 0xFF00..=0xFF60 | 0xFFE0..=0xFFE6
        | 0x1F300..=0x1F64F | 0x1F900..=0x1F9FF | 0x20000..=0x3FFFD => 2,
        _ => 1,
    }
}

fn disp_width_in(s: &str) -> usize {
    let mut w = 0usize;
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\x1b' {
            // Skip a CSI sequence: ESC [ params... final-byte(@..~).
            if it.peek() == Some(&'[') {
                it.next();
                for d in it.by_ref() {
                    if ('\x40'..='\x7e').contains(&d) { break; }
                }
            }
            continue;
        }
        w += char_width(c);
    }
    w
}

/// Clip plain (no ANSI) text to at most `cols` display columns.
fn clip_plain(s: &str, cols: usize) -> String {
    let mut out = String::new();
    let mut w = 0;
    for c in s.chars() {
        let cw = char_width(c);
        if w + cw > cols { break; }
        out.push(c);
        w += cw;
    }
    out
}

/// Clip text that contains ANSI escapes, keeping the escapes intact.
fn clip_ansi(s: &str, cols: usize) -> String {
    let mut out = String::new();
    let mut w = 0usize;
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\x1b' {
            out.push(c);
            if it.peek() == Some(&'[') {
                out.push(it.next().unwrap_or('['));
                for d in it.by_ref() {
                    out.push(d);
                    if ('\x40'..='\x7e').contains(&d) { break; }
                }
            }
            continue;
        }
        let cw = char_width(c);
        if w + cw > cols { break; }
        out.push(c);
        w += cw;
    }
    out.push_str("\x1b[0m");
    out
}

fn banner(app: &App) {
    if app.tui { return; }
    println!("perihel 0.1.0 | model={} | rounds={} | grammar={} | workspace={}",
        app.client.model,
        if app.max_rounds == 0 { "unlimited".into() } else { app.max_rounds.to_string() },
        if app.client.grammar.is_some() { "ON" } else { "OFF" },
        app.root.display());
    println!("Type /help for commands, Ctrl+C to stop. Thinking: {}",
        if app.client.thinking { "ON" } else { "OFF" });
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
    let mut client = Client::new(
        &cli.base_url,
        cli.model.as_deref().unwrap_or("local-model"),
        grammar,
        cli.thinking,
    );
    // No --model given: pick the first real model instead of a placeholder.
    if cli.model.is_none() {
        if let Some(first) = client.available_models(&root).into_iter().next() {
            client.model = first;
        }
    }
    let mut app = App {
        client,
        root,
        max_rounds: cli.max_rounds,
        tui: !cli.cli,
        round: 0,
        history: Vec::new(),
        picking_model: false,
        model_list: Vec::new(),
        model_cursor: 0,
        picking_session: false,
        session_list: Vec::new(),
        session_cursor: 0,
        session_id: format!("session-{}", chrono_stamp()),
        busy: false,
        scroll: 0,
        suggest_cursor: 0,
    };

    if app.tui {
        print!("\x1b[?1049h\x1b[?25l");
        let _ = io::stdout().flush();
    }

    banner(&app);

    if let Some(prompt) = &cli.prompt {
        app.run_turn(prompt);
        app.save_current_session();
    } else {
        'outer: loop {
            let cmd: String;
            if app.tui {
                if app.picking_session {
                    app.draw_session_picker();
                    match read_key() {
                        Key::Down | Key::Char('j', false) => {
                            let n = app.session_list.len().saturating_sub(1);
                            app.session_cursor = (app.session_cursor + 1).min(n);
                        }
                        Key::Up | Key::Char('k', false) => {
                            app.session_cursor = app.session_cursor.saturating_sub(1);
                        }
                        Key::Char(c, false) if c.is_ascii_digit() => {
                            let i = c.to_digit(10).unwrap_or(0) as usize;
                            if i < app.session_list.len() { app.session_cursor = i; }
                        }
                        Key::Enter => {
                            if let Some((path, _)) = app.session_list.get(app.session_cursor).cloned() {
                                match load_session(&mut app.client, &path) {
                                    Ok(()) => app.inform("session restored"),
                                    Err(e) => app.inform(&format!("session load failed: {e}")),
                                }
                            }
                            app.picking_session = false;
                        }
                        Key::Esc | Key::Char('q', false) => {
                            app.picking_session = false;
                            app.inform("session browse cancelled");
                        }
                        Key::Char('c', true) => break 'outer,
                        _ => {}
                    }
                    continue;
                }
                if app.picking_model {
                    app.draw_picker();
                    match read_key() {
                        Key::Down | Key::Char('j', false) => {
                            let n = app.model_list.len().saturating_sub(1);
                            app.model_cursor = (app.model_cursor + 1).min(n);
                        }
                        Key::Up | Key::Char('k', false) => {
                            app.model_cursor = app.model_cursor.saturating_sub(1);
                        }
                        Key::Char(c, false) if c.is_ascii_digit() => {
                            let i = c.to_digit(10).unwrap_or(0) as usize;
                            if i < app.model_list.len() {
                                app.model_cursor = i;
                                app.client.model = app.model_list[i].clone();
                                app.picking_model = false;
                                app.inform(&format!("model = {}", app.client.model));
                            }
                        }
                        Key::Enter => {
                            if let Some(m) = app.model_list.get(app.model_cursor).cloned() {
                                app.client.model = m;
                            }
                            app.picking_model = false;
                            app.inform(&format!("model = {}", app.client.model));
                        }
                        Key::Esc | Key::Char('q', false) => {
                            app.picking_model = false;
                            app.inform("model selection cancelled");
                        }
                        Key::Char('c', true) => break 'outer,
                        _ => {}
                    }
                    continue;
                }

                // Full-screen line editor with a visible block cursor.
                let mut buf = String::new();
                let mut cursor = 0usize;
                // Shell-style history walk: None means "editing a fresh line".
                let mut hist_pos: Option<usize> = None;
                let mut draft = String::new();
                app.suggest_cursor = 0;
                loop {
                    app.draw_cursor(&buf, cursor);
                    match read_key() {
                        Key::Char('c', true) => break 'outer,
                        Key::Down => {
                            let matches = command_matches(&buf);
                            if matches.is_empty() {
                                if let Some(i) = hist_pos {
                                    let next = {
                                        let hist = prompt_history().lock().unwrap();
                                        if i + 1 < hist.len() { Some(hist[i + 1].clone()) } else { None }
                                    };
                                    match next {
                                        Some(line) => { hist_pos = Some(i + 1); buf = line; }
                                        None => { hist_pos = None; buf = draft.clone(); }
                                    }
                                    cursor = buf.len();
                                }
                            } else {
                                let n = matches.len().saturating_sub(1);
                                app.suggest_cursor = (app.suggest_cursor + 1).min(n);
                            }
                        }
                        Key::Up => {
                            let matches = command_matches(&buf);
                            if matches.is_empty() {
                                let target = {
                                    let hist = prompt_history().lock().unwrap();
                                    match hist_pos {
                                        None if !hist.is_empty() => Some(hist.len() - 1),
                                        Some(0) => Some(0),
                                        Some(i) => Some(i - 1),
                                        None => None,
                                    }
                                    .map(|i| (i, hist[i].clone()))
                                };
                                if let Some((i, line)) = target {
                                    if hist_pos.is_none() { draft = buf.clone(); }
                                    hist_pos = Some(i);
                                    buf = line;
                                    cursor = buf.len();
                                }
                            } else {
                                app.suggest_cursor = app.suggest_cursor.saturating_sub(1);
                            }
                        }
                        Key::Char('\t', false) => {
                            let matches = command_matches(&buf);
                            if let Some((name, _)) = matches.get(app.suggest_cursor.min(matches.len().saturating_sub(1))) {
                                buf = format!("{} ", command_word(name));
                                cursor = buf.len();
                            }
                        }
                        Key::Char('w', true) | Key::BackspaceWord if cursor > 0 => {
                            let start = prev_word(&buf, cursor);
                            buf.replace_range(start..cursor, "");
                            cursor = start;
                            app.suggest_cursor = 0;
                        }
                        Key::Char('u', true) => {
                            buf.replace_range(..cursor, "");
                            cursor = 0;
                            app.suggest_cursor = 0;
                        }
                        Key::Char('k', true) => {
                            buf.truncate(cursor);
                            app.suggest_cursor = 0;
                        }
                        Key::Char('a', true) => cursor = 0,
                        Key::Char('e', true) => cursor = buf.len(),
                        Key::Char(ch, false) => {
                            buf.insert(cursor, ch);
                            cursor += ch.len_utf8();
                            app.suggest_cursor = 0;
                            hist_pos = None;
                        }
                        Key::Backspace if cursor > 0 => {
                            cursor = prev_boundary(&buf, cursor);
                            buf.remove(cursor);
                            app.suggest_cursor = 0;
                        }
                        Key::Delete if cursor < buf.len() => {
                            buf.remove(cursor);
                            app.suggest_cursor = 0;
                        }
                        Key::Left(word) => {
                            cursor = if word { prev_word(&buf, cursor) } else { prev_boundary(&buf, cursor) };
                        }
                        Key::Right(word) => {
                            cursor = if word { next_word(&buf, cursor) } else { next_boundary(&buf, cursor) };
                        }
                        Key::Home => cursor = 0,
                        Key::End => cursor = buf.len(),
                        Key::PageUp => {
                            let (_, rows) = term_size();
                            let step = rows.saturating_sub(8).max(1);
                            app.scroll = (app.scroll + step).min(10_000);
                        }
                        Key::PageDown => {
                            let (_, rows) = term_size();
                            let step = rows.saturating_sub(8).max(1);
                            app.scroll = app.scroll.saturating_sub(step);
                        }
                        Key::Enter => {
                            // Enter runs the highlighted command (like opencode/omp).
                            let matches = command_matches(&buf);
                            if !matches.is_empty() {
                                let i = app.suggest_cursor.min(matches.len() - 1);
                                buf = command_word(matches[i].0).to_string();
                            }
                            break;
                        }
                        Key::Esc => { buf.clear(); cursor = 0; app.suggest_cursor = 0; }
                        _ => {}
                    }
                }
                cmd = buf;
            } else {
                let mut line = String::new();
                match io::stdin().read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(e) => { eprintln!("input error: {e}"); break; }
                }
                cmd = line.trim().to_string();
            }

            // Model picker: arrows/j-k move, digits jump, Enter selects, q cancels.
            if cmd.is_empty() { continue; }
            remember_prompt(&cmd);
            if cmd.starts_with('/') {
                let (verb, rest) = cmd[1..].split_once(' ').unwrap_or((&cmd[1..], ""));
                match verb {
                    "quit" | "exit" => break,
                    "help" => app.inform("/help /quit /reset /model [name] /rounds <n> /thinking on|off /tools /workspace"),
                    "reset" => {
                        app.client.reset();
                        app.history.clear();
                        app.inform("conversation reset");
                    }
                    "model" => {
                        if !rest.is_empty() {
                            app.client.model = rest.trim().to_string();
                            app.inform(&format!("model = {}", app.client.model));
                        } else {
                            app.model_list = app.client.available_models(&app.root);
                            app.model_cursor = app.model_list.iter()
                                .position(|m| m == &app.client.model).unwrap_or(0);
                            if app.model_list.is_empty() {
                                app.inform("no models found; add a .gguf to a models/ folder");
                            } else {
                                app.picking_model = true;
                                if app.tui { app.draw_picker(); }
                            }
                        }
                    }
                    "session" => {
                        if !rest.trim().is_empty() {
                            let sessions = list_sessions();
                            if let Ok(i) = rest.trim().parse::<usize>() {
                                if let Some((path, _)) = sessions.get(i) {
                                    match load_session(&mut app.client, path) {
                                        Ok(()) => app.inform("session restored"),
                                        Err(e) => app.inform(&format!("session load failed: {e}")),
                                    }
                                } else { app.inform("session index out of range"); }
                            }
                            continue;
                        }
                        app.session_list = list_sessions();
                        app.session_cursor = 0;
                        if app.session_list.is_empty() {
                            app.inform("no saved sessions yet");
                        } else {
                            app.picking_session = true;
                            if app.tui { app.draw_session_picker(); }
                            else {
                                for (i, (_, title)) in app.session_list.iter().enumerate() {
                                    println!("  [{i}] {title}");
                                }
                                app.inform("use /session <n> in a TUI, or pick with arrows there");
                            }
                        }
                    }
                    "thinking" => {
                        match rest.trim() {
                            "on" => app.client.thinking = true,
                            "off" => app.client.thinking = false,
                            _ => {}
                        }
                        app.inform(&format!("thinking output = {}", if app.client.thinking { "ON" } else { "OFF" }));
                    }
                    "rounds" => {
                        if let Ok(n) = rest.trim().parse::<usize>() { app.max_rounds = n; }
                        app.inform(&format!("max rounds = {}", app.max_rounds));
                    }
                    "tools" => app.inform(&TOOLS.join(", ")),
                    "workspace" => app.inform(&app.root.display().to_string()),
                    _ => app.inform(&format!("unknown command: {verb}")),
                }
                continue;
            }
            app.run_turn(&cmd);
            app.save_current_session();
        }
    }

    if app.tui {
        print!("\x1b[?25h\x1b[?1049l");
        let _ = io::stdout().flush();
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn wrapped_lines_never_exceed_width() {
        let text = "Glad to hear it! Is there a specific coding task or project you would like help with today? Let me know and I will take a look. 日本語のテキストも混ざる場合があります。";
        for w in 16..=120 {
            for line in wrap(text, w) {
                assert!(disp_width_in(&line) <= w, "wrap({w}) produced {line:?}");
            }
        }
    }


    #[test]
    fn frame_fits_small_windows() {
        let app = App {
            client: Client::new("http://127.0.0.1:1", "tiny-model", None, false),
            root: PathBuf::from("."),
            max_rounds: 0,
            tui: true,
            round: 1,
            history: vec![
                ("user".to_string(), "hello there, can you explain how the wrapping works in this terminal user interface and why wide characters matter?".to_string()),
                ("assistant".to_string(), "Sure. The frame is rebuilt on every keystroke and every line is pre-wrapped to the terminal width so that nothing soft-wraps or gets clipped at the right edge, including CJK text like 日本語のテキスト。".to_string()),
                ("tool".to_string(), "web.search -> a fairly long tool result line that also needs to wrap cleanly without spilling over".to_string()),
            ],
            picking_model: false,
            model_list: vec![],
            model_cursor: 0,
            picking_session: false,
            session_list: vec![],
            session_cursor: 0,
            session_id: "t".to_string(),
            busy: false,
            scroll: 0,
            suggest_cursor: 0,
        };
        for (cols, rows) in [(30usize, 10usize), (40, 12), (64, 24), (120, 40)] {
            let frame = app.frame_with_size("/mo", 3, cols, rows);
            let mut used = 0;
            for line in frame.split('\n') {
                assert!(disp_width_in(line) <= cols, "cols={cols}: {line:?}");
                used += 1;
            }
            assert!(used <= rows, "cols={cols} used={used} rows={rows}");
        }
    }


    #[test]
    fn markdown_reply_fits_narrow_windows() {
        let app = App {
            client: Client::new("http://127.0.0.1:1", "tiny-model", None, false),
            root: PathBuf::from("."),
            max_rounds: 0,
            tui: true,
            round: 1,
            history: vec![(
                "assistant".to_string(),
                "## Result\n\nReplaced `wrap()` with an **ANSI-aware** renderer.\n\n```rust\nfn main() {\n    let answer: u32 = 42; // keep going\n    println!(\"answer = {answer}\");\n}\n```\n\n- tests pass\n- one long bullet that must wrap rather than spill past the right edge of a narrow terminal\n\n> links: [docs](https://example.com/a/long/path)\n".to_string(),
            )],
            picking_model: false,
            model_list: vec![],
            model_cursor: 0,
            picking_session: false,
            session_list: vec![],
            session_cursor: 0,
            session_id: "t".to_string(),
            busy: false,
            scroll: 0,
            suggest_cursor: 0,
        };
        for (cols, rows) in [(40usize, 30usize), (64, 30)] {
            let frame = app.frame_with_size("", 0, cols, rows);
            for line in frame.split('\n') {
                assert!(disp_width_in(line) <= cols, "cols={cols}: {line:?}");
            }
        }
    }

    #[test]
    fn word_motion_respects_boundaries() {
        let s = "hello  world, again";
        assert_eq!(next_word(s, 0), 5);
        assert_eq!(next_word(s, 5), 13);
        assert_eq!(next_word(s, s.len()), s.len());
        assert_eq!(prev_word(s, s.len()), 14);
        assert_eq!(prev_word(s, 13), 7);
        assert_eq!(prev_word(s, 5), 0);
        assert_eq!(prev_word(s, 0), 0);

        // UTF-8: never split a codepoint when jumping words.
        let cjk = "你好 世界";
        assert_eq!(next_word(cjk, 0), 6);
        assert_eq!(prev_word(cjk, cjk.len()), 7);
        assert!(cjk.is_char_boundary(next_word(cjk, 0)));
        assert!(cjk.is_char_boundary(prev_word(cjk, cjk.len())));
    }

    #[test]
    fn assistant_rows_are_markdown_but_tool_rows_are_plain() {
        let app = App {
            client: Client::new("http://127.0.0.1:1", "tiny-model", None, false),
            root: PathBuf::from("."),
            max_rounds: 0,
            tui: true,
            round: 1,
            history: vec![
                ("assistant".to_string(), "Use `cargo test` and **ship** it.\n\n```\nlet x = 1;\n```".to_string()),
                ("tool".to_string(), "web.search -> `quoted` and **stars** stay raw".to_string()),
            ],
            picking_model: false,
            model_list: vec![],
            model_cursor: 0,
            picking_session: false,
            session_list: vec![],
            session_cursor: 0,
            session_id: "t".to_string(),
            busy: false,
            scroll: 0,
            suggest_cursor: 0,
        };
        let frame = app.frame_with_size("", 0, 64, 20);
        let assistant_part: String = frame.lines().filter(|l| !l.contains("stay raw")).collect::<Vec<_>>().join("\n");
        assert!(frame.contains("\x1b[48;5;236m"), "code block background missing");
        assert!(frame.contains("\x1b[38;5;222m"), "inline code colour missing");
        assert!(!assistant_part.contains("```"), "fences leaked into the frame");
        assert!(frame.contains("`quoted`"), "tool rows must stay raw");
    }

    #[test]
    fn body_text_is_never_erased() {
        let app = App {
            client: Client::new("http://127.0.0.1:1", "tiny-model", None, false),
            root: PathBuf::from("."),
            max_rounds: 0,
            tui: true,
            round: 1,
            history: vec![("user".to_string(), "hello there".to_string())],
            picking_model: false,
            model_list: vec![],
            model_cursor: 0,
            picking_session: false,
            session_list: vec![],
            session_cursor: 0,
            session_id: "t".to_string(),
            busy: false,
            scroll: 0,
            suggest_cursor: 0,
        };
        let frame = app.frame_with_size("", 0, 64, 24);
        let line = frame.lines().find(|l| l.contains("hello there"))
            .expect("user message missing from frame");
        // ESC[2K erases the whole row, including the text just written.
        assert!(!line.contains("\x1b[2K"), "body line must not use erase-whole-line: {line:?}");
        assert!(line.ends_with("\x1b[0K"), "body line must end with erase-to-EOL: {line:?}");
    }

    #[test]
    fn scrolling_reaches_the_oldest_message() {
        let history: Vec<(String, String)> = (0..40)
            .map(|i| ("user".to_string(), format!("line {i}")))
            .collect();
        let mut app = App {
            client: Client::new("http://127.0.0.1:1", "tiny-model", None, false),
            root: PathBuf::from("."),
            max_rounds: 0,
            tui: true,
            round: 1,
            history,
            picking_model: false,
            model_list: vec![],
            model_cursor: 0,
            picking_session: false,
            session_list: vec![],
            session_cursor: 0,
            session_id: "t".to_string(),
            busy: false,
            scroll: 0,
            suggest_cursor: 0,
        };
        let newest = app.frame_with_size("", 0, 64, 24);
        assert!(newest.contains("line 39"), "newest view should show the last line");
        assert!(!newest.contains("SCROLLED"));

        app.scroll = 10_000; // clamped to the top
        let oldest = app.frame_with_size("", 0, 64, 24);
        assert!(oldest.contains("line 0"), "scrolled view should reach the first line");
        assert!(oldest.contains("SCROLLED"));
    }

    #[test]
    fn slash_menu_has_a_movable_pointer() {
        let app = App {
            client: Client::new("http://127.0.0.1:1", "tiny-model", None, false),
            root: PathBuf::from("."),
            max_rounds: 0,
            tui: true,
            round: 0,
            history: vec![],
            picking_model: false,
            model_list: vec![],
            model_cursor: 0,
            picking_session: false,
            session_list: vec![],
            session_cursor: 0,
            session_id: "t".to_string(),
            busy: false,
            scroll: 0,
            suggest_cursor: 2,
        };
        let frame = app.frame_with_size("/", 1, 64, 24);
        let selected = frame.lines().find(|l| l.contains("/session")).expect("/session row");
        assert!(selected.contains("\x1b[32m>"), "selected row needs the pointer: {selected:?}");
        let other = frame.lines().find(|l| l.contains("/help")).expect("/help row");
        assert!(!other.contains("\x1b[32m>"), "unselected row must not have the pointer");

        assert_eq!(command_word("/model [name]"), "/model");
        assert_eq!(command_word("/thinking on|off"), "/thinking");
        assert_eq!(command_matches("/mo").len(), 1);
        assert_eq!(command_matches("/").len(), 6);
        assert!(command_matches("hello").is_empty());
    }

    #[test]
    fn clip_plain_counts_display_columns() {
        assert_eq!(clip_plain("abcdef", 3), "abc");
        assert_eq!(disp_width_in(&clip_plain("日本語テキスト", 6)), 6);
        assert_eq!(clip_plain("abc", 0), "");
    }
}

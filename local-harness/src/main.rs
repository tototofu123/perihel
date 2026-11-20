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
use std::process::{Child, Command, Stdio};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
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
- {"tool":"run.cmd","args":{"command":"python","args":["-m","pytest","-q"]}}  run an allowlisted command and wait
- {"tool":"run.bg","args":{"command":"python","args":["-u","-m","http.server","8000"]}}  start a long-running command in the background
- {"tool":"run.logs","args":{"id":"j1"}}  read a background job's console output (stdout+stderr)
- {"tool":"run.stop","args":{"id":"j1"}}  stop a background job
- {"tool":"run.test","args":{"path":"tests/test_x.py"}}  run tests on one file
- {"tool":"web.search","args":{"query":"latest news"}}  keyless web search (Bing RSS)
- {"tool":"web.fetch","args":{"url":"https://example.com"}}  fetch a page as readable text
- {"tool":"memory.add","args":{"scope":"local","text":"..."}}  append a note to MEMORY.md (scope: local|global)
- {"tool":"reply.text","args":{"text":"..."}}  finish the turn
Rules: one tool per reply; paths are relative to the workspace, no absolute paths or `..`;
inspect before writing, verify after writing; never claim success you did not verify.
`reply.text` is the right tool for greetings, a bare "test", or any prompt with no concrete
task: do not call fs/web/run/memory tools just because the grammar requires one tool per reply.
Never invent a URL: if you do not know the exact address of a source, call `web.search` first
and fetch one of the links it returns. `web.fetch` only accepts links from a recent search
(or a localhost URL you are verifying).
`memory.add` is ONLY for durable facts that will still be true next week (project commands,
conventions, stable user preferences). Never save the current prompt, "no task given",
"waiting for instructions", a test result, or anything about this conversation.
The Memory section below is your persistent notes file; trust it and keep it current.

## Verify before you answer
- After writing or changing code, run the project's real check (`run.test`, or `run.cmd` with
  the project's test / lint / compile command) and read the output.
- For servers, GUIs and watchers use `run.bg`, then `run.logs` to read the console messages
  (exceptions, "listening on ..."), and `web.fetch http://127.0.0.1:PORT` to confirm the page
  really serves. Use `run.stop` when done so no port is left hijacked.
- If a check fails, fix it and check again. If you could not verify something, say exactly
  what is unverified in `reply.text` instead of claiming it works. If several web calls fail,
  stop guessing URLs and answer with that caveat - do not keep trying new links."#;

const TOOLS: &[&str] = &[
    "fs.list", "fs.read", "fs.write", "fs.mkdir", "fs.copy",
    "run.cmd", "run.bg", "run.logs", "run.stop", "run.test",
    "web.search", "web.fetch", "memory.add", "reply.text",
];

const COMMANDS: &[(&str, &str)] = &[
    ("/help", "show this help"),
    ("/model [name]", "pick a model (no arg opens picker)"),
    ("/session", "browse and restore past sessions"),
    ("/thinking on|off", "toggle streamed reasoning"),
    ("/rounds <n>", "set max tool rounds (0 = unlimited)"),
    ("/tools", "list available tools"),
    ("/workspace", "show the sandbox root"),
    ("/memory", "show global and project memory files"),
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

// GGUF discovery. The harness is often launched from a folder with no
// models/ of its own, so also look at sibling projects, PERIHEL_MODELS_DIR
// and the user profile. This is what makes "local-model" resolve to the
// real Qwen file instead of failing with "model file not found".
//
// When several GGUFs exist, prefer the local coding model (Qwen coder) over
// unrelated ones such as OCR/embedding models. Override with PERIHEL_MODEL.
fn model_score(name: &str) -> i32 {
    let n = name.to_lowercase();
    let mut score = 0;
    if n.contains("qwen") { score += 100; }
    if n.contains("coder") { score += 20; }
    if n.contains("3d") { score += 10; }
    if n.contains("ocr") || n.contains("embed") || n.contains("mmproj") { score -= 100; }
    score
}

fn model_search_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(v) = std::env::var("PERIHEL_MODELS_DIR") {
        for part in v.split([';', ',']) {
            let p = part.trim();
            if !p.is_empty() {
                dirs.push(PathBuf::from(p));
            }
        }
    }
    dirs.push(root.join("models"));
    dirs.push(root.to_path_buf());
    dirs.push(PathBuf::from("models"));
    if let Some(parent) = root.parent() {
        if let Ok(rd) = fs::read_dir(parent) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    dirs.push(p.join("models"));
                    dirs.push(p);
                }
            }
        }
    }
    if let Ok(home) = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) {
        let h = PathBuf::from(home);
        dirs.push(h.join(".llama").join("models"));
        dirs.push(h.join("models"));
    }
    dirs
}

fn discover_models(root: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for dir in model_search_dirs(root) {
        if let Ok(rd) = fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|s| s.to_str()) == Some("gguf")
                    && !p.to_string_lossy().to_lowercase().contains("mmproj")
                {
                    out.push(p);
                }
            }
        }
    }
    out.sort_by(|a, b| {
        let sa = model_score(&a.file_name().unwrap_or_default().to_string_lossy());
        let sb = model_score(&b.file_name().unwrap_or_default().to_string_lossy());
        sb.cmp(&sa).then_with(|| a.cmp(b))
    });
    out.dedup();
    out
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
        for p in discover_models(root) {
            let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
            if !out.contains(&name) {
                out.push(name);
            }
        }
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
                // With a grammar attached every completion is a tool-call JSON
                // document, so typing it out only to have the caller print the
                // parsed reply afterwards shows the same text twice. Stream it
                // only when the grammar is off (free-form prose).
                if self.grammar.is_none() {
                    print!("{content}");
                    let _ = io::stdout().flush();
                }
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

fn argv_from(args: &Value) -> Result<Vec<String>, String> {
    let prog = arg_str(args, "command")?.to_string();
    let rest: Vec<String> = args.get("args").and_then(|v| v.as_array())
        .ok_or("args must be a list of strings")?
        .iter().map(|v| v.as_str().map(str::to_string).ok_or("bad arg"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut full = vec![prog];
    full.extend(rest);
    Ok(full)
}

fn allow_command(argv: &[String]) -> Result<(), String> {
    let prog = Path::new(&argv[0]).file_name().and_then(|s| s.to_str()).unwrap_or("");
    let rest = &argv[1..];
    let first = rest.first().map(|s| s.as_str());
    let allowed = match prog {
        "python" | "python3" | "py" => {
            // Leading -u/-B only affect buffering; strip them before checking.
            let cleaned: Vec<&str> = rest.iter().map(|s| s.as_str())
                .skip_while(|s| *s == "-u" || *s == "-B").collect();
            let script = cleaned.len() == 1 && cleaned[0].ends_with(".py");
            let module = cleaned.first().map(|s| *s == "-m").unwrap_or(false);
            let pytest = module && cleaned.get(1) == Some(&"pytest");
            let http = module && cleaned.get(1) == Some(&"http.server");
            let json_tool = module && cleaned.get(1) == Some(&"json.tool");
            let compile = module && cleaned.get(1) == Some(&"py_compile") && cleaned.len() == 3;
            script || pytest || http || json_tool || compile
        }
        "node" => (rest.len() == 1 && rest[0].ends_with(".js"))
            || (rest.len() >= 2 && rest[0] == "-e")
            || (rest.len() >= 2 && rest[0] == "--test"),
        "npm" => matches!(first, Some("test" | "start" | "run" | "build" | "ci" | "--version")),
        "npx" => matches!(first, Some("tsc" | "vite" | "next" | "jest" | "eslint" | "prettier" | "serve")),
        "cargo" => matches!(first,
            Some("build" | "test" | "check" | "clippy" | "fmt" | "run" | "metadata" | "tree" | "--version")),
        "go" => matches!(first, Some("build" | "test" | "run" | "vet" | "mod")),
        "dotnet" => matches!(first, Some("build" | "test" | "run" | "watch")),
        "git" => matches!(first, Some("status" | "diff" | "log"))
            || (rest.len() == 3 && rest[0] == "branch" && rest[1] == "--show-current"),
        "rg" | "pytest" => true,
        _ => false,
    };
    if allowed { Ok(()) } else { Err(format!("command/arguments refused: {argv:?}")) }
}

fn run_process(argv: &[String], cwd: &Path) -> Result<(bool, String), String> {
    run_process_within(argv, cwd, cmd_timeout())
}

fn cmd_timeout() -> Duration {
    let secs = std::env::var("PERIHEL_CMD_TIMEOUT").ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(120);
    Duration::from_secs(secs)
}

/// Run to completion, capturing stdout+stderr to a file while we wait.
///
/// Pipes would deadlock as soon as a chatty command filled the 64 KB buffer,
/// and a hung command would block the whole TUI, so output goes to a temp file
/// and the waiter is bounded by `limit`. On timeout the whole process tree is
/// killed and the partial output is returned instead of an opaque error.
fn run_process_within(argv: &[String], cwd: &Path, limit: Duration) -> Result<(bool, String), String> {
    allow_command(argv)?;
    let log = temp_log_path("cmd");
    if let Some(parent) = log.parent() { fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .current_dir(cwd)
        .stdin(Stdio::null());
    attach_log(&mut cmd, &log)?;
    let mut child = cmd.spawn().map_err(|e| format!("cannot run {}: {e}", argv[0]))?;
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let text = fs::read_to_string(&log).unwrap_or_default();
                let _ = fs::remove_file(&log);
                return Ok((status.success(), tail_chars(&text, 6000)));
            }
            Ok(None) => {
                if started.elapsed() > limit {
                    kill_tree(child.id());
                    let _ = child.wait();
                    let mut text = fs::read_to_string(&log).unwrap_or_default();
                    let _ = fs::remove_file(&log);
                    if !text.ends_with('\n') { text.push('\n'); }
                    text.push_str(&format!(
                        "[timeout] command exceeded {}s and was stopped (raise PERIHEL_CMD_TIMEOUT to wait longer)",
                        limit.as_secs()));
                    return Ok((false, tail_chars(&text, 6000)));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                let _ = fs::remove_file(&log);
                return Err(e.to_string());
            }
        }
    }
}

fn tail_chars(s: &str, max: usize) -> String {
    if s.len() <= max { return s.to_string(); }
    let mut start = s.len() - max;
    while start < s.len() && !s.is_char_boundary(start) { start += 1; }
    format!("... [truncated]\n{}", &s[start..])
}

fn is_python(prog: &str) -> bool {
    let name = Path::new(prog).file_name().and_then(|s| s.to_str()).unwrap_or("");
    name == "python" || name == "python3" || name == "py"
}

fn attach_log(cmd: &mut Command, log: &Path) -> Result<(), String> {
    let out = fs::File::create(log).map_err(|e| e.to_string())?;
    let err = out.try_clone().map_err(|e| e.to_string())?;
    cmd.stdout(Stdio::from(out)).stderr(Stdio::from(err));
    Ok(())
}

fn temp_log_path(kind: &str) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    std::env::temp_dir()
        .join("perihel")
        .join(format!("{kind}-{}-{n}.log", std::process::id()))
}

fn kill_tree(pid: u32) {
    let pid_s = pid.to_string();
    #[cfg(windows)]
    {
        // /T also stops children (npm -> node, python -> subprocess).
        let _ = Command::new("taskkill")
            .args(["/PID", &pid_s, "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(not(windows))]
    {
        let _ = Command::new("kill").args(["-9", &pid_s]).status();
    }
}

// ------------------------------------------------------------- background jobs

struct Job {
    id: String,
    command: String,
    log: PathBuf,
    child: std::process::Child,
}

fn jobs() -> &'static Mutex<Vec<Job>> {
    static JOBS: OnceLock<Mutex<Vec<Job>>> = OnceLock::new();
    JOBS.get_or_init(|| Mutex::new(Vec::new()))
}

fn jobs_dir(root: &Path) -> PathBuf { root.join(".perihel").join("jobs") }

fn start_background(argv: &[String], root: &Path) -> Result<String, String> {
    allow_command(argv)?;
    let dir = jobs_dir(root);
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
    let id = format!("j{}", SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst));
    let log = dir.join(format!("{id}.log"));
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..]).current_dir(root).stdin(Stdio::null());
    if is_python(&argv[0]) { cmd.env("PYTHONUNBUFFERED", "1"); }
    attach_log(&mut cmd, &log)?;
    let child = cmd.spawn().map_err(|e| format!("cannot start {}: {e}", argv[0]))?;
    let pid = child.id();
    let command = argv.join(" ");
    jobs().lock().unwrap().push(Job { id: id.clone(), command, log: log.clone(), child });
    Ok(format!(
        "job {id} started (pid {pid})\nlog: {}\nread console output with {{\"tool\":\"run.logs\",\"args\":{{\"id\":\"{id}\"}}}}; stop it with run.stop",
        log.display()))
}

fn job_logs(id: &str) -> Result<String, String> {
    let mut guard = jobs().lock().unwrap();
    let job = guard.iter_mut().find(|j| j.id == id)
        .ok_or_else(|| format!("unknown job: {id} (jobs live only for this process)"))?;
    let state = match job.child.try_wait() {
        Ok(Some(status)) => format!("exited with {}", status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".into())),
        Ok(None) => "running".to_string(),
        Err(e) => format!("state unknown: {e}"),
    };
    let text = fs::read_to_string(&job.log).unwrap_or_default();
    let body = if text.trim().is_empty() { "(no output yet - the process may still be starting)".to_string() }
               else { tail_chars(&text, 6000) };
    Ok(format!("job {id}: {} | {}\nlog: {}\n{body}", job.command, state, job.log.display()))
}

fn stop_job(id: &str) -> Result<String, String> {
    let mut guard = jobs().lock().unwrap();
    let job = guard.iter_mut().find(|j| j.id == id)
        .ok_or_else(|| format!("unknown job: {id}"))?;
    match job.child.try_wait() {
        Ok(Some(status)) => Ok(format!("job {id} already stopped ({:?})", status.code())),
        _ => {
            kill_tree(job.child.id());
            let _ = job.child.wait();
            Ok(format!("job {id} stopped"))
        }
    }
}


// ------------------------------------------------------------------ sessions

fn perihel_home() -> PathBuf {
    if let Ok(p) = std::env::var("PERIHEL_HOME") { return PathBuf::from(p); }
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".perihel")
}

fn sessions_dir() -> PathBuf {
    if let Ok(p) = std::env::var("PERIHEL_SESSIONS_DIR") { return PathBuf::from(p); }
    perihel_home().join("sessions")
}

// ------------------------------------------------------------------- memory

/// Notes that follow you between projects.
fn global_memory_path() -> PathBuf { perihel_home().join("MEMORY.md") }

/// Notes about this workspace, checked in next to the code.
fn local_memory_path(root: &Path) -> PathBuf { root.join("MEMORY.md") }

const MEMORY_LIMIT: usize = 4000;

fn read_memory(path: &Path) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    let text = text.trim().to_string();
    if text.is_empty() { return None; }
    Some(truncate_chars(&text, MEMORY_LIMIT))
}

fn memory_block(root: &Path) -> String {
    let global = global_memory_path();
    let local = local_memory_path(root);
    let mut out = String::from("\n\n## Memory\n");
    out.push_str(&format!("\n### Global memory ({})\n", global.display()));
    out.push_str(&read_memory(&global).unwrap_or_else(|| "(empty)".into()));
    out.push_str(&format!("\n\n### Project memory ({})\n", local.display()));
    out.push_str(&read_memory(&local).unwrap_or_else(|| "(empty)".into()));
    out.push_str("\n\nUse {\"tool\":\"memory.add\",\"args\":{\"scope\":\"local\",\"text\":\"...\"}} \
to keep a durable fact or a project rule; use scope \"global\" for things true in every project. \
Memory is re-read at the start of every turn. Never write notes about this conversation, a bare \
\"test\", or a prompt with no task - stale notes are worse than none.\n");
    out
}

fn system_with_memory(root: &Path) -> String {
    format!("{}{}", DEFAULT_SYSTEM, memory_block(root))
}

fn refresh_system(client: &mut Client, root: &Path) {
    let sys = system_with_memory(root);
    if client.messages.is_empty() {
        client.messages.push(json!({"role": "system", "content": sys}));
    } else {
        client.messages[0] = json!({"role": "system", "content": sys});
    }
}

fn memory_add(root: &Path, scope: &str, text: &str) -> Result<String, String> {
    memory_add_at(&perihel_home(), root, scope, text)
}

fn memory_add_at(home: &Path, root: &Path, scope: &str, text: &str) -> Result<String, String> {
    let path = match scope {
        "global" => home.join("MEMORY.md"),
        "local" => local_memory_path(root),
        _ => return Err("scope must be \"global\" or \"local\"".into()),
    };
    let note = text.lines().map(str::trim).filter(|l| !l.is_empty())
        .collect::<Vec<_>>().join(" ");
    if note.is_empty() { return Err("text is empty".into()); }
    if let Some(parent) = path.parent() { fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
    let mut body = fs::read_to_string(&path).unwrap_or_default();
    if body.trim().is_empty() {
        body = if scope == "global" {
            "# Global memory\n\nDurable facts and preferences that apply in every project.\n".to_string()
        } else {
            "# Project memory\n\nFacts, commands and rules for this workspace.\n".to_string()
        };
    }
    if !body.ends_with('\n') { body.push('\n'); }
    body.push_str(&format!("- {note}\n"));
    fs::write(&path, body).map_err(|e| e.to_string())?;
    Ok(format!("remembered ({scope}) -> {}\ntakes effect on the next turn", path.display()))
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
struct WebRate { search: VecDeque<Instant>, fetch: VecDeque<Instant> }
static WEB_RATE: OnceLock<Mutex<WebRate>> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WebKind { Search, Fetch }

#[derive(Debug, PartialEq, Eq)]
enum SlotError {
    /// A same-kind call happened less than 5 s ago; we can just wait it out.
    Gap(Duration),
    /// 10 calls in the last minute; retry after this delay.
    Quota(Duration),
}

impl WebRate {
    fn bucket(&mut self, kind: WebKind) -> &mut VecDeque<Instant> {
        match kind { WebKind::Search => &mut self.search, WebKind::Fetch => &mut self.fetch }
    }

    fn reserve(&mut self, kind: WebKind, now: Instant) -> Result<Instant, SlotError> {
        self.search.retain(|t| now.saturating_duration_since(*t) < Duration::from_secs(60));
        self.fetch.retain(|t| now.saturating_duration_since(*t) < Duration::from_secs(60));
        if self.search.len() + self.fetch.len() >= web_quota() {
            let oldest = self.search.iter().chain(self.fetch.iter()).min().copied().unwrap_or(now);
            let retry = Duration::from_secs(60).saturating_sub(now.saturating_duration_since(oldest));
            return Err(SlotError::Quota(retry.max(Duration::from_secs(1))));
        }
        let bucket = self.bucket(kind);
        if let Some(last) = bucket.back() {
            let wait = web_gap().saturating_sub(now.saturating_duration_since(*last));
            if !wait.is_zero() { return Err(SlotError::Gap(wait)); }
        }
        bucket.push_back(now);
        Ok(now)
    }

    fn unreserve(&mut self, kind: WebKind, when: Instant) {
        let bucket = self.bucket(kind);
        if let Some(i) = bucket.iter().position(|t| *t == when) { bucket.remove(i); }
    }
}

/// Keyless search endpoints do not publish a hard quota. Keep a small gap to
/// be polite and a generous per-minute ceiling; both are tunable with env vars
/// so a heavy research session is not artificially throttled.
fn web_gap() -> Duration {
    let ms = std::env::var("PERIHEL_WEB_GAP_MS").ok()
        .and_then(|v| v.parse::<u64>().ok()).unwrap_or(1000);
    Duration::from_millis(ms)
}

fn web_quota() -> usize {
    std::env::var("PERIHEL_WEB_MAX_MIN").ok()
        .and_then(|v| v.parse::<usize>().ok()).unwrap_or(60)
}

/// Wait out a short gap so a model-driven search -> fetch pair just works;
/// only a real per-minute quota is surfaced to the model as an error.
fn web_slot(kind: WebKind) -> Result<Instant, String> {
    for _ in 0..2 {
        let attempt = {
            let mut state = WEB_RATE.get_or_init(|| Mutex::new(WebRate::default()))
                .lock().map_err(|e| e.to_string())?;
            state.reserve(kind, Instant::now())
        };
        match attempt {
            Ok(when) => return Ok(when),
            Err(SlotError::Gap(wait)) => std::thread::sleep(wait),
            Err(SlotError::Quota(retry)) => return Err(format!(
                "web rate limit: {} calls per minute reached, try again in {}s (set PERIHEL_WEB_MAX_MIN to raise it)",
                web_quota(), retry.as_secs())),
        }
    }
    Err("web rate limit: busy, try again".into())
}

fn web_unreserve(kind: WebKind, when: Instant) {
    if let Ok(mut state) = WEB_RATE.get_or_init(|| Mutex::new(WebRate::default())).lock() {
        state.unreserve(kind, when);
    }
}

// Only URLs that actually came back from a search (plus localhost dev servers)
// may be fetched. The model otherwise invents plausible-looking weather/news
// URLs from memory and burns turns on 404s.
static SEARCH_LINKS: OnceLock<Mutex<VecDeque<String>>> = OnceLock::new();

fn remember_links(known: &mut VecDeque<String>, links: &[String]) {
    for link in links {
        let link = normalise_url(link);
        if link.is_empty() { continue; }
        if !known.contains(&link) { known.push_back(link); }
    }
    while known.len() > 40 { known.pop_front(); }
}

fn remember_search_links(links: &[String]) {
    if let Ok(mut known) = SEARCH_LINKS.get_or_init(|| Mutex::new(VecDeque::new())).lock() {
        remember_links(&mut known, links);
    }
}

fn normalise_url(url: &str) -> String {
    let url = url.trim().trim_end_matches('/').to_lowercase();
    url.replace("http://www.", "http://").replace("https://www.", "https://")
}

fn is_local_url(url: &str) -> bool {
    let u = normalise_url(url);
    u.starts_with("http://127.0.0.1")
        || u.starts_with("http://localhost")
        || u.starts_with("http://[::1]")
}

fn url_host(url: &str) -> String {
    let u = url.trim().to_lowercase();
    let rest = u.split_once("://").map(|(_, r)| r).unwrap_or(&u);
    let host = rest.split(['/', '?', '#']).next().unwrap_or("");
    host.strip_prefix("www.").unwrap_or(host).to_string()
}

fn fetch_url_allowed(url: &str, known: &VecDeque<String>) -> bool {
    if is_local_url(url) { return true; }
    if known.contains(&normalise_url(url)) { return true; }
    // Once a domain appeared in the results, a different path on that same
    // site is fine - models often remember the right site but not the URL.
    let host = url_host(url);
    !host.is_empty() && known.iter().any(|k| url_host(k) == host)
}

fn fetch_url_guard(url: &str) -> Result<(), ToolResult> {
    let known = SEARCH_LINKS.get_or_init(|| Mutex::new(VecDeque::new()))
        .lock().map_err(|e| ToolResult { ok: false, output: e.to_string(), refused: None })?;
    if fetch_url_allowed(url, &known) { return Ok(()); }
    Err(ToolResult {
        ok: false,
        output: format!(
            "URL was not returned by a search, so it was not fetched: {url}\n\
             Call {{\"tool\":\"web.search\",\"args\":{{\"query\":\"...\"}}}} first, then fetch one of the links it returns. \
             (Localhost URLs are always allowed for verifying a dev server.)"),
        refused: None,
    })
}

#[derive(Debug, PartialEq, Eq)]
enum WebGuardAction { Run, Refuse, Stop }

/// Three failed web calls in a row are enough: a 4xx or a DNS failure almost
/// never fixes itself by guessing another URL.
fn web_guard_action(failures: usize) -> WebGuardAction {
    if failures >= 5 { WebGuardAction::Stop }
    else if failures >= 3 { WebGuardAction::Refuse }
    else { WebGuardAction::Run }
}

const WEB_GUARD_NUDGE: &str = "WEB GUARD: several web calls failed. Do NOT call web.search or web.fetch again. \
Use reply.text now and say exactly what you could not verify.";

fn html_unescape(s: &str) -> String {
    s.replace("&amp;", "&").replace("&lt;", "<").replace("&lt;", "<")
        .replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

fn strip_tags(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find('<') {
        out.push_str(&html_unescape(&rest[..i]));
        if let Some(end) = rest[i..].find('>') { rest = &rest[i + end + 1..]; } else { break; }
    }
    out.push_str(&html_unescape(rest));
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn hex_digit(c: u8) -> Option<u8> {
    match c { b'0'..=b'9' => Some(c - b'0'), b'a'..=b'f' => Some(c - b'a' + 10), b'A'..=b'F' => Some(c - b'A' + 10), _ => None }
}

fn url_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(a), Some(c)) = (hex_digit(b[i + 1]), hex_digit(b[i + 2])) {
                out.push(a * 16 + c); i += 3; continue;
            }
        }
        out.push(if b[i] == b'+' { b' ' } else { b[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

/// DuckDuckGo wraps result links in `//duckduckgo.com/l/?uddg=<encoded url>`.
fn ddg_result_url(href: &str) -> String {
    let href = html_unescape(href);
    if let Some((_, tail)) = href.split_once("uddg=") {
        let encoded = tail.split('&').next().unwrap_or(tail);
        let decoded = url_decode(encoded);
        if decoded.starts_with("http://") || decoded.starts_with("https://") { return decoded; }
    }
    if href.starts_with("//") { format!("https:{href}") } else { href }
}

/// Search results as (title, url, snippet).
fn parse_ddg_results(body: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for block in body.split("class=\"result results_").skip(1) {
        let title = block.split_once("class=\"result__a\"")
            .and_then(|(_, r)| r.split_once('>'))
            .and_then(|(a, rest)| rest.split_once("</a>").map(|(t, _)| (a, t)))
            .map(|(_, t)| strip_tags(t)).unwrap_or_default();
        let href = block.split_once("class=\"result__a\"")
            .and_then(|(_, r)| r.split_once("href=\""))
            .and_then(|(_, r)| r.split_once('"').map(|(h, _)| h))
            .map(ddg_result_url).unwrap_or_default();
        let snippet = block.split_once("class=\"result__snippet\"")
            .and_then(|(_, r)| r.split_once('>'))
            .and_then(|(_, r)| r.split_once("</a>").map(|(s, _)| strip_tags(s)))
            .unwrap_or_default();
        if !title.is_empty() && href.starts_with("http") {
            out.push((title, href, snippet));
        }
        if out.len() >= 8 { break; }
    }
    out
}

fn ddg_search(query: &str) -> Option<Vec<(String, String, String)>> {
    let url = format!("https://html.duckduckgo.com/html/?q={}", url_encode(query));
    let (status, body) = http_get_status(&url, 15).ok()?;
    if status != 200 { return None; }
    let results = parse_ddg_results(&body);
    if results.is_empty() { None } else { Some(results) }
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

/// Returns the HTTP status as well as the body: a 404 must be data the model
/// can react to (try the next link), not an opaque transport error.
fn http_get_status(url: &str, timeout_s: u64) -> Result<(u16, String), String> {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err("only http(s) URLs are allowed".into());
    }
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(timeout_s)))
        .http_status_as_error(false)
        .build();
    let agent = ureq::Agent::new_with_config(config);
    let resp = agent.get(url)
        .header("User-Agent", "Mozilla/5.0 (compatible; perihel-local/0.1)")
        .call()
        .map_err(|e| format!("request failed for {url}: {e}"))?;
    let status = resp.status();
    let bytes = resp.into_body().read_to_vec().map_err(|e| format!("read failed for {url}: {e}"))?;
    let limited = if bytes.len() > 2 * 1024 * 1024 { &bytes[..2 * 1024 * 1024] } else { &bytes };
    Ok((status.into(), String::from_utf8_lossy(limited).to_string()))
}

fn web_search(args: &Value) -> ToolResult {
    let query = match arg_str(args, "query") { Ok(q) => q, Err(e) => return ToolResult { ok: false, output: String::new(), refused: Some(e) } };
    let slot = match web_slot(WebKind::Search) {
        Ok(s) => s,
        Err(e) => return ToolResult { ok: false, output: e, refused: None },
    };
    // DuckDuckGo HTML is the primary backend: keyless, no API signup, and it
    // returns real ranked results instead of the Wikipedia-heavy Bing RSS feed.
    // Bing RSS stays as a fallback in case DDG is blocked on this network.
    let mut results = ddg_search(query).unwrap_or_default();
    if results.is_empty() {
        let url = format!("https://www.bing.com/search?format=rss&q={}", url_encode(query));
        if let Ok(body) = http_body(&url, 15, slot, WebKind::Search) {
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
                if !title.is_empty() && !link.is_empty() { results.push((title, link, desc)); }
                if results.len() >= 8 { break; }
            }
        }
    }
    let text = if results.is_empty() {
        web_unreserve(WebKind::Search, slot);
        "no results (DuckDuckGo and Bing RSS returned nothing)".into()
    } else {
        remember_search_links(&results.iter().map(|r| r.1.clone()).collect::<Vec<_>>());
        results.iter()
            .map(|(t, u, s)| format!("- {t}\n  {u}\n  {}", truncate_chars(s, 300)))
            .collect::<Vec<_>>().join("\n")
    };
    ToolResult { ok: true, output: text, refused: None }
}

/// A failed request does not burn one of the 10 slots, and a 4xx/5xx is
/// reported with its URL and body so the model can try the next result.
fn http_body(url: &str, timeout_s: u64, slot: Instant, kind: WebKind) -> Result<String, ToolResult> {
    match http_get_status(url, timeout_s) {
        Ok((status, body)) if status < 400 => Ok(body),
        Ok((status, body)) => {
            web_unreserve(kind, slot);
            Err(ToolResult {
                ok: false,
                output: format!("HTTP {status} for {url}\n{}", truncate_chars(body.trim(), 300)),
                refused: None,
            })
        }
        Err(e) => {
            web_unreserve(kind, slot);
            Err(ToolResult { ok: false, output: e, refused: None })
        }
    }
}

/// Many news/weather pages are JavaScript shells. Their `<link rel="alternate">`
/// usually points at an RSS/Atom feed with the actual data, so follow it when
/// the visible text is suspiciously thin. Generic - no per-site rules.
fn find_feed_link(html: &str) -> Option<String> {
    let mut from = 0usize;
    while let Some(rel) = html[from..].find("<link") {
        let start = from + rel;
        let end = start + html[start..].find('>')?;
        let tag = &html[start..end];
        let lower = tag.to_ascii_lowercase();
        if lower.contains("application/rss+xml") || lower.contains("application/atom+xml") {
            if let Some(href) = tag.split_once("href=\"")
                .and_then(|(_, r)| r.split_once('"').map(|(h, _)| h)) {
                let href = html_unescape(href);
                if href.starts_with("http://") || href.starts_with("https://") { return Some(href); }
            }
        }
        from = end + 1;
    }
    None
}

fn looks_like_js_shell(text: &str) -> bool {
    // A rendered page usually has prose; a JS shell leaks CSS/JS into the
    // readable-text extractor instead.
    let markers = ["function ", "var ", "display:", "!important", "$(function", "/*", "window.open"];
    let hits = markers.iter().filter(|m| text.contains(**m)).count();
    hits >= 2 || text.chars().count() < 500
}

fn parse_feed_items(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let blocks = body.split("<item>").skip(1).chain(body.split("<entry>").skip(1));
    for raw in blocks {
        let block = raw.split("</item>").next().unwrap_or(raw)
            .split("</entry>").next().unwrap_or(raw);
        let field = |name: &str| -> String {
            let open = format!("<{name}>");
            let close = format!("</{name}>");
            block.split_once(&open)
                .and_then(|(_, r)| r.split_once(&close).map(|(v, _)| v))
                .map(|v| strip_tags(v.trim().trim_start_matches("<![CDATA[").trim_end_matches("]]>")))
                .unwrap_or_default()
        };
        let title = field("title");
        let detail = field("description");
        let detail = if detail.is_empty() { field("content") } else { detail };
        let item = if detail.is_empty() { title } else { format!("{title}: {detail}") };
        if !item.is_empty() { out.push(truncate_chars(&item, 700)); }
        if out.len() >= 3 { break; }
    }
    out
}

fn read_feed(url: &str) -> Option<String> {
    let (status, body) = http_get_status(url, 10).ok()?;
    if status != 200 { return None; }
    let items = parse_feed_items(&body);
    if items.is_empty() { return None; }
    Some(format!("Feed from {url}:\n{}", items.join("\n")))
}

fn web_fetch(args: &Value) -> ToolResult {
    let url = match arg_str(args, "url") { Ok(u) => u, Err(e) => return ToolResult { ok: false, output: String::new(), refused: Some(e) } };
    if let Err(t) = fetch_url_guard(url) { return t; }
    let slot = match web_slot(WebKind::Fetch) {
        Ok(s) => s,
        Err(e) => return ToolResult { ok: false, output: e, refused: None },
    };
    let html = match http_body(url, 15, slot, WebKind::Fetch) { Ok(h) => h, Err(t) => return t };
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
    let mut out = truncate_chars(&collapsed.join("\n"), 6000);
    // JS-rendered pages yield almost no visible text. If the page advertises a
    // feed, use its items as the structured version of the same content.
    if looks_like_js_shell(&out) {
        if let Some(link) = find_feed_link(&html) {
            if let Some(feed) = read_feed(&link) {
                out.push_str("\n\n");
                out.push_str(&feed);
            }
        }
    }
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
            let full = argv_from(args)?;
            let (ok, out) = run_process(&full, root)?;
            Ok(ToolResult { ok, output: out, refused: None })
        })()),
        "run.bg" => wrap((|| {
            let full = argv_from(args)?;
            let msg = start_background(&full, root)?;
            Ok(ToolResult { ok: true, output: msg, refused: None })
        })()),
        "run.logs" => wrap((|| {
            let id = arg_str(args, "id")?;
            let msg = job_logs(id)?;
            Ok(ToolResult { ok: true, output: msg, refused: None })
        })()),
        "run.stop" => wrap((|| {
            let id = arg_str(args, "id")?;
            let msg = stop_job(id)?;
            Ok(ToolResult { ok: true, output: msg, refused: None })
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
        "memory.add" => wrap((|| {
            let scope = arg_str(args, "scope")?;
            let text = arg_str(args, "text")?;
            let msg = memory_add(root, scope, text)?;
            Ok(ToolResult { ok: true, output: msg, refused: None })
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

// ------------------------------------------------------------------ llama sidecar

/// Owns llama-server for the lifetime of the harness. It is started on the
/// first request (never at launch), kept loaded while the harness is open, and
/// killed when the harness exits or the model changes.
struct Sidecar {
    child: Option<Child>,
    owned: bool,                 // false when we adopted a server someone else started
    model: String,
    base: String,
    port: String,
    root: PathBuf,
    probe: ureq::Agent,
}

impl Sidecar {
    fn new(base: &str, root: &Path) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(2)))
            .build();
        Self {
            child: None,
            owned: false,
            model: String::new(),
            base: base.trim_end_matches('/').to_string(),
            port: base.rsplit(':').next().unwrap_or("8090").to_string(),
            root: root.to_path_buf(),
            probe: ureq::Agent::new_with_config(config),
        }
    }

    fn healthy(&self) -> bool {
        self.probe.get(&format!("{}/health", self.base)).call().is_ok()
    }

    /// True when a server with this exact model is already up.
    fn is_loaded(&self, model: &str) -> bool {
        self.model == model && self.child.is_some()
    }

    fn ensure(&mut self, model: &str) -> Result<(), String> {
        if self.is_loaded(model)
            && self.child.as_mut().is_some_and(|c| c.try_wait().ok().flatten().is_none())
        {
            return Ok(());
        }
        // An llama-server someone else started is fine to use — we just don't
        // own it, so we won't kill it on exit.
        if self.child.is_none() && self.healthy() {
            self.model = model.to_string();
            return Ok(());
        }

        self.stop();
        let server = find_llama_server().ok_or_else(|| {
            "llama-server.exe not found — install llama.cpp (winget install llama.cpp)".to_string()
        })?;
        let gguf = resolve_gguf(&self.root, model).ok_or_else(|| {
            format!("model file for '{model}' not found under {}", self.root.display())
        })?;

        let mut cmd = Command::new(&server);
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW — no extra console
        cmd.args([
            "-m", gguf.to_string_lossy().as_ref(),
            "--host", "127.0.0.1",
            "--port", &self.port,
            "--ctx-size", "32768",
            "--parallel", "1",
            "--n-gpu-layers", "99",
            "--flash-attn", "on",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--no-webui",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null());

        let child = cmd.spawn().map_err(|e| format!("cannot start llama-server: {e}"))?;
        self.child = Some(child);
        self.owned = true;
        self.model = model.to_string();

        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(120) {
            if self.healthy() {
                return Ok(());
            }
            if self.child.as_mut().is_some_and(|c| c.try_wait().ok().flatten().is_some()) {
                self.stop();
                return Err("llama-server exited during startup".into());
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        self.stop();
        Err("llama-server did not become healthy within 120s".into())
    }

    fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            if self.owned {
                #[cfg(windows)]
                {
                    let _ = Command::new("taskkill")
                        .args(["/PID", &child.id().to_string(), "/T", "/F"])
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status();
                }
                #[cfg(not(windows))]
                {
                    let _ = child.kill();
                }
            }
            let _ = child.wait();
        }
        self.owned = false;
        self.model.clear();
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        self.stop();
    }
}

fn find_llama_server() -> Option<PathBuf> {
    let finder = if cfg!(windows) { "where" } else { "which" };
    if let Ok(out) = Command::new(finder).arg("llama-server").output() {
        if out.status.success() {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                let p = PathBuf::from(line.trim());
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }
    #[cfg(windows)]
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let base = PathBuf::from(local).join("Microsoft/WinGet/Packages");
        if let Ok(rd) = std::fs::read_dir(&base) {
            for e in rd.flatten() {
                let direct = e.path().join("llama-server.exe");
                if direct.exists() {
                    return Some(direct);
                }
                if let Ok(inner) = std::fs::read_dir(e.path()) {
                    for ie in inner.flatten() {
                        let p = ie.path().join("llama-server.exe");
                        if p.exists() {
                            return Some(p);
                        }
                    }
                }
            }
        }
    }
    for env in ["PERIHEL_LLAMA_SERVER", "LLAMA_BIN"] {
        if let Ok(v) = std::env::var(env) {
            let p = PathBuf::from(&v);
            let exe = if p.is_dir() { p.join("llama-server.exe") } else { p };
            if exe.exists() {
                return Some(exe);
            }
        }
    }
    None
}

fn resolve_gguf(root: &Path, name: &str) -> Option<PathBuf> {
    if let Ok(v) = std::env::var("PERIHEL_MODEL") {
        let p = PathBuf::from(v);
        if p.exists() {
            return Some(p);
        }
    }
    let wanted = name.trim().to_lowercase();
    let candidates = discover_models(root);
    if let Some(p) = candidates.iter().find(|p| {
        let n = p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
        n == wanted || n == format!("{wanted}.gguf")
    }) {
        return Some(p.clone());
    }
    // The "local-model" placeholder means: use whatever GGUF we can find.
    if wanted.is_empty() || wanted == "local-model" {
        return candidates.into_iter().next();
    }
    candidates.into_iter().find(|p| p.to_string_lossy().to_lowercase().contains(&wanted))
}

// ------------------------------------------------------------------ agent

struct App {
    client: Client,
    sidecar: Sidecar,
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

    /// Switch model and bring its server up right away (the same "starting
    /// model" announcement a request produces).
    fn set_model(&mut self, name: String) {
        if self.client.model == name && self.sidecar.is_loaded(&name) {
            self.inform(&format!("model = {name}"));
            return;
        }
        self.sidecar.stop();
        self.client.model = name.clone();
        self.inform(&format!("model = {name}"));
        self.inform(&format!("starting model {name}..."));
        match self.sidecar.ensure(&name) {
            Ok(()) => self.inform("model ready"),
            Err(e) => {
                if !self.tui { eprintln!("[error] model start failed: {e}"); }
                self.add("error", &format!("model start failed: {e}"));
            }
        }
    }

    fn run_turn(&mut self, user: &str) {
        // Pick up MEMORY.md edits and notes added by the previous turn.
        refresh_system(&mut self.client, &self.root);
        // llama-server is started here — on the first request, never at launch.
        let starting = !self.sidecar.is_loaded(&self.client.model);
        if starting {
            self.inform(&format!("starting model {}...", self.client.model));
        }
        if let Err(e) = self.sidecar.ensure(&self.client.model) {
            if !self.tui { eprintln!("[error] model start failed: {e}"); }
            self.add("error", &format!("model start failed: {e}"));
            return;
        }
        if starting {
            self.inform("model ready");
        }
        self.client.messages.push(json!({"role":"user","content":user}));
        self.add("user", user);
        let mut round = 0;
        let mut web_failures = 0usize;
        loop {
            round += 1;
            self.round = round;
            if self.tui { self.draw(""); } else { println!("[round {}] thinking...", round); }
            self.busy = true;
            if self.tui { self.draw(""); }
            let raw = match self.client.chat(!self.tui) {
                Ok(r) => r,
                Err(e) => {
                    self.busy = false;
                    // `add` only redraws in TUI mode; a CLI user would never
                    // see the error (unreachable server, bad JSON, retry fail).
                    if !self.tui { eprintln!("[error] {e}"); }
                    self.add("error", &e);
                    return;
                }
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
            let web_tool = name == "web.search" || name == "web.fetch";
            // A model that cannot find a working weather URL will happily try
            // dozens of them. After 3 failures, stop spending web calls and
            // force it to answer with an explicit "unverified" caveat.
            match web_guard_action(web_failures) {
                WebGuardAction::Stop if web_tool => {
                    let msg = "web guard: too many failed web attempts; ending the turn to avoid an endless retry loop";
                    self.add("error", msg);
                    if !self.tui { eprintln!("[guard] {msg}"); }
                    return;
                }
                WebGuardAction::Refuse if web_tool => {
                    web_failures += 1;
                    let nudge = WEB_GUARD_NUDGE;
                    self.client.messages.push(json!({"role":"user","content":nudge}));
                    self.inform("web guard: refused another fetch; asked the model to answer");
                    continue;
                }
                _ => {}
            }
            let result = execute(&call, &self.root);
            if web_tool {
                if result.ok {
                    web_failures = 0;
                } else {
                    web_failures += 1;
                    if web_failures == 3 {
                        self.client.messages.push(json!({"role":"user","content":WEB_GUARD_NUDGE}));
                        self.inform("web guard: 3 failed web calls; asked the model to answer without more fetches");
                    }
                }
            }
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
        sidecar: Sidecar::new(&cli.base_url, &root),
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
                                let name = app.model_list[i].clone();
                                app.picking_model = false;
                                app.set_model(name);
                            }
                        }
                        Key::Enter => {
                            let name = app.model_list.get(app.model_cursor).cloned();
                            app.picking_model = false;
                            if let Some(name) = name {
                                app.set_model(name);
                            }
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
                    "help" => app.inform("/help /quit /reset /model [name] /rounds <n> /thinking on|off /tools /workspace /memory"),
                    "reset" => {
                        app.client.reset();
                        app.history.clear();
                        app.inform("conversation reset");
                    }
                    "model" => {
                        if !rest.is_empty() {
                            app.set_model(rest.trim().to_string());
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
                    "memory" => {
                        for (label, p) in [("global", global_memory_path()), ("local", local_memory_path(&app.root))] {
                            let lines = fs::read_to_string(&p).map(|t| t.lines().count()).unwrap_or(0);
                            let state = if p.exists() { "exists" } else { "not created yet" };
                            app.inform(&format!("{label}: {} ({lines} lines, {state})", p.display()));
                        }
                    }
                    _ => app.inform(&format!("unknown command: {verb}")),
                }
                continue;
            }
            app.run_turn(&cmd);
            app.save_current_session();
        }
    }

    app.sidecar.stop();   // llama-server lives only as long as this harness

    if app.tui {
        print!("\x1b[?25h\x1b[?1049l");
        let _ = io::stdout().flush();
    }
}

#[cfg(test)]
mod tool_tests {
    use super::*;

    fn python_available() -> bool {
        Command::new("python").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("perihel-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn feed_fallback_parses_linked_rss() {
        let html = r#"<html><head><link rel="alternate" type="application/rss+xml" title="Current Weather" href="https://rss.weather.gov.hk/rss/CurrentWeather.xml"></head><body>shell</body></html>"#;
        assert_eq!(find_feed_link(html).as_deref(), Some("https://rss.weather.gov.hk/rss/CurrentWeather.xml"));
        assert!(find_feed_link("<html><body>nothing</body></html>").is_none());

        let feed = r#"<?xml version="1.0"?><rss><channel><item>
          <title>Bulletin updated at 15:02 HKT</title>
          <description><![CDATA[<p>Air temperature : 13 degrees Celsius<br/>Relative Humidity : 91 per cent</p>]]></description>
        </item></channel></rss>"#;
        let items = parse_feed_items(feed);
        assert_eq!(items.len(), 1);
        assert!(items[0].contains("13 degrees Celsius"), "{:?}", items);
        assert!(items[0].contains("91 per cent"), "{:?}", items);
    }

    #[test]
    fn web_fetch_rejects_guessed_urls_but_allows_search_and_local() {
        let mut known = VecDeque::new();
        remember_links(&mut known, &[
            "https://www.Weather.com/HK/".to_string(),
            "https://example.com/a".to_string(),
        ]);
        assert_eq!(known.len(), 2);
        assert!(fetch_url_allowed("https://weather.com/HK", &known), "case/www/trailing slash normalised");
        assert!(fetch_url_allowed("https://www.weather.com/forecast/today", &known), "same site, different path");
        assert!(fetch_url_allowed("http://127.0.0.1:8123/", &known), "localhost dev server always allowed");
        assert!(!fetch_url_allowed("https://nowweather.com/weather/hong-kong", &known), "invented domain refused");
        assert!(!fetch_url_allowed("", &VecDeque::new()));

        let many: Vec<String> = (0..50).map(|i| format!("https://example.com/{i}")).collect();
        remember_links(&mut known, &many);
        assert_eq!(known.len(), 40, "link memory is bounded");
    }

    #[test]
    fn web_guard_stops_retry_loops() {
        assert_eq!(web_guard_action(0), WebGuardAction::Run);
        assert_eq!(web_guard_action(2), WebGuardAction::Run);
        assert_eq!(web_guard_action(3), WebGuardAction::Refuse);
        assert_eq!(web_guard_action(4), WebGuardAction::Refuse);
        assert_eq!(web_guard_action(5), WebGuardAction::Stop);
        assert_eq!(web_guard_action(20), WebGuardAction::Stop);
        assert!(WEB_GUARD_NUDGE.contains("reply.text"));
    }

    #[test]
    fn web_rate_separates_search_and_fetch_and_releases_failed_calls() {
        let mut r = WebRate::default();
        let t0 = Instant::now();
        r.reserve(WebKind::Search, t0).unwrap();
        // The normal search -> fetch flow must not trip a shared gap.
        assert!(r.reserve(WebKind::Fetch, t0).is_ok());
        assert_eq!(r.reserve(WebKind::Fetch, t0), Err(SlotError::Gap(web_gap())));

        let later = t0 + web_gap() + Duration::from_millis(1);
        let reserved = r.reserve(WebKind::Fetch, later).unwrap();
        r.unreserve(WebKind::Fetch, reserved);
        assert!(r.reserve(WebKind::Fetch, later).is_ok(), "failed calls release their slot");
    }

    #[test]
    fn web_rate_quota_expires_after_a_minute() {
        let mut r = WebRate::default();
        let t0 = Instant::now();
        let step = web_gap().max(Duration::from_millis(1));
        // Alternate buckets so the per-kind gap does not interfere.
        for i in 0..web_quota() {
            let kind = if i % 2 == 0 { WebKind::Search } else { WebKind::Fetch };
            r.reserve(kind, t0 + step * (i as u32)).unwrap();
        }
        let just_before = t0 + step * (web_quota() as u32) - Duration::from_millis(1);
        assert!(matches!(r.reserve(WebKind::Search, just_before), Err(SlotError::Quota(_))));
        assert!(r.reserve(WebKind::Search, t0 + Duration::from_secs(61)).is_ok());
    }

    #[test]
    fn ddg_results_decode_wrapped_links_and_snippets() {
        let body = r#"<div class="result results_links web-result">
          <h2 class="result__title"><a rel="nofollow" class="result__a"
            href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.accuweather.com%2Fen%2Fhk%2Fweather&amp;rut=abc">Hong Kong Current Weather</a></h2>
          <a class="result__snippet" href="//duckduckgo.com/l/?uddg=ignored">Get the <b>Hong Kong</b> weather forecast</a>
        </div>"#;
        let results = parse_ddg_results(body);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "Hong Kong Current Weather");
        assert_eq!(results[0].1, "https://www.accuweather.com/en/hk/weather");
        assert_eq!(results[0].2, "Get the Hong Kong weather forecast");
    }

    #[test]
    fn allowlist_accepts_verification_commands() {
        let ok = |v: &[&str]| allow_command(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert!(ok(&["cargo", "test"]).is_ok());
        assert!(ok(&["cargo", "check", "--all-targets"]).is_ok());
        assert!(ok(&["python", "-u", "app.py"]).is_ok());
        assert!(ok(&["python", "-m", "http.server", "8000"]).is_ok());
        assert!(ok(&["python", "-u", "-m", "http.server", "8000"]).is_ok());
        assert!(ok(&["python", "-m", "pytest", "-q"]).is_ok());
        assert!(ok(&["npm", "run", "dev"]).is_ok());
        assert!(ok(&["npx", "vite"]).is_ok());
        assert!(ok(&["go", "test", "./..."]).is_ok());
        assert!(ok(&["git", "status"]).is_ok());
        assert!(ok(&["rm", "-rf", "/"]).is_err());
        assert!(ok(&["curl", "http://example.com"]).is_err());
        assert!(ok(&["python", "-c", "import os"]).is_err());
    }

    #[test]
    fn tail_chars_never_splits_utf8() {
        let s = "日本語テキスト".repeat(30);
        let t = tail_chars(&s, 10);
        assert!(t.starts_with("... [truncated]\n"));
        assert!(t.ends_with('ト'));
        assert_eq!(tail_chars("short", 100), "short");
    }

    #[test]
    fn long_command_is_stopped_and_partial_output_kept() {
        if !python_available() { return; }
        let dir = temp_dir("timeout");
        fs::write(dir.join("sleepy.py"),
            "import time\nprint('started', flush=True)\ntime.sleep(30)\nprint('done')\n").unwrap();
        let argv = vec!["python".to_string(), "sleepy.py".to_string()];
        let started = Instant::now();
        let (ok, out) = run_process_within(&argv, &dir, Duration::from_millis(600)).unwrap();
        assert!(!ok, "timed out command must report failure");
        assert!(started.elapsed() < Duration::from_secs(10), "killed promptly");
        assert!(out.contains("started"), "partial output kept: {out:?}");
        assert!(out.contains("[timeout]"), "timeout explained: {out:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn background_job_streams_logs_and_stops() {
        if !python_available() { return; }
        let dir = temp_dir("jobs");
        fs::write(dir.join("hello.py"),
            "import time\nprint('bg-hello', flush=True)\ntime.sleep(30)\n").unwrap();
        let argv = vec!["python".to_string(), "-u".to_string(), "hello.py".to_string()];
        let started = start_background(&argv, &dir).unwrap();
        let id = started.split_whitespace().nth(1).expect("job id").to_string();

        let mut logs = String::new();
        for _ in 0..60 {
            logs = job_logs(&id).unwrap();
            if logs.contains("bg-hello") { break; }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(logs.contains("bg-hello"), "console output captured: {logs:?}");
        assert!(logs.contains("running"), "job state reported: {logs:?}");

        assert!(stop_job(&id).unwrap().contains("stopped"));
        std::thread::sleep(Duration::from_millis(200));
        assert!(job_logs(&id).unwrap().contains("exited"), "job is reaped");
        let _ = fs::remove_dir_all(&dir);
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
            sidecar: Sidecar::new("http://127.0.0.1:1", Path::new(".")),
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
            sidecar: Sidecar::new("http://127.0.0.1:1", Path::new(".")),
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
            sidecar: Sidecar::new("http://127.0.0.1:1", Path::new(".")),
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
    fn memory_files_are_created_and_read_back() {
        let base = std::env::temp_dir().join(format!("perihel-mem-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let home = base.join("home");
        let ws = base.join("ws");
        fs::create_dir_all(&ws).unwrap();

        memory_add_at(&home, &ws, "local", "always run cargo test").unwrap();
        memory_add_at(&home, &ws, "local", "second\nnote").unwrap();
        memory_add_at(&home, &ws, "global", "prefer ASCII output").unwrap();
        assert!(memory_add_at(&home, &ws, "bogus", "x").is_err());

        let local = fs::read_to_string(ws.join("MEMORY.md")).unwrap();
        assert!(local.starts_with("# Project memory"));
        assert!(local.contains("- always run cargo test"));
        assert!(local.contains("- second note"), "multi-line notes are joined: {local:?}");

        let global = fs::read_to_string(home.join("MEMORY.md")).unwrap();
        assert!(global.starts_with("# Global memory"));
        assert!(global.contains("- prefer ASCII output"));

        assert_eq!(read_memory(&ws.join("MEMORY.md")).unwrap().matches("- ").count(), 2);
        assert!(read_memory(&ws.join("missing.md")).is_none());
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn body_text_is_never_erased() {
        let app = App {
            client: Client::new("http://127.0.0.1:1", "tiny-model", None, false),
            sidecar: Sidecar::new("http://127.0.0.1:1", Path::new(".")),
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
            sidecar: Sidecar::new("http://127.0.0.1:1", Path::new(".")),
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
            sidecar: Sidecar::new("http://127.0.0.1:1", Path::new(".")),
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

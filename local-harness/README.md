# Perihel (Rust)

The `perihel` command — a local tool-calling agent harness for llama.cpp.
Replaces the Python prototype in `../perihel-cli-lab` (kept as reference).

## Why Rust

- Single self-contained binary, no Python environment needed.
- Installed on PATH as `perihel` (in `%USERPROFILE%\.cargo\bin`).
- Ctrl+C stops it at any point, including mid-generation.

## Install / rebuild

```powershell
cargo build --release
Copy-Item .\target\release\perihel-rs.exe "$env:USERPROFILE\.cargo\bin\perihel.exe" -Force
perihel --version
```

## Use

```powershell
perihel                                   # interactive CLI
perihel --tui                             # full-screen harness-style UI
perihel --prompt "write hello.py and test it"
perihel --model my-model                  # choose a model at startup
perihel --workspace C:\some\project       # sandbox root
```

In-session commands: `/help`, `/model [name]`, `/session`, `/memory`,
`/rounds <n>`, `/thinking on|off`, `/reset`, `/tools`, `/workspace`, `/quit`.
`/model` with **no argument lists available models** (from llama.cpp
`/v1/models` plus local `.gguf` files). `/rounds 0` (the default) means
**unlimited** — there is no 8-round cap; the model can keep checking syntax
and fixing code until it calls `reply.text`. Ctrl+C stops whenever you want.

Reasoning can be streamed live in dim text under a `[thinking]` header with
`--thinking` or `/thinking on`. It is **off by default**: measured on this
machine, even "hi" burned 8,000 reasoning tokens in ~160 s with thinking on,
while the same request took **0.78 s** with it off. If reasoning consumes the
whole budget without producing a tool call, the harness retries once with
thinking off. Every generation prints `[stats] N tokens in X.Xs = Y.Y tok/s`
(in the TUI, in the status bar).

In `--cli` mode the completion is streamed, but with the grammar on every
completion is a tool-call JSON document rather than prose, so the raw JSON is
no longer typed out and then printed again as the parsed reply: you see
`[round N] thinking...`, the stats line, and then `[assistant] ...` once. With
`--no-grammar` the model's text still streams typewriter-style.

The llama.cpp server is managed for you: it is **not started when the harness
opens**. The first request (or a `/model` switch) starts it, the TUI shows
`starting model <name>...` / `model ready`, and it stays loaded for the whole
session. It is stopped when the harness exits (quit, Ctrl+C, or `--prompt`
finishing), so no `llama-server.exe` is left behind. An already-running server
is reused and left alone on exit. GGUF files are discovered in
`<workspace>/models/`, sibling project folders, `PERIHEL_MODELS_DIR`, and
`~/.llama/models` — so launching from a folder with no models still finds the
local Qwen coder model, which is preferred automatically. Override the choice
with `--model`, `PERIHEL_MODEL` (full path), or `/model`. llama-server is found
on PATH, under WinGet packages, or via `PERIHEL_LLAMA_SERVER` / `LLAMA_BIN`.

The llama.cpp server here runs with `--parallel 1`, so a long generation blocks
later requests. Keep thinking off for interactive use.

Typing `/` in the TUI opens a live **command menu**. Up/Down (or `j`/`k`) move
the `>` pointer, **Tab** completes the highlighted command, and **Enter** runs
it (opencode/oh-my-pi style). Typing filters the list; the menu resets when the
text changes. `/session` opens a session browser with a `>` pointer over saved
conversations (`~/.perihel/sessions/`); Enter restores one, and `/session <n>`
does the same from the CLI. Every turn is saved automatically.

When no `--model` is given, the first available model is auto-selected instead
of the `local-model` placeholder. llama.cpp reports the full GGUF path, so the
display name is shortened to the file name. `/model` opens a **model picker**
with a `>` pointer: arrows or `j`/`k` move, a number jumps to that entry, Enter
selects, `q` cancels. `/model <name>` sets it directly.

While a message is generating, the TUI shows a `generating... (Ctrl+C stops)`
line so the session does not look locked; the status bar shows the round and the
last timing stats. The input line is cleared at the start of every redraw to
avoid overlapping text on short windows.

The layout uses the **real terminal size** reported by the console, not the
`COLUMNS` environment variable (Windows never sets it). Before this, the TUI
laid itself out for 120 columns inside a narrow window: lines were cut off at
the right edge, the status bar wrapped onto a second row, and the body scrolled
past the header. Long input now scrolls horizontally with a `<` marker instead
of wrapping into the status bar. `cargo test` covers the invariants (no rendered
row wider than the window, no frame taller than the window).

**PgUp / PgDn** scroll the conversation (new output snaps back to the bottom);
the status bar shows `SCROLLED +N` while you are reading older messages.
`--tui` is the default and is still accepted as a flag for the README and
desktop shortcut.

Assistant replies are rendered as **Markdown** instead of raw text: fenced code
blocks get a dark background (`48;5;236`) and a small built-in highlighter
(keywords, strings, comments, numbers, `diff` +/-) for rust, python, js/ts,
json, shell and friends; inline code is amber (`38;5;222`), `**bold**` is bold,
`*italic*` is italic, headings are bold blue, list items get a `-` indent, `>`
quotes get a dim bar, and `[text](url)` keeps the text with a dim URL. Wrapping
happens on the styled spans themselves, so every rendered row still fits the
window at any width (covered at 40 and 64 columns). User, tool and error lines
stay plain, and `--cli` output stays raw so it can be piped.

The input line is a real line editor, not append-only: **Ctrl+W** deletes the
word before the cursor, **Ctrl+U** clears to the start, **Ctrl+K** kills to the
end, **Ctrl+A**/**Ctrl+E** jump to start/end, **Ctrl+Left/Right** (or
**Alt+Left/Right**) move a word at a time, and Home/End/Delete behave as
**Up/Down** scroll the conversation one line at a time when the slash menu is
closed (the status bar shows `SCROLLED +N`); while the menu is open they move the
`>` pointer. **Ctrl+Up/Ctrl+Down** (or Alt) walk prompts from this session;
editing a recalled line keeps it as your draft. **PgUp/PgDn** scroll by a screen.

## Memory

Two plain Markdown files are read at the start of **every turn** and appended to
the system prompt:

| Scope | File | Use it for |
|---|---|---|
| global | `~/.perihel/MEMORY.md` | durable facts and preferences true in every project |
| local | `<workspace>/MEMORY.md` | build commands, conventions and quirks of this repo |

The model can append notes itself with
`{"tool":"memory.add","args":{"scope":"local","text":"..."}}` (or `"global"`),
which writes a `- note` line under a heading and re-reads the file next turn.
You can edit either file by hand, check it into the repo, or just ask the model
to remember something. `/memory` prints both paths and their line counts.
Memory is capped at 4,000 characters per file so it cannot crowd out the prompt.

## Background commands and verification

`run.cmd` waits for the command; for servers, watchers and anything long-lived
use `run.bg`. It starts the process with stdin closed, `stdout`+`stderr` going
to `<workspace>/.perihel/jobs/<id>.log`, and returns a job id. `run.logs` reads
the tail of that log (with the job's running/exited state) so the model can see
console messages such as tracebacks or `Serving HTTP on ...`, and `run.stop`
kills the whole process tree. Python jobs get `PYTHONUNBUFFERED=1` so output
appears immediately.

Foreground commands are bounded too: output goes to a temp file (a full pipe
buffer can no longer deadlock the harness) and after **120 s** the process tree
is killed and the partial output plus a `[timeout]` line is returned instead of
an opaque error. Set `PERIHEL_CMD_TIMEOUT` (seconds) to change it.

Because the model can now start something, read its console, fetch
`http://127.0.0.1:PORT` with `web.fetch`, and stop it again, the system prompt
tells it to verify before answering: run the project's real check, read the
console, fix failures, and say exactly what is unverified instead of claiming
success. `web.fetch` accepts `http://127.0.0.1:*`, so a dev server can be
smoke-tested end to end without leaving the machine.

## How tool calling works

llama.cpp has no native tool-call API, so `tools.gbnf` is embedded in the binary
and sent as the `grammar` field of `/v1/chat/completions`. The sampler physically
cannot emit an unknown tool or malformed arguments. One JSON tool call per turn;
its result is fed back as a user message (Perihel `copilot.rs` pattern).

Tools: `fs.list`, `fs.read`, `fs.write`, `fs.mkdir`, `fs.copy`, `run.cmd`,
`run.bg`, `run.logs`, `run.stop`, `run.test`, `web.search`, `web.fetch`,
`memory.add`, `reply.text`. All paths are
workspace-sandboxed, commands are
allowlisted with no shell, bounded output, 120 s timeout (see above).
`web.search` uses keyless **DuckDuckGo HTML** (8 ranked results, Bing RSS as
fallback) and `web.fetch` strips script/style/nav from a page (2 MB cap, 6k
chars). Search and fetch have **separate** rate-limit buckets (1 s gap, 60
calls/min total by default; tune with `PERIHEL_WEB_GAP_MS` and
`PERIHEL_WEB_MAX_MIN`), so the normal search -> fetch flow is never refused: a
short gap is waited out inside the tool, and failed calls do not consume a
slot. A 404/5xx is returned with
its URL and body so the model can try the next link instead of answering from
memory. When a page turns out to be a JavaScript shell (CSS/JS leaked into the
extracted text), `web.fetch` follows the page's advertised RSS/Atom `<link>`
and includes the feed items, so live data can be read without any per-site tool.
After **3 consecutive failed web calls** the harness injects a nudge
("answer with what you know, state what is unverified") and refuses further
fetches; after 5 it ends the turn rather than letting the model guess dozens of
URLs. `enable_thinking` is off to avoid the reasoning loops measured with
Qwen3.5-4B.

**No invented URLs.** The system prompt requires `web.search` first whenever
the exact address is unknown. `web.fetch` enforces that generically: it only
fetches an exact link returned by a recent search (or any URL on a domain that
appeared in one), plus `127.0.0.1`/`localhost` for verifying a dev server.
Everything else is refused with a `call web.search first` message. No
per-topic or per-site tools are hard-coded.

`memory.add` is for durable facts only (project commands, conventions, stable
preferences). For greetings, a bare `test`, or a prompt with no concrete task,
the model is told to answer with `reply.text` immediately - no tool calls and
no notes about the conversation itself.

The allowlist covers the usual verification commands: `python script.py`,
`python -m pytest|http.server|py_compile|json.tool`, `node`/`npm`/`npx`
(`test`, `start`, `run <script>`, `build`, `tsc`, `vite`, `next`, ...), `cargo`
(`build`, `test`, `check`, `clippy`, `fmt`, `run`), `go`, `dotnet`, `git`
(read-only), `rg` and `pytest`.

## Desktop shortcuts

`Perihel CLI.lnk` runs `perihel` in a console; `Perihel TUI.lnk` runs
`perihel --tui`. Both default to workspace `threejs-model-lab` and the
llama.cpp server on `http://127.0.0.1:8090`.

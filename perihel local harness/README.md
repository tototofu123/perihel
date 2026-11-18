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

In-session commands: `/help`, `/model [name]`, `/session`, `/rounds <n>`,
`/thinking on|off`, `/reset`, `/tools`, `/workspace`, `/quit`.
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

The llama.cpp server here runs with `--parallel 1`, so a long generation blocks
later requests. Keep thinking off for interactive use.

Typing `/` in the TUI opens a live **command suggestion** popup (Tab completes,
Enter runs). `/session` opens a session browser with a `>` pointer over saved
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

## How tool calling works

llama.cpp has no native tool-call API, so `tools.gbnf` is embedded in the binary
and sent as the `grammar` field of `/v1/chat/completions`. The sampler physically
cannot emit an unknown tool or malformed arguments. One JSON tool call per turn;
its result is fed back as a user message (Perihel `copilot.rs` pattern).

Tools: `fs.list`, `fs.read`, `fs.write`, `fs.mkdir`, `fs.copy`, `run.cmd`,
`run.test`, `web.search`, `web.fetch`, `reply.text`. All paths are
workspace-sandboxed, commands are
allowlisted with no shell, 60 s timeout, bounded output. `web.search` uses
keyless Bing RSS (8 results, 5 s gap, 10/min) and `web.fetch` strips
script/style/nav from a page (2 MB cap, 6k chars). `enable_thinking` is
off to avoid the reasoning loops measured with Qwen3.5-4B.

## Desktop shortcuts

`Perihel CLI.lnk` runs `perihel` in a console; `Perihel TUI.lnk` runs
`perihel --tui`. Both default to workspace `threejs-model-lab` and the
llama.cpp server on `http://127.0.0.1:8090`.

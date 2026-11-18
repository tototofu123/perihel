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

In-session commands: `/help`, `/model [name]`, `/rounds <n>`,
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

## How tool calling works

llama.cpp has no native tool-call API, so `tools.gbnf` is embedded in the binary
and sent as the `grammar` field of `/v1/chat/completions`. The sampler physically
cannot emit an unknown tool or malformed arguments. One JSON tool call per turn;
its result is fed back as a user message (Perihel `copilot.rs` pattern).

Tools: `fs.list`, `fs.read`, `fs.write`, `fs.mkdir`, `fs.copy`, `run.cmd`,
`run.test`, `reply.text`. All paths are workspace-sandboxed, commands are
allowlisted with no shell, 60 s timeout, bounded output. `enable_thinking` is
off to avoid the reasoning loops measured with Qwen3.5-4B.

## Desktop shortcuts

`Perihel CLI.lnk` runs `perihel` in a console; `Perihel TUI.lnk` runs
`perihel --tui`. Both default to workspace `threejs-model-lab` and the
llama.cpp server on `http://127.0.0.1:8090`.

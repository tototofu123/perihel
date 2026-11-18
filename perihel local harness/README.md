# Perihel Local Harness

A standalone Rust binary that brings Perihel's tool-calling copilot pattern to a
terminal: the model decides one structured tool call per turn, deterministic code
executes it, and the observation is fed back. Built for local llama.cpp servers;
no cloud and no Python runtime required.

## Why Rust

- One self-contained binary; install it as `perihel` on PATH.
- Ctrl+C stops it at any point, including mid-generation.
- No fixed round cap — the model can keep checking syntax, running tests and
  fixing files until it calls `reply.text`.

## Build and install

```powershell
cd "perihel local harness"
cargo build --release
# copy target/release/perihel-rs.exe to a directory on PATH, e.g.:
Copy-Item .\target\release\perihel-rs.exe "$env:USERPROFILE\.cargo\bin\perihel.exe" -Force
perihel --version
```

## Use

```powershell
perihel                          # full-screen harness UI (default)
perihel --cli                    # plain line-oriented CLI
perihel --prompt "write hello.py and test it"
perihel --model my-model         # choose a model at startup
perihel --workspace C:\some\project
```

In-session: `/help`, `/model <name>`, `/rounds <n>`, `/reset`, `/tools`,
`/workspace`, `/quit`. `/rounds 0` (default) means unlimited; use `/rounds 25`
if you want a cap. Ctrl+C stops at any time.

## Tool calling without a native API

llama.cpp has no native tool-call API, so `tools.gbnf` is embedded in the binary
and sent as the `grammar` field of `/v1/chat/completions`. The sampler physically
cannot emit an unknown tool or malformed arguments. This follows
`desktop/grammar/perihel-tools.gbnf` and `copilot.rs` (bounded tool rounds,
observations fed back as user messages, graceful degrade-to-text).

Tools: `fs.list`, `fs.read`, `fs.write`, `fs.mkdir`, `fs.copy`, `run.cmd`,
`run.test`, `reply.text`. Paths are workspace-sandboxed (absolute paths and `..`
are refused), commands are allowlisted with no shell, 60 s timeout and bounded
output. `enable_thinking` is disabled to avoid reasoning loops consuming the
token budget with thinking models.

## Configuration

Defaults: llama.cpp at `http://127.0.0.1:8090`, model `local-model`, workspace
`.`, temperature 0.4 / top-p 0.95 / top-k 20. Override with `--base-url`,
`--model`, `--workspace`, `--grammar`, or `--no-grammar`.

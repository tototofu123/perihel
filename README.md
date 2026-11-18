# Perihel

_Perihel is a monorepo of small, self-hosted tools built around one idea: your
machine and your data stay yours._

- A **capture companion** that understands your thoughts — desktop app, Android
  app, and a tiny self-hosted AI server. Everything runs on free tiers; your
  data lives on your own VM.
- Two **local coding-agent harnesses** that drive llama.cpp with constrained
  tool calling: one on your desktop (`local-harness`), one on the VM
  (`vm-harness`).

```
local-agent/     Rust egui capture companion - dictation (sherpa-onnx), local
                 llama.cpp copilot, capture resurfacing
local-harness/   Rust CLI/TUI coding agent - llama.cpp + GBNF-constrained tool
                 calls, web search/fetch, sessions
vm/              Python sync server + AI processor (NVIDIA NIM tool calling,
                 long-term memory, reminders), Caddy TLS, systemd units
vm-harness/      systemd units + scripts that keep the agent loop alive on the VM
mobile/          Kotlin + Compose Android app - AI chat capture, filterable
                 logbook, alerts
ops/             budget kill-switch Cloud Function (auto-stops the VM when the
                 bill crosses HK$0.01)
docs/            SELF-HOST.md - rebuild the whole stack from scratch, $0/month
```

These are independent components that happen to share one repository: the
capture product (`local-agent`, `mobile`, `vm`) and the two harnesses
(`local-harness` runs on your machine, `vm-harness` runs on the server). Each
folder has its own README and its own build.

## How it works

You send a thought — on the phone's chat, or `Ctrl+Shift+P` on the desktop:

1. **It is saved first.** The capture hits the server's SQLite before any AI
   call, so a dead API can never lose a thought.
2. **AI processes it** (toggleable, default on): expands what you meant,
   assigns a category (`idea · todo · link · note · question · person ·
   event`), extracts reminders when you mention a time, and files durable
   facts about you into long-term memory that improves future processing.
3. **It resurfaces at the right moment**: your explicit reminders fire at
   their time (any hour); smart nudges arrive at most every 4 hours, only
   during active hours (default 09:00–21:00), via ntfy pushes to your phone.
4. **Everything syncs** — phone, desktop, and server converge by
   last-write-wins; each device keeps a full local copy and works offline.
   Deleting is a 30-day soft delete, recoverable from the server DB.

## Stack

- **Local agent** — Rust, egui, sherpa-onnx (on-device STT), llama.cpp sidecar
  for the offline copilot, NVIDIA NIM (free API) for the cloud copilot.
- **Local harness** — Rust single binary on top of llama.cpp's
  OpenAI-compatible endpoint. Tool calls are forced by an embedded GBNF
  grammar (no native tool API), so the sampler cannot emit a malformed call.
  Full-screen TUI, `/` command menu, session browser, web search/fetch.
- **VM** — Python stdlib only (no pip on the VM): HTTPS via Caddy +
  Let's Encrypt, bearer-token auth, SQLite. AI via NVIDIA NIM's
  OpenAI-compatible API with tool calls; a rate limiter and retry-once
  policy protect the free tier.
- **Mobile** — Kotlin + Jetpack Compose, zero networking dependencies.
- **Hosting** — GCP always-free e2-micro, free static IP, sslip.io domain,
  budget alert at HK$0.01 wired to a Cloud Function that stops the VM.

## Quick start

**Local agent** — see `local-agent/README.md`.

**Local harness** — needs a llama.cpp server on `http://127.0.0.1:8090`:

```powershell
cd local-harness
cargo build --release
Copy-Item .\target\release\perihel-rs.exe "$env:USERPROFILE\.cargo\bin\perihel.exe"
perihel                     # full-screen TUI (default)
perihel --cli               # plain line-oriented CLI
```

**VM** — `docs/SELF-HOST.md` walks through the whole free-tier build, and
`ops/deploy.sh` ships `vm/*.py` plus `vm/systemd/*` to the server.

**Mobile** — needs the Android SDK:

```bash
cd mobile
./gradlew assembleDebug -PPERIHEL_URL=https://your-vm -PPERIHEL_TOKEN=yourtoken
# APK at app/build/outputs/apk/debug/ — or just launch it and enter URL+token
```

**Server** — `docs/SELF-HOST.md` walks through the whole free-tier build.

## Principles

- **Free forever** — every component sits on an always-free tier, guarded by
  a HK$0.01 budget alert + auto-kill.
- **Save first, think second** — no AI failure can lose a capture.
- **Local-first** — sync is additive; offline Perihel is complete Perihel.

MIT

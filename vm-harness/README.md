# vm-harness

Snapshot of what is **actually running** on the Perihel VM
(`perihel-sync`, project `perihel-tototofu`, `us-west1-b`), captured
2026-11-18. This is the runtime scaffolding around the app code: the
systemd units/timers, their model drop-ins, and the helper scripts that
only exist on the VM.

The app code itself (`server.py`, `ai.py`, `digest.py`, …) lives in
`vm/` and is deployed by `ops/deploy.sh`. This folder
does **not** duplicate it — except for `tick.py` (see drift note 2).

## Contents

| Path | What it is |
| --- | --- |
| `systemd/perihel-*.service`, `*.timer` | Live unit files from `/etc/systemd/system/` (8 files, byte-identical to the VM except `perihel-ntfy.service`, see drift note 1) |
| `systemd/perihel-*.service.d/model.conf` | Drop-ins pinning the model env vars (`PERIHEL_AI_MODEL`, `PERIHEL_BOT_MODEL`, `PERIHEL_SEARCH_MODEL` = `nvidia/nemotron-3-ultra-550b-a55b`). No secrets. |
| `scripts/tick.py` | The every-10-minutes brain the ntfy timer runs: AI queue, reminders, nudges, 30-day purge, daily budget-fallback check. Live copy — see drift note 2. |
| `scripts/budget_fallback.py` | Called by `tick.py` once a day: creates the Monitoring fallback alert (issue #3) as soon as the billing metric exists after first spend. Token comes from the VM metadata server — no key file, no secret. |
| `scripts/telegram.py` | Telegram push helper. Token is read from `/opt/perihel-sync/telegram.token` (chmod 600, never committed). |
| `scripts/ntfy_push.py` | Superseded by `tick.py`. Still on the VM, no longer run by anything. Kept here for reference. |

Live state on 2026-11-18: `perihel-sync` and `perihel-telegram`
enabled + active; `perihel-digest.timer` and `perihel-ntfy.timer`
enabled + active; `perihel-watchdog.timer` **disabled** (watchdog
disabled at the owner's request, 2026-11-15 — see issue #38).

## Reinstall on a fresh VM

```bash
# units + drop-ins (paths mirror /etc/systemd/system/ exactly)
sudo cp vm-harness/systemd/*.service vm-harness/systemd/*.timer /etc/systemd/system/
sudo cp -r vm-harness/systemd/*.service.d /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now perihel-sync perihel-telegram \
    perihel-digest.timer perihel-ntfy.timer
# watchdog timer stays disabled unless deliberately re-enabled

# helper scripts
sudo cp vm-harness/scripts/*.py /opt/perihel-sync/
sudo chown root:root /opt/perihel-sync/*.py   # match live ownership
```

## Deliberately NOT in this folder

These live on the VM and must never be committed: `nim.key`,
`github.key`, `telegram.key`, `telegram.token`, `token`, `key.pem`,
`cert.pem`, `cert-public.pem`, `ntfy-topic` (the topic name acts as
its password), `sync.db`, `media/`. The scripts here read them from
`/opt/perihel-sync/` at runtime; nothing in this folder contains a
secret value.

## Drift vs the rest of the repo (read before changing either side)

1. **`perihel-ntfy.service` runs `tick.py` on the VM, but
   `vm/systemd/perihel-ntfy.service` still points at
   `ntfy_push.py`.** `ops/deploy.sh` even documents this ("nothing on
   the VM runs tick.py today", issue #10) — that claim is now out of
   date: the live unit was switched to `tick.py` outside of deploy.sh.
   The copy in this folder is the live one.
2. **Live `tick.py` has a `check_budget_fallback()` step that
   `vm/tick.py` does not.** The repo copy is behind
   live; updating it (and the deploy script) is follow-up work, not
   part of this snapshot.
3. **The VM's deployed `ai.py` predates main.** PR #39 (Gemini
   fallback) merged 2026-11-16 but has not been deployed yet — the
   `gemini.key` placement is still outstanding. Deploying main will
   replace the VM's `ai.py`/`search.py`; nothing in this folder is
   affected.

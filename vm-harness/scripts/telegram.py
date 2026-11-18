"""Telegram push channel for Perihel — primary notification path.

Bot: t.me/perihelcloudbot. Token lives at /opt/perihel-sync/telegram.token
(chmod 600, never committed). The chat_id is discovered automatically: the
first chat that ever messages the bot claims it, and the value is stored
ONCE (INSERT OR IGNORE — first-wins) so a later stranger cannot hijack the
feed. To rebind, clear the `telegram_chat_id` row in the settings table.
"""
import json
import os
import urllib.request

DATA_DIR = os.environ.get("PERIHEL_SYNC_DIR", "/opt/perihel-sync")
TOKEN_PATH = os.path.join(DATA_DIR, "telegram.token")
API = "https://api.telegram.org"

_cache = {"token": None}


def _token() -> str:
    if _cache["token"] is None:
        with open(TOKEN_PATH) as f:
            _cache["token"] = f.read().strip()
    return _cache["token"]


def _call(method: str, payload: dict, timeout: int = 15) -> dict:
    req = urllib.request.Request(
        f"{API}/bot{_token()}/{method}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def chat_id(conn) -> str | None:
    row = conn.execute(
        "SELECT value FROM settings WHERE key = 'telegram_chat_id'").fetchone()
    v = row["value"] if row else None
    return v if v else None


def configured() -> bool:
    return os.path.isfile(TOKEN_PATH)


def discover(conn) -> str | None:
    """Poll getUpdates once; the first chat to message the bot becomes ours."""
    if not configured() or chat_id(conn):
        return chat_id(conn)
    try:
        out = _call("getUpdates", {"timeout": 0, "limit": 100})
    except Exception as e:
        print(f"telegram getUpdates failed: {e}", flush=True)
        return None
    ids = []
    for upd in out.get("result", []):
        msg = upd.get("message") or upd.get("edited_message") or {}
        cid = (msg.get("chat") or {}).get("id")
        if cid is not None:
            ids.append(cid)
    if not ids:
        return None
    cid = str(ids[-1])
    conn.execute(
        "INSERT OR IGNORE INTO settings (key, value) VALUES ('telegram_chat_id', ?)",
        (cid,))
    conn.commit()
    print(f"telegram chat_id locked in: {cid}", flush=True)
    send(conn, "✅ Perihel connected — reminders and nudges will arrive here.")
    return cid


def send(conn, text: str) -> bool:
    cid = chat_id(conn)
    if not cid:
        return False
    try:
        _call("sendMessage", {"chat_id": int(cid), "text": text[:4000],
                              "disable_web_page_preview": True})
        return True
    except Exception as e:
        print(f"telegram send failed: {e}", flush=True)
        return False

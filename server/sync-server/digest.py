#!/usr/bin/env python3
"""Perihel daily digest — AI news, repo PRs/issues summary, and plans.

Stdlib only. Builds the morning digest and sends it to every Telegram
chat the bot has seen. Triggered daily by the perihel-digest systemd
timer (08:00 VM-local time), or on demand via POST /v1/digest.

Why a VM timer and not the agent-bus scheduler (agent-bus #15): the
bus is localhost-only on the user's Windows machine and the VM cannot
see it (there is no Telegram bridge code in agent-bus), so a bus
task.offer can never reach the VM. The digest therefore runs on the
VM itself; the bus scheduler stays the mechanism for scheduled jobs
generally, but it cannot trigger this one.

Sections — defaults, each configurable (see below):
  - AI news: real news *articles* from a fixed list of trusted
    sources (NEWS_FEEDS), each read straight from its own RSS/Atom
    feed — no search engine in the loop at all (Bing News was tried
    2026-11-15 and dropped the same evening: its results were mostly
    stock-market spam, and it still returned section pages). Each
    article is fetched and summarised to 2 sentences by the worker
    role's model — a real summary of one story, in the model's own
    words, not cropped fragments off a section homepage (that was
    the 2026-11-15 bug: homepages are a jumble of headlines, so the
    model just lifted sentences off the page, and once asked "Which
    specific article would you like me to summarise?"). General-tech
    feeds are filtered to AI stories by keyword. An unfetchable
    article is dropped and replaced; a failed feed is skipped and
    the other feeds fill in. Every item carries a subheading with
    its source and publish date AND time (HKT) next to the link.
  - Repos: open PRs and open issues (most recently updated first, max 5
    each) for PERIHEL_DIGEST_REPOS (comma-separated owner/repo;
    default agent-bus, perihel-v2, hkust-climbing-app), via the GitHub
    REST API. Unauthenticated (60 req/hour) is enough for a daily
    digest of public repos; a read-only github.key is needed only for
    private repos.
  - Plans: plans.md in DATA_DIR (one plan per line) if it exists; else
    the top open issues across the configured repos, with a note that
    plans.md is missing.

The digest never fabricates content: a failed search, a failed GitHub
call, or an empty section is reported as such in that section.
"""
import html as html_lib
import json
import os
import re
import sqlite3
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

DATA_DIR = os.environ.get("PERIHEL_SYNC_DIR", "/opt/perihel-sync")
DB_PATH = os.path.join(DATA_DIR, "sync.db")
GITHUB_KEY_PATH = os.path.join(DATA_DIR, "github.key")
PLANS_PATH = os.path.join(DATA_DIR, "plans.md")

DEFAULT_REPOS = ["tototofu123/agent-bus", "tototofu123/perihel-v2",
                 "HKUST-Climbing-Society/hkust-climbing-app"]
# Trusted news sources, read straight from their own RSS/Atom feeds.
# (name, feed URL). A feed that fails or has nothing fresh is skipped
# and the others fill in. Reuters is NOT here: it blocks fetches
# (401) and its feed is unusable from the VM. AI News is NOT here
# either: its feed returns non-XML. To add or remove a source, edit
# this list — nothing else needs to change.
NEWS_FEEDS = [
    ("ScienceDaily", "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml"),
    ("MIT News", "https://news.mit.edu/rss/topic/artificial-intelligence2"),
    ("TechCrunch", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("Wired", "https://www.wired.com/feed/tag/ai/latest/rss"),
    ("BBC", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("CNBC", "https://www.cnbc.com/id/19854910/device/rss/rss.html"),
    ("The Verge", "https://www.theverge.com/rss/index.xml"),
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
    ("The Guardian", "https://www.theguardian.com/technology/rss"),
]
NEWS_MAX_RESULTS = 5
FEED_MAX_ITEMS = 4  # newest items taken from each feed (top 2 first,
                    # then the rest if more slots need filling)
SUMMARY_MAX_CHARS = 380  # per-item summary cap — the whole digest
                         # must fit in ONE Telegram message (4096
                         # chars), so summaries stay to 2 sentences
REPO_MAX_ITEMS = 5        # per repo, per kind (PRs / issues)
PLANS_MAX_LINES = 10
GITHUB_TIMEOUT_S = 15
GITHUB_UA = "Perihel-Digest"
TG_CHUNK_LIMIT = 4000     # Telegram messages cap at 4096 chars

_github_token_cache = None


def now_local() -> datetime:
    return datetime.now().astimezone()  # VM local time (set it to HKT)


def db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------- config ----------------

def parse_repos(value: str | None) -> list[str]:
    """Comma-separated owner/repo list (PERIHEL_DIGEST_REPOS). Junk
    entries (no slash, extra slashes, blanks) are dropped; an empty
    result falls back to DEFAULT_REPOS."""
    repos = []
    for part in (value or "").split(","):
        part = part.strip()
        if part.count("/") == 1 and not part.startswith("/") \
                and not part.endswith("/"):
            repos.append(part)
    return repos or list(DEFAULT_REPOS)


def get_repos() -> list[str]:
    return parse_repos(os.environ.get("PERIHEL_DIGEST_REPOS"))


# ---------------- GitHub ----------------

def _github_token():
    """Read-only PAT from github.key, or None (unauthenticated). Cached;
    never printed or logged anywhere."""
    global _github_token_cache
    if _github_token_cache is not None:
        return _github_token_cache
    try:
        with open(GITHUB_KEY_PATH) as f:
            _github_token_cache = f.read().strip() or None
    except OSError:
        _github_token_cache = None
    return _github_token_cache


def github_get(url: str):
    """One GitHub REST GET. Returns the parsed JSON. Raises HTTPError /
    URLError — the caller turns those into a section note (never a
    crash, never a fabricated item)."""
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": GITHUB_UA,
               "X-GitHub-Api-Version": "2022-11-28"}
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=GITHUB_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode())


def parse_pulls(payload) -> list[dict]:
    """A /pulls response -> [{number, title, url, updated_at}]. Pure."""
    out = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        out.append({"number": item.get("number"),
                    "title": (item.get("title") or "").strip(),
                    "url": item.get("html_url") or "",
                    "updated_at": item.get("updated_at") or ""})
    return out


def parse_issues(payload) -> list[dict]:
    """An /issues response -> issues only. The issues endpoint also
    returns PRs (they carry a "pull_request" key) — those are dropped
    here, they are listed under PRs. Pure."""
    out = []
    for item in payload or []:
        if not isinstance(item, dict) or "pull_request" in item:
            continue
        out.append({"number": item.get("number"),
                    "title": (item.get("title") or "").strip(),
                    "url": item.get("html_url") or "",
                    "updated_at": item.get("updated_at") or ""})
    return out


def _github_error_note(repo: str, e: Exception) -> str:
    """A short honest note for a failed repo fetch. No token, no URL."""
    if isinstance(e, urllib.error.HTTPError):
        if e.code == 404:
            return (f"{repo}: unreachable (private repo or wrong name — "
                    "a github.key token is needed for private repos)")
        if e.code == 403:
            remaining = ""
            try:
                remaining = e.headers.get("X-RateLimit-Remaining", "")
            except Exception:
                pass
            if remaining == "0":
                return (f"{repo}: GitHub rate limit exhausted "
                        "(unauthenticated, 60/hr) — add a github.key")
            return f"{repo}: GitHub denied access (403)"
        return f"{repo}: GitHub returned HTTP {e.code}"
    return f"{repo}: GitHub request failed ({type(e).__name__})"


def fetch_repo_summary(repo: str) -> dict:
    """Open PRs + open issues for one repo, most recently updated first.
    Never raises: a failure becomes an "error" note for that repo."""
    base = f"https://api.github.com/repos/{repo}"
    qs = urllib.parse.urlencode({"state": "open", "sort": "updated",
                                 "direction": "desc",
                                 "per_page": REPO_MAX_ITEMS})
    try:
        pulls = parse_pulls(github_get(f"{base}/pulls?{qs}"))
    except Exception as e:
        return {"pulls": [], "issues": [],
                "error": _github_error_note(repo, e)}
    try:
        issues = parse_issues(github_get(f"{base}/issues?{qs}"))
    except Exception as e:
        return {"pulls": pulls, "issues": [],
                "error": _github_error_note(repo, e)}
    return {"pulls": pulls, "issues": issues, "error": None}


# ---------------- plans ----------------

def parse_plans(text: str) -> list[str]:
    """plans.md: one plan per line. Blank lines and '#' comments are
    skipped; capped at PLANS_MAX_LINES. Pure."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
        if len(out) >= PLANS_MAX_LINES:
            break
    return out


def read_plans() -> list[str] | None:
    """The plans.md file, or None when it does not exist (the caller
    falls back to the top open issues)."""
    try:
        with open(PLANS_PATH) as f:
            return parse_plans(f.read())
    except OSError:
        return None


# ---------------- AI news ----------------

def get_search_module():
    """The search worker, or None when search.py is not deployed. Lazy
    so this module stays importable without it."""
    try:
        import search as search_mod
        return search_mod
    except ImportError:
        return None


ITEM_FETCH_CHARS = 6000  # chars of each news article sent to the model


def _domain(url: str) -> str:
    """The registrable-ish domain of a URL (netloc, no www.), for
    de-duplication and source names. Pure."""
    host = (urllib.parse.urlparse(url or "").netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


_SOURCE_NAMES = {
    "reuters.com": "Reuters",
    "sciencedaily.com": "ScienceDaily",
    "cnbc.com": "CNBC",
    "cnn.com": "CNN",
    "news.mit.edu": "MIT News",
    "artificialintelligence-news.com": "AI News",
    "theverge.com": "The Verge",
    "techcrunch.com": "TechCrunch",
    "wired.com": "WIRED",
    "theguardian.com": "The Guardian",
    "bbc.com": "BBC",
    "nytimes.com": "NY Times",
    "pcmag.com": "PCMag",
    "nsf.gov": "NSF",
    "msn.com": "MSN",
    "whitehouse.gov": "White House",
    "bbc.com": "BBC",
    "arstechnica.com": "Ars Technica",
}

# A "summary" that is just the site describing itself, not news
# (e.g. "Explore the latest artificial intelligence news with
# Reuters"). Such items are dropped and replaced, not shown.
_GENERIC_PHRASES = (
    "explore the latest",
    "latest news on",
    "latest updates in",
    "delivers the latest",
    "insights powering",
    "most recent coverage",
    "find the latest",
    "news and insights surrounding",
    # Model refusals (the fetched page had almost no text, e.g. a
    # portal that blocks scraping): "You haven't provided an article
    # to summarize — paste the article text..."
    "you haven't provided",
    "paste the article",
    "cannot summarize",
    "no article to summarize",
    "as an ai language model",
    "i don't have access to",
    # The model asking the reader a question instead of summarising
    # ("This text is a homepage listing... Which specific article
    # would you like me to summarise?") — that happened 2026-11-15
    # when a section homepage was summarised instead of an article.
    "which specific article",
    "which article would you like",
    "would you like me to summarise",
    "would you like me to summarize",
)


def _looks_generic(summary: str) -> bool:
    """True when a summary is a site blurb, a model refusal, or the
    model asking the reader a question — rather than news. A real
    2-sentence summary never contains a question mark. Pure."""
    text = (summary or "").strip().lower()
    if not text:
        return True
    if len(text) < 40:
        return True
    if "?" in text:
        return True
    return any(phrase in text for phrase in _GENERIC_PHRASES)


# Words that mark a story as AI-related. The general-tech feeds (BBC,
# CNBC, The Verge, Ars, Guardian) cover all of tech, so their items
# are filtered to AI stories; the AI-specific feeds pass everything.
# Single words are matched on word boundaries ("ai" must not match
# "said"; "agent" must not match "management"); multi-word phrases
# are matched as substrings.
_AI_WORDS = (
    "ai", "llm", "gpu", "agent", "agents", "robot", "robots",
    "neural", "gemini", "chatbot",
    # NOTE: "copilot" is deliberately NOT here — too many non-AI
    # Microsoft stories mention Copilot in passing (outages, 365
    # updates) and would slip through the filter.
)
_AI_PHRASES = (
    "artificial intelligence", "machine learning", "openai",
    "anthropic", "deepmind", "nvidia", "generative", "chatgpt",
)


def _is_ai_relevant(item: dict) -> bool:
    """True when a news item's title/snippet is about AI. Pure."""
    text = f"{item.get('title') or ''} {item.get('snippet') or ''}".lower()
    words = set(re.findall(r"[a-z0-9]+", text))
    if words & set(_AI_WORDS):
        return True
    return any(phrase in text for phrase in _AI_PHRASES)


def _source_name(url: str) -> str:
    """A friendly source name for a URL: known sites by name, otherwise
    the domain with dashes/underscores as spaces, title-cased. Pure."""
    domain = _domain(url)
    if domain in _SOURCE_NAMES:
        return _SOURCE_NAMES[domain]
    base = domain.rsplit(".", 2)[0] if domain.count(".") >= 1 else domain
    return base.replace("-", " ").replace("_", " ").title() or "Web"


def _format_datetime(pub_date: str | None) -> tuple[str, str]:
    """An RSS pubDate ('Sat, 15 Nov 2026 08:00:00 GMT') -> ('15 Nov
    2026', '16:00') in VM-local time (HKT). ('', '') when there is no
    usable date — including a stale one: section homepages sometimes
    carry an old cached date (ScienceDaily once showed 2024), and a
    wrong date is worse than none. Dates more than 14 days old, or in
    the future, are dropped. Pure."""
    if not pub_date:
        return "", ""
    try:
        from datetime import timezone
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(pub_date.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_days = (now - dt).total_seconds() / 86400
        if age_days < -1 or age_days > 14:
            return "", ""
        local = dt.astimezone()  # VM local time (set it to HKT)
        return local.strftime("%d %b %Y"), local.strftime("%H:%M")
    except Exception:
        return "", ""


# Titles/descriptions that mark stock-market content, not tech news.
# Bing News is full of it for AI queries ("Which AI Stock Is a Better
# Buy", "AI Stocks Under $10"). Such items are filtered out before
# fetching, so the digest is news, not investing tips.
_MARKET_WORDS = (
    "stock", "stocks", "stock market", "s&p", "nasdaq", "dow jones",
    "shares", "dividend", "better buy", "biggest win", "under $10",
    "rally", "market today", "market news",
)


def _is_market_junk(item: dict) -> bool:
    """True when a news item is stock-market content. Pure."""
    text = f"{item.get('title') or ''} {item.get('snippet') or ''}".lower()
    return any(word in text for word in _MARKET_WORDS)


# Sentence-ending abbreviations: "and U.S." ends with a period but is
# NOT a finished sentence (this fooled the clean-end check on a CNBC
# item, 2026-11-15 — the digest showed a dangling "and U.S.").
_ABBREV_TAILS = ("u.s.", "u.k.", "mr.", "mrs.", "ms.", "dr.",
                 "e.g.", "i.e.", "etc.", "vs.", "no.")


def _pub_age_hours(pub_date: str | None) -> float | None:
    """Age of an RSS pubDate in hours, or None when it cannot be
    parsed. Pure."""
    if not pub_date:
        return None
    try:
        from datetime import timezone
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(pub_date.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except Exception:
        return None


def _ends_cleanly(text: str) -> bool:
    """True when a summary ends on a finished sentence, not mid-thought
    ('...clinical operations, research,' or '...and U.S.'). Pure."""
    tail = (text or "").strip().rstrip("\"'”’)]")
    if not tail:
        return False
    if tail.lower().endswith(_ABBREV_TAILS):
        return False
    return tail[-1] in ".!?…"


def _trim_to_last_sentence(text: str) -> str:
    """Cut a summary back to its last finished sentence. '' when it
    has no finished sentence at all. Pure."""
    text = (text or "").strip()
    if not text:
        return ""
    if _ends_cleanly(text):
        return text
    cuts = [m.end() for m in re.finditer(r"[.!?]\s+", text)]
    for cut in reversed(cuts):
        candidate = text[:cut].strip()
        if _ends_cleanly(candidate):
            return candidate
    return ""


def _summarise_item(search_mod, item: dict) -> str:
    """One news item -> a 2-sentence summary of what is on its page.
    Fetches the page and summarises the page text (search-result titles
    and snippets alone are mostly site descriptions, not news). Returns
    '' when the page cannot be fetched at all — the caller drops the
    item instead of showing a bare link with a site blurb (that was
    the Reuters item). A fetch that works but a summary that fails
    falls back to the item's snippet. A summary cut off mid-sentence
    (the model sometimes stops early) is retried once, then trimmed
    back to its last finished sentence — a dangling half-sentence is
    never shown. Never returns raw page text."""
    snippet = (item.get("snippet") or "").strip()
    url = item.get("url") or ""
    if not url:
        return ""
    try:
        page_text = search_mod.fetch(url, max_chars=ITEM_FETCH_CHARS)
    except Exception:
        return ""
    if len((page_text or "").strip()) < 300:
        # Almost no readable text (portal / blocked page): there is
        # nothing to summarise, and the model just refuses. Drop it.
        return ""
    summary = (search_mod.summarise(page_text,
                                    system=search_mod.ITEM_SYSTEM) or "").strip()
    if summary and not _ends_cleanly(summary):
        retry = (search_mod.summarise(
            page_text, system=search_mod.ITEM_SYSTEM) or "").strip()
        if retry:
            summary = retry
    if summary:
        summary = _trim_to_last_sentence(summary)
        if len(summary) > SUMMARY_MAX_CHARS:
            # Cap the length (one Telegram message, 4096 chars total):
            # cut back to the last finished sentence inside the cap.
            summary = _trim_to_last_sentence(summary[:SUMMARY_MAX_CHARS])
        if summary:
            return summary
    return snippet


def _keep_items(search_mod, items: list[dict],
                require_date: bool = False,
                ai_filter: bool = False,
                max_age_hours: float | None = None) -> list[dict]:
    """Summarise a list of candidate items, keeping the first
    NEWS_MAX_RESULTS usable ones: one per domain, not market junk, a
    real (non-generic, finished) summary. With require_date, an item
    with no fresh publish date is skipped too — a news article with
    no date is usually an evergreen/landing page, not news. With
    max_age_hours, only items published within that many hours are
    kept (the daily digest wants the last 24 hours). With ai_filter,
    only AI-related stories are kept. The source is the feed name
    when the item came from a feed, else the domain. Pure except for
    the fetches and NIM calls inside _summarise_item."""
    kept = []
    seen_domains = set()
    for item in items:
        if len(kept) >= NEWS_MAX_RESULTS:
            break
        if _is_market_junk(item):
            continue
        if ai_filter and not _is_ai_relevant(item):
            continue
        domain = _domain(item.get("url") or "")
        if domain and domain in seen_domains:
            continue
        date, time = _format_datetime(item.get("pub_date"))
        if require_date and not date:
            continue
        if max_age_hours is not None:
            age = _pub_age_hours(item.get("pub_date"))
            if age is None or age < -1 or age > max_age_hours:
                continue
        summary = _summarise_item(search_mod, item)
        if not summary or _looks_generic(summary):
            continue
        if domain:
            seen_domains.add(domain)
        item["summary"] = summary
        item["source"] = item.get("feed") or _source_name(item.get("url") or "")
        item["date"], item["time"] = date, time
        kept.append(item)
    return kept


def fetch_news(search_mod) -> dict:
    """Top AI-news articles, each with a 2-sentence summary of the
    article itself (worker role's model), plus its source and publish
    date and time. Sources are the fixed trusted feeds in NEWS_FEEDS:
    the newest items of each feed first (top 2 per feed), then the
    rest if slots remain. A failed feed is skipped; an unfetchable
    article is dropped and replaced. Never raises: a failure becomes
    an "error" note for the section. The returned items carry
    "summary", "source", "date" and "time" keys; the top-level
    "summary" is kept (empty) for callers that read it. No raw search
    text is ever shown as a summary — and no search engine is used
    at all for this section."""
    if search_mod is None:
        return {"available": False, "count": 0, "items": [],
                "summary": "",
                "error": "search worker not deployed"}
    primary, secondary = [], []
    for name, feed_url in NEWS_FEEDS:
        try:
            items = search_mod.fetch_rss_feed(feed_url,
                                              max_items=FEED_MAX_ITEMS)
        except Exception:
            continue
        for i, item in enumerate(items):
            item["feed"] = name
            (primary if i < 2 else secondary).append(item)
    kept = _keep_items(search_mod, primary,
                       require_date=True, ai_filter=True,
                       max_age_hours=24)
    if len(kept) < NEWS_MAX_RESULTS:
        # Fill the remaining slots with slightly older stories (up to
        # 7 days) rather than sending a thin digest on a slow day.
        seen = {_domain(i.get("url") or "") for i in kept}
        for item in _keep_items(search_mod, secondary,
                               require_date=True, ai_filter=True,
                               max_age_hours=24 * 7):
            if len(kept) >= NEWS_MAX_RESULTS:
                break
            if _domain(item.get("url") or "") in seen:
                continue
            kept.append(item)
    if not kept:
        return {"available": False, "count": 0, "items": [],
                "summary": "",
                "error": "AI news unavailable today (no trusted feed could be read)"}
    return {"available": True, "count": len(kept), "items": kept,
            "summary": "", "error": None}


# ---------------- build + format ----------------

def build_digest() -> dict:
    """Build the digest. Returns {"date", "sections", "text"}; never
    raises — a failed section is reported inside its own section."""
    date = now_local().strftime("%a %d %b %Y")

    news = fetch_news(get_search_module())

    repos = {}
    for repo in get_repos():
        repos[repo] = fetch_repo_summary(repo)
    repos_available = any(not v["error"] for v in repos.values())

    plans = read_plans()
    if plans is None:
        # fallback: the top open issues across the configured repos
        # (the planned work), most recently updated first
        all_issues = []
        for repo, data in repos.items():
            for item in data["issues"]:
                all_issues.append({**item, "repo": repo})
        all_issues.sort(key=lambda i: i["updated_at"], reverse=True)
        top = all_issues[:PLANS_MAX_LINES]
        plans_data = {
            "source": "open-issues" if top else "none",
            "items": [f"{i['repo']}#{i['number']}: {i['title']}" for i in top],
            "links": {f"{i['repo']}#{i['number']}": i["url"] for i in top},
            "note": ("plans.md not found in " + DATA_DIR +
                     " — showing the top open issues instead")
            if top else ("plans.md not found in " + DATA_DIR +
                         " and no open issues to show"),
        }
    else:
        plans_data = {"source": "plans.md", "items": plans,
                      "links": {}, "note": None}

    sections = {
        "news": news,
        "repos": {"available": repos_available, "repos": repos},
        "plans": plans_data,
    }
    data = {"date": date, "sections": sections}
    data["text"] = format_digest(data)
    # One Telegram message, not two: the cap is 4096 chars, and the
    # bot splits anything longer. Shed the least important parts
    # first (plans, then the oldest news items) until it fits.
    while len(data["text"]) > TG_CHUNK_LIMIT - 100 and (
            plans_data["items"] or len(news["items"]) > 3):
        if plans_data["items"]:
            plans_data["items"] = []
            plans_data["links"] = {}
            plans_data["note"] = None
        elif len(news["items"]) > 3:
            news["items"] = news["items"][:-1]
            news["count"] = len(news["items"])
        data["text"] = format_digest(data)
    return data


def _esc(text: str) -> str:
    return html_lib.escape(text or "")


def _link(url: str, label: str) -> str:
    return f'<a href="{html_lib.escape(url or "", quote=True)}">{_esc(label)}</a>'


def format_digest(data: dict) -> str:
    """The digest as Telegram HTML: bold headings, inline links for
    sources. All data values are escaped first; only the markup this
    function writes is HTML. Pure."""
    sections = data["sections"]
    lines = [f"<b>☀️ Daily digest — {data['date']}</b>", ""]

    # --- AI news ---
    lines.append("<b>📰 AI news</b>")
    news = sections["news"]
    if not news["available"]:
        lines.append(news["error"] or "AI news unavailable today.")
    else:
        for item in news["items"]:
            source = _esc(item.get("source") or "Web")
            date = _esc(item.get("date") or "")
            time = _esc(item.get("time") or "")
            subhead = f"🗞️ <b>{source}</b>"
            if date:
                subhead += f"  ·  📅 {date}"
            if time:
                subhead += f"  ·  🕐 {time}"
            lines.append(subhead)
            lines.append(f"• {_link(item['url'], item['title'])}")
            if item.get("summary"):
                lines.append(f"<i>{_esc(item['summary'])}</i>")
            lines.append("")
    lines.append("")

    # --- repos ---
    lines.append("<b>📋 Repos — open PRs &amp; issues</b>")
    for repo, summary in sections["repos"]["repos"].items():
        lines.append(f"<b>{_esc(repo)}</b>")
        if summary["error"]:
            lines.append(summary["error"])
            continue
        for pr in summary["pulls"]:
            lines.append(f"• 🔀 PR #{pr['number']}: {_link(pr['url'], pr['title'])}")
        for issue in summary["issues"]:
            lines.append(f"• ❗ Issue #{issue['number']}: {_link(issue['url'], issue['title'])}")
        if not summary["pulls"] and not summary["issues"]:
            lines.append("• ✅ nothing open")
    lines.append("")

    # --- plans ---
    lines.append("<b>📅 Plans</b>")
    plans = sections["plans"]
    if plans["note"]:
        lines.append(f"<i>{_esc(plans['note'])}</i>")
    for item in plans["items"]:
        if item in plans.get("links", {}):
            lines.append(f"• 📌 {_link(plans['links'][item], item)}")
        else:
            lines.append(f"• 📌 {_esc(item)}")
    if not plans["items"]:
        lines.append("• nothing planned")
    return "\n".join(lines)


def split_message(text: str, limit: int = TG_CHUNK_LIMIT) -> list[str]:
    """Split a long message into <=limit-char chunks on blank-line
    boundaries (single newlines as fallback), so a long digest never
    hits Telegram's 4096-char cap. Pure."""
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for block in text.split("\n\n"):
        if len(current) + len(block) + 2 > limit and current:
            chunks.append(current)
            current = ""
        if len(block) > limit:  # one huge block: split on newlines
            for line in block.split("\n"):
                if len(current) + len(line) + 1 > limit and current:
                    chunks.append(current)
                    current = ""
                current = f"{current}\n{line}" if current else line
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        chunks.append(current)
    return chunks


# ---------------- delivery ----------------

def get_telegram():
    """(get_chat_ids, send_telegram) from telegram_bot.py, or (None,
    None) when the bot module is not deployed. Lazy so this module
    stays importable without it."""
    try:
        from telegram_bot import get_chat_ids, send_telegram
        return get_chat_ids, send_telegram
    except ImportError:
        return None, None


def send_digest() -> dict:
    """Build the digest and send it to every Telegram chat the bot has
    seen. Returns {"sent", "chats", "chunks", "data", "error"}; never
    raises. "sent" is True iff every chunk reached at least one chat."""
    try:
        data = build_digest()
    except Exception as e:  # should not happen (build never raises),
        return {"sent": False, "chats": 0, "chunks": 0, "data": {},
                "error": f"digest build failed: {type(e).__name__}"}  # but be safe
    get_chat_ids, send_telegram = get_telegram()
    if get_chat_ids is None or send_telegram is None:
        return {"sent": False, "chats": 0, "chunks": 0, "data": data,
                "error": "Telegram bot not deployed — no chats to send to"}
    try:
        with db() as conn:
            chat_ids = get_chat_ids(conn)
    except Exception as e:
        return {"sent": False, "chats": 0, "chunks": 0, "data": data,
                "error": f"could not read Telegram chats: {type(e).__name__}"}
    if not chat_ids:
        return {"sent": False, "chats": 0, "chunks": 0, "data": data,
                "error": "no Telegram chats known yet — the bot has not seen any chat"}
    chunks = split_message(data["text"])
    # Pictures: Telegram shows a preview card (with the article's
    # image) for one link per message. Point it at the top story so
    # the single digest message carries an image.
    news_items = data.get("sections", {}).get("news", {}).get("items") or []
    preview_url = news_items[0].get("url") if news_items else None
    sent_any = False
    for i, chunk in enumerate(chunks):
        chunk_sent = False
        for chat_id in chat_ids:
            if send_telegram(chat_id, chunk, raw_html=True,
                             preview_url=preview_url if i == 0 else None):
                chunk_sent = True
        sent_any = sent_any or chunk_sent
    if not sent_any:
        return {"sent": False, "chats": len(chat_ids),
                "chunks": len(chunks), "data": data,
                "error": "Telegram rejected every chunk"}
    return {"sent": True, "chats": len(chat_ids), "chunks": len(chunks),
            "data": data, "error": None}


def main():
    result = send_digest()
    if result["sent"]:
        print(f"digest sent to {result['chats']} chat(s) "
              f"({result['chunks']} message(s))", flush=True)
        return 0
    print(f"digest not sent: {result['error']}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())

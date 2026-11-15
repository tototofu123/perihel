#!/usr/bin/env python3
"""Perihel web search/fetch worker — keyless search + readable page text,
with optional AI summaries via NVIDIA NIM.

Stdlib only. Consumed over HTTP by the Telegram bot and the daily digest
through POST /v1/search (see server.py); nothing here touches the database
or the ntfy path.

Search provider: Bing's keyless RSS endpoint (no key, no signup). It was
chosen because the preferred first choice — DuckDuckGo's keyless HTML
endpoint — returns its bot-check ("anomaly") page instead of results from
a cloud VM, Brave's search page 429s from the same VM, and Mojeek serves
a captcha (all verified Nov 2026 from the dev VM). Bing's HTML results
page is the fallback if the RSS endpoint ever stops returning items.
Bing's RSS terms limit the results to personal, non-commercial use,
which is what Perihel is.

Rate limits (in-process, enforced by the /v1/search endpoint, not by the
functions below — the functions stay pure so tests can call them freely):
at most one search/fetch per 5 s (MIN_GAP_S, same as ai.py's
MIN_CALL_GAP_S) and at most 10 per rolling minute (MAX_PER_MINUTE).
Over the limit the endpoint answers HTTP 429 with a retry_after hint.

Summarising reuses ai.py's nim_call_fallback (lazy import inside
summarise() so this module stays importable without nim.key — a missing
key only breaks summarisation, not search/fetch). A failed summary never
fails the request: summarise() returns '' and the caller shows the
snippet instead. It never returns raw source text as a "summary" — a
dump of titles/URLs/snippets is not a summary (that was the daily
digest bug of 2026-11-15).
"""
import html as html_lib
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import deque
from html.parser import HTMLParser

SEARCH_MODEL = os.environ.get("PERIHEL_SEARCH_MODEL") or None

BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
SEARCH_TIMEOUT_S = 15       # one search request (RSS or HTML page)
FETCH_TIMEOUT_S = 15        # one page fetch
MAX_FETCH_BYTES = 2 * 1024 * 1024   # never download more than 2 MB
SUMMARY_INPUT_CHARS = 6000  # chars of text sent to the model

# ---------------- rate limit ----------------

MIN_GAP_S = 5.0             # min gap between search/fetch calls
MAX_PER_MINUTE = 10         # max search/fetch calls per rolling 60 s
WINDOW_S = 60.0

_rate_lock = threading.Lock()
_last_call = 0.0
_call_times: deque = deque()  # timestamps of calls inside the window


def check_rate_limit(now: float | None = None) -> tuple[bool, float]:
    """One search/fetch slot, or (False, retry_after_seconds).

    `now` is injectable for tests; callers pass nothing. A granted call is
    recorded immediately, so concurrent callers cannot both slip through.
    """
    global _last_call
    t = time.time() if now is None else now
    with _rate_lock:
        while _call_times and _call_times[0] <= t - WINDOW_S:
            _call_times.popleft()
        gap = MIN_GAP_S - (t - _last_call)
        if gap > 0:
            return False, round(gap, 1)
        if len(_call_times) >= MAX_PER_MINUTE:
            retry = WINDOW_S - (t - _call_times[0])
            return False, round(max(retry, 0.1), 1)
        _last_call = t
        _call_times.append(t)
        return True, 0.0


# ---------------- HTML -> readable text ----------------

_SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside",
              "noscript", "svg", "form", "button", "select", "textarea",
              "iframe"}  # note: <head> is NOT skipped — <title> is useful,
              # and meta/link carry no text; script/style inside it are
              # skipped individually
_BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "div",
               "section", "article", "blockquote", "pre", "br", "title"}


class _TextExtractor(HTMLParser):
    """Visible text only: skips script/style/nav/etc, keeps block
    boundaries as newlines so the text stays readable."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def _collapse(text: str) -> str:
    """Collapse whitespace: strip each line, drop empties, single spaces."""
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def html_to_text(page_html: str) -> str:
    """Strip a page down to readable text. Pure function — unit-testable
    without any network."""
    parser = _TextExtractor()
    parser.feed(page_html)
    parser.close()
    return _collapse(html_lib.unescape(parser.text()))


# ---------------- search ----------------

def _clean(text: str | None) -> str:
    if not text:
        return ""
    return _collapse(html_lib.unescape(re.sub(r"<[^>]+>", "", text)))


def _search_bing_rss(query: str, max_results: int) -> list[dict]:
    """Bing RSS: clean XML, direct destination links. Raises on HTTP /
    network / parse failure; returns [] only when Bing genuinely has no
    items for the query."""
    params = urllib.parse.urlencode(
        {"q": query, "format": "rss", "count": max_results})
    req = urllib.request.Request(
        f"https://www.bing.com/search?{params}",
        headers={"User-Agent": BROWSER_UA})
    with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT_S) as resp:
        tree = ET.fromstring(resp.read())
    out = []
    for item in tree.iter("item"):
        link = _clean(item.findtext("link"))
        if not link:
            continue
        out.append({"title": _clean(item.findtext("title")),
                    "url": link,
                    "snippet": _clean(item.findtext("description")),
                    "pub_date": _clean(item.findtext("pubDate"))})
        if len(out) >= max_results:
            break
    return out


ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _atom_link(entry) -> str:
    """The article URL of an Atom entry: the rel=alternate link, else
    the first link's href. Pure."""
    for link in entry.findall(f"{ATOM_NS}link"):
        if link.get("rel") in (None, "alternate"):
            return link.get("href") or ""
    return ""


def fetch_rss_feed(url: str, max_items: int = 5) -> list[dict]:
    """Fetch one RSS 2.0 or Atom feed and return its newest items as
    [{title, url, snippet, pub_date}]. Raises on HTTP / network /
    parse failure; returns [] only when the feed genuinely has no
    items. The caller decides what to do with a failed feed (the
    digest just skips it and uses the other feeds)."""
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
    with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT_S) as resp:
        tree = ET.fromstring(resp.read())
    out = []
    # RSS 2.0: channel/item
    for item in tree.iter("item"):
        link = _clean(item.findtext("link"))
        if not link:
            continue
        out.append({"title": _clean(item.findtext("title")),
                    "url": link,
                    "snippet": _clean(item.findtext("description")),
                    "pub_date": _clean(item.findtext("pubDate"))})
        if len(out) >= max_items:
            return out
    # Atom: feed/entry (namespaced)
    for entry in tree.iter(f"{ATOM_NS}entry"):
        link = _atom_link(entry)
        if not link:
            continue
        out.append({"title": _clean(entry.findtext(f"{ATOM_NS}title")),
                    "url": link,
                    "snippet": _clean(entry.findtext(f"{ATOM_NS}summary")),
                    "pub_date": _clean(entry.findtext(f"{ATOM_NS}published"))
                                or _clean(entry.findtext(f"{ATOM_NS}updated"))})
        if len(out) >= max_items:
            break
    return out


def _resolve_bing_link(href: str) -> str:
    """Bing HTML wraps destinations in /ck/a redirect links. Follow the
    redirect (headers only, short timeout) to get the real URL; keep the
    original link when resolution fails — it still works in a browser."""
    if not href.startswith("https://www.bing.com/ck/a"):
        return href
    try:
        req = urllib.request.Request(href, method="GET",
                                     headers={"User-Agent": BROWSER_UA})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.geturl()
    except Exception:
        return href


def _search_bing_html(query: str, max_results: int) -> list[dict]:
    """Bing HTML results page (fallback when RSS yields nothing). Parses
    the b_algo result blocks; the page layout is Bing's, so this is the
    brittle half — RSS is the primary path for a reason."""
    params = urllib.parse.urlencode({"q": query, "count": max_results})
    req = urllib.request.Request(
        f"https://www.bing.com/search?{params}",
        headers={"User-Agent": BROWSER_UA})
    with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT_S) as resp:
        page = resp.read().decode("utf-8", errors="replace")
    out = []
    for block in re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', page, re.S):
        m = re.search(r'<h2><a[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>',
                      block, re.S)
        if not m:
            continue
        href = html_lib.unescape(m.group(1))
        snip = re.search(r'<div class="b_caption"><p[^>]*>(.*?)</p>',
                         block, re.S)
        out.append({"title": _clean(m.group(2)),
                    "url": _resolve_bing_link(href),
                    "snippet": _clean(snip.group(1)) if snip else ""})
        if len(out) >= max_results:
            break
    return out


def search(query: str, max_results: int = 5) -> list[dict]:
    """Web search. Returns [{title, url, snippet, pub_date}]. RSS first;
    HTML only when RSS fails or returns nothing (HTML items carry no
    pub_date). Raises only when both providers fail (network / HTTP
    error) — the endpoint turns that into a 502. An empty list means
    the search worked and found nothing."""
    query = (query or "").strip()
    if not query:
        raise ValueError("empty query")
    try:
        results = _search_bing_rss(query, max_results)
        if results:
            return results
    except Exception:
        pass  # fall through to the HTML page
    return _search_bing_html(query, max_results)


# ---------------- fetch ----------------

JINA_READER_PREFIX = "https://r.jina.ai/"


def _fetch_jina(url: str, max_chars: int) -> str:
    """Fetch a page via Jina AI Reader (https://r.jina.ai/<url>): a
    hosted reader that returns clean Markdown, JS-rendered, no key
    needed on the free tier. Used only as a fallback when the direct
    fetch fails or returns almost no text (JS-heavy / thin pages).
    It does NOT bypass access controls — a site that 401s still 401s.
    Raises on failure, like fetch()."""
    req = urllib.request.Request(
        JINA_READER_PREFIX + url,
        headers={"User-Agent": BROWSER_UA,
                 "Accept": "text/plain, text/markdown, */*;q=0.1"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        raw = resp.read(MAX_FETCH_BYTES)
    text = raw.decode("utf-8", errors="replace")
    # Jina prepends a metadata header ("Title: ... URL Source: ...
    # Markdown Content:") — drop everything up to the content marker.
    marker = "Markdown Content:"
    if marker in text:
        text = text.split(marker, 1)[1]
    return _collapse(text)[:max_chars]


def fetch(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL and return readable text, capped at max_chars.
    Rejects non-http(s) URLs and non-text content types. When the
    direct fetch fails or returns almost no readable text (a
    JS-heavy page, a thin portal), retries once through Jina Reader
    before giving up."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("only http(s) URLs can be fetched")
    direct_error = None
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": BROWSER_UA,
                          "Accept": "text/html, text/plain, */*;q=0.1"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
            ctype = resp.headers.get_content_type()
            if ctype not in ("text/html", "application/xhtml+xml") \
                    and not ctype.startswith("text/"):
                raise ValueError(f"not a text page (Content-Type: {ctype})")
            charset = resp.headers.get_content_charset() or "utf-8"
            raw = resp.read(MAX_FETCH_BYTES)
        text = raw.decode(charset, errors="replace")
        if ctype in ("text/html", "application/xhtml+xml"):
            text = html_to_text(text)
        else:
            text = _collapse(text)
        if len(text.strip()) >= 300:
            return text[:max_chars]
        direct_error = ValueError("page returned almost no readable text")
    except Exception as e:
        direct_error = e
    try:
        text = _fetch_jina(url, max_chars)
    except Exception:
        raise direct_error
    if len(text.strip()) < 300:
        raise direct_error
    return text


# ---------------- summarise ----------------

SUMMARY_SYSTEM = ("You summarise web text. Reply with a short, factual "
                  "summary (3-6 sentences, no preamble, no invented "
                  "details).")

ITEM_SYSTEM = ("You summarise one news article. Reply with exactly "
               "2 sentences, in your own words: sentence 1 — what "
               "happened and who is involved; sentence 2 — why it "
               "matters, with enough context that a reader who never "
               "opens the article understands the story. Exactly 2 "
               "sentences, no more, no fewer. Do NOT copy sentences "
               "from the article — write a real summary. No preamble, "
               "no invented details, no repeating the site's own "
               "description of itself.")


def summarise(text: str, query: str | None = None,
              model: str | None = None, system: str | None = None) -> str:
    """Summarise fetched/search text with the NIM model. `model` is a
    per-call override; None means PERIHEL_SEARCH_MODEL, and that unset
    means the server default (ai.py MODEL). `system` overrides the
    system prompt (the digest passes ITEM_SYSTEM for per-item
    summaries). Never raises: on empty text returns ''; on a NIM
    failure returns '' too — the caller falls back to the snippet.
    Raw source text is never returned as a "summary".
    Uses nim_call_fallback (not _nim_call) so a dead model — e.g. the
    retired default llama-3.2-11b-vision-instruct, which 410s — falls
    through to Laguna / Bolt / MiniMax instead of failing outright."""
    text = (text or "").strip()
    if not text:
        return ""
    use_model = model or SEARCH_MODEL or None
    user = "Summarise this text"
    if query:
        user += f" for the question: {query}"
    user += f":\n\n{text[:SUMMARY_INPUT_CHARS]}"
    messages = [{"role": "system", "content": system or SUMMARY_SYSTEM},
                {"role": "user", "content": user}]
    try:
        from ai import nim_call_fallback  # lazy: keeps this module
        # importable without nim.key (only summarisation needs the key)
        resp, _used = nim_call_fallback(messages, timeout=30,
                                        model=use_model, role="worker")
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    # NIM chain down: try Gemini (Google AI Studio free tier) with the
    # same messages before giving up. No gemini.key on the VM just
    # means this step is skipped, same as before.
    try:
        from ai import gemini_call  # lazy, same reason as above
        resp = gemini_call(messages, timeout=30)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

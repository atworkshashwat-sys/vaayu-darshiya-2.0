"""
TinyFish Search + Fetch client — used ONLY by the AI Insights section.

Nothing in the ML pipeline (the /api/forecast, /api/stations, /api/metrics,
/api/weekly-trend and /api/time endpoints, database.py or models/) imports this
module. It exists so the AI Insights panel can attach real, externally
retrieved Delhi NCR pollution news and advisories next to its ML cards.

Environment:
    TINYFISH_API_KEY   TinyFish API key — backend only, never sent to the browser

Docs: https://docs.tinyfish.ai/search-api · https://docs.tinyfish.ai/fetch-api
Stdlib only (urllib), so backend/requirements.txt stays unchanged.
"""

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

SEARCH_ENDPOINT = "https://api.search.tinyfish.ai"
FETCH_ENDPOINT = "https://api.fetch.tinyfish.ai"

# Tight timeouts: this runs inside the AI Insights request path and must never
# hold the UI for long. The router treats any timeout as "unavailable".
SEARCH_TIMEOUT = 6.0
FETCH_TIMEOUT = 8.0

# Excerpt of the top article, cleaned to plain-ish text for the panel.
EXCERPT_CHARS = 480


class TinyFishError(Exception):
    """Transport / HTTP / payload problem talking to TinyFish.

    Raised only inside tinyfish_service and the AI Insights router; the router
    converts it into a `status: "unavailable"` payload so no request ever fails.
    """


def api_key():
    """Backend-only key; None when unset so callers fail closed."""
    key = os.environ.get("TINYFISH_API_KEY", "").strip()
    return key or None


def _site_name(url):
    netloc = urllib.parse.urlparse(str(url)).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def _call(url, data=None, timeout=SEARCH_TIMEOUT):
    """One HTTP call to TinyFish. Returns a dict payload or raises TinyFishError."""
    headers = {"X-API-Key": api_key() or "", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if data is not None else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:  # 401 / 402 / 429 / 5xx …
        raise TinyFishError(f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TinyFishError(str(exc) or exc.__class__.__name__) from exc
    except ValueError as exc:
        raise TinyFishError("TinyFish returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise TinyFishError("Unexpected TinyFish payload")
    return payload


def search(query, domain_type="news", page_size=5, timeout=SEARCH_TIMEOUT):
    """Ranked TinyFish Search results for `query`.

    Returns a list of {title, url, snippet, site_name, publisher, date}.
    Raises TinyFishError when the key is missing or the call fails.
    """
    if not api_key():
        raise TinyFishError("TINYFISH_API_KEY is not set")
    params = {
        "query": query,
        "domain_type": domain_type,
        "location": "IN",
        "language": "en",
        "purpose": "Delhi NCR pollution news and official air quality advisories "
                   "for the Vaayu Darshiya AI Insights panel",
    }
    payload = _call(f"{SEARCH_ENDPOINT}?{urllib.parse.urlencode(params)}", timeout=timeout)

    items = []
    for raw in payload.get("results") or []:
        if not isinstance(raw, dict):
            continue
        url, title = raw.get("url"), raw.get("title")
        if not url or not title:
            continue  # a result without a link cannot be cited
        items.append({
            "title": str(title),
            "url": str(url),
            "snippet": str(raw.get("snippet") or "").strip(),
            "site_name": str(raw.get("site_name") or _site_name(url)),
            "publisher": str(raw.get("publisher") or ""),
            "date": str(raw.get("date") or ""),
        })
        if len(items) >= page_size:
            break
    return items


def _clean_excerpt(text, limit=EXCERPT_CHARS):
    """Markdown page text → a short plain excerpt (never invents wording)."""
    if not text:
        return ""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # links → label
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]  # drop headings
    joined = re.sub(r"\s+", " ", " ".join(lines)).strip()
    joined = re.sub(r"\*\*([^*]+)\*\*", r"\1", joined)
    joined = re.sub(r"\*([^*]+)\*", r"\1", joined)
    if len(joined) <= limit:
        return joined
    return joined[:limit].rsplit(" ", 1)[0] + " …"


def fetch_text(url, timeout=FETCH_TIMEOUT):
    """Clean extracted text for one URL via TinyFish Fetch.

    Returns {url, title, site_name, text} or raises TinyFishError. The caller
    treats failure as "no excerpt" — search results still stand on their own.
    """
    if not api_key():
        raise TinyFishError("TINYFISH_API_KEY is not set")
    body = json.dumps({
        "urls": [str(url)],
        "format": "markdown",
        "purpose": "Extract the article text so the AI Insights panel can quote "
                   "a short, faithful excerpt beside the headline",
    }).encode("utf-8")
    payload = _call(FETCH_ENDPOINT, data=body, timeout=timeout)

    results = payload.get("results") or []
    if not results or not isinstance(results[0], dict):
        errors = [e for e in (payload.get("errors") or []) if isinstance(e, dict)]
        detail = "; ".join(str(e.get("error")) for e in errors if e.get("error"))
        raise TinyFishError(f"Fetch returned no content{': ' + detail if detail else ''}")

    first = results[0]
    final_url = str(first.get("final_url") or first.get("url") or url)
    return {
        "url": final_url,
        "title": str(first.get("title") or ""),
        "site_name": _site_name(final_url),
        "text": _clean_excerpt(str(first.get("text") or "")),
    }

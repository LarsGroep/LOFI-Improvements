"""
LabelRadar — signing announcements from curated label Bandcamp pages.

A strong label's A&R decision is a free expert vote (docs/agentic_os.md §3.1):
when a label first lists a new artist in its discography, that is often earlier
and cleaner than any streaming metric. This listener fetches each configured
label page (through the cached, polite osk.scrape.fetch), parses the messy HTML
defensively, diffs the artist roster against the previous run, and emits a
`label_signing` for each first-seen artist.

Zero credentials required — it reads public pages. Empty label list → skipped.
Bandcamp HTML is inconsistent, so parsing is belt-and-braces: the embedded
data-blob JSON island first, a regex over /album/ tiles as a fallback, and a
per-label try/except so one broken page never stops the sweep.
"""
from __future__ import annotations

import html as _html
import json
import re

from osk.listeners.resolve import index_from_ctx, resolve
from osk.listeners.sources import labels as _configured_labels
from osk.manifest import AgentManifest, Budget
from osk.scrape import fetch

_DATA_BLOB_RE = re.compile(r'data-blob="([^"]*)"')
_TILE_RE = re.compile(
    r'href="[^"]*/album/[^"]*".*?class="title">(?P<body>.*?)</p>', re.DOTALL)
_OVERRIDE_RE = re.compile(
    r'<span class="artist-override">(?P<name>.*?)</span>', re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


# ── pure parsing (unit-tested) ───────────────────────────────────────────────

def _clean(text: str) -> str:
    return " ".join(_html.unescape(_TAG_RE.sub("", text or "")).split())


def _walk_items(obj, out: list[dict]) -> None:
    """Recursively collect {artist, title} from any nested structure that
    looks like a Bandcamp discography item — robust to blob-shape drift."""
    if isinstance(obj, dict):
        title = obj.get("title")
        artist = (obj.get("artist") or obj.get("band_name")
                  or obj.get("artist_name") or obj.get("artist_override"))
        if title and artist:
            out.append({"artist": str(artist).strip(),
                        "title": str(title).strip()})
        for value in obj.values():
            _walk_items(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _walk_items(value, out)


def _parse_data_blob(html: str) -> list[dict]:
    m = _DATA_BLOB_RE.search(html or "")
    if not m:
        return []
    try:
        blob = json.loads(_html.unescape(m.group(1)))
    except Exception:
        return []
    out: list[dict] = []
    _walk_items(blob, out)
    return out


def _parse_album_tiles(html: str) -> list[dict]:
    out = []
    for m in _TILE_RE.finditer(html or ""):
        body = m.group("body")
        om = _OVERRIDE_RE.search(body)
        artist = _clean(om.group("name")) if om else ""
        title = _clean(_OVERRIDE_RE.sub("", body))
        # tiles often prefix the override with "by " — drop it
        artist = re.sub(r"^by\s+", "", artist, flags=re.IGNORECASE)
        if title and artist:
            out.append({"artist": artist, "title": title})
    return out


def parse_label_page(html: str) -> list[dict]:
    """[{artist, title}, ...]. Data-blob island first, /album/ tiles as
    fallback. A page we can't parse yields [] — never an exception."""
    items = _parse_data_blob(html)
    if items:
        return items
    return _parse_album_tiles(html)


def _dedup_artists(items: list[dict]) -> list[dict]:
    """First release per artist name (case-folded), preserving order."""
    seen, out = set(), []
    for it in items:
        key = (it.get("artist") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


class LabelRadar:
    manifest = AgentManifest(
        name="label_radar",
        description="first-seen artist signings from curated label Bandcamps",
        writes=("signal",),
        schedule="daily@02:40",
        budget=Budget(max_runs_per_day=3),
    )

    def run(self, ctx) -> str:
        labels = _configured_labels()
        if not labels:
            return "skipped: no labels configured (sources.yaml)"

        index = index_from_ctx(ctx)
        observed_at = ctx.now.date().isoformat()
        emitted, swept = 0, 0
        for label in labels:
            url = label["bandcamp"]
            try:
                status, body = fetch(url)
                if not body:
                    continue
                items = _dedup_artists(parse_label_page(body))
            except Exception:
                continue  # a single unparseable label never stops the sweep
            swept += 1
            key = f"roster:{label['name']}"
            prev = ctx.state_get(key)
            current_names = [it["artist"] for it in items]
            ctx.state_set(key, sorted({n.lower() for n in current_names}))
            if prev is None:
                continue  # first sight = baseline; nobody was "signed just now"
            prev_set = set(prev)
            for it in items:
                if it["artist"].lower() in prev_set:
                    continue
                name = it["artist"]
                ctx.emit("signal", {
                    "source": self.manifest.name,
                    "kind": "label_signing",
                    "value": None,
                    "evidence": {
                        "label": label["name"],
                        "release": it.get("title"),
                        "url": url,
                    },
                    "observed_at": observed_at,
                }, artist_id=resolve(name, index), artist_name=name)
                emitted += 1
        return f"{emitted} label_signing signal(s) over {swept} label(s)"

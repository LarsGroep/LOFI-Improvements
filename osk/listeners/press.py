"""
PressListener — first-coverage events from scene-press feeds.

RA news, Mixmag, DJ Mag and friends run "ones to watch" pieces and first
features well before an artist shows up in the quantitative pipeline
(docs/agentic_os.md §3.1). This listener fetches each configured RSS/Atom feed
(through the cached osk.scrape.fetch), diffs item GUIDs against the previous
run, and emits a `press_mention` per new item. When a tracked artist's name is
in the headline it resolves the artist_id; otherwise it emits with
artist_name=None and the headline in evidence — a name we can link later.

Parsing is stdlib xml.etree only (no feedparser), and lenient: a malformed feed
or item is skipped, never fatal. Zero credentials; empty feed list → skipped.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from osk.listeners.resolve import index_from_ctx, match_in_title
from osk.listeners.sources import press_feeds as _configured_feeds
from osk.manifest import AgentManifest, Budget
from osk.scrape import fetch


# ── pure parsing (unit-tested) ───────────────────────────────────────────────

def _localname(tag: str) -> str:
    """Strip the XML namespace: '{http://www.w3.org/2005/Atom}entry' -> 'entry'."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _child_text(elem, name: str) -> str | None:
    for child in elem:
        if _localname(child.tag) == name:
            return (child.text or "").strip() or None
    return None


def _child_link(elem) -> str | None:
    """RSS <link>text</link> or Atom <link href="..."/>."""
    for child in elem:
        if _localname(child.tag) != "link":
            continue
        if child.text and child.text.strip():
            return child.text.strip()
        href = child.attrib.get("href")
        if href:
            return href
    return None


def parse_feed(xml_text: str) -> list[dict]:
    """[{guid, title, link}, ...] from RSS <item> or Atom <entry>. A parse
    failure on the whole document returns []; a bad single item is skipped."""
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    out = []
    for elem in root.iter():
        if _localname(elem.tag) not in ("item", "entry"):
            continue
        try:
            title = _child_text(elem, "title")
            link = _child_link(elem)
            guid = (_child_text(elem, "guid") or _child_text(elem, "id")
                    or link or title)
            if not guid:
                continue
            out.append({"guid": guid, "title": title, "link": link})
        except Exception:
            continue
    return out


class PressListener:
    manifest = AgentManifest(
        name="press_listener",
        description="new items from scene-press RSS/Atom feeds",
        writes=("signal",),
        schedule="every:6h",
        budget=Budget(max_runs_per_day=8),
    )

    def run(self, ctx) -> str:
        feeds = _configured_feeds()
        if not feeds:
            return "skipped: no press feeds configured (sources.yaml)"

        index = index_from_ctx(ctx)
        observed_at = ctx.now.date().isoformat()
        emitted = 0
        for feed in feeds:
            try:
                status, body = fetch(feed["url"])
                if not body:
                    continue
                items = parse_feed(body)
            except Exception:
                continue  # one bad feed never stops the sweep
            key = f"guids:{feed['name']}"
            prev = ctx.state_get(key)
            current_guids = [it["guid"] for it in items]
            ctx.state_set(key, current_guids)
            if prev is None:
                continue  # baseline the feed; don't replay its backlog
            prev_set = set(prev)
            for it in items:
                if it["guid"] in prev_set:
                    continue
                name, aid = match_in_title(it.get("title") or "", index)
                ctx.emit("signal", {
                    "source": self.manifest.name,
                    "kind": "press_mention",
                    "value": None,
                    "evidence": {
                        "outlet": feed["name"],
                        "title": it.get("title"),
                        "link": it.get("link"),
                        "guid": it.get("guid"),
                    },
                    "observed_at": observed_at,
                }, artist_id=aid, artist_name=name)
                emitted += 1
        return f"{emitted} press_mention signal(s) over {len(feeds)} feed(s)"

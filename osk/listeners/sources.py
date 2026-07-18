"""
Loader for the curated listener sources (sources.yaml).

Path is LOFI_OS_SOURCES or the bundled sources.yaml. A missing or corrupt file
is not an error — it returns {} and the listeners that depend on it skip
gracefully. Same graceful-degradation contract as everywhere else in the OS.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT = Path(__file__).parent / "sources.yaml"


def sources_path() -> Path:
    return Path(os.environ.get("LOFI_OS_SOURCES") or _DEFAULT)


def load_sources() -> dict:
    """Return the parsed sources dict, or {} if unreadable/empty/corrupt."""
    try:
        import yaml
        text = sources_path().read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def labels() -> list[dict]:
    """[{name, bandcamp}, ...] — only well-formed entries with a bandcamp URL."""
    out = []
    for entry in (load_sources().get("labels") or []):
        if isinstance(entry, dict) and entry.get("bandcamp"):
            out.append({"name": entry.get("name") or entry["bandcamp"],
                        "bandcamp": entry["bandcamp"]})
    return out


def playlists() -> list[dict]:
    """[{name, id}, ...] — only entries with a Spotify playlist id."""
    out = []
    for entry in (load_sources().get("playlists") or []):
        if isinstance(entry, dict) and entry.get("id"):
            out.append({"name": entry.get("name") or entry["id"],
                        "id": str(entry["id"])})
    return out


def press_feeds() -> list[dict]:
    """[{name, url}, ...] — only entries with a feed URL."""
    out = []
    for entry in (load_sources().get("press_feeds") or []):
        if isinstance(entry, dict) and entry.get("url"):
            out.append({"name": entry.get("name") or entry["url"],
                        "url": entry["url"]})
    return out

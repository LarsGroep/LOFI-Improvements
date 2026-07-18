"""
Listener tests — pure parse/normalise on inline fixtures, scrape-cache logic
against a monkeypatched httpx, state-diff on a tmp SQLite blackboard, and the
skip-when-unconfigured contract. No network, no Supabase, no LLM (same policy
as tests/test_osk.py and tests/test_predict.py).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from osk import scrape  # noqa: E402
from osk.blackboard import SQLiteBlackboard  # noqa: E402
from osk.listeners import (LISTENER_AGENTS, LabelRadar, PlaylistListener,  # noqa: E402
                           PressListener, SoundCloudListener)
from osk.listeners import label_radar, playlists, press, soundcloud  # noqa: E402
from osk.listeners.resolve import (build_index, match_in_title,  # noqa: E402
                                   normalise, resolve)
from osk.orchestrator import Context  # noqa: E402

UTC = dt.timezone.utc


def _ctx(agent, bb, now=None):
    return Context(agent.manifest, bb, {}, now or dt.datetime(2026, 7, 18, tzinfo=UTC))


# ── scrape: pure freshness helpers ───────────────────────────────────────────

def test_cache_key_and_freshness():
    assert scrape.cache_key("http://a") == scrape.cache_key("http://a ")
    assert scrape.cache_key("http://a") != scrape.cache_key("http://b")

    now = dt.datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    fresh = (now - dt.timedelta(seconds=100)).isoformat()
    stale = (now - dt.timedelta(seconds=500)).isoformat()
    assert scrape.is_fresh(fresh, now, 200)
    assert not scrape.is_fresh(stale, now, 200)
    assert not scrape.is_fresh(None, now, 200)
    assert not scrape.is_fresh("not-a-date", now, 200)


# ── scrape: fetch against a monkeypatched httpx ──────────────────────────────

def test_fetch_caches_conditionals_and_degrades(tmp_path, monkeypatch):
    monkeypatch.setenv("LOFI_OS_SCRAPE_DB", str(tmp_path / "cache.db"))
    calls = {"n": 0, "headers": None}

    def fake_get(url, headers, timeout):
        calls["n"] += 1
        calls["headers"] = headers
        return 200, "BODY", "etag1"

    monkeypatch.setattr(scrape, "_http_get", fake_get)

    # first fetch hits the network and caches
    assert scrape.fetch("http://x") == (200, "BODY")
    assert calls["n"] == 1
    assert calls["headers"]["User-Agent"] == "LOFI-Scout/0.1"

    # within max_age → served from cache, no second call
    assert scrape.fetch("http://x") == (200, "BODY")
    assert calls["n"] == 1

    # stale (max_age 0) → re-fetch, and the stored etag is replayed
    assert scrape.fetch("http://x", max_age_seconds=0) == (200, "BODY")
    assert calls["n"] == 2
    assert calls["headers"].get("If-None-Match") == "etag1"

    # a 304 refreshes the timestamp and serves the cached body
    def fake_304(url, headers, timeout):
        calls["n"] += 1
        assert headers.get("If-None-Match") == "etag1"
        return 304, "", ""

    monkeypatch.setattr(scrape, "_http_get", fake_304)
    assert scrape.fetch("http://x", max_age_seconds=0) == (200, "BODY")
    assert calls["n"] == 3

    # network error on a cached url → stale body served (allow_stale default)
    def boom(url, headers, timeout):
        raise RuntimeError("network down")

    monkeypatch.setattr(scrape, "_http_get", boom)
    assert scrape.fetch("http://x", max_age_seconds=0) == (200, "BODY")
    # network error on an unknown url → honest (0, "")
    assert scrape.fetch("http://never", max_age_seconds=0) == (0, "")


# ── resolve: normalise / build_index / resolve / title match ─────────────────

def test_resolve_is_exact_after_normalisation_only():
    assert normalise("Kölsch") == "kolsch"
    assert normalise("  DJ  Test! ") == "dj test"

    index = build_index([{"artist_id": "a1", "artist_name": "Kölsch"},
                         {"artist_id": "a2", "artist_name": "DJ Test"},
                         {"artist_id": None, "artist_name": "No Id"}])
    assert resolve("kolsch", index) == "a1"
    assert resolve("KÖLSCH ", index) == "a1"
    assert resolve("no id", index) is None      # unresolved (no artist_id)
    assert resolve("someone else", index) is None
    assert resolve("anything", {}) is None       # no index → None

    name, aid = match_in_title("Interview: DJ Test plays tonight", index)
    assert (name, aid) == ("dj test", "a2")
    assert match_in_title("nobody we know here", index) == (None, None)


# ── label_radar: defensive Bandcamp parsing ──────────────────────────────────

_DATA_BLOB_HTML = (
    '<div id="pagedata" data-blob="{&quot;discography&quot;:['
    '{&quot;title&quot;:&quot;Album One&quot;,&quot;artist&quot;:&quot;Artist Alpha&quot;},'
    '{&quot;title&quot;:&quot;Album Two&quot;,&quot;band_name&quot;:&quot;Artist Beta&quot;}'
    ']}"></div>')

_TILE_HTML = (
    '<li class="music-grid-item"><a href="/album/some-slug">'
    '<p class="title">Great Release'
    '<span class="artist-override">by Tile Artist</span></p></a></li>')


def test_parse_label_page_data_blob_and_tiles():
    blob_items = label_radar.parse_label_page(_DATA_BLOB_HTML)
    assert {i["artist"] for i in blob_items} == {"Artist Alpha", "Artist Beta"}

    tile_items = label_radar.parse_label_page(_TILE_HTML)
    assert tile_items == [{"artist": "Tile Artist", "title": "Great Release"}]

    # unparseable input degrades to [], never raises
    assert label_radar.parse_label_page("<html>nothing here</html>") == []


# ── press: stdlib RSS / Atom parsing ─────────────────────────────────────────

_RSS = (
    '<?xml version="1.0"?><rss version="2.0"><channel><title>Feed</title>'
    '<item><title>New one to watch: Someone</title>'
    '<link>http://x/1</link><guid>g1</guid></item>'
    '<item><title>Another story</title>'
    '<link>http://x/2</link><guid>g2</guid></item>'
    '</channel></rss>')

_ATOM = (
    '<feed xmlns="http://www.w3.org/2005/Atom"><title>Feed</title>'
    '<entry><title>Atom Story</title><id>a1</id>'
    '<link href="http://x/a1"/></entry></feed>')


def test_parse_feed_rss_and_atom():
    rss = press.parse_feed(_RSS)
    assert [i["guid"] for i in rss] == ["g1", "g2"]
    assert rss[0]["link"] == "http://x/1"

    atom = press.parse_feed(_ATOM)
    assert atom == [{"guid": "a1", "title": "Atom Story", "link": "http://x/a1"}]

    assert press.parse_feed("<<not xml") == []   # lenient on garbage


# ── playlists: Spotify tracklist parse + diff ────────────────────────────────

_SPOTIFY = {"items": [
    {"added_at": "2026-07-01", "track": {"id": "t1", "name": "Song",
        "artists": [{"id": "sp1", "name": "Alpha"},
                    {"id": "sp2", "name": "Beta"}]}},
    {"added_at": "2026-07-02", "track": {"id": "t2", "name": "Song2",
        "artists": [{"id": "sp3", "name": "Gamma"}]}},
    {"added_at": None, "track": None},          # removed/local track → null
]}


def test_parse_playlist_items_and_new_entries():
    rows = playlists.parse_playlist_items(_SPOTIFY)
    assert [r["artist_name"] for r in rows] == ["Alpha", "Beta", "Gamma"]
    assert playlists.track_ids(rows) == ["t1", "t2"]

    fresh = playlists.new_entries(rows, ["t1"])
    assert [r["artist_name"] for r in fresh] == ["Gamma"]
    assert playlists.new_entries(rows, ["t1", "t2"]) == []


# ── soundcloud: plays-velocity maths ─────────────────────────────────────────

def test_soundcloud_velocity_windows_and_maths():
    now = dt.datetime(2026, 7, 18, tzinfo=UTC)
    assert soundcloud.plays_velocity(1000, "2026-07-16T00:00:00Z", now) == 500.0
    # same-day upload: age floored at 1 day, not divide-by-zero
    assert soundcloud.plays_velocity(300, "2026-07-18T00:00:00Z", now) == 300.0
    assert soundcloud.plays_velocity(None, "2026-07-16T00:00:00Z", now) is None

    tracks = [
        {"id": 1, "title": "Fresh", "playback_count": 1000,
         "created_at": "2026-07-16T00:00:00Z", "permalink_url": "u1"},
        {"id": 2, "title": "Old", "playback_count": 5000,
         "created_at": "2026-01-01T00:00:00Z"},   # > 14d → excluded
    ]
    sigs = soundcloud.recent_track_signals(tracks, now)
    assert len(sigs) == 1 and sigs[0]["velocity"] == 500.0


# ── state-diff: second run emits only new items (label_radar) ─────────────────

def test_label_radar_state_diff_emits_only_new(tmp_path, monkeypatch):
    bb = SQLiteBlackboard(tmp_path / "os.db")
    monkeypatch.setattr(label_radar, "_configured_labels",
                        lambda: [{"name": "TestLabel",
                                  "bandcamp": "http://label"}])
    pages = {"n": 0}

    def fake_fetch(url, *a, **k):
        pages["n"] += 1
        if pages["n"] == 1:
            return 200, _blob(["Artist Alpha", "Artist Beta"])
        return 200, _blob(["Artist Alpha", "Artist Beta", "Artist Gamma"])

    monkeypatch.setattr(label_radar, "fetch", fake_fetch)
    agent = LabelRadar()

    # first run baselines the roster — nobody was "signed just now"
    agent.run(_ctx(agent, bb))
    assert bb.read(kinds=["signal"]) == []

    # second run: only the newcomer is emitted
    agent.run(_ctx(agent, bb))
    sigs = bb.read(kinds=["signal"])
    assert [s.artist_name for s in sigs] == ["Artist Gamma"]
    assert sigs[0].payload["kind"] == "label_signing"
    assert sigs[0].payload["evidence"]["label"] == "TestLabel"
    bb.close()


def _blob(artists):
    items = ",".join(
        '{&quot;title&quot;:&quot;Rel %s&quot;,&quot;artist&quot;:&quot;%s&quot;}'
        % (a, a) for a in artists)
    return '<div data-blob="{&quot;d&quot;:[%s]}"></div>' % items


# ── skip-when-unconfigured: all four listeners ───────────────────────────────

def test_all_listeners_skip_when_unconfigured(tmp_path, monkeypatch):
    for var in ("SOUNDCLOUD_CLIENT_ID", "SPOTIFY_CLIENT_ID",
                "SPOTIFY_CLIENT_SECRET", "SUPABASE_URL", "SUPABASE_KEY"):
        monkeypatch.delenv(var, raising=False)
    # point sources at a missing file so label/press have nothing to sweep
    monkeypatch.setenv("LOFI_OS_SOURCES", str(tmp_path / "absent.yaml"))

    bb = SQLiteBlackboard(tmp_path / "os.db")
    for cls in (SoundCloudListener, PlaylistListener, LabelRadar, PressListener):
        agent = cls()
        outcome = agent.run(_ctx(agent, bb))
        assert outcome.startswith("skipped:"), (cls.__name__, outcome)
    assert bb.read(kinds=["signal"]) == []      # nothing emitted while skipping
    bb.close()


def test_listener_roster_names_and_schedules():
    got = [(a.manifest.name, a.manifest.schedule) for a in LISTENER_AGENTS]
    assert got == [
        ("soundcloud_listener", "daily@02:00"),
        ("playlist_listener", "daily@02:20"),
        ("label_radar", "daily@02:40"),
        ("press_listener", "every:6h"),
    ]
    for a in LISTENER_AGENTS:
        assert a.manifest.llm is False
        assert a.manifest.writes == ("signal",)
        a.manifest.validate()

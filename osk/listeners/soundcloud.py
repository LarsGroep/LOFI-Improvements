"""
SoundCloudListener — plays-velocity on fresh releases, the classic pre-Spotify
heat signal (docs/agentic_os.md §3.1).

For each tracked artist it resolves the SoundCloud user, pulls recent tracks,
and computes plays/day for anything uploaded in the last 14 days. A track
climbing fast on SoundCloud days after release, before any editorial playlist
or Spotify bump, is exactly the cheap-booking window the OS is built to catch.

Tracked artists = the venue watchlist (predict/data/watchlist.json) plus, when
Supabase is configured, the top-N ranked Scout candidates. Auth is the public
api-v2 client_id (SOUNDCLOUD_CLIENT_ID). No client_id → skipped, never a crash.
"""
from __future__ import annotations

import datetime as _dt
import os

from osk.listeners.resolve import index_from_ctx, resolve
from osk.manifest import AgentManifest, Budget

_API = "https://api-v2.soundcloud.com"
_MAX_AGE_DAYS = 14


# ── pure velocity maths (unit-tested) ────────────────────────────────────────

def _parse_created(value) -> _dt.datetime | None:
    """SoundCloud api-v2 stamps like '2026-07-01T12:00:00Z' (and older pages
    like '2026/07/01 12:00:00 +0000'). Parse both, leniently."""
    if not value:
        return None
    text = str(value).strip()
    try:
        return _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%Y/%m/%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S %z"):
        try:
            return _dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def track_age_days(created_at, now: _dt.datetime) -> float | None:
    ts = _parse_created(created_at)
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_dt.timezone.utc)
    return (now - ts).total_seconds() / 86400.0


def plays_velocity(playback_count, created_at, now: _dt.datetime) -> float | None:
    """plays / day since upload. Age floored at 1 day so a same-day upload with
    big numbers doesn't divide to infinity. None if we can't compute it."""
    if playback_count is None:
        return None
    age = track_age_days(created_at, now)
    if age is None:
        return None
    return round(float(playback_count) / max(age, 1.0), 1)


def recent_track_signals(tracks, now: _dt.datetime,
                         max_age_days: int = _MAX_AGE_DAYS) -> list[dict]:
    """From a SoundCloud tracks payload, the velocity for each track uploaded
    within the window, hottest first."""
    out = []
    for t in tracks or []:
        age = track_age_days(t.get("created_at"), now)
        if age is None or age > max_age_days:
            continue
        vel = plays_velocity(t.get("playback_count"), t.get("created_at"), now)
        if vel is None:
            continue
        out.append({
            "title": t.get("title"),
            "track_id": t.get("id"),
            "permalink_url": t.get("permalink_url"),
            "plays": t.get("playback_count"),
            "age_days": round(age, 1),
            "velocity": vel,
        })
    out.sort(key=lambda r: r["velocity"], reverse=True)
    return out


# ── network (never reached by the test suite) ────────────────────────────────

def _search_user(name: str, client_id: str) -> dict | None:
    import httpx
    r = httpx.get(f"{_API}/search/users",
                  params={"q": name, "client_id": client_id, "limit": 1},
                  headers={"User-Agent": "LOFI-Scout/0.1"}, timeout=20)
    items = (r.json() or {}).get("collection") or []
    return items[0] if items else None


def _user_tracks(user_id, client_id: str) -> list[dict]:
    import httpx
    r = httpx.get(f"{_API}/users/{user_id}/tracks",
                  params={"client_id": client_id, "limit": 20},
                  headers={"User-Agent": "LOFI-Scout/0.1"}, timeout=20)
    return (r.json() or {}).get("collection") or []


# ── tracked-artist roster ────────────────────────────────────────────────────

def _tracked_names(ctx) -> list[str]:
    from predict.watchlist import load_watchlist
    names = list(load_watchlist())
    if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"):
        try:
            from osk.agents_builtin import _load_candidates
            top_n = int(os.environ.get("LOFI_OS_SC_TOPN", "100"))
            ranked = sorted(_load_candidates(ctx),
                            key=lambda c: (c.get("momentum") or 0),
                            reverse=True)
            names += [c.get("artist_name") for c in ranked[:top_n]
                      if c.get("artist_name")]
        except Exception:
            pass  # Supabase hiccup must not sink the whole sweep
    # de-dup, preserve order
    seen, out = set(), []
    for n in names:
        key = (n or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(n)
    return out


class SoundCloudListener:
    manifest = AgentManifest(
        name="soundcloud_listener",
        description="plays-velocity on <14d releases for tracked artists",
        writes=("signal",),
        schedule="daily@02:00",
        budget=Budget(max_runs_per_day=3),
    )

    def run(self, ctx) -> str:
        client_id = os.environ.get("SOUNDCLOUD_CLIENT_ID")
        if not client_id:
            return "skipped: SOUNDCLOUD_CLIENT_ID not set"

        names = _tracked_names(ctx)
        if not names:
            return ("skipped: no tracked artists (empty watchlist and no "
                    "Supabase candidates)")

        index = index_from_ctx(ctx)
        observed_at = ctx.now.date().isoformat()
        emitted = 0
        for name in names:
            try:
                user = _search_user(name, client_id)
                if not user:
                    continue
                tracks = _user_tracks(user.get("id"), client_id)
            except Exception:
                continue  # one bad artist never stops the sweep
            hot = recent_track_signals(tracks, ctx.now)
            if not hot:
                continue
            top = hot[0]
            ctx.emit("signal", {
                "source": self.manifest.name,
                "kind": "sc_plays_velocity",
                "value": top["velocity"],
                "evidence": {
                    "track": top.get("title"),
                    "url": top.get("permalink_url"),
                    "plays": top.get("plays"),
                    "age_days": top.get("age_days"),
                    "soundcloud_user": user.get("permalink"),
                    "n_recent_tracks": len(hot),
                },
                "observed_at": observed_at,
            }, artist_id=resolve(name, index), artist_name=name)
            emitted += 1
        return f"{emitted} velocity signal(s) over {len(names)} tracked artist(s)"

"""
PlaylistListener — first adds to scene-defining Spotify playlists.

A tastemaker/editorial playlist adding an artist is an early, human-curated
vote (docs/agentic_os.md §3.1). This listener diffs each watched playlist's
tracklist against the previous run and emits a `playlist_add` for every artist
that just appeared. State lives on the blackboard (ctx.state_get/set), so a
fresh deploy simply starts diffing from its second run — no back-fill needed.

Auth is the Spotify client-credentials flow (SPOTIFY_CLIENT_ID /
SPOTIFY_CLIENT_SECRET). Missing either → skipped, never a crash.
"""
from __future__ import annotations

from osk.listeners.resolve import index_from_ctx, resolve
from osk.listeners.sources import playlists as _configured_playlists
from osk.manifest import AgentManifest, Budget

_TOKEN_URL = "https://accounts.spotify.com/api/token"
_API = "https://api.spotify.com/v1"


# ── pure parse / diff (unit-tested) ──────────────────────────────────────────

def parse_playlist_items(response: dict) -> list[dict]:
    """Flatten a Spotify playlist-tracks response to one row per (track,
    artist): {track_id, track_name, artist_name, artist_id, added_at}. Defends
    against the nulls Spotify sprinkles through `items` (removed/local tracks)."""
    rows = []
    for item in (response or {}).get("items") or []:
        track = (item or {}).get("track") or {}
        tid = track.get("id")
        if not tid:
            continue
        for artist in track.get("artists") or []:
            name = artist.get("name")
            if not name:
                continue
            rows.append({
                "track_id": tid,
                "track_name": track.get("name"),
                "artist_name": name,
                "artist_id": artist.get("id"),   # Spotify's id, not ours
                "added_at": item.get("added_at"),
            })
    return rows


def new_entries(current: list[dict], seen_track_ids) -> list[dict]:
    """Rows whose track_id was not present on the previous run."""
    seen = set(seen_track_ids or [])
    return [r for r in current if r["track_id"] not in seen]


def track_ids(rows: list[dict]) -> list[str]:
    """Deduplicated, sorted track-id snapshot to persist as state."""
    return sorted({r["track_id"] for r in rows})


# ── network (never reached by the test suite) ────────────────────────────────

def _get_token(client_id: str, client_secret: str) -> str | None:
    import httpx
    r = httpx.post(_TOKEN_URL,
                   data={"grant_type": "client_credentials"},
                   auth=(client_id, client_secret),
                   headers={"User-Agent": "LOFI-Scout/0.1"}, timeout=20)
    return (r.json() or {}).get("access_token")


def _fetch_playlist(playlist_id: str, token: str) -> dict:
    import httpx
    r = httpx.get(f"{_API}/playlists/{playlist_id}/tracks",
                  params={"fields": "items(added_at,track(id,name,"
                                    "artists(id,name)))", "limit": 100},
                  headers={"Authorization": f"Bearer {token}",
                           "User-Agent": "LOFI-Scout/0.1"}, timeout=20)
    return r.json() or {}


class PlaylistListener:
    manifest = AgentManifest(
        name="playlist_listener",
        description="new artist adds to tastemaker Spotify playlists",
        writes=("signal",),
        schedule="daily@02:20",
        budget=Budget(max_runs_per_day=3),
    )

    def run(self, ctx) -> str:
        import os
        cid = os.environ.get("SPOTIFY_CLIENT_ID")
        secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
        if not (cid and secret):
            return "skipped: SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET not set"
        watched = _configured_playlists()
        if not watched:
            return "skipped: no playlists configured (sources.yaml)"

        token = None
        try:
            token = _get_token(cid, secret)
        except Exception:
            token = None
        if not token:
            return "skipped: Spotify auth failed"

        index = index_from_ctx(ctx)
        observed_at = ctx.now.date().isoformat()
        emitted = 0
        for pl in watched:
            pid = pl["id"]
            try:
                current = parse_playlist_items(_fetch_playlist(pid, token))
            except Exception:
                continue  # one playlist failing must not stop the sweep
            prev_ids = ctx.state_get(f"tracks:{pid}")
            fresh = new_entries(current, prev_ids)
            ctx.state_set(f"tracks:{pid}", track_ids(current))
            if prev_ids is None:
                continue  # first sight of this playlist = baseline, no alerts
            for row in fresh:
                name = row["artist_name"]
                ctx.emit("signal", {
                    "source": self.manifest.name,
                    "kind": "playlist_add",
                    "value": None,
                    "evidence": {
                        "playlist": pl["name"],
                        "playlist_id": pid,
                        "track": row.get("track_name"),
                        "track_id": row.get("track_id"),
                        "spotify_artist_id": row.get("artist_id"),
                    },
                    "observed_at": observed_at,
                }, artist_id=resolve(name, index), artist_name=name)
                emitted += 1
        return f"{emitted} playlist_add signal(s) over {len(watched)} playlist(s)"

"""
The Sentinel — the OS's standing attention system (docs/agentic_os.md §3.4).

Subscribes to every `signal`, maintains a per-artist composite HEAT with
per-source decay (a press mention fades in weeks, a label signing doesn't),
and promotes an artist to a `candidate` + human `alert` when heat crosses a
threshold — or when a Judge's re-hearing watch is tripped, regardless of heat
(the mechanism Phase D's dissent summaries will feed).

Heat is a pure function so the Feed console can render the same leaderboard the
promoter uses. This is the blackboard-heat sentinel; the score-spike
`watchlist_sentinel` in agents_builtin.py is a different animal and the two
coexist deliberately.
"""
from __future__ import annotations

import datetime as _dt
import math
import os

from osk.detectors._common import signal_views
from osk.manifest import AgentManifest, Budget

# Per-kind half-lives in days — how fast a signal's contribution decays.
HALF_LIVES = {
    "press_mention": 21, "playlist_add": 45, "sc_plays_velocity": 30,
    "pf_fans_delta": 30, "label_signing": 120,
    "divergence": 45, "ignition": 60, "fee_lag": 30, "venue_ladder": 90,
    "scene_centrality": 90,
}
# Weights — milestone kinds 1.0, derived detector kinds 2.0 (already
# aggregates), a label signing 3.0 (a free A&R expert vote).
WEIGHTS = {
    "press_mention": 1.0, "playlist_add": 1.0, "sc_plays_velocity": 1.0,
    "pf_fans_delta": 1.0, "label_signing": 3.0,
    "divergence": 2.0, "ignition": 2.0, "fee_lag": 2.0, "venue_ladder": 2.0,
    "scene_centrality": 2.0,
}
DEFAULT_HALF_LIFE = 30.0
DEFAULT_WEIGHT = 1.0

DEFAULT_PROMOTE_MIN = 4.0
COOLDOWN_DAYS = 30
HEAT_WINDOW_DAYS = 90


def _reason(view: dict) -> str:
    """A human-readable line for the top-reasons list — the detector's explain
    when it has one, otherwise the raw kind."""
    if view.get("explain"):
        return view["explain"]
    kind = (view.get("kind") or "signal").replace("_", " ")
    val = view.get("value")
    return f"{kind}" + (f" ({val:g})" if isinstance(val, (int, float)) else "")


# ── heat (pure) ───────────────────────────────────────────────────────────────

def compute_heat(signals: list[dict], now: _dt.datetime,
                 half_lives: dict | None = None,
                 weights: dict | None = None) -> dict:
    """`signals` are signal_views dicts (need key, kind, when, artist_name).
    heat = Σ weight(kind) · exp(−age_days / half_life(kind)). Returns
    {key: {heat, top_reasons, artist_name}}."""
    half_lives = half_lives or HALF_LIVES
    weights = weights or WEIGHTS
    heat: dict = {}
    reasons: dict = {}
    names: dict = {}
    for s in signals or []:
        key, kind, when = s.get("key"), s.get("kind"), s.get("when")
        if not key or when is None:
            continue
        age = max((now - when).total_seconds() / 86400.0, 0.0)
        hl = half_lives.get(kind, DEFAULT_HALF_LIFE)
        w = weights.get(kind, DEFAULT_WEIGHT)
        contrib = w * math.exp(-age / hl)
        heat[key] = heat.get(key, 0.0) + contrib
        names.setdefault(key, s.get("artist_name") or key)
        reasons.setdefault(key, []).append((contrib, _reason(s)))
    out = {}
    for key, h in heat.items():
        ranked = sorted(reasons[key], key=lambda t: t[0], reverse=True)
        out[key] = {"heat": round(h, 3),
                    "top_reasons": [txt for _, txt in ranked[:3]],
                    "artist_name": names[key]}
    return out


def match_watches(watches: list[dict], signals: list[dict]) -> dict:
    """Re-hearing hook: {key: {name, reason, value}} for artists whose incoming
    signal matches a standing watch above its threshold. A watch is
    {artist_name, kind, threshold, reason} (Phase D writes these from Judge
    dissent summaries; the mechanism lives here now)."""
    out: dict = {}
    for w in watches or []:
        want_name = (w.get("artist_name") or "").strip().lower()
        want_kind = w.get("kind")
        try:
            thr = float(w.get("threshold") or 0.0)
        except (TypeError, ValueError):
            thr = 0.0
        for s in signals:
            if s.get("kind") != want_kind:
                continue
            name = (s.get("artist_name") or "").strip().lower()
            if want_name and name != want_name:
                continue
            if float(s.get("value") or 0.0) < thr:
                continue
            key = s.get("key")
            val = float(s.get("value") or 0.0)
            cur = out.get(key)
            if cur is None or val > cur["value"]:
                out[key] = {"name": s.get("artist_name") or key,
                            "reason": w.get("reason") or "watch tripped",
                            "value": val}
    return out


# ── agent shell ───────────────────────────────────────────────────────────────

class Sentinel:
    manifest = AgentManifest(
        name="sentinel",
        description="blackboard-heat early-warning: promote + alert",
        reads=("signal", "verdict"), writes=("candidate", "alert"),
        schedule="daily@06:00", budget=Budget(max_runs_per_day=4))

    def _threshold(self, ctx) -> float:
        env = os.environ.get("LOFI_OS_PROMOTE_MIN")
        if env:
            try:
                return float(env)
            except ValueError:
                pass
        stored = ctx.state_get("promotion_threshold")
        return float(stored) if stored is not None else DEFAULT_PROMOTE_MIN

    def _dissent_watches(self, ctx) -> list[dict]:
        """Re-hearing watches derived from Judge dissent (Phase D §3.3): a
        verdict's dissent block becomes a standing watch — artist, signal kind
        and threshold — so the condition that would flip the verdict re-hears
        the artist regardless of heat. Complements the state `watches` list."""
        since = (ctx.now - _dt.timedelta(days=180)).isoformat(timespec="seconds")
        out = []
        for r in ctx.read(["verdict"], since=since, limit=1000):
            d = (r.payload or {}).get("dissent") or {}
            kind = d.get("kind")
            if not kind:
                continue
            out.append({
                "artist_name": r.artist_name,
                "kind": kind,
                "threshold": d.get("threshold"),
                "reason": f"dissent: {d.get('reason') or d.get('condition') or 'watch'}",
            })
        return out

    def _in_cooldown(self, ctx, key: str) -> bool:
        from osk.blackboard import parse_ts
        last = parse_ts(ctx.state_get(f"promoted:{key}"))
        if last is None:
            return False
        return (ctx.now - last) < _dt.timedelta(days=COOLDOWN_DAYS)

    def _promote(self, ctx, key, artist_id, name, priority, trigger) -> None:
        payload = {"priority": round(priority, 3),
                   "trigger_signals": trigger, "heat": round(priority, 3)}
        ctx.emit("candidate", payload, artist_id=artist_id, artist_name=name)
        alert = {"artist_name": name, "artist_id": artist_id,
                 "reasons": trigger, "genres": [], "heat": round(priority, 3)}
        ctx.emit("alert", alert, artist_id=artist_id, artist_name=name)
        try:
            from predict.watchlist import push_webhook
            push_webhook([alert])
        except Exception:
            pass                                    # webhook is best-effort
        ctx.state_set(f"promoted:{key}", ctx.now.isoformat(timespec="seconds"))

    def run(self, ctx) -> str:
        since = (ctx.now - _dt.timedelta(days=HEAT_WINDOW_DAYS)
                 ).isoformat(timespec="seconds")
        views = signal_views(ctx.read(["signal"], since=since, limit=5000))
        if not views:
            return "no signals yet — heat accrues nightly"

        idmap = {v["key"]: (v["artist_id"], v["artist_name"]) for v in views}
        heat = compute_heat(views, ctx.now)
        threshold = self._threshold(ctx)
        watches = list(ctx.state_get("watches") or []) + self._dissent_watches(ctx)
        forced = match_watches(watches, views)

        promoted = []
        for key in set(heat) | set(forced):
            info = heat.get(key)
            h = info["heat"] if info else 0.0
            fr = forced.get(key)
            if not (h >= threshold or fr):
                continue
            if self._in_cooldown(ctx, key):
                continue
            artist_id, name = idmap.get(key, (None, key))
            if info:
                name = info["artist_name"]
            trigger = list(info["top_reasons"]) if info else []
            priority = h
            if fr:
                trigger = [f"re-hearing: {fr['reason']}"] + trigger
                priority = max(h, fr["value"])
                name = fr["name"] or name
            self._promote(ctx, key, artist_id, name, priority, trigger)
            promoted.append(name)

        if promoted:
            return f"promoted {len(promoted)}: {', '.join(promoted[:5])}"
        return f"heat computed for {len(heat)} artist(s), none crossed threshold"

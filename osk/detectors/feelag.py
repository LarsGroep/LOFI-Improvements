"""
Fee-lag arbitrage — the business case in one record (§5.6).

When an artist has fresh underground heat (a positive divergence or an ignition
in the last 14 days) but the comparable-fee basis `predict/fees.py` prices them
from is still flat, that is the "book in the gap" moment `predict/window.py`
models. This detector is Economist × Sentinel: it only fires when heat is
rising AND the modelled fee is comparable-based (a static corpus) AND that base
hasn't moved since we last looked.

Airtable is the fee ground truth. No Airtable → the detector degrades to a
clean "skipped", never a guess.
"""
from __future__ import annotations

import datetime as _dt
import os

from osk.detectors._common import HONEST_EMPTY, id_index, signal_views
from osk.manifest import AgentManifest, Budget

DEFAULT_WINDOW_DAYS = 14
DEFAULT_TOLERANCE = 0.10


def _airtable_missing() -> str | None:
    if not (os.environ.get("AIRTABLE_TOKEN") and os.environ.get("AIRTABLE_BASE_ID")):
        return "skipped: Airtable not configured"
    return None


# ── detection (pure) ──────────────────────────────────────────────────────────

def heat_rising(signals: list[dict], now: _dt.datetime,
                window_days: int = DEFAULT_WINDOW_DAYS) -> dict:
    """Artists with a POSITIVE divergence or an ignition inside the window →
    {artist: [reasons]}. Negative divergences (playlist-inflated) don't count."""
    horizon = now - _dt.timedelta(days=window_days)
    out: dict = {}
    for s in signals or []:
        key, kind, when = s.get("artist"), s.get("kind"), s.get("when")
        if kind not in ("divergence", "ignition") or when is None or not key:
            continue
        if when < horizon:
            continue
        if kind == "divergence" and float(s.get("value") or 0.0) <= 0:
            continue
        out.setdefault(key, []).append(kind)
    return out


def is_fee_lag(fee: dict, prev_base: float | None,
               tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """True when the fee is comparable-based (a static corpus, not the artist's
    own moving gages) and its base hasn't shifted beyond `tolerance` since the
    stored value — i.e. the market comparable is lagging the heat."""
    if not fee or fee.get("method", "insufficient") == "insufficient":
        return False
    if not str(fee.get("method", "")).startswith("comparables"):
        return False
    base = fee.get("base")
    if base is None:
        return False
    if prev_base is None:
        return True
    return abs(base - prev_base) <= tolerance * prev_base


# ── agent shell ───────────────────────────────────────────────────────────────

class FeeLagDetector:
    manifest = AgentManifest(
        name="fee_lag_detector",
        description="heat rising while comparable fee basis stays flat",
        reads=("signal",), writes=("signal",),
        schedule="daily@05:30", budget=Budget(max_runs_per_day=2))

    def run(self, ctx) -> str:
        if (why := _airtable_missing()):
            return why
        since = (ctx.now - _dt.timedelta(days=DEFAULT_WINDOW_DAYS)
                 ).isoformat(timespec="seconds")
        views = signal_views(ctx.read(["signal"], since=since, limit=5000),
                             kinds=("divergence", "ignition"))
        if not views:
            return HONEST_EMPTY
        idx = id_index(views)
        signals = [{"artist": v["key"], "kind": v["kind"], "value": v["value"],
                    "when": v["when"]} for v in views]
        rising = heat_rising(signals, ctx.now)
        if not rising:
            return HONEST_EMPTY

        from predict.fees import estimate_fee_for
        flat_by_id = {p.get("artist_id"): p
                      for p in (ctx.shared.get("flat_profiles") or [])}
        emitted = 0
        for key, reasons in rising.items():
            meta = idx[key]
            profile = flat_by_id.get(meta["artist_id"]) or {}
            genres = profile.get("genres") or []
            try:
                fee = estimate_fee_for(meta["artist_name"], genres)
            except Exception:               # graceful — Airtable hiccup, never crash
                continue
            prev_base = ctx.state_get(f"fee_base:{key}")
            if not is_fee_lag(fee, prev_base):
                ctx.state_set(f"fee_base:{key}", fee.get("base"))
                continue
            ctx.emit("signal", {
                "source": "fee_lag_detector", "kind": "fee_lag",
                "value": fee.get("base"), "inputs": meta["ids"],
                "explain": ("heat rising, comparable fee basis unchanged → "
                            "book-in-the-gap window"),
                "fee_method": fee.get("method"), "heat_reasons": reasons,
                "observed_at": ctx.now.date().isoformat(),
            }, artist_id=meta["artist_id"], artist_name=meta["artist_name"])
            ctx.state_set(f"fee_base:{key}", fee.get("base"))
            emitted += 1
        return f"{emitted} fee-lag signal(s) over {len(rising)} heating artists"

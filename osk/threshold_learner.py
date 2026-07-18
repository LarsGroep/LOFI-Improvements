"""
Learned promotion threshold — the Sentinel's trigger tunes itself.

docs/agentic_os.md §8: the Sentinel promotes an artist to `candidate` when
composite HEAT crosses `promotion_threshold` (default 4.0, hand-set). But the
right cutoff is a question the OS's own history can answer: of the artists it
already promoted, which heats actually preceded a booking or a book_now/act_fast
verdict, and which were noise? This agent labels matured promotions (≥60 days
old — long enough for the outcome to reveal itself) and fits the heat cutoff
that best separates hits from misses.

It writes the answer as a `promotion_threshold` signal; the Sentinel reads the
latest such signal (via its `promotion_threshold` state / env override) and
retunes. The fit is intentionally conservative, in the spirit of
predict/rank_weights.py: a thin, unbalanced label set returns None and the
threshold is left exactly where it was — honest insufficiency over a guessed
number. Pure-Python (llm=False); no Supabase configured → outcome-labels come
from verdicts alone.
"""
from __future__ import annotations

import datetime as _dt

from osk.blackboard import parse_ts
from osk.manifest import AgentManifest, Budget

MATURE_DAYS = 60                 # a promotion's outcome needs time to show
_MIN_SAMPLES = 10                # matured promotions needed before we trust a fit
_MIN_POSITIVES = 3               # …and enough of them positive to have a signal
_CLAMP_LO, _CLAMP_HI = 2.0, 10.0
_POSITIVE_VERDICTS = {"book_now", "act_fast"}


# ── the fit (pure) ────────────────────────────────────────────────────────────

def learn_threshold(samples: list[dict]) -> dict | None:
    """samples = [{"heat": float, "positive": bool}] — one per matured promotion.

    Scan every observed heat as a candidate cutoff (predict positive ⇔
    heat ≥ cutoff) and keep the one maximising F1 over the labels. Conservative
    like predict/rank_weights.fit_weights: needs ≥10 samples with ≥3 positives,
    and the winning cutoff is clamped to [2.0, 10.0]; otherwise None (the caller
    keeps the current threshold). Ties resolve to the lower cutoff — the OS errs
    toward surfacing an artist rather than missing one.
    """
    rows = [s for s in samples or [] if s.get("heat") is not None]
    positives = sum(1 for s in rows if s.get("positive"))
    if len(rows) < _MIN_SAMPLES or positives < _MIN_POSITIVES:
        return None

    cutpoints = sorted({float(s["heat"]) for s in rows})
    best_cut, best_f1 = cutpoints[0], -1.0
    for cut in cutpoints:
        tp = sum(1 for s in rows if float(s["heat"]) >= cut and s.get("positive"))
        fp = sum(1 for s in rows if float(s["heat"]) >= cut and not s.get("positive"))
        fn = sum(1 for s in rows if float(s["heat"]) < cut and s.get("positive"))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:                     # strict > → lowest cutoff wins ties
            best_f1, best_cut = f1, cut

    threshold = max(_CLAMP_LO, min(_CLAMP_HI, best_cut))
    return {"threshold": round(threshold, 3), "f1": round(best_f1, 3),
            "n": len(rows), "positives": positives}


# ── labelling helpers (pure) ──────────────────────────────────────────────────

def _verdict_value(payload: dict) -> str | None:
    """Verdict vocabulary lives under `verdict` (predict/window.py); accept
    `window` as an alias so we speak the same language as the deal radar."""
    v = payload.get("verdict") or payload.get("window")
    return str(v).strip().lower() if v else None


def build_samples(matured: list, verdicts: list,
                  booked: dict[str, bool] | None = None) -> list[dict]:
    """Turn matured `candidate` Records into labelled samples. Positive when the
    artist later drew a book_now/act_fast verdict, or (Supabase configured) has
    lofi_booked truthy. `booked` maps both artist_id and lower-cased name → bool.
    """
    booked = booked or {}
    pos_ids, pos_names = set(), set()
    for r in verdicts or []:
        if _verdict_value(getattr(r, "payload", {}) or {}) in _POSITIVE_VERDICTS:
            if r.artist_id:
                pos_ids.add(r.artist_id)
            if r.artist_name:
                pos_names.add(r.artist_name.strip().lower())

    samples = []
    for c in matured:
        payload = getattr(c, "payload", {}) or {}
        heat = payload.get("priority")
        if heat is None:
            heat = payload.get("heat")
        name = (c.artist_name or "").strip().lower()
        positive = bool(
            (c.artist_id and c.artist_id in pos_ids)
            or (name and name in pos_names)
            or (c.artist_id and booked.get(c.artist_id))
            or (name and booked.get(name)))
        samples.append({"heat": float(heat or 0.0), "positive": positive})
    return samples


# ── agent shell ───────────────────────────────────────────────────────────────

class ThresholdLearner:
    """Weekly retune of the Sentinel's promotion threshold from matured
    promotions (docs/agentic_os.md §8, verdict-calibration sibling)."""

    manifest = AgentManifest(
        name="threshold_learner",
        description="learn the Sentinel promotion threshold from matured promotions",
        reads=("candidate", "verdict", "signal"),
        writes=("signal",),
        schedule="weekly@tue 05:00",
        budget=Budget(max_runs_per_day=2),
    )

    def _booked_labels(self, ctx) -> dict[str, bool]:
        """lofi_booked per artist (id + lower-name) when Supabase is configured;
        {} otherwise. Reuses the per-tick candidate sweep (ctx.shared)."""
        import os
        if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY")):
            return {}
        try:
            from osk.agents_builtin import _load_candidates
            _load_candidates(ctx)                       # fills flat_profiles
            out: dict[str, bool] = {}
            for p in ctx.shared.get("flat_profiles") or []:
                flag = bool(p.get("lofi_booked"))
                if p.get("artist_id"):
                    out[p["artist_id"]] = flag
                if p.get("artist_name"):
                    out[p["artist_name"].strip().lower()] = flag
            return out
        except Exception:
            return {}                                   # degrade, never crash

    def run(self, ctx) -> str:
        cutoff = ctx.now - _dt.timedelta(days=MATURE_DAYS)
        candidates = ctx.read(["candidate"], limit=5000)
        matured = [c for c in candidates
                   if (ts := parse_ts(c.created_at)) is not None and ts <= cutoff]
        verdicts = ctx.read(["verdict"], limit=5000)
        booked = self._booked_labels(ctx)

        samples = build_samples(matured, verdicts, booked)
        result = learn_threshold(samples)
        n = len(samples)
        positives = sum(1 for s in samples if s["positive"])
        if result is None:
            return (f"not enough matured promotions ({n}/{_MIN_SAMPLES}) — "
                    "keeping current threshold")

        t = result["threshold"]
        ctx.emit("signal", {
            "source": "threshold_learner",
            "kind": "promotion_threshold",
            "value": t,
            "explain": (f"learned from {n} promotions ({positives} positive), "
                        f"F1={result['f1']:.2f}"),
            "observed_at": ctx.now.isoformat(timespec="seconds"),
        })
        return (f"learned promotion threshold {t} from {n} promotions "
                f"({positives} positive)")

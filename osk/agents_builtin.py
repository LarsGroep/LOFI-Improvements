"""
Phase A roster — the four existing cron jobs, re-homed as OS agents.

Each wraps a predict/ module verbatim (the logic stays where it was and
remains runnable standalone); the agent adds scheduling, auditing, and
blackboard records. All four are pure-Python (llm=False) — this roster can
run forever at zero LLM cost.

Every agent degrades gracefully: no Supabase configured → "skipped", never
a crash. Candidates load once per tick via ctx.shared, so a morning where
three agents fire costs one Supabase sweep, not three.
"""
from __future__ import annotations

import os

from osk.manifest import AgentManifest, Budget

_CANDIDATES_KEY = "candidates"


def _supabase_missing() -> str | None:
    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY")):
        return "skipped: SUPABASE_URL/SUPABASE_KEY not set"
    return None


def _load_candidates(ctx) -> list[dict]:
    """Ranked Scout candidates, cached per tick across agents."""
    if _CANDIDATES_KEY in ctx.shared:
        return ctx.shared[_CANDIDATES_KEY]
    from scout.data import load_flat_profiles, load_ml_features, make_client
    from scout.ranking import build_candidates, load_predictions

    client = make_client()
    flat = load_flat_profiles(client)
    candidates = build_candidates(flat, load_ml_features(client),
                                  load_predictions())
    ctx.shared[_CANDIDATES_KEY] = candidates
    ctx.shared["flat_profiles"] = flat
    return candidates


class ForecastLogger:
    """Nightly forecast snapshot → predict/data/forecast_log.csv accrues, so
    the calibration report fills itself in (FEEDBACK.md item 5)."""

    manifest = AgentManifest(
        name="forecast_logger",
        description="snapshot today's XGBoost forecasts for backtesting",
        writes=("signal",),
        schedule="daily@03:10",
        budget=Budget(max_runs_per_day=2),
    )

    def run(self, ctx) -> str:
        if (why := _supabase_missing()):
            return why
        from predict.backtest import log_forecasts
        n = log_forecasts(_load_candidates(ctx))
        ctx.emit("signal", {"source": "predict.backtest",
                            "kind": "forecast_log", "logged": n})
        return f"logged {n} forecasts"


class WatchlistSentinel:
    """Morning spike sweep → alert records + the existing webhook push
    (FEEDBACK.md item 7). Snapshot state lives on the blackboard now, so a
    fresh VPS deploy starts diffing from its second run automatically."""

    manifest = AgentManifest(
        name="watchlist_sentinel",
        description="momentum-spike detection over the watchlist",
        writes=("signal", "alert"),
        schedule="daily@07:30",
        budget=Budget(max_runs_per_day=4),
    )

    def run(self, ctx) -> str:
        if (why := _supabase_missing()):
            return why
        from predict.watchlist import (
            detect_spikes, load_watchlist, push_webhook, snapshot)
        candidates = _load_candidates(ctx)
        watch_all = os.environ.get("LOFI_OS_WATCH_ALL", "").strip().lower() \
            in ("1", "true", "yes", "on")
        watch = load_watchlist()
        if not watch and not watch_all:
            # mirror the CLI: no implicit whole-pool sweep. Still snapshot,
            # so the delta rule works from day one once a watchlist exists.
            ctx.state_set("snapshot", snapshot(candidates))
            return ("watchlist empty — add names to predict/data/"
                    "watchlist.json or set LOFI_OS_WATCH_ALL=1")
        prev = ctx.state_get("snapshot") or {}
        alerts = detect_spikes(candidates, watch=None if watch_all else watch,
                               prev=prev)
        ctx.state_set("snapshot", snapshot(candidates))
        for a in alerts:
            ctx.emit("alert", a, artist_id=a.get("artist_id"),
                     artist_name=a.get("artist_name"))
        pushed = push_webhook(alerts)
        scope = "whole pool" if watch_all else f"watchlist({len(watch)})"
        return (f"{len(alerts)} spike(s) over {scope}"
                + (", pushed to webhook" if pushed else ""))


class RankWeightsRefit:
    """Weekly learning-to-rank refit from booking outcomes (FEEDBACK.md
    item 6). scout/ranking.py picks the JSON up automatically."""

    manifest = AgentManifest(
        name="rank_weights_refit",
        description="refit Scout-score weights from booked-vs-not labels",
        writes=("signal",),
        schedule="weekly@mon 04:00",
        budget=Budget(max_runs_per_day=2),
    )

    def run(self, ctx) -> str:
        if (why := _supabase_missing()):
            return why
        from predict.rank_weights import fit_weights, save_weights
        from scoring.five_scores import compute_five_scores
        from scout.data import load_ml_features, make_client
        from scout.ranking import load_predictions

        _load_candidates(ctx)                      # fills flat_profiles too
        flat = ctx.shared["flat_profiles"]
        ml_by_id = load_ml_features(make_client())
        predictions = load_predictions()
        rows = []
        for p in flat:
            aid = p.get("artist_id")
            if not aid:
                continue
            scores = compute_five_scores(p, ml_by_id.get(aid) or {})
            rows.append(scores | {"forecast_90d": predictions.get(aid),
                                  "lofi_booked": bool(p.get("lofi_booked"))})
        weights = fit_weights(rows)
        if weights is None:
            ctx.emit("signal", {"source": "predict.rank_weights",
                                "kind": "refit", "status": "insufficient"})
            return "not enough booked labels — kept hand-tuned defaults"
        n_booked = sum(1 for r in rows if r["lofi_booked"])
        save_weights(weights, n_labels=n_booked)
        ctx.emit("signal", {"source": "predict.rank_weights", "kind": "refit",
                            "weights": weights, "n_booked_labels": n_booked})
        return f"refit from {len(rows)} artists ({n_booked} booked labels)"


class BacktestReporter:
    """Weekly honest hit-rate report on mature forecast rows (FEEDBACK.md
    item 5) — the report also renders in the app's Model health tab."""

    manifest = AgentManifest(
        name="backtest_reporter",
        description="score mature logged forecasts against reality",
        writes=("signal",),
        schedule="weekly@mon 04:30",
        budget=Budget(max_runs_per_day=2),
    )

    def run(self, ctx) -> str:
        if (why := _supabase_missing()):
            return why
        from predict.backtest import evaluate, read_log
        _load_candidates(ctx)
        current = {p.get("artist_id"): p.get("spotify_listeners")
                   for p in ctx.shared["flat_profiles"]
                   if p.get("spotify_listeners")}
        report = evaluate(read_log(), current)
        ctx.emit("signal", {"source": "predict.backtest",
                            "kind": "calibration_report", **report})
        n = report.get("n_mature", 0)
        return (f"{n} mature rows scored" if n
                else "no mature rows yet — report stays honest-empty")


BUILTIN_AGENTS = (ForecastLogger, WatchlistSentinel, RankWeightsRefit,
                  BacktestReporter)

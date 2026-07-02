"""
Forecast backtesting — trust the +32% or don't, but know which.

FEEDBACK.md item 5: the XGBoost 90-day forecast carries 20% of the Scout
score with an unmeasured hit rate. This module (a) logs today's forecasts
with today's listener counts, and (b) once logged rows are ≥ ~90 days old,
scores them against what actually happened. Run from cron / GitHub Actions:

    python -m predict.backtest log        # snapshot today's forecasts
    python -m predict.backtest report     # calibration report on mature rows

The log lives in predict/data/forecast_log.csv (gitignored). The report is
also rendered in the app's Model health tab. Until enough rows mature the
report says exactly that — no fake confidence.
"""
from __future__ import annotations

import csv
import datetime as _dt
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_LOG = Path(__file__).parent / "data" / "forecast_log.csv"
_FIELDS = ["date", "artist_id", "artist_name", "spotify_listeners", "forecast_90d"]
HORIZON_DAYS = 90


# ── logging ───────────────────────────────────────────────────────────────────

def log_forecasts(candidates: list[dict], path: Path = _LOG,
                  today: str | None = None) -> int:
    """Append one row per artist with a forecast + listener count. Idempotent
    per (date, artist): re-running on the same day adds nothing."""
    today = today or _dt.date.today().isoformat()
    rows = read_log(path)
    seen = {(r["date"], r["artist_id"]) for r in rows}
    new = []
    for c in candidates or []:
        if c.get("forecast_90d") is None or not c.get("spotify_listeners"):
            continue
        key = (today, c.get("artist_id"))
        if key in seen:
            continue
        new.append({"date": today, "artist_id": c.get("artist_id"),
                    "artist_name": c.get("artist_name"),
                    "spotify_listeners": c["spotify_listeners"],
                    "forecast_90d": c["forecast_90d"]})
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        empty = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDS)
            if empty:
                w.writeheader()
            w.writerows(new)
    return len(new)


def read_log(path: Path = _LOG) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── evaluation (pure) ─────────────────────────────────────────────────────────

def evaluate(logged: list[dict], current_listeners: dict[str, float],
             today: str | None = None, horizon_days: int = HORIZON_DAYS,
             hit_tolerance: float = 15.0) -> dict:
    """Score mature rows (age ≥ horizon): realized 90d growth vs predicted.

    Metrics a booker can read:
      direction_hit_rate — did it at least get up-vs-down right?
      within_tolerance   — |error| ≤ 15 points of growth
      mae / bias         — average miss size, and whether it systematically
                           over- or under-promises (bias > 0 = over-promises)
    """
    today_d = _dt.date.fromisoformat(today or _dt.date.today().isoformat())
    errs, hits_dir, hits_tol = [], 0, 0
    n = 0
    for r in logged or []:
        try:
            age = (today_d - _dt.date.fromisoformat(r["date"])).days
            then = float(r["spotify_listeners"])
            pred = float(r["forecast_90d"])
        except (KeyError, TypeError, ValueError):
            continue
        if age < horizon_days or then <= 0:
            continue
        now = current_listeners.get(r.get("artist_id"))
        if not now:
            continue
        realized = (float(now) - then) / then * 100.0
        err = pred - realized
        errs.append(err)
        n += 1
        if (pred >= 0) == (realized >= 0):
            hits_dir += 1
        if abs(err) <= hit_tolerance:
            hits_tol += 1

    if not n:
        return {"n_mature": 0, "note": "no logged forecasts are ≥ "
                f"{horizon_days} days old yet — run `python -m predict.backtest "
                "log` on a schedule and the report fills itself in"}
    return {
        "n_mature": n,
        "direction_hit_rate_pct": round(hits_dir / n * 100.0, 1),
        "within_tolerance_pct": round(hits_tol / n * 100.0, 1),
        "tolerance_points": hit_tolerance,
        "mae_points": round(sum(abs(e) for e in errs) / n, 1),
        "bias_points": round(sum(errs) / n, 1),
    }


def _main(argv: list[str]) -> int:
    from scout.data import load_flat_profiles, load_ml_features, make_client
    from scout.ranking import build_candidates, load_predictions

    client = make_client()
    flat = load_flat_profiles(client)
    candidates = build_candidates(flat, load_ml_features(client),
                                  load_predictions())
    if "log" in argv:
        n = log_forecasts(candidates)
        print(f"Logged {n} forecasts → {_LOG}")
    if "report" in argv or not argv:
        current = {p.get("artist_id"): p.get("spotify_listeners")
                   for p in flat if p.get("spotify_listeners")}
        print(evaluate(read_log(), current))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

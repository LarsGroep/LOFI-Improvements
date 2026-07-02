"""
Learned ranking weights — the tool gets better every season LOFI uses it.

FEEDBACK.md item 6: the Scout-score weights (0.35/0.30/0.20/0.15) are
hand-tuned. LOFI's own booking decisions are labels: fit a tiny logistic
regression (pure Python — 5 features, no sklearn needed) on booked vs
unbooked artists, convert the positive coefficient mass into ranking
weights, and write them where the ranker looks:

    python -m predict.rank_weights        # fit from Supabase, write JSON

scout/ranking.py picks predict/data/rank_weights.json up automatically when
it exists; delete the file to fall back to the hand-tuned defaults. The fit
is intentionally conservative: weights are floored and re-normalised so a
thin label set can shift emphasis but never zero out a signal.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

WEIGHTS_PATH = Path(__file__).parent / "data" / "rank_weights.json"
FEATURES = ["future_potential", "growth", "forecast_norm", "momentum"]
DEFAULTS = {"future_potential": 0.35, "growth": 0.30,
            "forecast_norm": 0.20, "momentum": 0.15}
_MIN_LABELS = 10       # booked artists needed before we trust a fit
_FLOOR = 0.05          # no signal ever drops below 5% weight


def _standardize(X: list[list[float]]) -> list[list[float]]:
    cols = list(zip(*X))
    out_cols = []
    for col in cols:
        mean = sum(col) / len(col)
        sd = math.sqrt(sum((v - mean) ** 2 for v in col) / len(col)) or 1.0
        out_cols.append([(v - mean) / sd for v in col])
    return [list(row) for row in zip(*out_cols)]


def fit_logistic(X: list[list[float]], y: list[int], l2: float = 0.1,
                 lr: float = 0.1, iters: int = 800) -> list[float]:
    """Plain batch gradient descent — the data is tiny, this is instant."""
    n, d = len(X), len(X[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(iters):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            z = sum(wj * xj for wj, xj in zip(w, xi)) + b
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            e = p - yi
            for j in range(d):
                gw[j] += e * xi[j]
            gb += e
        w = [wj - lr * (gwj / n + l2 * wj / n) for wj, gwj in zip(w, gw)]
        b -= lr * gb / n
    return w


def weights_from_coefs(coefs: list[float]) -> dict[str, float]:
    """Positive coefficient mass → normalised, floored ranking weights.
    Every feature keeps the floor; the remaining mass is split proportionally
    to the coefficients — guaranteed ≥ floor and summing to 1."""
    pos = [max(0.0, c) for c in coefs]
    if sum(pos) <= 0:
        return dict(DEFAULTS)
    free = 1.0 - _FLOOR * len(FEATURES)
    return {f: round(_FLOOR + free * p / sum(pos), 3)
            for f, p in zip(FEATURES, pos)}


def fit_weights(rows: list[dict]) -> dict | None:
    """rows: candidate-shaped dicts INCLUDING booked artists, each with the
    five scores + forecast_90d + lofi_booked. None when labels are too thin."""
    X, y = [], []
    for r in rows or []:
        fc = r.get("forecast_90d")
        X.append([
            float(r.get("future_potential") or 0.0),
            float(r.get("growth") or 0.0),
            50.0 if fc is None else max(0.0, min(100.0, float(fc))),
            float(r.get("momentum") or 0.0),
        ])
        y.append(1 if r.get("lofi_booked") else 0)
    if sum(y) < _MIN_LABELS or sum(y) == len(y):
        return None
    coefs = fit_logistic(_standardize(X), y)
    return weights_from_coefs(coefs)


# ── persistence + the hook scout/ranking.py calls ─────────────────────────────

def save_weights(weights: dict, path: Path = WEIGHTS_PATH,
                 n_labels: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"weights": weights, "fitted": _dt.date.today().isoformat(),
         "n_booked_labels": n_labels}, indent=2), encoding="utf-8")


def learned_weights(path: Path = WEIGHTS_PATH) -> dict | None:
    """The ranker's hook: fitted weights, or None → hand-tuned defaults."""
    try:
        w = json.loads(path.read_text(encoding="utf-8")).get("weights") or {}
        if set(w) == set(FEATURES) and abs(sum(w.values()) - 1.0) < 0.02:
            return w
    except Exception:
        pass
    return None


def _main() -> int:
    from scoring.five_scores import compute_five_scores
    from scout.data import load_flat_profiles, load_ml_features, make_client
    from scout.ranking import load_predictions

    client = make_client()
    ml_by_id = load_ml_features(client)
    predictions = load_predictions()
    rows = []
    for p in load_flat_profiles(client):
        aid = p.get("artist_id")
        if not aid:
            continue
        scores = compute_five_scores(p, ml_by_id.get(aid) or {})
        rows.append(scores | {"forecast_90d": predictions.get(aid),
                              "lofi_booked": bool(p.get("lofi_booked"))})
    w = fit_weights(rows)
    if w is None:
        print(f"Not enough booked labels (need ≥ {_MIN_LABELS}) — "
              "keeping hand-tuned defaults.")
        return 1
    save_weights(w, n_labels=sum(1 for r in rows if r["lofi_booked"]))
    print(f"Fitted weights from {len(rows)} artists → {WEIGHTS_PATH}\n{w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

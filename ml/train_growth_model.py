"""
Train XGBoost growth-prediction model — aligned with future_predictor_v1 notebook.

Target  : 90-day forward % change of Chartmetric CPP score (audience_index),
          computed via shift(-FORECAST_DAYS) on each artist's full timeseries.
          This is a genuine forward-looking prediction, not a retrospective echo.

Features: 12 engineered features per metric (7d/30d/90d growth, acceleration,
          rolling mean/std, coefficient of variation) + 4 cross-platform ratios
          = ~100 features total.

Training: ALL historical rows where the 90-day future is known (each artist
          contributes ~275 training rows, not just one snapshot).

Saves:
  ml/models/growth_predictor.json
  ml/models/predictions.csv
  ml/models/model_meta.json

Run:
    python ml/train_growth_model.py [--output-dir ml/models]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import mean_absolute_error, r2_score
    from scipy.stats import spearmanr
    _HAS_SKL = True
except ImportError:
    _HAS_SKL = False

from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# ── Constants matching the notebook ──────────────────────────────────────────
METRIC_MAP: dict[tuple[str, str], str] = {
    ("spotify",         "followers"):    "spotify_followers",
    ("spotify",         "listeners"):    "spotify_listeners",
    ("instagram",       "followers"):    "instagram_followers",
    ("youtube_channel", "views"):        "youtube_channel_views",
    ("youtube_channel", "subscribers"):  "youtube_channel_subscribers",
    ("soundcloud",      "followers"):    "soundcloud_followers",
    ("facebook",        "likes"):        "facebook_likes",
    ("cpp",             "score"):        "chartmetric_cpp_score",
    ("cpp",             "rank"):         "chartmetric_cpp_rank",
}

TRAINED_METRICS = [
    "spotify_followers",
    "spotify_listeners",
    "youtube_channel_subscribers",
    "youtube_channel_views",
    "instagram_followers",
    "chartmetric_cpp_score",
    "chartmetric_cpp_rank",
    "soundcloud_followers",
]

TARGET_COL    = "chartmetric_cpp_score"   # "audience_index" in notebook
FORECAST_DAYS = 90
EPS           = 1e-9


# ── Data loading ─────────────────────────────────────────────────────────────

def _paginate(table: str, select: str, page_size: int = 50) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        page = (
            sb.schema("tinder").table(table)
            .select(select)
            .range(offset, offset + page_size - 1)
            .execute().data or []
        )
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return rows


def _expand_timeseries(all_cm: list[dict], name_map: dict[str, str]) -> pd.DataFrame:
    """Expand JSONB cm_timeseries into wide DataFrame (one row per artist x date)."""
    store: dict[str, dict[str, dict[str, float]]] = {}
    for row in all_cm:
        aid = row["artist_id"]
        ts  = row.get("cm_timeseries") or {}
        if not ts:
            continue
        for (platform, metric), col_name in METRIC_MAP.items():
            pts = (ts.get(platform) or {}).get(metric) or []
            for pt in pts:
                d = pt.get("date")
                v = pt.get("value")
                if d is None or v is None:
                    continue
                store.setdefault(aid, {}).setdefault(col_name, {})[d] = float(v)

    all_cols = list(METRIC_MAP.values())
    rows_out = []
    for aid, col_data in store.items():
        all_dates = sorted({d for col_vals in col_data.values() for d in col_vals})
        for date in all_dates:
            row = {"artist_id": aid, "artist_name": name_map.get(aid, aid), "date": date}
            for col in all_cols:
                row[col] = col_data.get(col, {}).get(date)
            rows_out.append(row)

    if not rows_out:
        return pd.DataFrame()

    df = pd.DataFrame(rows_out)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["artist_name", "date"]).reset_index(drop=True)

    # Forward-fill gaps within each artist
    df[all_cols] = df.groupby("artist_name")[all_cols].transform("ffill")
    df[all_cols] = df[all_cols].fillna(0)
    return df


# ── Feature engineering ──────────────────────────────────────────────────────

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Index each metric to 100 at first observation per artist."""
    df = df.copy()
    for col in list(METRIC_MAP.values()):
        first = df.groupby("artist_name")[col].transform("first")
        safe_first = first.replace(0, np.nan)  # avoid ZeroDivisionError in np.where
        df[col] = np.where(first != 0, df[col] / safe_first * 100, df[col])
    return df


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add 12 features per metric + 4 cross-metric ratios (matches notebook exactly)."""
    new_cols: dict[str, pd.Series] = {}

    for col in TRAINED_METRICS:
        if col not in df.columns:
            continue
        g = df.groupby("artist_name")[col]
        # Replace 0→NaN for growth rate denominator: _expand_timeseries fills missing
        # values with 0, so pct_change would hit a ZeroDivisionError on sparse artists.
        g_safe = df[col].where(df[col] != 0).groupby(df["artist_name"])

        # Growth rates
        new_cols[f"{col}_7d_growth"]  = g_safe.pct_change(7,  fill_method=None) * 100
        new_cols[f"{col}_30d_growth"] = g_safe.pct_change(30, fill_method=None) * 100
        new_cols[f"{col}_90d_growth"] = g_safe.pct_change(90, fill_method=None) * 100

        # Acceleration — second derivative of growth
        new_cols[f"{col}_accel_7v30"]  = new_cols[f"{col}_7d_growth"]  - new_cols[f"{col}_30d_growth"]
        new_cols[f"{col}_accel_30v90"] = new_cols[f"{col}_30d_growth"] - new_cols[f"{col}_90d_growth"]

        # Rolling means
        new_cols[f"{col}_7d_mean"]  = g.transform(lambda x: x.rolling(7,  min_periods=7 ).mean())
        new_cols[f"{col}_30d_mean"] = g.transform(lambda x: x.rolling(30, min_periods=30).mean())
        new_cols[f"{col}_90d_mean"] = g.transform(lambda x: x.rolling(90, min_periods=90).mean())

        # Rolling standard deviation (volatility)
        new_cols[f"{col}_30d_std"] = g.transform(lambda x: x.rolling(30, min_periods=30).std())
        new_cols[f"{col}_90d_std"] = g.transform(lambda x: x.rolling(90, min_periods=90).std())

        # Coefficient of variation (relative volatility)
        new_cols[f"{col}_30d_cv"] = new_cols[f"{col}_30d_std"] / (new_cols[f"{col}_30d_mean"].abs() + EPS)
        new_cols[f"{col}_90d_cv"] = new_cols[f"{col}_90d_std"] / (new_cols[f"{col}_90d_mean"].abs() + EPS)

    # Cross-metric ratios
    new_cols["listeners_per_follower"]   = df["spotify_listeners"]           / (df["spotify_followers"]           + EPS)
    new_cols["instagram_per_spotify"]    = df["instagram_followers"]         / (df["spotify_followers"]           + EPS)
    new_cols["youtube_subs_per_spotify"] = df["youtube_channel_subscribers"] / (df["spotify_followers"]           + EPS)
    new_cols["youtube_views_per_sub"]    = df["youtube_channel_views"]       / (df["youtube_channel_subscribers"] + EPS)

    # Join all new columns at once to avoid DataFrame fragmentation
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    meta = {"artist_id", "artist_name", "date", "audience_index", "target_growth"}
    base = set(METRIC_MAP.values())
    feature_cols = [c for c in df.columns if c not in meta and c not in base]
    return df, feature_cols


# ── Main data pipeline ───────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Returns (df_train, df_full, feature_cols).

    df_train — rows with a known 90-day future target, NaN->0 filled, upper outliers clipped.
    df_full  — all rows including latest (NaN preserved for missing_pct at inference time).
    """
    print("Loading timeseries from Supabase (paginated)...")
    all_cm    = _paginate("artist_chartmetric", "artist_id, cm_timeseries", page_size=50)
    name_rows = _paginate("artists", "id, name", page_size=500)
    name_map  = {r["id"]: r["name"] for r in name_rows}
    print(f"  Fetched {len(all_cm)} artist rows")

    df = _expand_timeseries(all_cm, name_map)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    print(f"  Expanded to {len(df):,} rows across {df['artist_name'].nunique()} artists")

    df = _normalize(df)

    # Target: genuine 90-day forward growth of CPP score
    df["audience_index"] = df[TARGET_COL]
    df["target_growth"]  = (
        df.groupby("artist_name")["audience_index"].shift(-FORECAST_DAYS)
        / (df["audience_index"] + EPS)
        - 1
    ) * 100

    df, feature_cols = _build_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    df_full = df.copy()   # preserve NaN for inference missing_pct

    # Training rows: only where future is known
    df_train = df.dropna(subset=["target_growth"]).copy()
    # Clip upper tail only (same as notebook — preserves declining artists)
    upper = df_train["target_growth"].quantile(0.98)
    df_train = df_train[df_train["target_growth"] <= upper].copy()
    df_train[feature_cols] = df_train[feature_cols].fillna(0)

    print(f"  Training rows: {len(df_train):,}  features: {len(feature_cols)}")
    return df_train, df_full, feature_cols


# ── Training ─────────────────────────────────────────────────────────────────

def train(output_dir: Path) -> None:
    if not _HAS_XGB:
        print("xgboost not installed — run: pip install xgboost")
        sys.exit(1)
    if not _HAS_SKL:
        print("scikit-learn or scipy not installed — run: pip install scikit-learn scipy")
        sys.exit(1)

    df_train, df_full, feature_cols = load_data()
    if df_train.empty:
        print("No usable training data.")
        sys.exit(1)

    X      = df_train[feature_cols].values
    y      = df_train["target_growth"].values
    groups = df_train["artist_name"].values

    # Group-aware split — no artist leaks across train/test
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X[train_idx], y[train_idx],
        eval_set=[(X[test_idx], y[test_idx])],
        verbose=50,
    )

    preds   = model.predict(X[test_idx])
    mae     = mean_absolute_error(y[test_idx], preds)
    r2      = r2_score(y[test_idx], preds)
    sp      = spearmanr(y[test_idx], preds).correlation
    dir_acc = float(np.mean(np.sign(preds) == np.sign(y[test_idx])))

    print(f"\nTest  MAE: {mae:.1f}%  R2: {r2:.3f}  Spearman: {sp:.3f}  Dir: {dir_acc:.2%}")

    imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
    print("\nTop-10 features:")
    for name, score in imp[:10]:
        print(f"  {name:<50} {score:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "growth_predictor.json"
    model.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Predictions: latest row per artist from full timeseries
    latest_raw = (
        df_full.sort_values(["artist_name", "date"])
        .groupby("artist_name").tail(1)
        .copy()
    )
    missing_pct        = (latest_raw[feature_cols].isna().mean(axis=1) * 100).values
    available_features = (~latest_raw[feature_cols].isna()).sum(axis=1).values

    latest = latest_raw.copy()
    latest[feature_cols] = latest[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    latest["predicted_growth_90d"] = model.predict(latest[feature_cols].values)
    latest["missing_pct"]          = missing_pct
    latest["available_features"]   = available_features
    latest["total_features"]       = len(feature_cols)

    pred_df = (
        latest[["artist_name", "artist_id", "date",
                "predicted_growth_90d", "missing_pct",
                "available_features", "total_features"]]
        .sort_values("predicted_growth_90d", ascending=False)
        .reset_index(drop=True)
    )
    pred_path = output_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path} ({len(pred_df)} artists)")

    # Write all predictions to Supabase
    model_version = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    supabase_rows = []
    for _, row in pred_df.iterrows():
        supabase_rows.append({
            "artist_id":            str(row["artist_id"]),
            "artist_name":          str(row["artist_name"]),
            "predicted_growth_90d": float(row["predicted_growth_90d"]),
            "missing_pct":          float(row["missing_pct"]),
            "available_features":   int(row["available_features"]),
            "total_features":       int(row["total_features"]),
            "prediction_date":      str(row["date"].date() if hasattr(row["date"], "date") else row["date"]),
            "model_version":        model_version,
            "predicted_at":         datetime.now(timezone.utc).isoformat(),
        })
    try:
        # Upsert in batches of 100 to avoid request size limits
        batch_size = 100
        for i in range(0, len(supabase_rows), batch_size):
            batch = supabase_rows[i:i + batch_size]
            sb.schema("tinder").table("xgboost_predictions").upsert(
                batch, on_conflict="artist_id"
            ).execute()
        print(f"Predictions upserted to Supabase ({len(supabase_rows)} artists)")
    except Exception as _e:
        print(f"  Supabase write failed (predictions still saved to CSV): {_e}")

    feat_imp  = {col: float(v) for col, v in zip(feature_cols, model.feature_importances_)}
    meta_path = output_dir / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "feature_cols":        feature_cols,
            "target":              "chartmetric_cpp_score_90d_forward_pct",
            "feature_importances": feat_imp,
            "n_training_rows":     len(df_train),
            "n_training_artists":  int(df_train["artist_name"].nunique()),
            "trained_at":          datetime.now().strftime("%Y-%m-%d %H:%M"),
            "test_mae":            round(float(mae), 1),
            "test_r2":             round(float(r2), 3),
            "test_spearman":       round(float(sp), 3),
            "test_dir_acc":        round(float(dir_acc), 3),
        }, f, indent=2)
    print(f"Metadata saved to {meta_path}")


def infer_artist(artist_id: str, output_dir: Path, artist_name: str = "") -> float | None:
    """
    Run inference for a single artist using the saved model.
    Updates predictions.csv and upserts the result into tinder.xgboost_predictions.
    Returns the predicted 90d growth %, or None if the model is missing or data insufficient.
    """
    if not _HAS_XGB:
        return None

    model_path = output_dir / "growth_predictor.json"
    meta_path  = output_dir / "model_meta.json"
    pred_path  = output_dir / "predictions.csv"

    if not model_path.exists() or not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols: list[str] = meta["feature_cols"]
    model_version = meta.get("trained_at", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    model = XGBRegressor()
    model.load_model(str(model_path))

    # Fetch only this artist's data
    name_rows = (
        sb.schema("tinder").table("artists")
        .select("id, name").eq("id", artist_id).execute().data or []
    )
    if not name_rows:
        return None
    resolved_name = name_rows[0]["name"] or artist_name or artist_id
    name_map = {name_rows[0]["id"]: resolved_name}

    cm_rows = (
        sb.schema("tinder").table("artist_chartmetric")
        .select("artist_id, cm_timeseries")
        .eq("artist_id", artist_id).execute().data or []
    )
    if not cm_rows or not (cm_rows[0].get("cm_timeseries") or {}):
        return None

    df = _expand_timeseries(cm_rows, name_map)
    if df.empty:
        return None

    df = _normalize(df)
    df["audience_index"] = df[TARGET_COL]
    df["target_growth"]  = np.nan  # unknown for new artist
    df, _ = _build_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    latest_raw = df.sort_values("date").tail(1).copy()
    missing_pct = float(latest_raw[feature_cols].isna().mean(axis=1).iloc[0] * 100)
    available   = int((~latest_raw[feature_cols].isna()).sum(axis=1).iloc[0])
    n_total     = len(feature_cols)
    latest_date = latest_raw["date"].iloc[0].date()

    # Build aligned feature matrix — fill any column gaps with 0
    X = pd.DataFrame(0.0, index=[0], columns=feature_cols)
    for col in feature_cols:
        if col in latest_raw.columns:
            X[col] = latest_raw[col].fillna(0).values[0]

    predicted_growth = float(model.predict(X.values)[0])

    new_row = {
        "artist_name":          resolved_name,
        "artist_id":            artist_id,
        "date":                 str(latest_date),
        "predicted_growth_90d": round(predicted_growth, 4),
        "missing_pct":          round(missing_pct, 1),
        "available_features":   available,
        "total_features":       n_total,
    }

    if pred_path.exists():
        preds_df = pd.read_csv(pred_path)
        preds_df = preds_df[preds_df["artist_id"] != artist_id]
        preds_df = pd.concat([preds_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        preds_df = pd.DataFrame([new_row])

    preds_df = preds_df.sort_values("predicted_growth_90d", ascending=False).reset_index(drop=True)
    preds_df.to_csv(pred_path, index=False)

    # Write prediction to Supabase
    try:
        sb.schema("tinder").table("xgboost_predictions").upsert({
            "artist_id":            artist_id,
            "artist_name":          resolved_name,
            "predicted_growth_90d": float(predicted_growth),
            "missing_pct":          float(missing_pct),
            "available_features":   int(available),
            "total_features":       int(n_total),
            "prediction_date":      str(latest_date),
            "model_version":        model_version,
            "predicted_at":         datetime.now(timezone.utc).isoformat(),
        }, on_conflict="artist_id").execute()
    except Exception as _e:
        print(f"    XGBoost Supabase write failed: {_e}")

    return predicted_growth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="ml/models")
    parser.add_argument("--artist-id", help="Inference-only for a single artist (skips training)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.artist_id:
        pred = infer_artist(args.artist_id, output_dir)
        if pred is not None:
            print(f"Predicted 90d growth: {pred:+.1f}%")
        else:
            print("Could not infer: model missing or insufficient timeseries data.")
    else:
        train(output_dir)


if __name__ == "__main__":
    main()

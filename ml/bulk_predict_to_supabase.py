"""
One-time (or periodic) bulk inference: run XGBoost predictions for ALL artists that
have timeseries data and write results to tinder.xgboost_predictions in Supabase.

Run once after training to populate the table, or re-run to refresh stale predictions.

Usage:
    python ml/bulk_predict_to_supabase.py [--output-dir ml/models]
"""
from __future__ import annotations

import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "ml"))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import json
import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# Import shared helpers from train_growth_model
from train_growth_model import (
    _paginate,
    _expand_timeseries,
    _normalize,
    _build_features,
    TARGET_COL,
)


def run_bulk_inference(output_dir: Path) -> None:
    if not _HAS_XGB:
        print("xgboost not installed — run: pip install xgboost")
        sys.exit(1)

    model_path = output_dir / "growth_predictor.json"
    meta_path  = output_dir / "model_meta.json"

    if not model_path.exists() or not meta_path.exists():
        print(f"Model not found at {output_dir}. Run train_growth_model.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols: list[str] = meta["feature_cols"]
    model_version = meta.get("trained_at", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    model = XGBRegressor()
    model.load_model(str(model_path))
    print(f"Model loaded from {model_path}  ({len(feature_cols)} features)")

    print("Loading timeseries from Supabase (paginated)...")
    all_cm    = _paginate("artist_chartmetric", "artist_id, cm_timeseries", page_size=50)
    name_rows = _paginate("artists", "id, name", page_size=500)
    name_map  = {r["id"]: r["name"] for r in name_rows}
    print(f"  Fetched {len(all_cm)} artist rows")

    df = _expand_timeseries(all_cm, name_map)
    if df.empty:
        print("No timeseries data found.")
        sys.exit(0)

    print(f"  Expanded to {len(df):,} rows across {df['artist_name'].nunique()} artists")

    df = _normalize(df)
    df["audience_index"] = df[TARGET_COL]
    df["target_growth"]  = np.nan
    df, _ = _build_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Latest row per artist
    latest_raw = (
        df.sort_values(["artist_name", "date"])
        .groupby("artist_name").tail(1)
        .copy()
    )

    missing_pct_vals        = (latest_raw[feature_cols].isna().mean(axis=1) * 100).values
    available_features_vals = (~latest_raw[feature_cols].isna()).sum(axis=1).values

    latest = latest_raw.copy()
    latest[feature_cols] = latest[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    predictions = model.predict(latest[feature_cols].values)

    now_ts = datetime.now(timezone.utc).isoformat()
    supabase_rows = []
    for i, (_, row) in enumerate(latest.iterrows()):
        artist_id = str(row["artist_id"])
        artist_name = name_map.get(artist_id, str(row.get("artist_name", artist_id)))
        latest_date = row["date"]
        if hasattr(latest_date, "date"):
            latest_date = latest_date.date()

        supabase_rows.append({
            "artist_id":            artist_id,
            "artist_name":          artist_name,
            "predicted_growth_90d": float(predictions[i]),
            "missing_pct":          float(missing_pct_vals[i]),
            "available_features":   int(available_features_vals[i]),
            "total_features":       len(feature_cols),
            "prediction_date":      str(latest_date),
            "model_version":        model_version,
            "predicted_at":         now_ts,
        })

    print(f"Upserting {len(supabase_rows)} predictions to Supabase...")
    batch_size = 100
    ok = 0
    for i in range(0, len(supabase_rows), batch_size):
        batch = supabase_rows[i:i + batch_size]
        try:
            sb.schema("tinder").table("xgboost_predictions").upsert(
                batch, on_conflict="artist_id"
            ).execute()
            ok += len(batch)
            print(f"  [{ok}/{len(supabase_rows)}] upserted")
        except Exception as e:
            print(f"  Batch {i}–{i+batch_size} failed: {e}")

    print(f"\nDone. {ok}/{len(supabase_rows)} predictions written to tinder.xgboost_predictions.")

    # Also save updated CSV
    pred_df = pd.DataFrame(supabase_rows)[
        ["artist_name", "artist_id", "prediction_date",
         "predicted_growth_90d", "missing_pct", "available_features", "total_features"]
    ].rename(columns={"prediction_date": "date"}).sort_values(
        "predicted_growth_90d", ascending=False
    ).reset_index(drop=True)
    pred_path = output_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"predictions.csv updated at {pred_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-predict growth for all artists and write to Supabase."
    )
    parser.add_argument("--output-dir", default="ml/models",
                        help="Directory containing growth_predictor.json and model_meta.json")
    args = parser.parse_args()
    run_bulk_inference(Path(args.output_dir))


if __name__ == "__main__":
    main()

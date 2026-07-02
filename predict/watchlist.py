"""
Watchlist alerts — the tool pings the scout, not the other way round.

FEEDBACK.md item 7: scouting is temporal. Watch a set of names (or the whole
unbooked pool), detect momentum spikes with simple explainable rules, and
push alerts to a Slack-compatible webhook (LOFI_ALERT_WEBHOOK). Run it from
cron / GitHub Actions:

    python -m predict.watchlist            # watchlist names only
    python -m predict.watchlist --all      # whole unbooked pool

Watchlist lives in predict/data/watchlist.json (gitignored — venue-private).
Detection is pure and injectable; alerts carry the reason in plain language.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DATA = Path(__file__).parent / "data"
_WATCHLIST = _DATA / "watchlist.json"

DEFAULT_THRESHOLDS = {
    "momentum": 75.0,        # crossing this = traction right now
    "growth": 70.0,          # strong acceleration signal
    "forecast_90d": 25.0,    # XGBoost sees a jump coming
    "momentum_delta": 15.0,  # jump vs the previous snapshot
}


# ── watchlist persistence ─────────────────────────────────────────────────────

def load_watchlist(path: Path = _WATCHLIST) -> list[str]:
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("names", []))
    except Exception:
        return []


def save_watchlist(names: list[str], path: Path = _WATCHLIST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"names": sorted(set(names))}, indent=2),
                    encoding="utf-8")


# ── detection (pure) ──────────────────────────────────────────────────────────

def detect_spikes(candidates: list[dict], watch: list[str] | None = None,
                  prev: dict[str, dict] | None = None,
                  thresholds: dict | None = None) -> list[dict]:
    """One alert per artist that trips a rule. `candidates` are the ranked
    Scout candidates; `watch` limits to watchlist names (None = everyone);
    `prev` is an optional previous snapshot {name: {momentum: ...}} that
    enables the delta rule ('momentum jumped 18 points since last run')."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    watch_set = {w.lower() for w in watch} if watch is not None else None
    alerts = []
    for c in candidates or []:
        name = c.get("artist_name") or ""
        if watch_set is not None and name.lower() not in watch_set:
            continue
        reasons = []
        if (c.get("momentum") or 0) >= th["momentum"]:
            reasons.append(f"momentum {c['momentum']:.0f} ≥ {th['momentum']:.0f}")
        if (c.get("growth") or 0) >= th["growth"]:
            reasons.append(f"growth {c['growth']:.0f} ≥ {th['growth']:.0f}")
        fc = c.get("forecast_90d")
        if fc is not None and fc >= th["forecast_90d"]:
            reasons.append(f"XGBoost forecast +{fc:.0f}% (90d)")
        if prev:
            was = (prev.get(name.lower()) or {}).get("momentum")
            if was is not None and (c.get("momentum") or 0) - was >= th["momentum_delta"]:
                reasons.append(
                    f"momentum jumped {c['momentum'] - was:+.0f} since last run")
        if reasons:
            alerts.append({
                "artist_name": name,
                "artist_id": c.get("artist_id"),
                "reasons": reasons,
                "momentum": c.get("momentum"),
                "growth": c.get("growth"),
                "forecast_90d": fc,
                "genres": (c.get("genres") or [])[:3],
            })
    alerts.sort(key=lambda a: (a.get("momentum") or 0), reverse=True)
    return alerts


def snapshot(candidates: list[dict]) -> dict[str, dict]:
    """The compact state the next run diffs against (keyed by lowercase name)."""
    return {(c.get("artist_name") or "").lower():
            {"momentum": c.get("momentum"), "growth": c.get("growth")}
            for c in candidates or []}


# ── delivery ──────────────────────────────────────────────────────────────────

def format_alerts(alerts: list[dict]) -> str:
    lines = ["⚡ LOFI Scout — watchlist alerts"]
    for a in alerts:
        genres = ", ".join(a.get("genres") or [])
        lines.append(f"• *{a['artist_name']}*"
                     + (f" ({genres})" if genres else "")
                     + " — " + "; ".join(a["reasons"]))
    return "\n".join(lines)


def push_webhook(alerts: list[dict], url: str | None = None) -> bool:
    """POST to a Slack-compatible webhook. No webhook configured or no alerts
    → quietly False (the CLI prints either way)."""
    import os
    url = url or os.environ.get("LOFI_ALERT_WEBHOOK", "")
    if not url or not alerts:
        return False
    import httpx
    try:
        httpx.post(url, json={"text": format_alerts(alerts)}, timeout=15)
        return True
    except Exception:
        return False


def _main(argv: list[str]) -> int:
    from scout.data import load_flat_profiles, load_ml_features, make_client
    from scout.ranking import build_candidates, load_predictions

    client = make_client()
    candidates = build_candidates(load_flat_profiles(client),
                                  load_ml_features(client), load_predictions())
    watch = None if "--all" in argv else load_watchlist()
    if watch == []:
        print("Watchlist is empty (predict/data/watchlist.json) — "
              "use --all for the whole pool.")
        return 0

    prev_path = _DATA / "watch_snapshot.json"
    prev = {}
    try:
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    alerts = detect_spikes(candidates, watch=watch, prev=prev)
    prev_path.parent.mkdir(parents=True, exist_ok=True)
    prev_path.write_text(json.dumps(snapshot(candidates)), encoding="utf-8")

    if not alerts:
        print("No spikes.")
        return 0
    print(format_alerts(alerts))
    if push_webhook(alerts):
        print("→ pushed to LOFI_ALERT_WEBHOOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

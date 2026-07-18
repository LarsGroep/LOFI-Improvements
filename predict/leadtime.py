"""
Detection lead time — does the OS actually find gems earlier?

docs/agentic_os.md §8: the whole promise of the agentic layer is *earliness*.
So we measure it the same honest way predict/backtest.py measures the forecast:
for every artist the OS promoted to `candidate`, find the day their Spotify
listeners first crossed an "established" bar (75k by default) — the moment the
mainstream would have noticed — and count the days between the promotion and
that crossing. **The headline metric is median days earlier.** If the OS keeps
promoting artists only *after* they've already crossed, the lead time is zero or
negative and this report says so; no fake head-start.

Its sibling is verdict calibration: of the book_now/act_fast verdicts old enough
to have an outcome (≥90 days), how many actually got booked? Precision and
recall on the positive class, maturity-gated exactly like the forecast report.

Both read the dated listener snapshots the OS has accrued since Phase A
(predict/data/forecast_log.csv) plus the blackboard's own promotion and verdict
history. Until enough has matured, each report says exactly that.

    python -m predict.leadtime        # print both reports
"""
from __future__ import annotations

import datetime as _dt
import os
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_ESTABLISHED = 75000.0
POSITIVE_VERDICTS = {"book_now", "act_fast"}
VERDICT_MATURE_DAYS = 90


def established_bar() -> float:
    try:
        return float(os.environ.get("LOFI_ESTABLISHED_LISTENERS")
                     or DEFAULT_ESTABLISHED)
    except ValueError:
        return DEFAULT_ESTABLISHED


def _as_date(value) -> _dt.date | None:
    if not value:
        return None
    try:
        return _dt.datetime.fromisoformat(
            str(value).replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return _dt.date.fromisoformat(str(value)[:10])
        except ValueError:
            return None


# ── detection lead time (pure) ────────────────────────────────────────────────

def _crossing_dates(listener_history: list[dict], bar: float) -> dict:
    """artist_id → the first logged date its listeners reached `bar` having been
    logged below it earlier. Artists never logged below the bar don't count —
    we can only claim a head-start we actually watched happen."""
    by_artist: dict[str, list[tuple[_dt.date, float]]] = {}
    for r in listener_history or []:
        aid = r.get("artist_id")
        d = _as_date(r.get("date"))
        try:
            v = float(r.get("spotify_listeners"))
        except (TypeError, ValueError):
            continue
        if not aid or d is None:
            continue
        by_artist.setdefault(aid, []).append((d, v))

    out: dict[str, _dt.date] = {}
    for aid, rows in by_artist.items():
        rows.sort(key=lambda t: t[0])
        below_seen = False
        for d, v in rows:
            if v >= bar and below_seen:
                out[aid] = d
                break
            if v < bar:
                below_seen = True
    return out


def evaluate_leadtime(promotions: list[dict], listener_history: list[dict],
                      established_bar: float, today=None) -> dict:
    """Median/mean days between a promotion and the artist's crossing of the
    established bar, over artists promoted BEFORE they crossed. `promotions` =
    [{"artist_id","artist_name","promoted_at"}]; `listener_history` = rows of
    predict/data/forecast_log.csv. Honest empty note when nothing has matured."""
    bar = float(established_bar)
    crossings = _crossing_dates(listener_history, bar)
    n_promoted = len(promotions or [])

    per_artist, leads = [], []
    for p in promotions or []:
        aid = p.get("artist_id")
        promoted = _as_date(p.get("promoted_at"))
        crossed = crossings.get(aid)
        if promoted is None or crossed is None or promoted >= crossed:
            continue
        lead = (crossed - promoted).days
        leads.append(lead)
        per_artist.append({
            "artist_id": aid,
            "artist_name": p.get("artist_name") or aid,
            "promoted_at": promoted.isoformat(),
            "crossed_at": crossed.isoformat(),
            "lead_days": lead,
        })

    if not leads:
        return {"n_promoted": n_promoted, "n_crossed": 0,
                "established_bar": bar,
                "note": (f"no promoted artist has crossed the {bar:,.0f}-listener "
                         "bar after promotion yet — lead time fills itself in as "
                         "the OS keeps logging listener snapshots")}

    per_artist.sort(key=lambda a: a["lead_days"], reverse=True)
    return {
        "n_promoted": n_promoted,
        "n_crossed": len(leads),
        "established_bar": bar,
        "median_lead_days": round(statistics.median(leads), 1),
        "mean_lead_days": round(statistics.fmean(leads), 1),
        "per_artist": per_artist,
    }


# ── verdict calibration (pure) ────────────────────────────────────────────────

def _verdict_value(payload: dict) -> str | None:
    v = (payload or {}).get("verdict") or (payload or {}).get("window")
    return str(v).strip().lower() if v else None


def evaluate_verdicts(verdicts: list[dict], outcomes: dict[str, bool],
                      min_age_days: int = VERDICT_MATURE_DAYS,
                      today=None) -> dict:
    """book_now/act_fast verdicts vs realised bookings, maturity-gated at
    `min_age_days`. `verdicts` = [{"artist_name","verdict","created_at"}];
    `outcomes` = artist_name → booked bool. Precision/recall on the positive
    class + a per-bucket booking breakdown, honest empty note otherwise."""
    today_d = _as_date(today) or _dt.date.today()
    outcomes = {str(k).strip().lower(): bool(v) for k, v in (outcomes or {}).items()}

    tp = fp = fn = tn = 0
    buckets: dict[str, dict] = {}
    n_mature = 0
    for v in verdicts or []:
        created = _as_date(v.get("created_at"))
        if created is None or (today_d - created).days < min_age_days:
            continue
        verdict = _verdict_value(v)
        name = (v.get("artist_name") or "").strip().lower()
        if not verdict or name not in outcomes:
            continue                        # can't score without a known outcome
        n_mature += 1
        booked = outcomes[name]
        b = buckets.setdefault(verdict, {"n": 0, "booked": 0})
        b["n"] += 1
        b["booked"] += int(booked)
        predicted_pos = verdict in POSITIVE_VERDICTS
        if predicted_pos and booked:
            tp += 1
        elif predicted_pos and not booked:
            fp += 1
        elif not predicted_pos and booked:
            fn += 1
        else:
            tn += 1

    if not n_mature:
        return {"n_mature": 0,
                "note": (f"no verdicts are ≥ {min_age_days} days old with a known "
                         "booking outcome yet — calibration fills in as the "
                         "chamber's verdicts mature")}
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "n_mature": n_mature,
        "positive_class": sorted(POSITIVE_VERDICTS),
        "precision_pct": None if precision is None else round(precision * 100, 1),
        "recall_pct": None if recall is None else round(recall * 100, 1),
        "n_positive_verdicts": tp + fp,
        "buckets": buckets,
    }


# ── CLI shell ─────────────────────────────────────────────────────────────────

def _load_from_blackboard():
    """(promotions, verdicts) from the blackboard, read-only. Returns
    (None, None) with a note when it isn't reachable."""
    try:
        from osk.blackboard import open_blackboard
        bb = open_blackboard()
    except Exception:
        return None, None, "OS blackboard not reachable"
    try:
        cands = bb.read(kinds=["candidate"], limit=5000)
        vers = bb.read(kinds=["verdict"], limit=5000)
    except Exception:
        return None, None, "OS blackboard not reachable"
    finally:
        try:
            bb.close()
        except Exception:
            pass
    promotions = [{"artist_id": c.artist_id, "artist_name": c.artist_name,
                   "promoted_at": c.created_at} for c in cands]
    verdicts = [{"artist_name": v.artist_name,
                 "verdict": _verdict_value(v.payload),
                 "created_at": v.created_at} for v in vers]
    return promotions, verdicts, None


def _load_outcomes() -> dict[str, bool] | None:
    """artist_name → lofi_booked from Supabase, or None when not configured."""
    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY")):
        return None
    try:
        from scout.data import load_flat_profiles, make_client
        return {p.get("artist_name", "").strip().lower(): bool(p.get("lofi_booked"))
                for p in load_flat_profiles(make_client())
                if p.get("artist_name")}
    except Exception:
        return None


def _main() -> int:
    from predict.backtest import read_log

    promotions, verdicts, note = _load_from_blackboard()
    if note:
        print(note)
        return 1

    bar = established_bar()
    lead = evaluate_leadtime(promotions, read_log(), bar)
    print("── Detection lead time ──")
    if lead.get("n_crossed"):
        print(f"median {lead['median_lead_days']} days earlier "
              f"(mean {lead['mean_lead_days']}) over {lead['n_crossed']} "
              f"of {lead['n_promoted']} promotions, bar={bar:,.0f}")
    else:
        print(lead["note"])

    outcomes = _load_outcomes()
    print("── Verdict calibration ──")
    if outcomes is None:
        print("Supabase not configured — no booking outcomes to calibrate "
              f"verdicts against ({len(verdicts)} verdict record(s) on file)")
    else:
        rep = evaluate_verdicts(verdicts, outcomes)
        print(rep.get("note") or
              f"precision {rep['precision_pct']}% / recall {rep['recall_pct']}% "
              f"over {rep['n_mature']} mature verdicts")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

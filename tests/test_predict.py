"""Unit tests for the predict/ layer — pure functions, synthetic data only
(no Supabase / Airtable / network). Run: python -m pytest tests/ -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from predict import backtest, draw, fees, rank_weights, routing, slots, twins, window
from predict.watchlist import detect_spikes, snapshot


# ── draw ──────────────────────────────────────────────────────────────────────

def test_quantile_interpolates():
    assert draw.quantile([100, 200, 300, 400, 500], 0.5) == 300
    assert draw.quantile([100, 200], 0.5) == 150
    assert draw.quantile([], 0.5) is None


def test_draw_own_history_dominates_with_enough_events():
    est = draw.estimate_draw([300, 320, 340, 310], [100, 110, 120])
    assert est["method"] == "own_history+comparables"
    # 4 own events → weight 4/7 ≈ 0.57 toward own numbers
    assert est["base"] > 200


def test_draw_comparables_only():
    est = draw.estimate_draw([], [100, 150, 200, 250, 300])
    assert est["method"] == "comparables"
    assert est["conservative"] < est["base"] < est["high"]


def test_draw_insufficient_never_guesses():
    est = draw.estimate_draw([], [100, 150])   # < 3 comparables, no own
    assert est["method"] == "insufficient"
    assert est["base"] is None


def test_draw_backtest_reports_coverage():
    events = [{"artist_name": f"a{i % 5}", "genre": "tech house",
               "actual_tickets": 200 + (i * 17) % 120} for i in range(30)]
    rep = draw.backtest(events)
    assert rep["n_events"] == 30
    assert 0 <= rep["interval_coverage_pct"] <= 100
    assert rep["median_ape_pct"] is not None


# ── fees ──────────────────────────────────────────────────────────────────────

def test_fee_own_gages_are_the_anchor():
    est = fees.estimate_fee([2000, 2500, 3000], [500, 600])
    assert est["method"] == "own_gages"
    assert est["low"] <= est["base"] <= est["high"]
    assert est["base"] == 2500


def test_fee_draw_adjustment_is_sublinear():
    plain = fees.estimate_fee([], [1000, 1200, 1400, 1600])
    doubled = fees.estimate_fee([], [1000, 1200, 1400, 1600], draw_ratio=2.0)
    assert doubled["method"] == "comparables_draw_adjusted"
    assert plain["base"] < doubled["base"] < plain["base"] * 2  # 2x draw ≠ 2x fee


def test_fee_insufficient():
    assert fees.estimate_fee([], [800])["method"] == "insufficient"


def test_quote_check_flags_overpriced():
    est = {"low": 2000, "base": 2500, "high": 3000}
    assert fees.quote_check(2800, est)["verdict"] == "fair"
    assert fees.quote_check(4000, est)["verdict"] == "above_range"
    assert fees.quote_check(9000, est)["verdict"] == "well_above_range"
    assert fees.quote_check(9000, {})["verdict"] == "no_model"


def test_booking_margin_scenarios():
    m = fees.booking_margin({"low": 1000, "base": 1500, "high": 2000},
                            {"conservative": 200, "base": 300, "high": 400},
                            ticket_price=20.0, variable_cost_pct=0.25)
    assert m["method"] == "modelled"
    # base: 300 * 15 net - 1500 = 3000
    assert m["base"] == 3000
    assert m["conservative"] < m["base"] < m["high"]


# ── window ────────────────────────────────────────────────────────────────────

def test_window_growth_prices_the_wait():
    ml = {"sp_listeners_30d_pct": 20.0, "sp_listeners_accel": 5.0}
    out = window.booking_window(50_000, ml, fee_base=2000.0)
    assert out["verdict"] == "book_now"
    assert out["horizons"]["6m"]["wait_cost"] > 0


def test_window_flat_trajectory_no_rush():
    out = window.booking_window(50_000, {"sp_listeners_30d_pct": -2.0},
                                fee_base=2000.0)
    assert out["verdict"] == "no_rush"


def test_window_capacity_crossing_says_act_fast():
    ml = {"sp_listeners_30d_pct": 40.0}
    out = window.booking_window(50_000, ml, fee_base=2000.0,
                                draw_base=450.0, capacity=500.0)
    assert out["verdict"] == "act_fast"


def test_window_insufficient_without_listeners():
    assert window.booking_window(None, {}, 1000.0)["method"] == "insufficient"


# ── twins ─────────────────────────────────────────────────────────────────────

def _vec(listeners, g30, g90, g180, accel=0.0, cpp=0.0):
    return twins.trajectory_vector(
        {"spotify_listeners": listeners},
        {"sp_listeners_30d_pct": g30, "sp_listeners_90d_pct": g90,
         "sp_listeners_180d_pct": g180, "sp_listeners_accel": accel,
         "cpp_score_30d_pct": cpp})


def test_twins_matches_shape_not_size_alone():
    target = _vec(50_000, 30, 80, 150)          # fast riser
    pool = [
        {"name": "fast_riser_twin", "vector": _vec(60_000, 28, 75, 140),
         "outcome": {"avg_tickets": 320}},
        {"name": "flat_giant", "vector": _vec(2_000_000, 1, 2, 4),
         "outcome": {"avg_tickets": 600}},
    ]
    out = twins.find_twins(target, pool, k=2)
    assert out[0]["name"] == "fast_riser_twin"
    assert out[0]["similarity"] > out[1]["similarity"]


def test_twins_needs_three_shared_dimensions():
    target = _vec(50_000, 30, None, None)
    sparse = {"name": "sparse", "vector": _vec(None, None, None, None),
              "outcome": {}}
    assert twins.find_twins(target, [sparse], k=5) == []


# ── slots ─────────────────────────────────────────────────────────────────────

def test_slot_ladder():
    assert slots.slot_fit({"conservative": 250, "base": 300, "high": 380},
                          cap=500)["slot"] == "co_headliner"
    assert slots.slot_fit({"conservative": 100, "base": 150, "high": 200},
                          cap=500)["slot"] == "support"
    assert slots.slot_fit({"conservative": 30, "base": 60, "high": 90},
                          cap=500)["slot"] == "opener"
    assert slots.slot_fit({"conservative": 700, "base": 900, "high": 1100},
                          cap=500)["slot"] == "too_big"
    assert slots.slot_fit({"base": None}, cap=500)["method"] == "insufficient"


def test_slot_headliner():
    assert slots.slot_fit({"conservative": 350, "base": 400, "high": 500},
                          cap=500)["slot"] == "headliner"


# ── watchlist ─────────────────────────────────────────────────────────────────

_CANDS = [
    {"artist_name": "Spiker", "artist_id": "1", "momentum": 82.0,
     "growth": 60.0, "forecast_90d": 10.0, "genres": ["tech house"]},
    {"artist_name": "Quiet", "artist_id": "2", "momentum": 40.0,
     "growth": 35.0, "forecast_90d": 2.0, "genres": ["house"]},
]


def test_spike_detection_thresholds():
    alerts = detect_spikes(_CANDS)
    assert [a["artist_name"] for a in alerts] == ["Spiker"]
    assert any("momentum" in r for r in alerts[0]["reasons"])


def test_spike_detection_watch_filter():
    assert detect_spikes(_CANDS, watch=["quiet"]) == []


def test_spike_detection_delta_rule():
    prev = {"quiet": {"momentum": 20.0, "growth": 30.0}}
    alerts = detect_spikes(_CANDS, prev=prev)
    assert any(a["artist_name"] == "Quiet" for a in alerts)


def test_snapshot_roundtrip():
    snap = snapshot(_CANDS)
    assert snap["spiker"]["momentum"] == 82.0


# ── backtest ──────────────────────────────────────────────────────────────────

def test_backtest_scores_only_mature_rows():
    logged = [
        {"date": "2026-01-01", "artist_id": "a", "artist_name": "A",
         "spotify_listeners": "10000", "forecast_90d": "50.0"},   # mature
        {"date": "2026-06-20", "artist_id": "b", "artist_name": "B",
         "spotify_listeners": "10000", "forecast_90d": "50.0"},   # too young
    ]
    rep = backtest.evaluate(logged, {"a": 14_000, "b": 20_000},
                            today="2026-07-01")
    assert rep["n_mature"] == 1
    assert rep["direction_hit_rate_pct"] == 100.0   # predicted up, went up
    assert rep["bias_points"] == 10.0               # predicted +50, realized +40


def test_backtest_empty_is_honest():
    assert backtest.evaluate([], {}, today="2026-07-01")["n_mature"] == 0


def test_log_forecasts_idempotent(tmp_path):
    p = tmp_path / "log.csv"
    cands = [{"artist_id": "a", "artist_name": "A", "spotify_listeners": 1000,
              "forecast_90d": 12.0}]
    assert backtest.log_forecasts(cands, path=p, today="2026-07-01") == 1
    assert backtest.log_forecasts(cands, path=p, today="2026-07-01") == 0
    assert len(backtest.read_log(p)) == 1


# ── rank weights ──────────────────────────────────────────────────────────────

def test_fit_weights_needs_labels():
    rows = [{"future_potential": 50, "growth": 50, "momentum": 50,
             "forecast_90d": 0, "lofi_booked": i < 3} for i in range(50)]
    assert rank_weights.fit_weights(rows) is None   # 3 < MIN_LABELS


def test_fit_weights_learns_the_separating_signal():
    rows = []
    for i in range(120):
        booked = i < 40
        rows.append({
            "future_potential": 80.0 if booked else 30.0,  # separates perfectly
            "growth": 50.0 + (i % 7),                       # noise
            "momentum": 50.0 - (i % 5),                     # noise
            "forecast_90d": 0.0,
            "lofi_booked": booked,
        })
    w = rank_weights.fit_weights(rows)
    assert w is not None
    assert abs(sum(w.values()) - 1.0) < 0.02
    assert w["future_potential"] == max(w.values())
    assert all(v >= 0.05 for v in w.values())       # floor keeps every signal


def test_learned_weights_roundtrip(tmp_path):
    p = tmp_path / "w.json"
    w = {"future_potential": 0.4, "growth": 0.3,
         "forecast_norm": 0.2, "momentum": 0.1}
    rank_weights.save_weights(w, path=p, n_labels=20)
    assert rank_weights.learned_weights(p) == w
    assert rank_weights.learned_weights(tmp_path / "missing.json") is None


# ── routing ───────────────────────────────────────────────────────────────────

def test_routing_signal_counts_eu_shows():
    events = [
        {"datetime": "2026-08-01T22:00:00",
         "venue": {"country": "Germany"}},
        {"datetime": "2026-08-15T23:00:00",
         "venue": {"country": "Netherlands"}},
        {"datetime": "2027-05-01T22:00:00",           # outside window
         "venue": {"country": "France"}},
        {"datetime": "2026-08-20T22:00:00",
         "venue": {"country": "United States"}},
    ]
    sig = routing.routing_signal("X", window_days=120, events=events)
    assert sig["eu_shows"] == 2
    assert sig["nl_shows"] == 1
    assert sig["next_eu_date"] == "2026-08-01"


def test_routing_unconfigured_is_graceful():
    assert routing.routing_signal("X", events=[])["method"] == "unavailable"

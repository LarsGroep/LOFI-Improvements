"""
LOFI Scout prediction layer — trained/statistical models UNDER the LLM.

The FEEDBACK.md thesis, implemented: the LLM explains numbers, it does not
invent them. Every module here is framework-agnostic (no Streamlit), pure
where possible (injectable data for tests), and graceful-empty: missing data
sources produce `method: "insufficient"`, never a crash and never a guess.

Modules
  draw.py          calibrated ticket-draw quantiles + leave-one-out backtest
  fees.py          draw-adjusted comparable fee model + quote check + margin
  window.py        booking-window economics: the euro cost of waiting
  twins.py         trajectory twins — path-alike booked artists with outcomes
  slots.py         slot-level fit (headliner / support / too big) vs capacity
  watchlist.py     momentum spike detection + webhook push
  backtest.py      forecast logging + calibration report (hit rate, MAE, bias)
  rank_weights.py  learn the Scout-score weights from booking outcomes
  routing.py       EU tour-routing signal (Bandsintown, optional)
  deal.py          the deal sheet — one dict combining all of the above
"""

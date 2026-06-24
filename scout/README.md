# Scout — Phase 1 (deterministic, no LLM)

The compliant, data-driven core of the booking agent. It ranks **unbooked,
on-feel** artists by growth + potential + the XGBoost forecast and explains each
pick in plain Dutch — **no external LLM, nothing leaves the dashboard**, so it
needs no permission under the Verwerkersovereenkomst.

## Modules
- `data.py` — Supabase read layer (framework-agnostic; paginated).
- `ranking.py` — reuses `scoring.five_scores` + `ml/models/predictions.csv` +
  the LOFI-feel taxonomy; deterministic ranking + rule-based Dutch rationale.
- `page.py` — `render_scout_page()`: filters, KPIs, a ranked table with visual
  score bars, an artist detail panel, and CSV export.
- `app.py` — standalone runner.

## Run it
```bash
# in the dashboard: pick "Scout" in the sidebar nav (already wired in)
streamlit run lofi_pipeline.py

# or standalone
streamlit run scout/app.py
```
Needs `SUPABASE_URL` and `SUPABASE_KEY` in the environment / `.env`.

## What Phase 2 adds (after Lofi's written permission)
The ranking + filters stay; `explain_nl()` is replaced by Claude-generated
rationales, and the per-artist chat is added — both routed through the single
LLM wrapper with EU residency + no-training (see `docs/agent_design.md`).

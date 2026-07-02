# LOFI Scout

AI booking scout for LOFI — find, rank, and pressure-test unbooked artists for
a booking. Extracted from the main [LOFI](https://github.com/LarsGroep/LOFI)
repo as a standalone tool, with its full git history.

## What it does

- **Deterministic ranking** (`scout/ranking.py`) — every unbooked artist gets a
  Scout score blending the five dashboard scores (momentum, growth, market
  relevance, future potential, data confidence) with the XGBoost 90-day CPP
  growth forecast, filtered through the LOFI-feel genre taxonomy. No LLM, no
  data leaves the machine.
- **AI booking analysis** (`scout/page.py` + `agents/core.py`) — pick a genre,
  pick up to 6 artists, and run a deep Claude-powered analysis grounded in
  LOFI's own booking history and economics, with optional web search for
  reputation/risk context. Follow-up chat included.
- **Per-artist validation** (`scout/validation.py`) — "can they sell tickets at
  LOFI, how many, do they fit?" with a book-now / monitor / too-early /
  not-a-fit verdict and a ticket estimate grounded in LOFI's own events.
- **Head-to-head compare** — 2–3 artists, one AI booking verdict.
- **Per-artist chat** (`scout/chat.py`) — ask anything about a candidate; the
  model sees a minimised context view (Supabase metrics + Airtable booking
  history + comparables), nothing more.

## Layout

| Path | Role |
|---|---|
| `scout/` | The Scout app: ranking, page, validation, chat, Airtable + events data layers |
| `agents/` | The single LLM wrapper (Claude, EU inference, no-training) all AI calls route through |
| `scoring/` | The five-scores engine + LOFI-feel taxonomy (shared lineage with the dashboard) |
| `ml/` | XGBoost growth model: training script, bulk predict, `models/predictions.csv` |
| `docs/agent_design.md` | Design spec: compliance-by-design architecture |

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in SUPABASE_URL / SUPABASE_KEY (+ ANTHROPIC_API_KEY for AI)
streamlit run scout/app.py
```

Without `LOFI_LLM_ENABLED=1` and an Anthropic key the app runs in **preview
mode**: rankings and filters work fully, AI buttons return mock output.

### Data expectations

- `ml/models/predictions.csv` — bundled; regenerate with
  `python ml/train_growth_model.py` + `python ml/bulk_predict_to_supabase.py`.
- `lineup_recommender/data/lofi/artists+events_clean.zip` — **not** in the repo
  (NDA); drop it in locally to ground ticket estimates in real event history.
  Everything degrades gracefully without it.
- Airtable booking economics need `AIRTABLE_TOKEN` + `AIRTABLE_BASE_ID`
  (read-only scope).

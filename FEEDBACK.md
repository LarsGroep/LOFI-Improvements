# Field notes — a week of scouting with LOFI Scout

*Written from the chair of a talent scout / booker for a 350–700 cap club,
using the tool as it ships in this repo.*

## The short version

As a **shortlist machine and due-diligence assistant, I'm satisfied** — this is
already better than how most clubs scout (group chats, gut feel, and whoever
the agents push this month). The killer feature is that every AI verdict is
grounded in *our own* ticketing history and gages, not generic hype.

As a **booking co-pilot, not yet.** The tool tells me *who* looks good. It
doesn't yet tell me *what they'll cost, what they'll draw (with calibrated
confidence), when to strike, or what the booking is worth in euros*. Those are
the four decisions I actually get paid for, and today three of them lean on
LLM reasoning where a trained model should sit underneath.

## What works — and I'd fight to keep

- **Genre → 6 artists → deep analysis** matches how I actually think ("I need
  a house name for the spring party"), not how databases think.
- **The deterministic backbone.** Rankings never hallucinate; AI runs only
  when I press the button. I trust the table because it's the same five scores
  every day.
- **Grounding in our own events + Airtable gages.** "Toman did X tickets at Y
  fee" is the only comparable that matters. No agency deck can argue with it.
- **The honest failure modes.** "Too early", visible data-confidence, forecast
  downturns shown instead of hidden ("let op: forecast −20%"). A tool that can
  say "don't book" is a tool I can defend to my boss.
- **Producer-led vs DJ-led lens** in validation — that distinction is real:
  streams don't fill rooms for producer-led acts the way club reputation does.

## What I'm missing — in the order it costs me money

### 1. A fee/gage prediction model (the biggest gap)
The Airtable join knows what LOFI paid historically. Nothing *predicts* what an
artist should cost **today**, or flags when an agent's quote is 2× fair value.
Train a regression on gage vs (listeners, CPP, growth stage, territory, agency
tier) → show **predicted fee range next to predicted draw = predicted margin
per booking**. That single number would change how I negotiate every week.

### 2. Calibrated draw prediction as a model, not a Claude estimate
The validation panel's ticket range comes from an LLM reasoning over
comparables. Good provenance, but I need **calibrated intervals**: a small
gradient-boosted model on `artist_events_clean` (features: NL listeners, genre
bucket, day-of-week, slot, season, support vs headline) with a backtested error
bar. Let Claude *explain* the model's number instead of *inventing* its own.

### 3. A booking-window model ("book now" needs a price on waiting)
"Too early / monitor / book now" is qualitative. Fees inflect *after* growth
inflects — the whole game is booking in the gap. Predict **when** an artist's
fee will jump (change-point / survival model on the growth curve) and show:
"booking now vs in 4 months ≈ €X difference, at Y% risk they blow up and
skip our room size entirely."

### 4. Trajectory twins, not similar-artist name lists
The design doc promises *"looks like Toman a year ago — booked for €X, sold
out."* Today's comparables are Last.fm/Chartmetric name lists (sound-alikes,
not path-alikes). Build it for real: nearest-neighbour matching on the
listener/CPP **time-series shape** (12-month window), restricted to artists
LOFI actually booked, surfacing their fee + door outcome. This is the single
most persuasive artefact I could bring into a booking meeting.

### 5. Backtest the forecast, or I can't trust it
The XGBoost 90-day forecast carries 20% of the Scout score, and I have no idea
if "+32%" has ever been right. Ship a monthly **hit-rate report** (predicted vs
realized growth, by genre and career stage) inside the app. If the model is
well-calibrated, say so and I'll lean on it harder; if not, I need to know that
even more.

### 6. Close the loop: outcomes should retrain the ranking
The rank weights (0.35/0.30/0.20/0.15) are hand-tuned. We log feedback and we
know which validated artists got booked and how the door did — that's a
learning-to-rank label. The tool should get **better every season we use it**;
right now it's frozen.

### 7. A watchlist that pings me, not a dashboard I refresh
Scouting is temporal. I want change-point alerts: "⚡ artist X: playlist adds
spiking, crossed your momentum threshold, still in your fee window — 3 similar
past spikes led to a fee jump within 8 weeks." Push it (mail/Slack/WhatsApp);
don't wait for me to open Streamlit on a Tuesday.

### 8. Routing & availability context
An artist already touring the EU in March is a cheaper, warmer booking than a
fly-in. Join tour-date data (RA/Songkick/Bandsintown) and flag routing
opportunities on the shortlist. Cheap feature, real fee leverage.

### 9. Slot-level thinking
"Book" isn't one decision — headliner at 700 cap vs 01:00 support slot are
different bets. The `lineup_recommender` lineage exists; Scout should say
*which slot* an artist is a fit for and simulate the night's lineup synergy.

## Verdict

**7/10 as a scout today, with the ceiling of a 10.** The architecture is right
(deterministic core, on-demand AI, own-data grounding, compliance-first) —
what's missing isn't more LLM, it's **more trained models under the LLM**:
fee, draw, and timing predictions with backtests, a real trajectory-comparable
engine, and a feedback loop so the tool learns from every booking we make.
Items 1–3 alone would move this from "useful second opinion" to "the first
thing I open every morning."

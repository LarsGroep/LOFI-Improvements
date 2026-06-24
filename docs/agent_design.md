# LOFI Agent — Design Spec (v0, design-only)

> Status: **design only, no implementation.** The LLM-powered parts are gated on
> Lofi's prior written permission (Verwerkersovereenkomst Art. 3.4 / 3.5 / Bijlage 2).
> This document is a Project Result and is the property of Lofi (Art. 6).

## 1. Purpose

Turn LOFI's existing artist intelligence into something a **non-technical booking
team** can act on, by combining the **two worlds** that currently live apart:

- **The dashboard** (Supabase): rich, current signal — scraped + Chartmetric
  metrics, the five scores, the XGBoost 90-day growth forecast, the LOFI-feel
  taxonomy.
- **Lofi's Airtable** (later phase, read-only): the ground truth — who Lofi
  actually booked, **what they paid (gages)**, set-times, event outcomes. Sparse,
  but real economics and history.

The agent's value is the **join**: recommendations and answers grounded in Lofi's
own booking history and economics, not just public metrics.

> *Example: "This rising artist looks like Toman a year ago — booked for €X, sold
> out — and is on a steeper Spotify curve. Worth a conversation."*

## 2. Compliance by design (drives the architecture)

The Verwerkersovereenkomst is not an afterthought; it shapes the build.

| Requirement | Article | How the design honours it |
|---|---|---|
| No personal/company data into external AI/LLM without written permission | 3.5, Bijlage 2 | **Permission gate** before any real data reaches Claude; a central LLM wrapper is the *only* egress point; data-minimised "model view" |
| EEA-only processing & storage | 3.7 | Claude configured for **EU data residency**; Supabase **EU region**; no non-EEA stores introduced |
| No new sub-processors without consent | 3.4 | Anthropic named + approved in the permission request; no others added silently |
| Confidentiality (incl. Airtable structure, source) | 5 | Secrets out of code; don't send the Airtable schema/source to any non-approved service |
| IP to Lofi, no copies kept | 6, 10 | Everything lives in Lofi's repo; project-end deletion (14 days) |

**Data minimisation (the practical core of Art. 3.5):** a single `to_model_view()`
layer decides exactly what leaves for the LLM. Default allow-list: artist display
name, genres, the five computed scores, the forecast number, aggregate public
platform metrics. **Gages and other Lofi business data are only included when the
permission explicitly covers it** — that grounding is valuable but it is
confidential business info going to an external service, so it must be named.

## 3. One brain, two surfaces (Airtable read = later phase)

A single **reasoning core** (shared data access, system prompt, tools), exposed
through the surfaces you described.

### Surface A — Scout page (dashboard, full page)
Proactive + interactive discovery of who to book next.

- **Input (context):** unbooked artists with the five scores + XGBoost forecast +
  genres/career status; the LOFI-feel taxonomy (tiers + benchmark artists);
  *(Phase 3)* Lofi Airtable booking history for analogy/grounding; optional user
  filters (genre, gage budget, region).
- **Instructions (system prompt essence):** you are LOFI's scout; surface rising,
  on-feel, not-yet-booked artists; ground reasoning in Lofi's own history and
  economics; explain in plain Dutch for non-technical bookers; never invent
  artists or IDs; respect the data allow-list.
- **Output:** ranked shortlist; per artist → fit score, one-line Dutch rationale,
  a comparable past booking *(Phase 3)*, suggested next action. In-dashboard only;
  any Airtable write-back is a separate, explicitly-instructed step.

### Surface B — Per-artist chat (dashboard, small dock, bottom-left of every artist page)
Context-scoped assistant for the artist currently on screen.

- **Input:** that one artist's full profile (scores, platform metrics, RA/shows,
  similar artists, milestones); *(Phase 3)* their Lofi booking history;
  conversation history; *(Phase 4)* web-search results.
- **Instructions:** answer the booker's questions about **this** artist in Dutch;
  compare to benchmark artists and to Lofi's past bookings; give booking verdicts
  with reasoning; draft short pitches; say which data each claim rests on; say when
  unsure; don't surface other artists' confidential gage data unless relevant.
- **Output:** conversational Dutch answers, comparisons, fit verdicts, draft
  outreach text. **Capabilities for v1: Q&A + comparison + web search. No system
  writes.** (Per your decision.)

**UI note:** Streamlit has no native floating/docked widget. Options for the
bottom-left dock: (a) a small custom component (e.g. a `streamlit-float`-style
container) — closest to the brief; (b) a sidebar chat (`st.sidebar` +
`st.chat_input`) — simplest; (c) a pinned expander. Recommendation: start with (b)
to validate behaviour, move to (a) for the docked look. This is a UI choice, not a
blocker.

## 4. Reasoning core & tools (function-calling design)

The core exposes read-only tools the model can call (Anthropic tool use):

| Tool | Reads | Phase |
|---|---|---|
| `list_candidates(filters)` | scored unbooked artists (Supabase) | 1 |
| `get_artist(artist_id)` | one artist's full profile (Supabase) | 1 |
| `get_booking_history(artist)` | Lofi Airtable rows (read-only) | 3 |
| `compare(a, b)` | two artists side by side | 2 |
| `web_search(query)` | recent public activity (data-minimised query) | 4 |

Every tool result passes through `to_model_view()` before reaching the model.

## 5. Architecture & where it lives

- `agents/core.py` — reasoning core: the **single LLM wrapper** (model
  `claude-opus-4-8`; EU residency + zero-retention + no-train config; tool
  definitions; the `to_model_view()` minimisation layer; no payload logging).
- Shared data-access module — refactor the dashboard's Supabase read helpers
  (currently inside `lofi_pipeline.py`) into a module both the dashboard and the
  agent import, so there's one source of truth.
- Streamlit surfaces — a Scout page; a chat dock component on the artist profile.
- Deterministic fallback — the Scout's ranking works **without** the LLM (Phase 1),
  so the product is useful and demoable before permission lands.

## 6. Phased roadmap

| Phase | What | Compliance status |
|---|---|---|
| **0** | Lofi written permission (Anthropic no-train + EU residency; Supabase EU; read-only Airtable) | **gate** |
| **1** | No-LLM core: shared data layer + deterministic scoring/ranking + Scout page (rule-based) | compliant now |
| **2** | LLM reasoning: rationales, per-artist chat (Q&A + comparison) via the core wrapper | needs Phase 0 |
| **3** | "Both worlds" join: read-only Lofi Airtable history into Scout + chat | needs Phase 0 + Airtable read scope |
| **4** | Web search in the chat (data-minimised queries) | needs Phase 0 (explicitly) |

## 7. Open questions

- **Gage data to the LLM** — include comparable bookings' gages in prompts (high
  value, but confidential)? Needs to be named in the permission, or kept
  dashboard-side only.
- **Per-artist chat UI** — sidebar (fast) vs docked custom component (matches the
  brief). Pick after Phase 1.
- **Web search provider** — which search API is EEA-compatible and acceptable
  under Art. 3.5/3.7; what may appear in a query.
- **Airtable structure** — needs the real schema (read-only token or your
  description) before Phase 3 can be designed concretely.

## 8. Out of scope / risks

- No writes to Lofi systems in v1 (Airtable stays read-only per Bijlage 1).
- No special-category data (Art. 9/10) is processed.
- The whole LLM layer is inert until Phase 0 permission is confirmed in writing.

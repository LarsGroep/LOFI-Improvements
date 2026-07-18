# LOFI Agentic OS — Design Spec (v0)

> Status: **design spec**. Successor to `docs/agent_design.md` ("one brain, two
> surfaces"). That doc designed a single agent; this one designs the **operating
> system** the next generation of the tool runs on: many specialised agents,
> a shared kernel, persistent memory in Supabase, and a deliberation protocol —
> all in service of one job: **detect artists on the rise before the market
> prices them in.**

## 0. Why an OS, not a bigger agent

Everything valuable this repo has shipped follows one pattern
(`FEEDBACK.md`'s verdict): *deterministic models underneath, LLM reasoning on
top, grounded in LOFI's own history*. The current architecture runs that
pattern **on demand** — a booker opens Streamlit, presses a button, one Claude
call happens, the result evaporates when the session ends.

Early detection is the opposite kind of problem. It is **continuous**
(signals spike on a Tuesday night), **multi-perspective** (a hype signal and a
fee signal disagree, and the disagreement *is* the information), and
**cumulative** (this month's verdict should remember last month's). That
demands three things a single request/response agent can't provide:

1. **Persistence** — agents that run on schedules, write their findings to
   Supabase, and pick up where they left off.
2. **Deliberation** — multiple reasoning stances that argue about a candidate
   and produce an auditable verdict, not one prompt's opinion.
3. **Division of labour** — scraping, quantitative analysis, qualitative
   judgement, and economics are different jobs with different tools, costs,
   and failure modes. One mega-prompt does all of them badly.

The OS metaphor is load-bearing:

| OS concept | LOFI Agentic OS equivalent |
|---|---|
| Kernel | Orchestrator: schedules agents, routes messages, enforces the egress/permission layer |
| Processes | Agents (scout, skeptic, economist, curator, sentinel, …) |
| Filesystem | Supabase (`tinder` schema + new `agentic` tables): dossiers, signals, debates, run logs |
| IPC | The blackboard: agents communicate by reading/writing typed records, never by sharing raw prompts |
| Syscalls | The tool registry — the *only* way an agent touches data or the network |
| Ring 0 security | `to_model_view()` + the single LLM wrapper in `agents/core.py`, unchanged as the sole egress point |
| Cron | The scheduler (extends today's `predict.backtest log` / `predict.watchlist` cron jobs) |
| Syslog | `agent_runs` audit table — every run, every tool call, every token spent |

## 1. Design principles (inherited, non-negotiable)

These carry over from `docs/agent_design.md` and the Verwerkersovereenkomst,
and every agent in the OS is bound by them:

- **Single LLM egress.** All model calls route through `agents/core.py`. Agents
  never construct their own Anthropic clients. Mock mode
  (`LOFI_LLM_ENABLED` off) must keep the whole OS runnable and demoable.
- **Data minimisation at the boundary.** Every payload passes a
  `to_model_view()`-family function. New agents get new view builders, not
  exemptions. Gage data stays behind its explicit permission.
- **Deterministic under, LLM over.** Agents *anchor to* `predict/` and
  `scoring/` outputs and explain them; they never invent numbers the models
  can compute. (This is enforced today via `model_estimates` +
  `grounding.ticket_estimate_allowed` in `scout/context.py` — the OS
  generalises that pattern to every agent.)
- **Honest failure modes.** `method="insufficient"` beats a confident guess.
  An agent with no data says so in its output schema.
- **Graceful degradation.** No Airtable → economics agent degrades; no
  Anthropic key → deliberation is mocked; no scraper credentials → that
  listener sleeps. Nothing hard-crashes the OS.
- **Everything auditable.** If it can't be replayed from `agent_runs` +
  the blackboard, it didn't happen.

## 2. The kernel

A small new package, `osk/` (OS kernel), pure Python, no Streamlit:

```
osk/
  orchestrator.py   # run loop: due jobs → agent instances → blackboard writes
  registry.py       # agent + tool registry, capability manifests
  blackboard.py     # typed read/write API over the Supabase agentic tables
  scheduler.py      # cron-style schedules + event triggers (signal-driven wakeups)
  budget.py         # per-agent token/cost budgets, backoff, kill switch
  audit.py          # agent_runs writer: inputs-hash, tools called, cost, outcome
```

### 2.1 Agent contract

Every agent is a class with a **manifest** — the kernel refuses to run
anything without one:

```python
class AgentManifest:
    name: str                 # "skeptic", "playlist_listener", ...
    tools: list[str]          # allow-list; kernel enforces, not the prompt
    reads: list[str]          # blackboard record types it may read
    writes: list[str]         # record types it may emit
    llm: bool                 # False → pure-Python agent, no egress possible
    view_builder: str | None  # the to_model_view()-family fn if llm=True
    budget: Budget            # max tokens / calls per run and per day
    schedule: str | None      # cron expr, or None = event-triggered only
```

Two agent species share this contract:

- **Pure-Python agents** (`llm=False`): scrapers, detectors, aggregators.
  Cheap, run often, can never leak data to an external model *by
  construction* — the kernel doesn't hand them the LLM wrapper at all.
- **Reasoning agents** (`llm=True`): run rarely, on curated minimised views,
  under budget. These are the only token spenders.

This split is the compliance story *and* the cost story: the OS can watch
2,000 artists nightly for ~zero LLM cost, and spend reasoning tokens only on
the ~10 that tripped a detector.

### 2.2 The blackboard (IPC)

Agents never call each other. They emit **typed records** to Supabase and
subscribe to record types. This gives replayability, decoupling (a new agent
just subscribes), and a natural UI (the console renders the blackboard).

Core record types:

| Record | Emitted by | Consumed by | Payload essence |
|---|---|---|---|
| `signal` | listeners, detectors | sentinel, orchestrator | artist_id, source, kind, magnitude, evidence URL/snapshot |
| `candidate` | sentinel | deliberation chamber | artist_id, trigger signals, priority score |
| `brief` | analyst agents | deliberation chamber | minimised evidence pack (quant + scraped + economics) |
| `argument` | advocate/skeptic | judge | stance, claims[], each claim → evidence pointer |
| `verdict` | judge | curator, sentinel, UI | book_now / act_fast / monitor / too_early / not_a_fit + confidence + dissent summary |
| `dossier_update` | curator | UI, future runs | diff to the living artist dossier |
| `alert` | sentinel | webhook (existing `LOFI_ALERT_WEBHOOK`) | human-facing push |

### 2.3 Scheduler & triggers

Two wake-up modes, mirroring how scouting actually works:

- **Cron** — nightly listener sweeps, weekly full re-rank, monthly backtest
  report (absorbs today's four cron jobs from the README).
- **Event** — a `signal` above threshold wakes the sentinel; a `candidate`
  wakes the deliberation chamber; a `verdict` of book_now/act_fast fires an
  `alert`. Event chains are depth-limited (default 3) so the OS can't
  self-amplify into a token fire.

## 3. The agent roster

### 3.1 Listeners (pure-Python scrapers — "ears on the scene")

The current pipeline sees an artist once they're in Chartmetric/Supabase.
Hidden gems, by definition, spike **before or beside** that. Each listener is
a thin adapter with the same shape: `fetch() → normalise() → emit signals +
cache raw snapshot`. All listeners respect robots.txt/ToS, use official APIs
where they exist, and store snapshots in `scrape_cache` so re-analysis never
re-fetches.

| Listener | Source | The early signal it captures | Notes |
|---|---|---|---|
| `chartmetric_listener` | existing Supabase ingest | baseline metrics, five scores input | already exists; wraps `scout/data.py` |
| `ra_listener` | Resident Advisor events | lineup adjacency: *who opens for whom*, venue-tier climbing | extends the RA data already in `scout/context.py` |
| `bandsintown_listener` | Bandsintown API | EU/NL routing windows | exists as `predict/routing.py`; becomes a listener emitting signals |
| `soundcloud_listener` | SoundCloud API/pages | plays/reposts velocity in the first 72h of a release — the classic pre-Spotify heat signal | new |
| `playlist_listener` | Spotify playlists (editorial + tastemaker) | first adds to scene-defining playlists; add→remove churn | new; official API |
| `label_radar` | label Bandcamp/IG/press pages for a curated label list | *signing announcements* — a strong label A&R decision is a free expert vote | new; the label list is itself LOFI-taxonomy-curated |
| `partyflock_listener` | Partyflock | NL fanbase growth (`pf_fans` delta) | data already flows in; becomes delta-aware |
| `press_listener` | RSS of scene press (RA news, Mixmag, DJ Mag, 3voor12) | first-coverage events, "ones to watch" lists | new |
| `youtube_listener` | YouTube Data API | Boiler Room / HÖR / venue-channel set uploads + view velocity | a filmed set at a taste-making channel is a career change-point |
| `shazam_charts_listener` | city-level charts where available | offline discovery signal, NL cities | optional |

Every raw metric a listener emits becomes a **first-class time series** in
`signals` — which is exactly the input the divergence detectors (§5) and
`predict/twins.py`-style shape matching need. Snapshots accruing is the whole
game; the OS makes it automatic instead of "still open: enabling the cron
jobs" (FEEDBACK.md, closing note).

### 3.2 Analysts (mostly pure-Python, thin LLM where noted)

| Agent | Job | Built on |
|---|---|---|
| `quant_analyst` | five scores, rank score, forecast — refreshed, with deltas vs last run | `scoring/`, `scout/ranking.py`, `ml/` |
| `economist` | deal sheet: draw quantiles, fee range, margin, booking window, slot fit | `predict/deal.build_deal_sheet` verbatim |
| `scene_cartographer` | graph of artists ↔ labels ↔ lineups ↔ B2Bs from listener data; computes *scene centrality* and *rising-cluster membership* | new; pure graph code |
| `dossier_curator` | maintains the living per-artist dossier (LLM-written, evidence-linked, diffed) | `scout/context.build_artist_view` as its view builder |

### 3.3 The Deliberation Chamber (the "reason amongst agents" core)

The user-facing ask — *"reason amongst agents about possible artist talent"* —
is formalised as a **structured adversarial review**, because a debate with
assigned stances surfaces the failure modes a single prompt smooths over.
Three reasoning roles, one shared evidence pack, strict grounding:

- **The Advocate** — builds the strongest *bull case*: momentum, scene
  position, trajectory-twin precedents (`predict/twins.py`), why LOFI
  specifically wins by moving early. May call `web_search` (Phase-4-gated)
  for reputation/context.
- **The Skeptic** — paid to kill the deal: producer-led vs DJ-led drawing
  power (the distinction FEEDBACK.md says is real), playlist-inflated
  numbers, no NL footprint, agency-hype patterns, forecast downside,
  data-confidence holes. Must attack the Advocate's *specific claims*, not
  produce a generic risk list.
- **The Judge** — does not add new evidence. Weighs both arguments **against
  the deterministic layer** (`model_estimates` is binding: the Judge may not
  overrule a calibrated draw quantile, only contextualise it), issues the
  verdict record, and — crucially — writes the **dissent summary**: what
  would change the verdict, and which future `signal` should trigger a
  re-hearing. That line becomes a standing sentinel watch.

Protocol (fixed, cheap, auditable):

```
brief (analysts, no LLM) → advocate argument → skeptic rebuttal
  → [optional 1 reply each iff Judge requests] → verdict + dissent
```

- Hard cap: 5 LLM calls per deliberation, one shared evidence pack, claims
  must carry evidence pointers into the brief (a claim without a pointer is
  struck — same philosophy as `grounding.ticket_estimate_allowed`).
- Every turn is an `argument` record: the full debate is replayable in the UI
  and *diffable across months* ("the Skeptic's NL-footprint objection from
  March is now resolved — 4 NL shows since").
- Verdict vocabulary matches `predict/window.py` / `scout/validation.py`
  verdicts so debates, deal radar, and validation speak one language.

Head-to-head compare (existing feature) generalises to a **docket**: the
chamber can hear 2–3 candidates jointly for one slot, which is the real
booking decision shape.

### 3.4 The Sentinel (early-warning process)

Extends `predict/watchlist.py` from threshold rules into the OS's standing
attention system:

- Subscribes to all `signal` records; maintains per-artist composite
  **heat** with per-source decay (a press mention decays in weeks, a label
  signing doesn't).
- Promotes to `candidate` when: heat crosses a learned threshold, **or** a
  divergence detector fires (§5), **or** a Judge's dissent condition is met
  (re-hearing).
- Emits `alert` to the existing webhook with the evidence trail attached —
  "ping me, don't make me refresh" (FEEDBACK.md §7), now with *why* built in.

## 4. Supabase as the OS filesystem

New `agentic` schema (keeping the app's `tinder` schema untouched), all
EU-region as required:

```sql
create schema agentic;

create table agentic.agent_runs (      -- syslog
  id uuid primary key default gen_random_uuid(),
  agent text not null, started_at timestamptz, finished_at timestamptz,
  trigger text, input_hash text, tools_used jsonb,
  llm_calls int default 0, tokens_in int, tokens_out int, cost_eur numeric,
  outcome text, error text);

create table agentic.signals (         -- the time-series spine
  id bigint generated always as identity primary key,
  artist_id text, artist_name text,    -- artist_id nullable: pre-Supabase discoveries!
  source text not null, kind text not null,
  value numeric, magnitude numeric,    -- raw + normalised
  evidence jsonb, observed_at timestamptz not null,
  emitted_by text references null);    -- agent name

create table agentic.scrape_cache (
  url_hash text primary key, url text, fetched_at timestamptz,
  status int, body_ref text, etag text);

create table agentic.candidates (
  artist_id text, artist_name text, promoted_at timestamptz,
  priority numeric, trigger_signals bigint[], status text);

create table agentic.debates (
  id uuid primary key, artist_ids text[], docket_slot text,
  brief jsonb, verdict text, confidence numeric,
  dissent jsonb,                       -- {condition, watch_kind, threshold}
  created_at timestamptz);

create table agentic.debate_turns (
  debate_id uuid references agentic.debates,
  turn int, role text, claims jsonb, tokens int);

create table agentic.dossiers (        -- the living artist profile
  artist_id text primary key, artist_name text,
  body_md text,                        -- curator-maintained narrative
  facts jsonb,                         -- structured: scores, deal sheet, heat, scene position
  last_verdict text, updated_at timestamptz, revision int);

create table agentic.dossier_revisions (
  artist_id text, revision int, diff_md text, updated_at timestamptz,
  primary key (artist_id, revision));
```

Notes:

- `signals.artist_id` **nullable** is deliberate: the OS can track a name a
  listener found on a lineup or label announcement *before* the artist exists
  in Chartmetric/Supabase — that pre-database window is where the deepest
  hidden gems live. A resolver job links names → ids when they appear.
- `dossiers` is what "show supabase artist profiles" becomes: not a raw
  table dump but a **curated, versioned profile** — narrative + structured
  facts + verdict history — assembled by the curator from the same minimised
  views the LLM is allowed to see. Revisions make "what changed since I last
  looked" a first-class query.
- RLS: dashboard role reads everything; agents get per-table write grants
  matching their manifest's `writes` list — the manifest is enforced in the
  database too, not just the kernel.

## 5. Hidden-gem detection engines (pure-Python detectors)

The detectors are where "early" actually comes from. All are `llm=False`
processes over `agentic.signals`; each fires `signal(kind="divergence", …)`
that the sentinel weighs. Ordered by expected value:

1. **Cross-platform divergence.** SoundCloud/YouTube/Partyflock heat
   z-score minus Spotify-listener z-score. A big positive gap = underground
   heat the mainstream metric hasn't priced in → the cheap-booking window.
   (The inverse gap = playlist-inflated, feeds the Skeptic.)
2. **Trajectory-twin projection, generalised.** `predict/twins.py` matches
   booked artists; the OS runs the same z-scored shape matching over *all*
   tracked artists against the "pre-breakout year" segments of artists who
   later broke out. Match quality × twin outcome = a prior on breakout.
3. **Scene-graph centrality delta.** From the cartographer's graph: an
   artist whose lineup-adjacency to LOFI-core artists is rising (opening
   slots for benchmark names, signings to core labels, B2Bs) is being pulled
   into the scene's center — often *before* any streaming metric moves.
4. **Venue-tier ladder.** From RA history: median venue capacity trend.
   Climbing 100-cap → 300-cap rooms in two quarters is the single most
   booking-relevant curve there is, and no streaming platform shows it.
5. **First-signal combinatorics.** Rules over signal *co-occurrence* within
   a window: label signing + first editorial playlist + first NL show inside
   90 days ≈ the classic ignition pattern. Rules start hand-written (the
   watchlist pattern), later refit like `predict/rank_weights.py` — from
   outcomes, per season.
6. **Fee-lag arbitrage.** Economist × sentinel: heat rising while
   `predict/fees.py` comparable pricing is still flat = the "book in the
   gap" moment `predict/window.py` prices. This detector *is* the business
   case in one record.

## 6. Surfaces (Streamlit)

The Scout app gains an **OS console** (new pages, existing app shell):

- **Feed** — the blackboard, human-readable, newest first: signals, promoted
  candidates, verdicts, alerts. Filter by artist/agent/kind. This is the
  "first thing I open every morning" FEEDBACK.md asked for.
- **Dossier page** — the Supabase artist profile: narrative + facts panel
  (scores, deal sheet, heat sparkline, scene position) + verdict history +
  full debate transcripts (expandable, per-claim evidence links) + revision
  diff view. Per-artist chat (`scout/chat.py`) docks here unchanged.
- **Chamber** — run a deliberation on demand (today's "pick 6 artists"
  flow), watch the argument→rebuttal→verdict unfold, or replay past debates.
- **Ops** — `agent_runs` health: last run per agent, cost per day, budget
  headroom, error tail, mock/live compliance banner (extends
  `compliance_status()`).

Deterministic ranking, Deal radar, Alerts, and Model health remain exactly
where they are — the OS feeds them, it doesn't replace them.

## 7. Cost & control

- LLM spend is concentrated in deliberations (≤5 calls each) and dossier
  updates (1 call, only on material change — the curator diffs `facts` first
  and skips the LLM when nothing moved).
- `budget.py` enforces per-agent daily caps and a global kill switch
  (`LOFI_OS_PAUSED`); breaching agents are parked, not retried.
- Expected steady state: ~2,000 artists watched nightly for €0 LLM cost;
  ~10–20 deliberations/week ≈ low tens of euros/month at Opus-class pricing,
  less on a smaller model for the Advocate/Skeptic with the Judge kept on
  the strongest model (worth an experiment; the wrapper already takes
  `LOFI_LLM_MODEL` per call site).

## 8. Evaluation — does the OS actually find gems earlier?

Same honesty standard as `predict/backtest.py`:

- **Detection lead time** — for every artist who later crossed an
  "established" bar (listeners, fee tier, venue tier): date the OS first
  promoted them to `candidate` vs date the baseline (plain Scout rank top-N)
  would have surfaced them. The headline metric is *median days earlier*.
- **Verdict calibration** — book_now/act_fast verdicts vs realised outcomes
  (fee drift, draw when booked, breakout), maturity-gated like the forecast
  report.
- **Debate value-add** — quarterly: verdicts where the Judge diverged from
  the raw rank score; were the divergences right? If the chamber never beats
  the ranking, it's theatre — shrink it.
- All three live in Model health, honest empty states included.

## 9. Phased roadmap

| Phase | What ships | Depends on |
|---|---|---|
| **A** | **✅ shipped** — kernel + blackboard + `agentic` schema (`osk/`, `deploy/`); the four existing cron jobs run as the first agents (zero LLM) | nothing — compliant now, mirrors Phase 1 of the original design |
| **B** | **✅ shipped** — listeners (`osk/listeners/`): soundcloud, playlist, label_radar, press + signal time series + scrape cache (`osk/scrape.py`) | API keys per source |
| **C** | **✅ shipped** — detectors (`osk/detectors/`: divergence, ignition, ladder, fee-lag, scene fns) + sentinel (`osk/sentinel.py`) + Agent OS console (`scout/os_console.py`) | A + B accruing ≥ a few weeks of signals |
| **D** | Deliberation chamber + curator narratives + Chamber/Dossier UI | Phase-0 permission (existing gate), `agents/core.py` extended with the three role prompts |
| **E** | Learned promotion thresholds + first detection-lead-time report | a season of Phase C data |

Phases A–C are pure-Python and ship value (feed, signals, detectors) even if
the LLM permission conversation stalls — the same de-risking that made
Phase 1 of the original design work.

## 10. Open questions

- **Scraper ToS envelope.** SoundCloud/YouTube/Spotify official APIs are
  clean; RA and Partyflock scraping terms need a read before those listeners
  leave "cache-only, low-frequency" mode. The `scrape_cache` +
  robots-respecting fetcher is designed so frequency is a config, not a
  rewrite.
- **Pre-database artists.** How much of the pipeline should run on
  name-only entities (no Chartmetric id)? Proposal: signals + dossier stub
  only; no deliberation until resolved to a real profile.
- **Debate model mix.** Advocate/Skeptic on a cheaper model with the Judge
  on the strongest — measure verdict quality before committing.
- **Multi-venue future.** Nothing above is LOFI-specific except the
  taxonomy, the events corpus, and the Airtable join — worth keeping the
  kernel venue-agnostic in case the tool ever serves a second room.
- **Write-backs.** The OS stays read-only toward Lofi systems (Bijlage 1);
  the only outbound side effect remains the alert webhook. Revisit only with
  explicit permission.

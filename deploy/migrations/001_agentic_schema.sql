-- LOFI Agentic OS — blackboard schema (docs/agentic_os.md §4, Phase A subset).
-- Apply once to the Supabase project (SQL editor or supabase db push), then
-- set LOFI_OS_BACKEND=supabase. The SQLite backend mirrors these three tables
-- exactly, so switching backends later is a config change, not a migration
-- of code. Richer per-type tables (debates, dossiers, …) arrive with the
-- phases that write them.

create schema if not exists agentic;

-- the feed: every typed record agents emit (signal / candidate / alert / …)
create table if not exists agentic.records (
  id          bigint generated always as identity primary key,
  kind        text not null,
  artist_id   text,             -- nullable by design: pre-database discoveries
  artist_name text,
  payload     jsonb not null default '{}'::jsonb,
  emitted_by  text not null,
  created_at  timestamptz not null default now()
);
create index if not exists idx_agentic_records_kind
  on agentic.records (kind, created_at desc);
create index if not exists idx_agentic_records_artist
  on agentic.records (artist_id);

-- syslog: one row per agent run, whatever the outcome
create table if not exists agentic.agent_runs (
  id              bigint generated always as identity primary key,
  agent           text not null,
  trigger         text,
  started_at      timestamptz not null default now(),
  finished_at     timestamptz,
  outcome         text,
  error           text,
  llm_calls       int not null default 0,
  tokens_in       int not null default 0,
  tokens_out      int not null default 0,
  records_emitted int not null default 0
);
create index if not exists idx_agentic_runs_agent
  on agentic.agent_runs (agent, started_at desc);

-- per-agent key/value state (namespaced "<agent>:<key>" by the kernel)
create table if not exists agentic.state (
  key        text primary key,
  value      jsonb,
  updated_at timestamptz not null default now()
);

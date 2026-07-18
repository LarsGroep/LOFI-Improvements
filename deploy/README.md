# Deploying LOFI on a VPS

Everything needed to go from a fresh VPS to a running Scout dashboard + the
Agentic OS loop. Two paths — pick one.

## What runs

| Process | Command | Does |
|---|---|---|
| **os** | `python -m osk loop` | The kernel: runs the scheduled agents (forecast logging, watchlist spikes, weekly refits/reports), audits every run, writes the blackboard |
| **scout** | `streamlit run scout/app.py` | The dashboard on port 8501 |

The OS replaces the README's four cron jobs — do **not** also install crontab
entries, or forecasts get double-logged days they race.

## Path 1 — docker compose (recommended)

```bash
# on the VPS
git clone <this repo> lofi && cd lofi
cp .env.example .env && $EDITOR .env     # SUPABASE_URL/KEY minimum
docker compose -f deploy/docker-compose.yml up -d --build

# see it live
docker compose -f deploy/docker-compose.yml logs -f os
docker compose -f deploy/docker-compose.yml exec os python -m osk status
docker compose -f deploy/docker-compose.yml exec os python -m osk feed -n 20
```

State (watchlist, forecast log, learned weights, the SQLite blackboard)
lives on the `predict-data` and `osk-data` volumes and survives
rebuilds/restarts. Upgrade = `git pull && docker compose -f
deploy/docker-compose.yml up -d --build`.

## Path 2 — bare metal + systemd

```bash
sudo useradd -r -m -d /opt/lofi lofi
sudo -u lofi git clone <this repo> /opt/lofi/app
sudo -u lofi python3 -m venv /opt/lofi/venv
sudo -u lofi /opt/lofi/venv/bin/pip install -r /opt/lofi/app/requirements.txt
sudo -u lofi cp /opt/lofi/app/.env.example /opt/lofi/app/.env  # then edit

sudo cp /opt/lofi/app/deploy/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now lofi-os lofi-scout
journalctl -u lofi-os -f
```

## Operating the OS

```bash
python -m osk status          # agents, schedules, last run + outcome
python -m osk once            # single tick (whatever is due), then exit
python -m osk run <agent>     # force one agent now, e.g. watchlist_sentinel
python -m osk feed --kind alert   # what the sentinel found
python -m osk health          # 0/1 — wired into the docker healthcheck
```

- **Pause everything** without redeploying: set `LOFI_OS_PAUSED=1` in `.env`
  and restart the os service (or export it live in the container). The loop
  keeps ticking but runs nothing until unpaused.
- **Watchlist**: names in `predict/data/watchlist.json` (the app's Alerts tab
  edits it too), or `LOFI_OS_WATCH_ALL=1` to sweep the whole unbooked pool.
- **Alerts to Slack/Discord**: set `LOFI_ALERT_WEBHOOK`.

## Blackboard backend

Default is SQLite at `osk/data/os.db` — zero setup, right for a single VPS.
To move the blackboard into Supabase (so the dashboard and future phases
read the same feed):

1. Run `deploy/migrations/001_agentic_schema.sql` against the project
   (Supabase SQL editor).
2. Set `LOFI_OS_BACKEND=supabase` in `.env` and restart the os service.

Both backends have identical tables; nothing else changes.

## Security notes

- The compose file publishes Streamlit on `:8501` — put a reverse proxy with
  TLS and auth (Caddy is the least work) in front before exposing it; the
  systemd unit already binds `127.0.0.1` for exactly that reason.
- `.env` holds all secrets; it is gitignored and read by both services.
- The OS is read-only toward Lofi systems by design; its only outbound side
  effect is the alert webhook. The LLM stays off until `LOFI_LLM_ENABLED=1`
  (the compliance gate in `agents/core.py` — unchanged by any of this).

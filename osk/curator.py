"""
The Dossier Curator — the living, diffed per-artist dossier (docs/agentic_os.md
§3.2). For every artist with recent activity it recomputes a compact **facts**
snapshot, compares it to the last one it stored, and only writes a new
`dossier_update` when the facts moved *materially*. That gate keeps the feed (and
the LLM bill) quiet: an artist whose numbers are flat costs nothing and produces
no record.

Facts are pure Python. The narrative paragraph is the only LLM touch, and it runs
ONLY when agents.core.is_live() — in preview mode the payload carries
`"narrative": None` and the facts + plain-English diff are the value, with no
fabricated prose. The single egress point stays agents/core (via
core.complete()); a pure-Python agent could never reach it.
"""
from __future__ import annotations

import datetime as _dt
import json

from osk.detectors._common import artist_key, signal_views
from osk.manifest import AgentManifest, Budget
from osk.sentinel import compute_heat

ACTIVITY_WINDOW_DAYS = 30
SIGNAL_WINDOW_DAYS = 90
VERDICT_LOOKBACK_DAYS = 180
HEAT_MATERIAL_FRACTION = 0.20   # heat must move ≥20% to count as material


def _iso(now: _dt.datetime, days: int) -> str:
    return (now - _dt.timedelta(days=days)).isoformat(timespec="seconds")


class DossierCurator:
    manifest = AgentManifest(
        name="dossier_curator",
        description="maintains the living, diffed per-artist dossier "
                    "(facts pure; narrative LLM, live-only)",
        reads=("signal", "candidate", "verdict", "dossier_update"),
        writes=("dossier_update",),
        llm=True,
        schedule="daily@07:00",
        budget=Budget(max_runs_per_day=2, max_llm_calls_per_run=3),
    )

    # ── facts (pure) ─────────────────────────────────────────────────────────
    def _facts(self, key, sig_views, heat_map, verdicts) -> dict:
        sigs = [v for v in sig_views if v.get("key") == key]
        counts: dict = {}
        whens = []
        for v in sigs:
            k = v.get("kind") or "signal"
            counts[k] = counts.get(k, 0) + 1
            if v.get("when"):
                whens.append(v["when"])
        h = heat_map.get(key) or {}
        last_verdict = last_conf = None
        for r in verdicts:            # verdicts arrive newest-first (id desc)
            if artist_key(r) == key:
                last_verdict = (r.payload or {}).get("verdict")
                last_conf = (r.payload or {}).get("confidence")
                break
        return {
            "heat": round(float(h.get("heat") or 0.0), 3),
            "top_reasons": list(h.get("top_reasons") or [])[:3],
            "last_verdict": last_verdict,
            "last_confidence": last_conf,
            "signal_counts": counts,
            "first_seen": min(whens).isoformat() if whens else None,
        }

    def _material(self, prev: dict, cur: dict) -> bool:
        """Material change = new verdict, or a new signal kind, or heat moved
        ≥20%. Anything less is noise and is skipped (no record, no LLM)."""
        if cur.get("last_verdict") != prev.get("last_verdict"):
            return True
        if set(cur.get("signal_counts") or {}) - set(prev.get("signal_counts") or {}):
            return True
        ph = float(prev.get("heat") or 0.0)
        ch = float(cur.get("heat") or 0.0)
        if ph == 0.0:
            return ch > 0.0
        return abs(ch - ph) / ph >= HEAT_MATERIAL_FRACTION

    def _diff(self, prev: dict, cur: dict) -> list[str]:
        if not prev:
            n = sum((cur.get("signal_counts") or {}).values())
            lines = [f"First dossier — heat {cur.get('heat')}, {n} signal(s) "
                     "in the last 90 days."]
            if cur.get("last_verdict"):
                lines.append(f"Latest verdict: {cur['last_verdict']}.")
            return lines
        lines = []
        ph, ch = float(prev.get("heat") or 0.0), float(cur.get("heat") or 0.0)
        if ch != ph:
            lines.append(f"Heat {ph:.2f} → {ch:.2f}.")
        if cur.get("last_verdict") != prev.get("last_verdict"):
            lines.append(f"Verdict {prev.get('last_verdict')} → "
                         f"{cur.get('last_verdict')}.")
        for k in sorted(set(cur.get("signal_counts") or {})
                        - set(prev.get("signal_counts") or {})):
            lines.append(f"New signal kind: {k}.")
        return lines or ["Facts changed."]

    # ── narrative (LLM, live only) ───────────────────────────────────────────
    def _narrative(self, ctx, core, name, facts, diff) -> str | None:
        system = "\n".join([
            "You maintain LOFI's living artist dossier for a booking team "
            "(tech-house / house, Amsterdam). Respond in English.",
            "Write ONE short paragraph (3-4 sentences) capturing where this "
            "artist stands and what just changed.",
            "Ground strictly in the facts and diff provided; never invent "
            "numbers, names, or events.",
        ])
        user = json.dumps({"artist": name, "facts": facts, "diff": diff},
                          ensure_ascii=False)
        text, usage = core.complete(system, user, max_tokens=400, mock_text="")
        ctx.llm_calls += 1
        ctx.tokens_in += usage.get("input_tokens", 0)
        ctx.tokens_out += usage.get("output_tokens", 0)
        return text.strip() or None

    # ── run ──────────────────────────────────────────────────────────────────
    def run(self, ctx) -> str:
        now = ctx.now
        active: dict = {}
        for r in ctx.read(["signal", "candidate", "verdict"],
                          since=_iso(now, ACTIVITY_WINDOW_DAYS), limit=5000):
            k = artist_key(r)
            if k:
                active.setdefault(k, (r.artist_id, r.artist_name or k))
        if not active:
            return "no artist activity in the last 30 days"

        sig_views = signal_views(ctx.read(["signal"],
                                          since=_iso(now, SIGNAL_WINDOW_DAYS),
                                          limit=5000))
        heat_map = compute_heat(sig_views, now)
        verdicts = ctx.read(["verdict"], since=_iso(now, VERDICT_LOOKBACK_DAYS),
                            limit=2000)
        core = ctx.llm
        live = core.is_live()
        max_calls = self.manifest.budget.max_llm_calls_per_run

        updated = skipped = 0
        for key, (aid, name) in active.items():
            facts = self._facts(key, sig_views, heat_map, verdicts)
            prev = ctx.state_get(f"dossier:{name}") or {}
            prev_facts = prev.get("facts") or {}
            if prev_facts and not self._material(prev_facts, facts):
                skipped += 1
                continue
            revision = int(prev.get("revision", 0)) + 1
            diff = self._diff(prev_facts, facts)
            narrative = None
            if live and ctx.llm_calls < max_calls:
                narrative = self._narrative(ctx, core, name, facts, diff)
            ctx.emit("dossier_update",
                     {"facts": facts, "diff": diff, "revision": revision,
                      "narrative": narrative,
                      "mode": "live" if live else "mock"},
                     artist_id=aid, artist_name=name)
            ctx.state_set(f"dossier:{name}",
                          {"facts": facts, "revision": revision})
            updated += 1

        return f"updated {updated}, skipped {skipped} (no material change)"

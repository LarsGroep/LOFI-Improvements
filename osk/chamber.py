"""
The Deliberation Chamber — the OS's "reason amongst agents" core
(docs/agentic_os.md §3.3).

A candidate is not judged by a single prompt but by a **structured adversarial
review**: three reasoning roles argue over one shared, minimised evidence pack.

    brief (pure Python, no LLM)          the evidence pack + pointer ids
      → Advocate  (bull case)            argument record  (turn 1)
      → Skeptic   (kills the deal,       argument record  (turn 2)
                   rebuts the Advocate's SPECIFIC claims)
      → Judge     (weighs both, model    argument record  (turn 3)
                   estimates BINDING)     + verdict record (verdict + dissent)

Hard rules, enforced in code (not just prompted):
  - ≤5 LLM calls per deliberation (Budget also caps it; we stop at MAX_LLM_CALLS).
  - Every claim must carry an evidence pointer into the brief; a claim citing no
    valid pointer is STRUCK before it is passed on or stored — the same
    philosophy as grounding.ticket_estimate_allowed in scout/validation.
  - The verdict vocabulary matches predict/window.py + scout/validation.py.
  - Unparseable Judge output drops the debate (no verdict), never a crash.

Compliance: the single LLM egress point stays agents/core.complete(); MOCK mode
(LOFI_LLM_ENABLED off) makes ZERO network calls and runs the full protocol on
deterministic, fact-derived template arguments so the whole OS is demoable
before US-hosting permission is granted.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import uuid

from osk.detectors._common import artist_key, signal_views
from osk.manifest import AgentManifest, Budget
from osk.sentinel import compute_heat

CANDIDATE_WINDOW_DAYS = 14
VERDICT_LOOKBACK_DAYS = 30
SIGNAL_WINDOW_DAYS = 90
DEFAULT_DEBATES_PER_RUN = 3
MAX_LLM_CALLS = 5
MAX_SIGNALS_IN_VIEW = 8

# Matches predict/window.py + scout/validation.py so debates, deal radar and
# validation speak one verdict language.
VERDICTS = ("book_now", "act_fast", "monitor", "too_early", "not_a_fit")


def _debates_per_run() -> int:
    try:
        return max(1, int(os.environ.get("LOFI_OS_DEBATES_PER_RUN",
                                         DEFAULT_DEBATES_PER_RUN)))
    except ValueError:
        return DEFAULT_DEBATES_PER_RUN


# ── data minimisation (the only thing the model ever sees) ───────────────────

def _model_summary(me: dict) -> dict:
    """Allow-listed slice of a deal sheet — base numbers + method only."""
    def base(section: str) -> dict:
        s = me.get(section) or {}
        return {"base": s.get("base"),
                "method": s.get("method") or s.get("basis")}
    return {
        "draw": base("draw_estimate"),
        "fee": base("fee_estimate"),
        "margin": base("margin_estimate"),
        "window_verdict": (me.get("booking_window") or {}).get("verdict"),
    }


def chamber_view(brief: dict) -> dict:
    """Minimise a raw brief to the allow-listed fields the Chamber may reason
    over, giving each evidence item a stable pointer id (E1, E2, …) — the same
    discipline as agents.core.to_model_view. Nothing outside this allow-list
    (raw payload keys, internal ids, secrets) ever reaches the model."""
    evidence: list[dict] = []
    counter = {"n": 0}

    def _ptr() -> str:
        counter["n"] += 1
        return f"E{counter['n']}"

    cand = brief.get("candidate") or {}
    evidence.append({
        "id": _ptr(), "kind": "candidate",
        "priority": cand.get("priority") or cand.get("heat"),
        "reasons": list(cand.get("trigger_signals")
                        or cand.get("reasons") or [])[:6],
    })

    for s in (brief.get("signals") or [])[:MAX_SIGNALS_IN_VIEW]:
        when = s.get("when")
        evidence.append({
            "id": _ptr(), "kind": "signal",
            "signal_kind": s.get("kind"),
            "value": s.get("value"),
            "when": when.isoformat() if hasattr(when, "isoformat") else when,
            "explain": s.get("explain"),
        })

    heat = brief.get("heat") or {}
    evidence.append({
        "id": _ptr(), "kind": "heat",
        "heat": heat.get("heat"),
        "top_reasons": list(heat.get("top_reasons") or [])[:3],
    })

    me = brief.get("model_estimates")
    if me:
        evidence.append({"id": _ptr(), "kind": "model_estimates",
                         "summary": _model_summary(me)})

    return {
        "artist_id": brief.get("artist_id"),
        "artist_name": brief.get("artist_name"),
        "evidence": evidence,
        "has_model_estimates": bool(me),
    }


def strike_claims(claims, valid_pointers) -> list[dict]:
    """Enforce grounding in code: keep only claims that cite at least one valid
    evidence pointer, and drop the invalid pointers from the ones that survive.
    A claim with no valid pointer is struck entirely (cf. the
    grounding.ticket_estimate_allowed rule)."""
    valid = set(valid_pointers or ())
    out = []
    for c in claims or []:
        if not isinstance(c, dict):
            continue
        ptrs = [p for p in (c.get("evidence") or []) if p in valid]
        if not ptrs:
            continue
        out.append({"text": str(c.get("text") or ""), "evidence": ptrs})
    return out


# ── evidence lookups + parsing ───────────────────────────────────────────────

def _find(view: dict, kind: str) -> dict | None:
    for e in view.get("evidence") or []:
        if e.get("kind") == kind:
            return e
    return None


def _strongest_signal(view: dict):
    best = None
    for e in view.get("evidence") or []:
        if e.get("kind") != "signal":
            continue
        try:
            v = float(e.get("value") or 0.0)
        except (TypeError, ValueError):
            v = 0.0
        if best is None or v > best[0]:
            best = (v, e)
    return best


def _parse_obj(text: str) -> dict:
    """Parse a JSON object from the model's text; tolerate ``` fences / prose."""
    t = (text or "").strip()
    if not t:
        return {}
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            obj = json.loads(t[i:j + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


# ── prompts (live) + deterministic templates (mock) ──────────────────────────

_CLAIMS_CONTRACT = (
    'Return ONLY one JSON object, no prose around it: '
    '{"claims":[{"text":"...","evidence":["E1", ...]}]}. Every claim MUST cite '
    'at least one evidence pointer id from the brief; a claim with no valid '
    'pointer is discarded.')

_JUDGE_CONTRACT = (
    'Return ONLY one JSON object, no prose around it: '
    '{"claims":[{"text":"...","evidence":["E1"]}], '
    '"verdict": one of "book_now"|"act_fast"|"monitor"|"too_early"|"not_a_fit", '
    '"confidence": a number 0-1, "summary":"...", '
    '"dissent":{"condition":"...","kind":"<a signal kind>",'
    '"threshold": a number or null, "reason":"..."}}.')


def _live_prompt(role: str, view: dict, prior: dict | None) -> tuple[str, str]:
    L = [
        "You are part of LOFI's Deliberation Chamber judging ONE artist for a "
        "tech-house/house booking in Amsterdam. Respond in English.",
        "Ground EVERY claim in the shared evidence pack using its pointer ids "
        "(E1..En); never invent numbers, names, or facts.",
    ]
    if role == "advocate":
        L += ["You are the ADVOCATE. Build the strongest bull case: momentum, "
              "scene position, trajectory precedent, and why LOFI wins by moving "
              "early.", _CLAIMS_CONTRACT]
    elif role == "skeptic":
        L += ["You are the SKEPTIC, paid to kill the deal. Attack the Advocate's "
              "SPECIFIC claims below — not a generic risk list. Weigh producer-led "
              "vs DJ-led draw, playlist inflation, a missing NL footprint, "
              "forecast downside, and data-confidence holes.",
              "Advocate claims: "
              + json.dumps((prior or {}).get("advocate") or [],
                           ensure_ascii=False),
              _CLAIMS_CONTRACT]
    else:  # judge
        L += ["You are the JUDGE. Add no new evidence. Weigh both arguments "
              "STRICTLY against model_estimates when present: the calibrated "
              "numbers are BINDING — you may contextualise them but never "
              "overrule them. Issue a verdict and a dissent: what would change "
              "the verdict, and which future signal should trigger a re-hearing.",
              "Advocate: " + json.dumps((prior or {}).get("advocate") or [],
                                        ensure_ascii=False),
              "Skeptic: " + json.dumps((prior or {}).get("skeptic") or [],
                                       ensure_ascii=False),
              _JUDGE_CONTRACT]
    return "\n".join(L), "Brief (JSON):\n" + json.dumps(view, ensure_ascii=False,
                                                        default=str)


def _mock_advocate(view: dict) -> dict:
    name = view.get("artist_name") or "this artist"
    claims = []
    cand = _find(view, "candidate")
    if cand:
        claims.append({
            "text": f"{name} was promoted by the Sentinel at priority "
                    f"{cand.get('priority')} — the OS already flagged momentum "
                    "worth a hearing.",
            "evidence": [cand["id"]]})
    st = _strongest_signal(view)
    if st:
        _, e = st
        claims.append({
            "text": f"Strongest live signal is {e.get('signal_kind')} "
                    f"({e.get('value')}); moving early is where LOFI wins.",
            "evidence": [e["id"]]})
    heat = _find(view, "heat")
    if heat and heat.get("heat"):
        reasons = "; ".join(heat.get("top_reasons") or []) or "multiple sources"
        claims.append({
            "text": f"Composite heat sits at {heat.get('heat')} ({reasons}).",
            "evidence": [heat["id"]]})
    if not claims and view.get("evidence"):
        claims.append({"text": f"Evidence on file supports a first look at "
                       f"{name}.", "evidence": [view["evidence"][0]["id"]]})
    return {"claims": claims}


def _mock_skeptic(view: dict, prior: dict | None) -> dict:
    adv = (prior or {}).get("advocate") or []
    claims = []
    for c in adv[:2]:
        claims.append({
            "text": "The bull case leans on \""
                    + (c.get("text") or "")[:60]
                    + "\" — but one signal is not NL draw, and producer-led "
                    "metrics overstate a DJ's pull.",
            "evidence": list(c.get("evidence") or [])[:2]})
    me = _find(view, "model_estimates")
    if me:
        claims.append({
            "text": "The calibrated model estimates are the ceiling here; the "
                    "Advocate's optimism must be checked against them.",
            "evidence": [me["id"]]})
    else:
        heat = _find(view, "heat")
        if heat:
            claims.append({
                "text": "Heat is decayed hype without an NL footprint to "
                        "confirm real demand.",
                "evidence": [heat["id"]]})
    if not claims and view.get("evidence"):
        claims.append({"text": "The evidence is too thin to commit to a booking "
                       "yet.", "evidence": [view["evidence"][0]["id"]]})
    return {"claims": claims}


def _mock_judge(view: dict, prior: dict | None) -> dict:
    name = view.get("artist_name") or "this artist"
    has_model = _find(view, "model_estimates") is not None
    anchor = (_find(view, "model_estimates") or _find(view, "heat")
              or (view.get("evidence") or [None])[0])
    claims = []
    if anchor:
        lead = ("Weighing both sides against the calibrated model estimates "
                "(binding): " if has_model
                else "Weighing both sides on the available signals: ")
        claims.append({"text": lead + f"the case for {name} is real but "
                       "unproven.", "evidence": [anchor["id"]]})
    st = _strongest_signal(view)
    kind = threshold = None
    if st:
        threshold, e = st
        kind = e.get("signal_kind")
    dissent = {
        "condition": (f"Re-hear if {kind} rises above {threshold}." if kind
                      else "Re-hear when a stronger, NL-facing signal lands."),
        "kind": kind,
        "threshold": threshold,
        "reason": (f"{kind} is the strongest current signal; a further rise "
                   "would move the verdict off monitor." if kind
                   else "No decisive signal yet — waiting for one."),
    }
    return {
        "claims": claims,
        "verdict": "monitor",
        "confidence": 0.5,
        "summary": f"Mock deliberation for {name}: monitor — grounded facts "
                   "recorded, no decisive edge yet. Enable the AI for a live, "
                   "web-grounded verdict.",
        "dissent": dissent,
    }


def _role_io(role: str, view: dict, prior: dict | None) -> tuple[str, str, str]:
    """(system, user, mock_text) for one role — mock_text is the deterministic
    JSON the egress point echoes back in preview mode."""
    system, user = _live_prompt(role, view, prior)
    if role == "advocate":
        mock = _mock_advocate(view)
    elif role == "skeptic":
        mock = _mock_skeptic(view, prior)
    else:
        mock = _mock_judge(view, prior)
    return system, user, json.dumps(mock, ensure_ascii=False)


def _clean_dissent(d) -> dict:
    d = d if isinstance(d, dict) else {}
    thr = d.get("threshold")
    if thr is not None:
        try:
            thr = float(thr)
        except (TypeError, ValueError):
            thr = None
    return {"condition": str(d.get("condition") or ""),
            "kind": d.get("kind"),
            "threshold": thr,
            "reason": str(d.get("reason") or "")}


# ── agent shell ───────────────────────────────────────────────────────────────

class Chamber:
    manifest = AgentManifest(
        name="chamber",
        description="structured adversarial review of candidates (Advocate / "
                    "Skeptic / Judge) with binding model estimates",
        reads=("candidate", "signal", "verdict", "brief"),
        writes=("brief", "argument", "verdict"),
        llm=True,
        schedule="daily@06:30",
        budget=Budget(max_runs_per_day=4, max_llm_calls_per_run=5),
    )

    # ── selection ────────────────────────────────────────────────────────────
    def _select_unheard(self, ctx) -> list:
        cand_since = (ctx.now - _dt.timedelta(days=CANDIDATE_WINDOW_DAYS)
                      ).isoformat(timespec="seconds")
        verdict_since = (ctx.now - _dt.timedelta(days=VERDICT_LOOKBACK_DAYS)
                         ).isoformat(timespec="seconds")
        heard = set()
        for v in ctx.read(["verdict"], since=verdict_since, limit=1000):
            k = artist_key(v)
            if k:
                heard.add(k)
        seen, unheard = set(), []
        for c in ctx.read(["candidate"], since=cand_since, limit=1000):
            k = artist_key(c)
            if not k or k in heard or k in seen:
                continue
            seen.add(k)
            unheard.append(c)
        return unheard

    # ── brief (pure Python, no LLM) ──────────────────────────────────────────
    def _model_estimates(self, cand) -> dict | None:
        supa = os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY")
        if not (supa or os.environ.get("AIRTABLE_TOKEN")):
            return None
        try:  # wrap like scout/context.py — None on any failure, never raises
            from predict.deal import build_deal_sheet
            genres = (cand.payload or {}).get("genres") or []
            return build_deal_sheet(cand.artist_name or "", genres)
        except Exception:
            return None

    def _build_brief(self, ctx, cand, all_views: list[dict]) -> dict:
        key = artist_key(cand)
        sigs = [v for v in all_views if v.get("key") == key]
        heat = (compute_heat(sigs, ctx.now).get(key)
                or {"heat": 0.0, "top_reasons": [],
                    "artist_name": cand.artist_name or key})
        return {
            "artist_id": cand.artist_id,
            "artist_name": cand.artist_name or key,
            "candidate": cand.payload or {},
            "signals": sigs,
            "heat": heat,
            "model_estimates": self._model_estimates(cand),
        }

    # ── one debate turn through the single egress point ──────────────────────
    def _call(self, ctx, core, role, view, prior, live) -> dict:
        system, user, mock = _role_io(role, view, prior)
        text, usage = core.complete(system, user, max_tokens=1500, mock_text=mock)
        ctx.llm_calls += 1
        if live:
            ctx.tokens_in += usage.get("input_tokens", 0)
            ctx.tokens_out += usage.get("output_tokens", 0)
        return _parse_obj(text)

    def _deliberate(self, ctx, core, cand, all_views) -> str | None:
        debate_id = uuid.uuid4().hex
        brief = self._build_brief(ctx, cand, all_views)
        view = chamber_view(brief)
        valid = {e["id"] for e in view["evidence"]}
        live = core.is_live()
        mode = "live" if live else "mock"
        aid, name = brief["artist_id"], brief["artist_name"]

        ctx.emit("brief", {"debate_id": debate_id, "mode": mode, "view": view},
                 artist_id=aid, artist_name=name)

        # Advocate → Skeptic → Judge, hard-capped at MAX_LLM_CALLS.
        adv = strike_claims(
            self._call(ctx, core, "advocate", view, None, live).get("claims"),
            valid)
        ctx.emit("argument", {"debate_id": debate_id, "turn": 1,
                              "role": "advocate", "claims": adv, "mode": mode},
                 artist_id=aid, artist_name=name)

        skp = strike_claims(
            self._call(ctx, core, "skeptic", view, {"advocate": adv},
                       live).get("claims"), valid)
        ctx.emit("argument", {"debate_id": debate_id, "turn": 2,
                              "role": "skeptic", "claims": skp, "mode": mode},
                 artist_id=aid, artist_name=name)

        if ctx.llm_calls >= MAX_LLM_CALLS:
            return None  # protocol exhausted its call budget before a verdict
        judgment = self._call(ctx, core, "judge", view,
                              {"advocate": adv, "skeptic": skp}, live)
        verdict = judgment.get("verdict")
        if verdict not in VERDICTS:
            return None  # unparseable / off-vocabulary → drop, never a verdict

        try:
            confidence = float(judgment.get("confidence"))
        except (TypeError, ValueError):
            confidence = None
        jclaims = strike_claims(judgment.get("claims"), valid)
        dissent = _clean_dissent(judgment.get("dissent"))
        summary = str(judgment.get("summary") or "")

        ctx.emit("argument", {"debate_id": debate_id, "turn": 3,
                              "role": "judge", "claims": jclaims, "mode": mode},
                 artist_id=aid, artist_name=name)
        ctx.emit("verdict", {"debate_id": debate_id, "verdict": verdict,
                             "confidence": confidence, "summary": summary,
                             "dissent": dissent, "mode": mode},
                 artist_id=aid, artist_name=name)
        return name

    def run(self, ctx) -> str:
        core = ctx.llm  # the agents.core module (llm=True agents only)
        unheard = self._select_unheard(ctx)
        if not unheard:
            return "no unheard candidates in the last 14 days"

        sig_since = (ctx.now - _dt.timedelta(days=SIGNAL_WINDOW_DAYS)
                     ).isoformat(timespec="seconds")
        all_views = signal_views(ctx.read(["signal"], since=sig_since,
                                          limit=5000))

        heard, dropped = [], 0
        for cand in unheard[:_debates_per_run()]:
            if ctx.llm_calls >= MAX_LLM_CALLS:
                break
            name = self._deliberate(ctx, core, cand, all_views)
            if name:
                heard.append(name)
            else:
                dropped += 1

        mode = "live" if core.is_live() else "mock"
        parts = [f"deliberated {len(heard)}"]
        if heard:
            parts.append(", ".join(heard[:5]))
        if dropped:
            parts.append(f"{dropped} dropped (no clean verdict)")
        return f"[{mode}] " + " · ".join(parts)

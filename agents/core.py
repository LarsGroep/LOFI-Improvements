"""
agents/core.py — the single LLM egress point for the LOFI agent.

COMPLIANCE GUARD (Verwerkersovereenkomst art. 3.5 / 3.7)
--------------------------------------------------------
Real inference only runs when LOFI_LLM_ENABLED is truthy. Until then the wrapper
runs in MOCK mode and makes **no network calls** — so the UI is fully
visualisable without sending any data to an external LLM.

Path C was chosen: the direct Anthropic API (US inference) under Anthropic's DPA
+ EU SCCs + zero-data-retention. The direct API cannot keep data in the EEA
(inference_geo only offers "us"/"global"; workspace geo is "us"-only), so
LOFI_LLM_ENABLED must ONLY be set once Lofi's *amended* written permission for
US-hosting-under-SCCs and account-level zero-retention are both confirmed.

Everything Claude ever sees passes through `to_model_view()` (data minimisation).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("LOFI_LLM_MODEL", "claude-opus-4-8")
# "us" = deterministic US (1.1x), cleaner to document for SCCs than "global".
INFERENCE_GEO = os.environ.get("LOFI_LLM_INFERENCE_GEO", "us")
MAX_TOKENS = 8000
LANG = os.environ.get("LOFI_LLM_LANG", "en")  # "en" | "nl"


def _lang_line() -> str:
    return ("Antwoord in het Nederlands." if LANG == "nl"
            else "Respond in English.")


# ── mode / compliance ────────────────────────────────────────────────────────

def _enabled_flag() -> bool:
    return os.environ.get("LOFI_LLM_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on")


def _anthropic_ready() -> tuple[bool, str]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY niet ingesteld"
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False, "anthropic SDK niet geïnstalleerd"
    return True, ""


def is_live() -> bool:
    return _enabled_flag() and _anthropic_ready()[0]


def compliance_status() -> dict:
    """Audit surface the UI can display so the active mode is always visible."""
    if not _enabled_flag():
        reason = "LOFI_LLM_ENABLED staat uit (voorbeeldmodus)"
        mode = "mock"
    else:
        ok, why = _anthropic_ready()
        mode, reason = ("live", "Claude actief") if ok else ("mock", why)
    return {"mode": mode, "model": MODEL, "inference_geo": INFERENCE_GEO,
            "reason": reason}


def _client():
    import anthropic
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY


def _create(**kwargs):
    """Central Claude call. Injects inference_geo robustly across SDK versions."""
    client = _client()
    kwargs.setdefault("model", MODEL)
    try:
        return client.messages.create(inference_geo=INFERENCE_GEO, **kwargs)
    except TypeError:
        # older SDK without a typed inference_geo kwarg
        extra = kwargs.pop("extra_body", {}) or {}
        extra["inference_geo"] = INFERENCE_GEO
        return client.messages.create(extra_body=extra, **kwargs)


# ── data minimisation (the only thing Claude ever sees) ──────────────────────

def to_model_view(c: dict, *, booking_history: list[dict] | None = None) -> dict:
    """Strip a candidate down to the allow-listed fields. `booking_history`
    (gages/past bookings) is Phase 3 and only included when Lofi approved it."""
    view = {
        "artist_id": c.get("artist_id"),
        "name": c.get("artist_name"),
        "genres": (c.get("genres") or [])[:6],
        "career_status": c.get("career_status"),
        "spotify_listeners": c.get("spotify_listeners"),
        "scores": {
            "momentum": c.get("momentum"),
            "growth": c.get("growth"),
            "market_relevance": c.get("market_relevance"),
            "future_potential": c.get("future_potential"),
            "confidence": c.get("confidence"),
        },
        "forecast_90d": c.get("forecast_90d"),
    }
    if booking_history:
        view["booking_history"] = booking_history  # Phase 3, gage-approved
    return view


# ── prompts ──────────────────────────────────────────────────────────────────

def _taxonomy_block() -> str:
    from scout.ranking import load_taxonomy
    t = load_taxonomy()
    g = t.get("genres", {})
    b = t.get("benchmark_artists", {})
    return (
        "Kerngenres: " + ", ".join(g.get("tier_1", [])) + ". "
        "Aanverwant: " + ", ".join(g.get("tier_2", [])) + ". "
        "Referentie-artiesten: "
        + ", ".join(b.get("tier_a_plus", []) + b.get("tier_a", [])) + "."
    )


def _scout_system() -> str:
    return "\n".join([
        "You are LOFI's scout for a tech-house/house booking agency, writing for "
        "a non-technical booking team.",
        _lang_line(),
        "Ground every pick in concrete signals from the data (growth, momentum, "
        "forecast, genre-fit).",
        "Be honest: a high growth score with a negative forecast is a "
        "trending-then-cooling signal — don't hide it.",
        "Use ONLY the provided data; never invent numbers or facts.",
    ])


def _chat_system(artist_view: dict) -> str:
    has_bookings = bool(artist_view.get("booking_history")
                        or artist_view.get("comparables")
                        or artist_view.get("lofi_record"))
    lines = [
        "You are LOFI's artist-intelligence assistant, helping a booking team "
        "decide about ONE artist. Think like an experienced LOFI booker "
        "(tech-house / house). You are a reasoning layer over LOFI's own data — "
        "not a source of facts about these (often niche) artists.",
        _lang_line(),
        "",
        "Work in two lenses — state which one the question needs:",
        "- SCOUTING: breakout potential in the next 6-18 months — growth signals, "
        "momentum, comparables, and whether it is simply too early. Best for "
        "small/rising artists.",
        "- VALIDATION: can this artist sell tickets at LOFI (especially the "
        "Amsterdam/NL audience), is there enough demand, does the sound fit, is "
        "the asked fee logical, and is the timing right (book now / keep "
        "monitoring / too early / too late)?",
        "",
        "First classify the artist as PRODUCER-LED or DJ-LED from the signal "
        "pattern, and say which:",
        "- Producer-led grows via releases, Spotify, viral tracks, social "
        "momentum, charts — digital metrics are strong predictors.",
        "- DJ-led grows via club reputation, extended sets, tastemaker status, "
        "word-of-mouth, live reputation — digital metrics UNDERESTIMATE them. "
        "Never write off a DJ-led artist for low Spotify or few releases; weight "
        "shows, venues and scene signals instead.",
        "Use `show_history` (RA + Partyflock show counts, recent venues) and "
        "`nl_signal` (NL audience score, Partyflock fans) to make this call: many "
        "shows and strong venue / Partyflock presence with modest Spotify = "
        "DJ-led.",
        "",
        "How to reason:",
        "- Ground every claim in the provided data and name the signal it rests "
        "on (a score, a metric, a show, a booking, booker feedback).",
        "- Explain a score in plain terms when relevant (what it measures, why it "
        "matters for a booking).",
        "- For 'comparable to' questions, ground on the provided `similar_artists` "
        "(Last.fm + Chartmetric scene adjacency) and the LOFI reference artists. "
        "That raw list is broad — filter it to genuine matches on sound + market "
        "position + audience + growth trajectory, and say why each one fits. Use "
        "`comparables` / `booking_history` (Airtable) only for fee/draw "
        "benchmarking. Never invent names you were not given.",
        "- For FEE and DRAW questions, ground strictly on LOFI's own numbers: "
        "`lofi_record` (this artist's LOFI economics — Sound/genre, `fee_range`, "
        "`last_fee_paid`, `last_event_visitors`, `avg_ticket_price`, "
        "`avg_bar_spend`, `momentum`, `fit_lofi`), `booking_history` (this "
        "artist's past LOFI gages/events) and `comparables` (genre-matched past "
        "LOFI bookings with their fees). Quote the figure and where it comes from; "
        "if a fee was never paid by LOFI, say so and reason from comparables "
        "instead of guessing. `fit_lofi` and `momentum` here are the team's own "
        "assessment — weight them.",
        "- Weight qualitative booker feedback highly (real-world > scraped) when "
        "it is present.",
        "- Use `show_history`, `nl_signal` and `milestones` as live-history "
        "evidence for draw, NL/Amsterdam demand, and trajectory — these are facts "
        "you have, not ticket predictions.",
        "",
        "Honesty (critical — the team does not fully trust the data yet):",
        "- NEVER invent ticket numbers, gages, comparable names, or facts. If you "
        "cannot ground something, say what data is missing.",
        "- Flag suspect data: if the genre does not match tech-house/house, the "
        "Chartmetric profile may be a name collision (wrong artist) — say so and "
        "lower confidence. A very low CM score or no releases for a DJ-led artist "
        "is expected, not a red flag by itself.",
        "- A high momentum/growth score with a negative forecast means "
        "trending-then-cooling — name it.",
        "- State your confidence and what extra data would change the answer.",
    ]
    if not has_bookings:
        lines.append(
            "- NOTE: No LOFI booking data (Airtable) is connected yet, so you "
            "cannot give gage indications, ticket predictions, or claims about "
            "past LOFI bookings — say that when asked. You CAN still speak to "
            "live draw and NL/Amsterdam demand from `show_history` and "
            "`nl_signal`.")
    lines += ["", _taxonomy_block(),
              "", "Artist data (JSON):",
              json.dumps(artist_view, ensure_ascii=False)]
    return "\n".join(lines)


# ── Scout rationales (structured output) ─────────────────────────────────────

_RATIONALE_SCHEMA = {
    "type": "object",
    "properties": {
        "rationales": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "artist_id": {"type": "string"},
                    "rationale_nl": {"type": "string"},
                    "fit_score": {"type": "integer"},
                },
                "required": ["artist_id", "rationale_nl", "fit_score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rationales"],
    "additionalProperties": False,
}


def generate_rationales(candidates: list[dict]) -> dict[str, dict]:
    """artist_id -> {"rationale_nl", "fit_score"}.

    MOCK mode reuses the deterministic `waarom`/`rank` so the UI is visualisable;
    LIVE mode asks Claude for a one-line Dutch rationale + fit per artist.
    """
    if not is_live():
        return {
            c["artist_id"]: {
                "rationale_nl": c.get("waarom", ""),
                "fit_score": round(c.get("rank", 0)),
            }
            for c in candidates
        }

    views = [to_model_view(c) for c in candidates]
    user = (
        "Geef voor elke artiest één korte Nederlandse toelichting (één zin) voor "
        "de boeking-shortlist, plus een fit_score 0-100. Kandidaten (JSON):\n"
        + json.dumps(views, ensure_ascii=False)
    )
    resp = _create(
        max_tokens=MAX_TOKENS,
        system=_scout_system(),
        output_config={"format": {"type": "json_schema",
                                  "schema": _RATIONALE_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    if getattr(resp, "stop_reason", None) == "refusal":
        raise RuntimeError(f"Claude weigerde het verzoek: {resp.stop_details}")
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    data = json.loads(text)
    return {
        r["artist_id"]: {"rationale_nl": r["rationale_nl"],
                         "fit_score": r["fit_score"]}
        for r in data.get("rationales", [])
    }


# ── Booker-in-the-loop: structure free-text feedback ─────────────────────────

_FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string",
                     "enum": ["performance", "correction", "intelligence", "other"]},
        "field_key": {"type": "string"},
        "field_value": {"type": "string"},
        "event_ref": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["category", "field_key", "field_value", "event_ref", "summary"],
    "additionalProperties": False,
}

_FEEDBACK_SYSTEM = "\n".join([
    "You structure a LOFI booker's free-text feedback about an artist into one "
    "record. Classify the category:",
    "- performance: real-world results (tickets sold, crowd response, sold out, "
    "room emptied, support over/under-delivered).",
    "- correction: the model/data is wrong (similar artists off, momentum too "
    "high, LOFI fit wrong, wrong Chartmetric profile).",
    "- intelligence: forward-looking, non-scrapeable industry info (upcoming "
    "release, agency switch, collab, festival booking).",
    "- other: anything else.",
    "field_key: a short snake_case slug (e.g. ticket_sales, crowd_response, "
    "agency_switch, momentum_too_high).",
    "field_value: the key extracted value, concise (e.g. '300 @ BRET', "
    "'switching to WME', 'too high').",
    "event_ref: the venue/event/date if mentioned, else an empty string.",
    "summary: one clean sentence capturing the feedback.",
    "Extract only what is stated; never invent details.",
])


def _fallback_feedback(raw_text: str) -> dict:
    return {"category": "other", "field_key": "booker_note",
            "field_value": raw_text.strip()[:120], "event_ref": "",
            "summary": raw_text.strip()}


def structure_feedback(raw_text: str) -> dict:
    """Classify + extract a booker's note. MOCK mode (or refusal) stores it as a
    plain note so capture always works; LIVE mode adds structure."""
    if not is_live():
        return _fallback_feedback(raw_text)
    resp = _create(
        max_tokens=1000,
        system=_FEEDBACK_SYSTEM,
        output_config={"format": {"type": "json_schema",
                                  "schema": _FEEDBACK_SCHEMA}},
        messages=[{"role": "user", "content": raw_text}],
    )
    if getattr(resp, "stop_reason", None) == "refusal":
        return _fallback_feedback(raw_text)
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        return json.loads(text)
    except Exception:
        return _fallback_feedback(raw_text)


# ── Per-artist validation (web-search grounded, structured) ──────────────────

# On Opus 4.8 the current web-search tool is web_search_20260209 (dynamic
# filtering). NOTE: web search emits citation-bearing text blocks, and citations
# are incompatible with output_config.format — so we ask for JSON in the prompt
# and parse it from the text, rather than using structured-output mode.
_WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": 4,
    "user_location": {"type": "approximate", "country": "NL",
                      "city": "Amsterdam", "timezone": "Europe/Amsterdam"},
}

_VALIDATION_CONTRACT = """Return ONLY one valid JSON object, no prose around it:
{
  "artist_type": "producer_led | dj_led | hybrid",
  "lens": "scouting | validation",
  "ticket_estimate": {"conservative": int, "base": int, "bull": int} | null,
  "ticket_basis": "own_history | comparables | insufficient",
  "sell_through": "low | medium | high | unknown",
  "lofi_fit": {"score": int (0-100), "explanation": "plain English"},
  "recommendation": "book_now | monitor | too_early | not_a_fit",
  "comparable_events": [{"event": "", "date": "", "tickets": int, "why": ""}],
  "key_signals": ["grounded positives, each naming its source"],
  "key_risks": [""],
  "data_confidence": "low | medium | high",
  "missing_data": ["what you'd need to sharpen this"],
  "web_context": [{"claim": "", "url": ""}]
}"""


def _validation_system(view: dict, use_web: bool = True) -> str:
    allowed = bool((view.get("grounding") or {}).get("ticket_estimate_allowed"))
    lines = [
        "You are a senior LOFI booker / A&R deciding whether to book ONE artist "
        "for LOFI in Amsterdam (tech-house / house / techno scene). Be concrete, "
        "skeptical, and grounded. Respond in English.",
        "",
        "Pick the lens and say which: SCOUTING (breakout potential, 6-18 months) "
        "for small/rising artists, or VALIDATION (can they sell tickets at LOFI "
        "now, does the sound fit, is the timing right) for established ones.",
        "",
        "First classify PRODUCER-LED vs DJ-LED and weight signals accordingly: "
        "producer-led grows via releases / Spotify / charts (digital metrics "
        "predict well); DJ-led grows via club reputation, residencies, RA / Boiler "
        "Room / Essential Mix, word-of-mouth (digital metrics UNDERESTIMATE them — "
        "lean on show_history, nl_signal, and web reputation instead).",
        "",
        "TICKET NUMBERS — trust-first, this is critical:",
        "- Anchor any ticket figure ONLY on LOFI's own data: `lofi_event_history` "
        "(this artist's past LOFI draw — the strongest signal) and "
        "`comparable_lofi_events` (genre-matched past LOFI events). Web search "
        "adds reputation and risk context; it NEVER sets the numbers.",
        "- Give conservative/base/bull as a RANGE, with the comparables that "
        "justify it. The comparables are the point; the number is the consequence.",
    ]
    if allowed:
        lines.append("- You HAVE enough grounding here (own history or >=3 "
                     "comparable events): give a range and set ticket_basis to "
                     "'own_history' or 'comparables'.")
    else:
        lines.append("- You do NOT have enough grounding here (no own LOFI "
                     "history and <3 comparable events): set ticket_estimate to "
                     "null and ticket_basis to 'insufficient', and say in "
                     "missing_data what would let you estimate. Do NOT guess a "
                     "number.")
    lines += [
        "- sell_through is a bucket (low/medium/high) tied to the occupancy_rate "
        "of the comparables — not a number you invent.",
        "",
    ]
    if use_web:
        lines += [
            "WEB SEARCH (<=4 searches): use it to check recent/upcoming releases, "
            "residencies, Boiler Room / RA Podcast / Essential Mix, agency/label, "
            "notable recent bookings, and scene buzz — ESPECIALLY when the data is "
            "thin or the artist is DJ-led (this fixes the low-digital-footprint "
            "blind spot). Put every web-sourced claim in web_context with its URL.",
            "- Search queries may contain ONLY the artist's public name and public "
            "scene context (genre, city, venue names). NEVER put LOFI's internal "
            "data into a web query — no fees/gages, ticket counts, occupancy, "
            "booker feedback, or comparable-event economics. Those stay inside this "
            "reasoning only.",
            "",
        ]
    else:
        lines += [
            "WEB SEARCH is OFF for this run: reason only from the provided LOFI + "
            "Chartmetric data, set web_context to [], and note in missing_data "
            "where web reputation would have sharpened the call (e.g. a DJ-led "
            "artist with thin streaming data).",
            "",
        ]
    lines += [
        "Honesty: never invent tickets, fees, comparable names, or facts. A high "
        "growth score with a negative forecast is trending-then-cooling — name it. "
        "If the genre is off tech-house/house/techno, flag a possible wrong "
        "Chartmetric profile and lower confidence. Weight booker_feedback highly.",
        "",
        _taxonomy_block(),
        "",
        _VALIDATION_CONTRACT,
        "",
        "Artist data (JSON):",
        json.dumps(view, ensure_ascii=False, default=list),
    ]
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of the model's final text (handles ``` fences)."""
    t = (text or "").strip()
    if "```" in t:
        seg = t.split("```")[1] if t.count("```") >= 2 else t
        t = seg[4:] if seg.lstrip().lower().startswith("json") else seg
        t = t.strip()
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return {}
    return {}


def _collect_citations(resp) -> list[dict]:
    """Web-search URLs the model actually cited — provenance for the UI."""
    out, seen = [], set()
    for block in getattr(resp, "content", []) or []:
        for c in (getattr(block, "citations", None) or []):
            url = getattr(c, "url", None)
            if url and url not in seen:
                seen.add(url)
                out.append({"title": getattr(c, "title", "") or url, "url": url})
    return out


def _web_search_status(resp) -> dict:
    """Inspect web_search_tool_result blocks so a silent 'no results' becomes a
    real diagnosis. Web-search errors return HTTP 200 with an error object in the
    block content (not an exception): a list content = results, an object/dict
    content = an error (e.g. error_code 'max_uses_exceeded', or web search not
    enabled in the Anthropic Console)."""
    ran, errors = 0, []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", "") != "web_search_tool_result":
            continue
        content = getattr(block, "content", None)
        if isinstance(content, list):
            ran += 1
        else:  # error object
            code = (getattr(content, "error_code", None)
                    or (content.get("error_code") if isinstance(content, dict) else None)
                    or getattr(content, "type", None) or "unknown_error")
            errors.append(str(code))
    return {"searches_ran": ran, "errors": errors}


def _mock_validation(view: dict) -> dict:
    return {
        "artist_type": "unknown",
        "lens": "validation",
        "ticket_estimate": None,
        "ticket_basis": "insufficient",
        "sell_through": "unknown",
        "lofi_fit": {"score": None,
                     "explanation": "Preview mode — enable LOFI_LLM_ENABLED for a "
                                    "real, web-grounded validation."},
        "recommendation": "monitor",
        "comparable_events": [
            {"event": e.get("event_name"), "date": e.get("event_date"),
             "tickets": e.get("actual_tickets"), "why": f"genre {e.get('genre')}"}
            for e in (view.get("comparable_lofi_events") or [])[:5]],
        "key_signals": [],
        "key_risks": [],
        "data_confidence": "low",
        "missing_data": ["AI assistant is off (preview mode)."],
        "web_context": [],
        "_mock": True,
    }


def validate_artist(view: dict, booker_note: str | None = None,
                    use_web: bool = True) -> dict:
    """Single grounded validation call → structured verdict.

    MOCK mode returns a labelled placeholder (UI stays visualisable). LIVE mode
    parses the JSON the model returns, attaches the real web citations, and
    enforces the trust-first rule in code: no ticket number unless the artist has
    LOFI history or >=3 comparable events. `use_web` toggles the web-search tool
    (off = reason on LOFI + Chartmetric data only, cheaper, no external query)."""
    if not is_live():
        return _mock_validation(view)

    user = "Validate this artist for a LOFI booking."
    if booker_note:
        user += f"\n\nBooker note to weigh: {booker_note}"

    messages = [{"role": "user", "content": user}]
    resp = None
    for _ in range(4):  # tolerate web-search pause_turn continuations
        kwargs = dict(max_tokens=3000,
                      system=_validation_system(view, use_web), messages=messages)
        if use_web:
            kwargs["tools"] = [_WEB_SEARCH_TOOL]
        resp = _create(**kwargs)
        if getattr(resp, "stop_reason", None) == "refusal":
            raise RuntimeError(f"Claude refused the request: {resp.stop_details}")
        if getattr(resp, "stop_reason", None) == "pause_turn":
            messages.append({"role": "assistant", "content": resp.content})
            continue
        break

    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    result = _extract_json(text)
    if not result:
        return {**_mock_validation(view),
                "missing_data": ["The AI returned an unparseable response."],
                "_error": "parse_failed", "_raw": text[:500]}

    # provenance: attach the actual cited URLs as a backstop to web_context
    cites = _collect_citations(resp)
    if cites and not result.get("web_context"):
        result["web_context"] = [{"claim": "", "url": c["url"]} for c in cites]
    result["_sources"] = cites
    if use_web:
        result["_web"] = _web_search_status(resp)  # diagnose silent 'no results'

    # trust-first enforcement (belt-and-suspenders over the prompt rule)
    if not (view.get("grounding") or {}).get("ticket_estimate_allowed"):
        result["ticket_estimate"] = None
        result["ticket_basis"] = "insufficient"
    return result


# ── Compare 2-3 artists (head-to-head, prose) ────────────────────────────────

def _compare_system() -> str:
    return "\n".join([
        "You are a senior LOFI booker comparing artists head-to-head for a "
        "booking decision (tech-house / house / techno, Amsterdam).",
        _lang_line(),
        "Classify each as producer-led or DJ-led and weight signals accordingly "
        "(DJ-led artists are underestimated by digital metrics).",
        "Compare them on: LOFI sound-fit, draw / market position, momentum vs "
        "trajectory (a high growth score with a negative forecast is "
        "trending-then-cooling — say so), and risk. Use ONLY the provided data; "
        "never invent numbers or names.",
        "Be concise: a short head-to-head, then a clear RANKED recommendation "
        "(who to prioritise and why), and one line on what extra data would "
        "change it.",
        "",
        _taxonomy_block(),
    ])


def compare_artists(candidates: list[dict]) -> str:
    """Head-to-head booking comparison of 2-3 artists → markdown text.

    MOCK mode returns a deterministic side-by-side so the UI stays visualisable;
    LIVE mode asks Claude for a booker's comparison + ranked recommendation."""
    views = [to_model_view(c) for c in candidates]
    if not is_live():
        out = ["**Preview mode** — enable LOFI_LLM_ENABLED for an AI comparison. "
               "Data-driven snapshot:"]
        for c, v in zip(candidates, views):
            s = v["scores"]
            fc = v.get("forecast_90d")
            out.append(
                f"- **{v['name']}** ({', '.join((v.get('genres') or [])[:2]) or '—'})"
                f" — scout {c.get('rank', '—')}, momentum {s['momentum']}, growth "
                f"{s['growth']}, potential {s['future_potential']}, forecast "
                f"{'—' if fc is None else f'{fc:+.0f}%'}")
        return "\n".join(out)

    user = ("Compare these artists for a LOFI booking decision. Artists (JSON):\n"
            + json.dumps(views, ensure_ascii=False, default=list))
    resp = _create(max_tokens=2000, system=_compare_system(),
                   messages=[{"role": "user", "content": user}])
    if getattr(resp, "stop_reason", None) == "refusal":
        raise RuntimeError(f"Claude refused the request: {resp.stop_details}")
    return "".join(b.text for b in resp.content
                   if getattr(b, "type", "") == "text").strip()


# ── Shortlist booking analysis (deep, grounded, ranked) ──────────────────────

def _shortlist_system(views: list[dict], use_web: bool = True) -> str:
    lines = [
        "You are a senior LOFI booker / A&R choosing which artist to book from a "
        "shortlist for LOFI in Amsterdam (tech-house / house / techno). Be "
        "concrete, skeptical, and grounded.",
        _lang_line(),
        "",
        "You are given up to 6 artists. Each carries LOFI's own grounding: "
        "`lofi_event_history` (their past LOFI draw — the strongest signal), "
        "`comparable_lofi_events` (genre-matched past LOFI events), `grounding` "
        "(whether a ticket estimate is allowed), plus the five scores, growth "
        "forecast, genres, show history and NL signal.",
        "",
        "For EACH artist, briefly: classify producer-led vs DJ-led; read their "
        "draw / LOFI sound-fit / timing; and give an expected LOFI draw ONLY when "
        "grounded (own history or >=3 comparable events) — otherwise say it is not "
        "groundable and do NOT invent a number. Same trust-first rule as the "
        "single-artist validation.",
        "",
        "THEN rank them from smartest to riskiest booking for LOFI, and state the "
        "#1 pick explicitly with why it beats #2. Sound-fit matters: weight the "
        "tech-house / house / techno core and flag anything off-genre. A high "
        "growth score with a negative forecast is trending-then-cooling — name it.",
    ]
    if use_web:
        lines += [
            "",
            "WEB SEARCH (<=8 searches total across the shortlist): check "
            "residencies, recent/upcoming releases, Boiler Room / RA / Essential "
            "Mix, agency/label and scene buzz — especially for DJ-led artists with "
            "thin streaming data. Cite every web claim with its URL. Queries may "
            "contain ONLY the artist's public name + public scene context, NEVER "
            "LOFI's internal data (fees, ticket counts, occupancy, booker notes).",
        ]
    else:
        lines += [
            "",
            "WEB SEARCH is OFF: reason only from the provided LOFI + Chartmetric "
            "data; note where web reputation would have sharpened the call.",
        ]
    lines += [
        "",
        "Format as readable markdown: a short line per artist, then a clear "
        "**Ranking** and a one-line **Bottom line** recommendation, then what "
        "extra data (a booker note, the fee ask) would change the ranking.",
        "",
        _taxonomy_block(),
        "",
        "Shortlist data (JSON):",
        json.dumps(views, ensure_ascii=False, default=list),
    ]
    return "\n".join(lines)


def _mock_recommend(views: list[dict]) -> dict:
    ranked = sorted(
        views, key=lambda v: (v.get("scores") or {}).get("future_potential") or 0,
        reverse=True)
    out = ["**Preview mode** — enable LOFI_LLM_ENABLED for the full booking "
           "analysis. Data-driven snapshot (ranked by potential):", ""]
    for i, v in enumerate(ranked, 1):
        s = v.get("scores") or {}
        fc = v.get("forecast_90d")
        g = v.get("grounding") or {}
        out.append(
            f"{i}. **{v.get('name')}** — potential {s.get('future_potential')}, "
            f"momentum {s.get('momentum')}, growth {s.get('growth')}, forecast "
            f"{'—' if fc is None else f'{fc:+.0f}%'} · "
            f"{'LOFI-grounded' if g.get('ticket_estimate_allowed') else 'thin LOFI data'}")
    if ranked:
        out += ["", f"**Bottom line (preview):** {ranked[0].get('name')} ranks "
                "highest on potential — enable the AI for the grounded verdict."]
    return {"analysis": "\n".join(out), "sources": []}


def recommend_booking(views: list[dict], use_web: bool = True) -> dict:
    """Deep, grounded analysis over up to 6 artists → ranked 'smartest to book'.

    Returns {"analysis": markdown, "sources": [...]}. MOCK mode returns a
    deterministic ranked snapshot so the UI stays visualisable."""
    if not is_live():
        return _mock_recommend(views)

    user = ("Analyse this shortlist and tell me which artist is smartest to book "
            "for LOFI, with a clear ranking.")
    messages = [{"role": "user", "content": user}]
    tool = {**_WEB_SEARCH_TOOL, "max_uses": 8}
    resp = None
    for _ in range(5):  # tolerate web-search pause_turn continuations
        kwargs = dict(max_tokens=6000,
                      system=_shortlist_system(views, use_web), messages=messages)
        if use_web:
            kwargs["tools"] = [tool]
        resp = _create(**kwargs)
        if getattr(resp, "stop_reason", None) == "refusal":
            raise RuntimeError(f"Claude refused the request: {resp.stop_details}")
        if getattr(resp, "stop_reason", None) == "pause_turn":
            messages.append({"role": "assistant", "content": resp.content})
            continue
        break

    text = "".join(b.text for b in resp.content
                   if getattr(b, "type", "") == "text").strip()
    out = {"analysis": text, "sources": _collect_citations(resp)}
    if use_web:
        out["web"] = _web_search_status(resp)  # diagnose silent 'no results'
    return out


def _shortlist_chat_system(views: list[dict], analysis: str) -> str:
    return "\n".join([
        "You are LOFI's booking assistant. The user is following up on a shortlist "
        "booking analysis you already produced. Answer grounded ONLY in that "
        "analysis and the artist data below; never invent numbers or names. Be "
        "concise and decisive.",
        _lang_line(),
        "",
        "The booking analysis you produced:",
        analysis or "(none)",
        "",
        "Artist data (JSON):",
        json.dumps(views, ensure_ascii=False, default=list),
    ])


def shortlist_chat_stream(views: list[dict], analysis: str,
                          history: list[dict], user_msg: str):
    """Follow-up chat over a finished shortlist analysis. Yields text chunks."""
    if not is_live():
        yield ("[Preview mode] The AI assistant is off. Once Claude is enabled "
               "I'll answer follow-ups about this shortlist here.")
        return
    client = _client()
    messages = list(history) + [{"role": "user", "content": user_msg}]
    kw = dict(model=MODEL, max_tokens=2000,
              system=_shortlist_chat_system(views, analysis), messages=messages)
    try:
        stream_cm = client.messages.stream(inference_geo=INFERENCE_GEO, **kw)
    except TypeError:
        stream_cm = client.messages.stream(
            extra_body={"inference_geo": INFERENCE_GEO}, **kw)
    with stream_cm as stream:
        for text in stream.text_stream:
            yield text


# ── Per-artist chat (streaming) ──────────────────────────────────────────────

def chat_stream(artist_view: dict, history: list[dict], user_msg: str):
    """Yields text chunks. MOCK mode yields a clearly-labelled placeholder."""
    if not is_live():
        yield ("[Preview mode] The AI assistant is off. Once Claude is enabled "
               "I'll answer your question about "
               f"{artist_view.get('name', 'this artist')} here.")
        return

    client = _client()
    messages = list(history) + [{"role": "user", "content": user_msg}]
    kw = dict(model=MODEL, max_tokens=4000,
              system=_chat_system(artist_view), messages=messages)
    try:
        stream_cm = client.messages.stream(inference_geo=INFERENCE_GEO, **kw)
    except TypeError:
        stream_cm = client.messages.stream(
            extra_body={"inference_geo": INFERENCE_GEO}, **kw)
    with stream_cm as stream:
        for text in stream.text_stream:
            yield text


# ── CLI self-test (a safe way to verify your token + where inference runs) ───

def _selftest() -> int:
    st = compliance_status()
    print(f"Mode: {st['mode']}  |  model: {st['model']}  |  "
          f"inference_geo: {st['inference_geo']}  |  {st['reason']}")
    sample = [{
        "artist_id": "demo", "artist_name": "Demo DJ", "genres": ["tech house"],
        "career_status": "mid", "spotify_listeners": 120000,
        "momentum": 78, "growth": 81, "market_relevance": 55,
        "future_potential": 84, "confidence": 90, "forecast_90d": 64.0,
        "waarom": "Sterke groeiversnelling, kerngenre (tech house).", "rank": 80,
    }]
    out = generate_rationales(sample)
    print("Rationale:", out.get("demo"))
    if is_live():
        print("Live call OK — check the Anthropic Console usage to confirm geo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())

# agents/ — LLM reasoning core (Phase 2)

`agents/core.py` is the **single egress point** to Claude. Everything it sends is
minimised by `to_model_view()`, and real inference is **off by default**.

## Mode: mock vs live
- **Mock (default)** — no network calls. `generate_rationales()` returns the
  deterministic `waarom`/`rank`; `chat_stream()` yields a labelled placeholder.
  The whole UI is visualisable with zero data egress.
- **Live** — set `LOFI_LLM_ENABLED=1`. Only do this once **both** are true:
  1. Lofi's *amended* written permission for US inference under Anthropic's
     DPA + EU SCCs (Path C) is in hand, and
  2. zero-data-retention is enabled on the Anthropic account/workspace.

> The direct Anthropic API cannot keep data in the EEA (`inference_geo` is
> `us`/`global` only; workspace geo is `us`-only). That's why live mode is gated
> behind an explicit flag and an amended Lofi approval — see `docs/agent_design.md`.

## Env
| Var | Meaning |
|---|---|
| `ANTHROPIC_API_KEY` | required for live mode (put it in `.env`, never in chat) |
| `LOFI_LLM_ENABLED` | `1`/`true` to allow real calls (default off) |
| `LOFI_LLM_MODEL` | default `claude-opus-4-8` |
| `LOFI_LLM_INFERENCE_GEO` | default `us` (deterministic, documentable for SCCs) |

## Verify your setup safely
```bash
python -m agents.core            # prints mode + a sample rationale
# mock:  "Mode: mock … (voorbeeldmodus)"
# live:  one real call; check usage.inference_geo + the Anthropic Console
```

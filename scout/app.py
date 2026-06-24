"""
Standalone entrypoint for the Scout page (Phase 1, no LLM).

    streamlit run scout/app.py

In the live dashboard this page is integrated via the nav instead (add "Scout"
to the sidebar radio in lofi_pipeline.py and call render_scout_page()).
"""
import sys
from pathlib import Path

# Ensure the repo root is importable when run via `streamlit run scout/app.py`.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except Exception:
    pass

from scout.page import render_scout_page  # noqa: E402

st.set_page_config(page_title="LOFI Scout", layout="wide")
render_scout_page()

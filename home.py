"""
FOOTNOTE–SOURCE VERIFIER (LAWS90286 Assignment 2) - entry point.

Upload a draft legal paper (.docx with real Word footnotes) and PDFs of the
cases it cites. For every footnote the app checks whether the cited source,
at the cited pinpoint, actually supports the proposition - and says why not.

    streamlit run home.py
"""

# Chroma needs a newer sqlite3 than Streamlit Community Cloud provides (Lab 2 README fix).
# This must run before anything imports chromadb. pysqlite3-binary is only installed
# on Linux (see requirements.txt), so on a Mac or Windows laptop this is skipped.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from reusable_code.ai_client import DEFAULT_MODEL, EFFORT_LEVELS, AIError
from reusable_code.api_key import api_key_module, require_api_key
from reusable_code.password_gate import require_password
from reusable_code.docx_footnotes import extract_draft
from reusable_code.pdf_ingest import guess_citation, ingest_pdf
from reusable_code.pipeline import run_verification
from reusable_code.results_view import show_results

load_dotenv()   # local development only: reads OPENAI_API_KEY from .env if present
st.set_page_config(page_title="Footnote–Source Verifier", page_icon="📚", layout="wide")
require_password()   # optional shared password (only if APP_PASSWORD is set in Streamlit secrets)

ARTEFACTS = Path(__file__).parent / "test_artefacts"
TEST_CASES = {
    "TC1 · Accurate citations (strengths)": "TC1_accurate_citations.docx",
    "TC2 · Seeded citation errors": "TC2_seeded_errors.docx",
    "TC3 · Limitations": "TC3_limitations.docx",
}


# ---------------------------------------------------------------------------
# Sidebar: credentials module + settings
# ---------------------------------------------------------------------------
def settings_sidebar() -> dict:
    st.sidebar.divider()
    st.sidebar.header("Settings")
    settings = {
        "model": st.sidebar.text_input("Model", value=DEFAULT_MODEL),
        "effort": st.sidebar.select_slider("Reasoning effort (judge and reviewer)", EFFORT_LEVELS, value="medium"),
        "max_footnotes": st.sidebar.number_input("Max footnotes to check (cost control)", 1, 100, 20),
        "max_turns": st.sidebar.slider("Max reflection turns per citation", 1, 3, 2),
        "check_pinpoints": st.sidebar.toggle("Check pinpoints", value=True),
        "suggest_fixes": st.sidebar.toggle("Suggest fixes", value=True),
        "exclude_cited_by": st.sidebar.toggle(
            "Experimental: exclude Jade 'cited by' annotations", value=False,
            help="BarNet Jade inserts quotations from LATER cases into its PDFs ('Following paragraph cited by:' "
                 "blocks and a trailing 'Cited by:' index). By default the tool trusts the file as given. "
                 "Turn this on to strip those annotations (a layout heuristic; see Test Case 3)."),
    }
    st.session_state.settings = settings
    with st.sidebar.expander("How it works"):
        st.markdown(
            "1. Parse the draft's footnotes and the PDFs' numbered paragraphs (code).\n"
            "2. AI identifies each source, extracts each claim and resolves each citation.\n"
            "3. For each citation: fetch the pinpoint, search the source, AI judges content, "
            "attribution, strength and location.\n"
            "4. Code checks every quotation and pinpoint against the source text.\n"
            "5. A reviewer AI loops until the verdict is well founded (hard stop).\n\n"
            "See **How it works** in the page list for details.")
    return settings


# ---------------------------------------------------------------------------
# Parsing (steps 1 and 2) - cached, so reruns don't re-parse the same bytes
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading the draft's footnotes...")
def parse_draft(data: bytes) -> dict:
    return extract_draft(data)


@st.cache_data(show_spinner="Reading the PDF and numbering its paragraphs...")
def parse_source(data: bytes, filename: str, exclude_cited_by: bool) -> dict:
    return ingest_pdf(data, filename, exclude_cited_by=exclude_cited_by)


def _load_inputs():
    """Return (draft_name, draft_bytes, [(pdf_name, pdf_bytes), ...]) from uploads or a bundled test case."""
    mode = st.radio("Input", ["Upload my own files", "Load a bundled test case"], horizontal=True)
    if mode == "Load a bundled test case":
        choice = st.selectbox("Test case (all three source PDFs are loaded with each)", list(TEST_CASES))
        draft_path = ARTEFACTS / "drafts" / TEST_CASES[choice]
        pdfs = [(p.name, p.read_bytes()) for p in sorted((ARTEFACTS / "sources").glob("*.pdf"))]
        st.caption(f"Loaded `{draft_path.name}` and {len(pdfs)} PDFs from `test_artefacts/`. "
                   "Expected results are on the **Test cases** page.")
        return draft_path.name, draft_path.read_bytes(), pdfs
    c1, c2 = st.columns(2)
    draft_file = c1.file_uploader("Draft paper (.docx with Word footnotes)", type=["docx"])
    pdf_files = c2.file_uploader("Cited sources (PDF, text-searchable)", type=["pdf"], accept_multiple_files=True)
    if not draft_file:
        return None, None, [(f.name, f.getvalue()) for f in pdf_files or []]
    return draft_file.name, draft_file.getvalue(), [(f.name, f.getvalue()) for f in pdf_files or []]


def _unique_ids(sources: list) -> list:
    """Two files with the same name would share a source_id; make each one unique."""
    seen = {}
    for s in sources:
        n = seen.get(s["source_id"], 0)
        seen[s["source_id"]] = n + 1
        if n:
            s["source_id"] = f"{s['source_id']}_{n + 1}"
    return sources


def _preview(draft: dict, sources: list):
    st.markdown("#### Ingestion preview (no AI used yet)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Footnotes found", len(draft["footnotes"]))
    c2.metric("Source PDFs", len(sources))
    c3.metric("Readable PDFs", sum(s["readable"] for s in sources))
    rows = [{"file": s["filename"],
             "citation on first pages (regex)": guess_citation(s) or "—",
             "pages": s["n_pages"],
             "paragraphs detected": len([p for p in s["paragraphs"] if p["para"]]),
             "numbering": s["numbering_style"],
             "readable": "✓" if s["readable"] else "✗ scanned / no text",
             "cited-by lines removed": s["excluded_cited_by_lines"]} for s in sources]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for s in sources:
        for w in s["warnings"]:
            st.warning(f"{s['filename']}: {w}")
    with st.expander("Footnotes as read from the .docx"):
        for f in draft["footnotes"]:
            st.markdown(f"**{f['fn']}.** {f['text']}")
            if f["quoted_phrases"]:
                st.caption("Quoted in the text: " + " · ".join(f"‘{q}’" for q in f["quoted_phrases"]))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
api_key_module()
settings = settings_sidebar()

st.title("📚 Footnote–Source Verifier")
st.write(
    "Checks whether each footnoted proposition in a draft legal paper is actually supported by the cited source, "
    "at the cited pinpoint - the check a careful editor or supervising lawyer would do. It looks for overstated "
    "holdings, words attributed to the wrong judge or speaker, wrong pinpoints and quotations that are not in the "
    "source (errors that generative AI makes easy to introduce). The AI judges only from the uploaded sources; "
    "quotations and pinpoints are verified by code against the source text."
)
require_api_key()

draft_name, draft_bytes, pdfs = _load_inputs()
if not draft_bytes:
    st.info("Upload a draft (.docx) and the PDFs it cites, or load a bundled test case.")
    st.stop()

draft = parse_draft(draft_bytes)
sources = _unique_ids([parse_source(data, name, settings["exclude_cited_by"]) for name, data in pdfs])
_preview(draft, sources)

if not draft["footnotes"]:
    st.error("No Word footnotes were found in this .docx. The tool needs real footnotes (References → Insert Footnote).")
    st.stop()
if not sources:
    st.warning("No source PDFs uploaded: every citation will be CANNOT_DETERMINE. You can still run the check.")

# A signature of the inputs, so old results are hidden if the files change.
signature = hashlib.sha256(draft_bytes + b"".join(d for _, d in pdfs) + str(settings).encode()).hexdigest()
if st.button("Run verification", type="primary"):
    try:
        st.session_state.run = run_verification(draft_name, draft, sources, settings)
        st.session_state.run_signature = signature
    except AIError as error:
        st.error(str(error))

run = st.session_state.get("run")
if run:
    if st.session_state.get("run_signature") != signature:
        st.caption("These results are from a previous run with different files or settings. Click **Run verification** to update.")
    show_results(run)

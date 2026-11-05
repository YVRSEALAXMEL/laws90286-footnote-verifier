"""Test cases: what to load, what to expect, and downloads of every test artefact."""

from pathlib import Path

import pandas as pd
import streamlit as st

from reusable_code.api_key import api_key_module
from reusable_code.password_gate import require_password

st.set_page_config(page_title="Test cases", page_icon="🧪", layout="wide")
require_password()
api_key_module()

ARTEFACTS = Path(__file__).resolve().parent.parent / "test_artefacts"

st.title("🧪 Test cases")
st.write("On the home page, choose **Load a bundled test case** (or upload the files below yourself), keep the default "
         "settings (model gpt-6.1-sol, effort medium, 2 reflection turns, pinpoints and fixes on, Jade toggle off) "
         "and click **Run verification**. Ground truth comes from `test_artefacts/ANSWER_KEY.md`, checked by hand.")

CASES = {
    "TC1 · Strengths: accurate citations": {
        "file": "TC1_accurate_citations.docx",
        "shows": "Footnote extraction; Ibid and (n X); a parallel citation (62 VR 491 ↔ [2020] VSCA 212); ranges and "
                 "multiple pinpoints; three paragraph-numbering formats.",
        "rows": [(1, "Brown (2020) 62 VR 491, [1]–[3]", "SUPPORTED"), (2, "Ibid [6], [29]", "SUPPORTED"),
                 (3, "Ibid [28]–[29]", "SUPPORTED"), (4, "Brown (n 1) [6], [56]–[58], [68]–[69]", "SUPPORTED"),
                 (5, "Kapua [2023] NSWCCA 14, [111]", "SUPPORTED"), (6, "Lojovic [2025] ONCA 319, [42]–[44]", "SUPPORTED"),
                 (7, "Kapua (n 5) [113]", "SUPPORTED")],
    },
    "TC2 · Catching errors: five planted errors and a control": {
        "file": "TC2_seeded_errors.docx",
        "shows": "Misattribution (only reasoning about WHO said it catches fn 1), overstatement, a wrong pinpoint, a "
                 "fabricated quotation, a quotation from the wrong case, and a control that must pass.",
        "rows": [(1, "Lojovic [42]–[44] - trial judge's words attributed to the Court of Appeal", "MISATTRIBUTED"),
                 (2, "Kapua [112]–[113] - 'never' vs 'may, or may not'", "OVERSTATED"),
                 (3, "Kapua (n 2) [113] - quote is at [111]", "WRONG_PINPOINT"),
                 (4, "Brown [6] - 'ordinarily operate to reduce…' is not in Brown", "UNSUPPORTED"),
                 (5, "Brown (n 4) [35] - words are from Kapua [113]", "UNSUPPORTED (wrong case)"),
                 (6, "Ibid [52] - control", "SUPPORTED")],
    },
    "TC3 · Limitations": {
        "file": "TC3_limitations.docx",
        "shows": "Where the tool cannot (or should not) give a confident answer, including one expected FAILURE.",
        "rows": [(1, "Brown [6] - 'exceptional' appears only in Jade's cited-by blocks", "UNSUPPORTED (tool expected to pass it: limitation)"),
                 (2, "R v Verdins (2007) 16 VR 269 - not uploaded", "CANNOT_DETERMINE (must not answer from memory)"),
                 (3, "Brown (n 1) 493 - a VR page pinpoint", "SUPPORTED; pinpoint unverifiable"),
                 ("4.1", "Brown (n 1) [81]–[82]", "SUPPORTED"),
                 ("4.2", "Strawson (1962) - not uploaded", "CANNOT_DETERMINE; 'language close to…' is the author's own argument")],
    },
}

for title, case in CASES.items():
    st.subheader(title)
    st.write(case["shows"])
    table = pd.DataFrame(case["rows"], columns=["fn", "citation / issue", "expected"]).astype({"fn": str})
    st.dataframe(table, hide_index=True, width="stretch")
    path = ARTEFACTS / "drafts" / case["file"]
    st.download_button(f"⬇️ {case['file']}", path.read_bytes(), file_name=case["file"], on_click="ignore",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

st.subheader("Source PDFs (upload all three with each test case)")
cols = st.columns(3)
for col, pdf in zip(cols, sorted((ARTEFACTS / "sources").glob("*.pdf"))):
    col.download_button(f"⬇️ {pdf.name}", pdf.read_bytes(), file_name=pdf.name, mime="application/pdf",
                        on_click="ignore", width="stretch")
st.caption("BarNet Jade and CanLII copies, included for educational testing only. TC3(1) can also be run with the "
           "experimental 'exclude Jade cited-by annotations' toggle ON to compare.")

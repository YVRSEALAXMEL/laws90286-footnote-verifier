"""
STEP 3 - IDENTIFY THE SOURCE (1 AI call per uploaded PDF)

The AI reads the first pages of each PDF and returns the case name,
medium-neutral citation, any reported citations PRINTED in the file, the
court, the date and the paragraph-numbering style, as strict JSON.

Rule: copy what is printed; never add a citation from memory. (Our Brown
file shows only "[2020] VSCA 212"; the "(2020) 62 VR 491" report citation
must NOT be invented here. Recognising that the two are the same case is
step 5's job, and step 5 has to show its reasoning.)

The result is shown to the user so they can check it before trusting the run.
"""

from reusable_code.ai_client import ask_ai

IDENTIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "case_name": {"type": "string", "description": "Parties as printed, e.g. 'Brown v The Queen'"},
        "medium_neutral_citation": {"type": ["string", "null"], "description": "e.g. '[2020] VSCA 212' or '2025 ONCA 319'"},
        "reported_citations": {"type": "array", "items": {"type": "string"},
                               "description": "Only report citations printed as THIS document's citation"},
        "court": {"type": "string"},
        "decision_date": {"type": "string"},
        "judges": {"type": "array", "items": {"type": "string"}},
        "document_type": {"type": "string", "enum": ["judgment", "legislation", "article_or_book", "other"]},
        "paragraph_numbering_style": {"type": "string", "enum": ["N.", "[N]", "none", "other"]},
        "short_title": {"type": "string", "description": "A short title, e.g. 'Brown'"},
        "notes": {"type": "string"},
    },
    "required": ["case_name", "medium_neutral_citation", "reported_citations", "court", "decision_date",
                 "judges", "document_type", "paragraph_numbering_style", "short_title", "notes"],
    "additionalProperties": False,
}

IDENTIFY_INSTRUCTIONS = """You identify legal source documents from their first pages.
Return the details of THIS document only - not of cases it cites or that cite it.
Copy names and citations exactly as printed. Never add a citation, date or judge from your own memory:
if something is not printed on these pages, leave it empty (or null). reported_citations must contain only
report citations printed as this document's own citation; return an empty list if there are none.
Publisher furniture (BarNet Jade, CanLII, AustLII) is not part of the case name."""


def identify_source(source: dict, effort: str = "low") -> dict:
    """Ask the AI who/what this PDF is. Unreadable PDFs are not sent."""
    if not source["readable"]:
        return {"case_name": source["filename"], "medium_neutral_citation": None, "reported_citations": [],
                "court": "", "decision_date": "", "judges": [], "document_type": "other",
                "paragraph_numbering_style": "none", "short_title": "",
                "notes": "Unreadable PDF (no text layer) - not identified."}
    prompt = (
        f"File name: {source['filename']}\n"
        f"Pages: {source['n_pages']}\n"
        f"Paragraph numbering detected by the parser: {source['numbering_style']}\n\n"
        f"First pages (publisher boilerplate removed):\n{source['first_pages_text']}"
    )
    return ask_ai(prompt, instructions=IDENTIFY_INSTRUCTIONS, effort=effort, schema=IDENTIFY_SCHEMA,
                  step=f"3 identify source: {source['filename']}")


def source_label(identity: dict, source: dict) -> str:
    """A human label such as 'Brown v The Queen [2020] VSCA 212'."""
    name = (identity or {}).get("case_name") or source["filename"]
    mnc = (identity or {}).get("medium_neutral_citation") or ""
    return f"{name} {mnc}".strip()

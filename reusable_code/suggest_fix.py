"""
STEP 10 - SUGGEST A FIX (1 AI call per flagged citation unit; can be switched off)

For anything not SUPPORTED, the AI proposes the smallest correction a careful
editor would make: a corrected pinpoint, softened wording, the correct
attribution, or removing / replacing a quotation that is not in the source.

The fix must be grounded in what was found: a corrected pinpoint may only use
paragraphs the code actually located, and replacement wording must come from
the verified supporting passage.

CANNOT_DETERMINE needs no AI: the fix is simply "upload the source" or
"upload a text-searchable copy", so it is written in code.
"""

from reusable_code.ai_client import ask_ai
from reusable_code.grounding import format_paras

FIX_TYPES = ["correct_pinpoint", "soften_wording", "correct_attribution", "remove_or_replace_quotation",
             "cite_correct_source", "upload_source", "no_change_needed"]

FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "fix_type": {"type": "string", "enum": FIX_TYPES},
        "suggested_sentence": {"type": "string", "description": "The corrected body text ('' if unchanged)"},
        "suggested_footnote": {"type": "string", "description": "The corrected footnote in AGLC style ('' if unchanged)"},
        "explanation": {"type": "string"},
    },
    "required": ["fix_type", "suggested_sentence", "suggested_footnote", "explanation"],
    "additionalProperties": False,
}

FIX_INSTRUCTIONS = """You are a legal editor proposing the smallest correction that makes a footnoted sentence accurate.
Use ONLY the verified evidence given. Keep the author's argument where you can; change only what is wrong:
- WRONG_PINPOINT: correct the pinpoint to the located paragraph(s).
- OVERSTATED: soften the wording to match the source's actual strength (e.g. "may" rather than "must").
- MISATTRIBUTED: attribute the words or view to the person who actually said it (e.g. "the trial judge reasoned ...").
- UNSUPPORTED: remove or replace a quotation that is not in the source; if the words belong to another uploaded
  source, cite that source instead (with its paragraph). Do not invent a quotation or a pinpoint.
Write the footnote in AGLC 4th edition style. Explain the change in one or two sentences."""


def deterministic_fix(record: dict, source) -> dict:
    if source is None:
        return {"fix_type": "upload_source", "suggested_sentence": "", "suggested_footnote": "",
                "explanation": f"Upload a copy of {record['full_citation']} and re-run to verify this citation."}
    return {"fix_type": "upload_source", "suggested_sentence": "", "suggested_footnote": "",
            "explanation": f"{source['filename']} has no text layer. Upload a text-searchable copy (or run OCR) and re-run."}


def suggest_fix(record: dict, source: dict, effort: str = "low") -> dict:
    judgement = record.get("judgement") or {}
    support = record.get("support_check") or {}
    pinpoint = record.get("pinpoint_check") or {}
    elsewhere = [f"‘{q['phrase']}’ is in {o['label']} {format_paras(o['paras'])}"
                 for q in record["quote_results"] for o in q["found_elsewhere"]]
    prompt = (
        f"DRAFT SENTENCE: {record['sentence']}\n"
        f"FOOTNOTE: {record['citation_text']}  (resolved: {record['full_citation']})\n"
        f"VERDICT: {record['verdict']} - {record['explanation']}\n"
        f"WHO IS SPEAKING IN THE SOURCE: {judgement.get('who_is_speaking', '?')} ({judgement.get('speaker_detail', '')})\n"
        f"VERIFIED SUPPORTING PASSAGE: {judgement.get('supporting_quote', '') if support.get('found') else '(none verified)'}\n"
        f"LOCATED AT: {format_paras(support.get('paras', []))}\n"
        f"PINPOINT CHECK: {pinpoint.get('detail', '')}\n"
        f"QUOTED PHRASES FOUND IN OTHER UPLOADED SOURCES: {elsewhere or 'none'}\n"
        f"AI REASONING: {judgement.get('reasoning', '')}\n"
        f"CITED SOURCE: {source['label']}"
    )
    return ask_ai(prompt, instructions=FIX_INSTRUCTIONS, effort=effort, schema=FIX_SCHEMA,
                  step=f"10 suggest fix (fn {record['unit_id']})")

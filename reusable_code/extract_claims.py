"""
STEP 4 - EXTRACT CLAIMS (1 AI call for the whole draft)

For each footnote, the AI identifies the proposition in the body text that
the footnote is cited for, who the draft says made or held it, and any
author's own argument mixed into the sentence (which is NOT a cited
proposition and should not be checked against the source).

Two safeguards in code:
  * Every quoted phrase the AI returns must appear VERBATIM in the draft;
    anything else is dropped (the AI must not "tidy" a quotation).
  * Quoted phrases are also found deterministically (docx_footnotes.py) in
    the footnote's span, so a phrase the AI missed is still checked.
"""

from reusable_code.ai_client import ask_ai
from reusable_code.grounding import normalise

CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fn": {"type": "integer"},
                    "sentence": {"type": "string", "description": "The sentence(s) the footnote marker attaches to, verbatim"},
                    "proposition": {"type": "string", "description": "What the cited source(s) are said to establish, keeping the draft's strength words"},
                    "attributed_to": {"type": "string", "description": "Who the draft says said/held it, e.g. 'the Court of Appeal (holding)', 'the trial judge', 'the defence expert'"},
                    "quoted_phrases": {"type": "array", "items": {"type": "string"}},
                    "authorial_comment": {"type": "string", "description": "The author's own argument or characterisation in the sentence, if any; '' if none"},
                },
                "required": ["fn", "sentence", "proposition", "attributed_to", "quoted_phrases", "authorial_comment"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

CLAIMS_INSTRUCTIONS = """You are a legal editor reading a law research paper. Footnote markers appear in the body text as [^n].
Under AGLC, a footnote supports the text that precedes it, back to the previous footnote marker or the start of the sentence.

For EVERY footnote listed, return one claim:
- sentence: the sentence(s) the marker attaches to, copied verbatim (without the [^n] markers).
- proposition: what the cited source(s) are relied on to establish. Keep the draft's own strength and attribution words
  ("held", "never", "must", "the trial judge treated...") - do not soften or correct them; we are checking them.
- attributed_to: who the draft says said, found or held it (e.g. "the Court of Appeal (holding)", "the sentencing judge",
  "the defence expert", "the five-member bench", "Strawson"). If the draft does not say, write "the source".
- quoted_phrases: every phrase in quotation marks in that span, copied EXACTLY character for character, without the quote marks.
- authorial_comment: any part of the sentence that is the author's own argument, evaluation or comparison rather than something
  attributed to the source (e.g. "language close to X's invitation"). Use "" if there is none.
Do not judge whether the claims are correct."""


def extract_claims(draft: dict, footnotes: list, effort: str = "medium") -> list:
    """Return one claim per footnote (in footnote order), validated against the draft."""
    footnote_list = "\n".join(f"[^{f['fn']}] {f['text']}" for f in footnotes)
    prompt = f"Body text:\n{draft['body_text']}\n\nFootnotes to process:\n{footnote_list}"
    result = ask_ai(prompt, instructions=CLAIMS_INSTRUCTIONS, effort=effort, schema=CLAIMS_SCHEMA,
                    step="4 extract claims")
    by_fn = {c["fn"]: c for c in result["claims"]}

    body_norm = normalise(draft["body_text"])
    claims = []
    for f in footnotes:
        claim = by_fn.get(f["fn"]) or {
            # The AI skipped this footnote: fall back to the deterministic span.
            "fn": f["fn"], "sentence": f["span"], "proposition": f["span"], "attributed_to": "the source",
            "quoted_phrases": [], "authorial_comment": "",
        }
        claim = dict(claim)
        ai_phrases = claim["quoted_phrases"]
        verified = [q for q in ai_phrases if normalise(q) and normalise(q) in body_norm]
        dropped = [q for q in ai_phrases if q not in verified]
        # Merge with the deterministic phrases from the span (no duplicates).
        merged = list(f["quoted_phrases"])
        for q in verified:
            if not any(normalise(q) == normalise(m) or normalise(q) in normalise(m) for m in merged):
                merged.append(q)
        claim["quoted_phrases"] = merged
        claim["quote_extraction_check"] = {
            "ai_phrases_verified": len(verified),
            "ai_phrases_dropped": dropped,   # not verbatim in the draft
            "from_span": f["quoted_phrases"],
        }
        claim["footnote_text"] = f["text"]
        claims.append(claim)
    return claims

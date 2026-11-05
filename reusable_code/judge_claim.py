"""
STEP 7 - JUDGE THE CLAIM (1 AI call per citation unit per turn)

The AI compares the proposition (and any quotation) with the evidence
excerpts and reports, as strict JSON, its findings on four ELEMENTS -
the questions a careful supervising lawyer asks of every footnote:

    1. CONTENT      does the source state the proposition?       full / partial / none
    2. ATTRIBUTION  who is actually speaking in that passage?     same / different
    3. STRENGTH     same force? ("may" vs "must", obiter vs held) same / draft_stronger / ...
    4. LOCATION     which paragraph is the best supporting passage in?

plus a verbatim supporting quote (checked by code in step 8), a reasoning
note, an overall verdict and a confidence score.

The code (verdict_rules.verdict_from_elements) then derives the verdict from
the elements, and flags any inconsistency with the AI's own verdict.

"Judge only from the excerpts; never from memory" is the key instruction:
the tool must not "verify" a citation from what the model thinks the case says.
"""

from reusable_code.ai_client import ask_ai
from reusable_code.retrieve_evidence import evidence_label
from reusable_code.verdict_rules import VERDICT_NAMES, WHO_IS_SPEAKING, definitions_text, speakers_text

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "draft_attribution": {"type": "string", "description": "Who the draft says said/held it"},
        "who_is_speaking": {"type": "string", "enum": list(WHO_IS_SPEAKING)},
        "speaker_detail": {"type": "string", "description": "e.g. 'the trial judge, as described by the Court of Appeal'"},
        "content_match": {"type": "string", "enum": ["full", "partial", "none"]},
        "attribution_match": {"type": "string", "enum": ["same", "different", "not_applicable"]},
        "strength_match": {"type": "string", "enum": ["same", "draft_stronger", "draft_weaker", "not_applicable"]},
        "found_at_para": {"type": ["integer", "null"]},
        "supporting_quote": {"type": "string", "description": "Copied exactly from ONE excerpt"},
        "verdict": {"type": "string", "enum": VERDICT_NAMES},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["draft_attribution", "who_is_speaking", "speaker_detail", "content_match", "attribution_match",
                 "strength_match", "found_at_para", "supporting_quote", "verdict", "reasoning", "confidence"],
    "additionalProperties": False,
}

JUDGE_INSTRUCTIONS = f"""You are a meticulous legal editor checking ONE footnote citation in a law research paper against
excerpts from the cited judgment. Judge ONLY from the excerpts provided. Never use your own knowledge of the case
or the law: if the excerpts do not show it, it is not supported, even if you believe it to be true.

Work through the elements in order, as a careful supervising lawyer would:
1. CONTENT - does the source state the proposition the draft cites it for? "full", "partial" (only part of it), or "none".
   Judge only the proposition attributed to this source; ignore the author's own commentary.
2. ATTRIBUTION - who is speaking in the passage that contains it? An appellate judgment routinely DESCRIBES the
   trial judge's reasons, SUMMARISES counsel's submissions, REPORTS expert evidence and QUOTES other cases before giving
   its own view; words appearing in the judgment are not necessarily the court's own words or holding. Watch for
   signals such as "the trial judge said", "it was submitted", "in Dr X's opinion", indented block quotes and quotation
   from other cases. Compare with who the draft attributes it to ("the Court held" = the deciding court's own holding).
   Speakers:
{speakers_text()}
   attribution_match: "same", "different" or "not_applicable" (only if content is "none").
3. STRENGTH - compare modal and quantifier words and the status of the statement: "may" vs "must", "can" vs
   "never"/"always", "likely" vs "will", "not irrelevant" vs "decisive", a discretionary consideration vs a rule,
   a reservation/obiter vs a holding. "draft_stronger" if the draft says more than the source; "draft_weaker" if less.
4. LOCATION - found_at_para: the paragraph number of the best supporting passage, from the excerpt label
   (null if there is none).

supporting_quote: copy ONE short passage (one or two sentences, at most about 60 words) EXACTLY, character for character,
from ONE excerpt - it will be checked by computer against the source text. Do not stitch passages together, paraphrase
or add ellipses. If nothing supports the claim, quote the passage that comes closest (or "" if none is relevant).

verdict - use these definitions:
{definitions_text()}
Your verdict must follow from your element findings. confidence: 0 to 1."""


def build_judge_prompt(unit: dict, claim: dict, source_short: str, source_label: str, evidence: list,
                       quote_results: list, work_record: str) -> str:
    cited = unit["pinpoint"].get("paras", [])
    quote_lines = []
    for q in quote_results:
        where = f"found at {q['paras']}" if q["found"] else "NOT found in the source"
        quote_lines.append(f"   ‘{q['phrase']}’ - computer check: {where}")
    excerpts = "\n\n".join(f"{evidence_label(source_short, e, cited)}\n{e['text']}" for e in evidence)
    return (
        f"DRAFT SENTENCE: {claim.get('sentence', '')}\n"
        f"PROPOSITION THIS SOURCE IS CITED FOR: {unit['unit_proposition'] or claim.get('proposition', '')}\n"
        f"DRAFT ATTRIBUTES IT TO: {claim.get('attributed_to', 'the source')}\n"
        f"AUTHOR'S OWN COMMENT (not to be verified): {claim.get('authorial_comment') or '(none)'}\n"
        f"CITATION: {unit['citation_text']}  ->  {unit['full_citation']}\n"
        f"CITED PINPOINT: {unit['pinpoint_text'] or '(none)'} ({unit['pinpoint']['type']} pinpoint)\n"
        f"QUOTED PHRASES:\n" + ("\n".join(quote_lines) or "   (none)") + "\n\n"
        f"CITED SOURCE: {source_label}\n"
        f"EXCERPTS (labelled with paragraph number and how each was found):\n\n{excerpts}\n\n"
        f"WORK RECORD OF EARLIER TURNS:\n{work_record or '(first turn)'}"
    )


def judge_claim(unit, claim, source_short, source_label, evidence, quote_results, work_record, effort="medium"):
    prompt = build_judge_prompt(unit, claim, source_short, source_label, evidence, quote_results, work_record)
    judgement = ask_ai(prompt, instructions=JUDGE_INSTRUCTIONS, effort=effort, schema=JUDGE_SCHEMA,
                       step=f"7 judge claim (fn {unit['unit_id']})")
    judgement["confidence"] = max(0.0, min(1.0, float(judgement["confidence"])))
    return judgement

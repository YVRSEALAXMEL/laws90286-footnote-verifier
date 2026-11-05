"""
VERDICT LOGIC - the legal reasoning, written down in one place.

This module is the heart of the "legal logic". It holds:

  1. VERDICTS: precise definitions, used in the AI instructions, the UI and
     the README (so all three always say the same thing).
  2. WHO_IS_SPEAKING: the speakers a judgment can contain. An appellate
     judgment routinely DESCRIBES the trial judge's reasons, SUMMARISES
     counsel's submissions and QUOTES other cases before it decides anything.
     Words being in the judgment does not make them the court's holding.
  3. verdict_from_elements(): a Lab 1-style if-then gate. The AI reports
     element findings (content / attribution / strength); the CODE turns
     them into a verdict, so the rule is visible and auditable.
  4. decide_verdict(): the rule-based PRECEDENCE that combines the AI's
     findings with the deterministic grounding checks (grounding.py).
     Where code can establish a fact (a quote is not in the source), the
     code wins over the AI.
"""

from reusable_code.grounding import format_paras

VERDICTS = {
    "SUPPORTED": (
        "The source at (or within one paragraph of) the pinpoint states the proposition, "
        "with the same strength and the same attribution."
    ),
    "OVERSTATED": (
        "The source supports only a weaker, qualified or partial version of the proposition "
        "(e.g. 'may' presented as 'must', 'can' as 'never', a discretionary consideration as a rule, "
        "obiter or a reservation as a holding)."
    ),
    "MISATTRIBUTED": (
        "The words or proposition are in the source but belong to a different speaker (the trial "
        "judge being described, counsel's submission, a dissent, an expert, another case being "
        "quoted), or to a different uploaded source."
    ),
    "WRONG_PINPOINT": (
        "The proposition is supported, but at a different paragraph from the one cited "
        "(more than one paragraph away)."
    ),
    "UNSUPPORTED": (
        "The source does not support the proposition, or a quoted phrase does not appear in it "
        "(a fabricated or altered quotation)."
    ),
    "CANNOT_DETERMINE": (
        "The tool cannot check this citation: the source was not provided, the source is "
        "unreadable (no text layer), or the evidence could not be verified."
    ),
}
VERDICT_NAMES = list(VERDICTS)

# Colours for the results table (traffic-light style).
VERDICT_COLOURS = {
    "SUPPORTED": "#d4edda",
    "OVERSTATED": "#fff3cd",
    "MISATTRIBUTED": "#f8d7da",
    "WRONG_PINPOINT": "#ffe5b4",
    "UNSUPPORTED": "#f5c6cb",
    "CANNOT_DETERMINE": "#e2e3e5",
}

# How serious each verdict is, for ordering the report (higher = worse).
SEVERITY = {"UNSUPPORTED": 5, "MISATTRIBUTED": 4, "OVERSTATED": 3, "WRONG_PINPOINT": 2,
            "CANNOT_DETERMINE": 1, "SUPPORTED": 0}

WHO_IS_SPEAKING = {
    "court_or_majority": "The court deciding the case (or the majority / leading judgment), in its own voice.",
    "concurring_judge": "A judge agreeing in the result, in separate reasons.",
    "dissenting_judge": "A judge in dissent.",
    "lower_court_judge_described": "The trial / sentencing / primary judge, whose reasons the court is describing or quoting.",
    "counsel_or_party_submission": "A submission or argument of counsel or a party.",
    "expert_or_witness_evidence": "Evidence of an expert or witness, as reported in the judgment.",
    "another_case_quoted": "A different judgment that is quoted or summarised.",
    "secondary_source_quoted": "Legislation, a practice note, an article or a textbook being quoted.",
    "editorial_or_headnote": "Catchwords, headnote or publisher material - not part of the reasons.",
    "unclear": "The speaker cannot be determined from the excerpt.",
}

CONFIDENCE_THRESHOLD = 0.6   # below this, the reflection loop takes another turn


def definitions_text() -> str:
    """The verdict definitions as a bulleted block for AI instructions."""
    return "\n".join(f"- {name}: {meaning}" for name, meaning in VERDICTS.items())


def speakers_text() -> str:
    return "\n".join(f"- {name}: {meaning}" for name, meaning in WHO_IS_SPEAKING.items())


# ---------------------------------------------------------------------------
# Lab 1-style gate: element findings -> verdict
# ---------------------------------------------------------------------------
def verdict_from_elements(judgement: dict) -> tuple:
    """
    Apply the elements in order, stopping at the first one that fails
    ("if an element is not satisfied, stop and explain why").
    Returns (verdict, explanation).
    """
    if judgement["content_match"] == "none":
        return "UNSUPPORTED", "Element 1 (content) fails: the excerpts do not state the proposition."
    if judgement["attribution_match"] == "different":
        return "MISATTRIBUTED", (
            f"Element 2 (attribution) fails: the draft attributes this to {judgement['draft_attribution'] or 'the court'}, "
            f"but in the source it is {judgement['who_is_speaking'].replace('_', ' ')}"
            f"{' (' + judgement['speaker_detail'] + ')' if judgement.get('speaker_detail') else ''}."
        )
    if judgement["strength_match"] == "draft_stronger":
        return "OVERSTATED", "Element 3 (strength) fails: the draft states the proposition more strongly than the source."
    if judgement["content_match"] == "partial":
        return "OVERSTATED", "Element 1 (content) is only partly met: the source supports part of the proposition, not all of it."
    return "SUPPORTED", "All elements met: content, attribution and strength match the source."


def _comparable(verdict: str) -> str:
    """WRONG_PINPOINT is a location problem, not a substance problem."""
    return "SUPPORTED" if verdict == "WRONG_PINPOINT" else verdict


# ---------------------------------------------------------------------------
# Rule-based precedence (applied in code around the AI verdict)
# ---------------------------------------------------------------------------
def decide_verdict(unit: dict, source, quote_results: list, judgement, support_check, pinpoint_check: dict,
                   check_pinpoints: bool = True, final_turn: bool = True) -> dict:
    """
    Combine everything into one verdict. Returns:
        {"verdict", "rule", "explanation", "flags": [...], "needs_reflection": bool,
         "deterministic": bool  (True if a code rule fixed the verdict, so AI reflection can't change it)}
    """
    flags = []

    # Rule 1. Source not provided: do NOT ask the AI to judge from memory.
    if unit["matched_source_id"] == "NOT_PROVIDED" or source is None:
        return _result("CANNOT_DETERMINE", "R1 source not provided",
                       f"The cited source ({unit['full_citation']}) was not uploaded, so the tool will not verify it. "
                       "Answering from the model's memory of the source would be unverifiable and is exactly the "
                       "error this tool exists to catch.", flags, deterministic=True)

    # Rule 2. Source unreadable (a scan with no text layer).
    if not source.get("readable", True):
        return _result("CANNOT_DETERMINE", "R2 source unreadable",
                       f"{source['filename']} has no usable text layer (it looks scanned). Upload a "
                       "text-searchable copy or run OCR, then re-run.", flags, deterministic=True)

    # Rule 3. A quotation in the draft that is not in the cited source.
    missing = [q for q in quote_results if not q["found"]]
    if missing:
        notes = []
        for q in missing:
            if q["found_elsewhere"]:
                where = "; ".join(f"{o['label']} {_paras(o['paras'])}" for o in q["found_elsewhere"])
                notes.append(f"‘{q['phrase']}’ is not in the cited source but IS in {where} - "
                             "it appears to be attributed to the wrong case")
            elif q["in_front_matter_only"]:
                notes.append(f"‘{q['phrase']}’ appears only in the catchwords/headnote, which are not part of the reasons")
            else:
                notes.append(f"‘{q['phrase']}’ does not appear anywhere in the cited source (fabricated or altered quotation)")
        if all(q["in_front_matter_only"] for q in missing):
            return _result("MISATTRIBUTED", "R3 quote only in headnote", "; ".join(notes) + ".", flags,
                           deterministic=True)
        return _result("UNSUPPORTED", "R3 quotation not in source", "; ".join(notes) + ".", flags,
                       deterministic=True)

    if judgement is None:
        return _result("CANNOT_DETERMINE", "AI judgement unavailable",
                       "The AI judgement step failed, so the citation could not be assessed.", flags)

    # The Lab 1 gate turns the AI's element findings into a verdict.
    verdict, explanation = verdict_from_elements(judgement)
    needs_reflection = False

    # Cross-check: does the AI's own overall verdict agree with its element findings?
    if _comparable(judgement["verdict"]) != verdict:
        flags.append(f"The AI's overall verdict ({judgement['verdict']}) is inconsistent with its element "
                     f"findings ({verdict}); the element-based verdict is used.")
        needs_reflection = True

    # Rule 4. The AI's supporting quote must really be in the source.
    if verdict != "UNSUPPORTED" and support_check is not None and not support_check["found"]:
        needs_reflection = True
        if final_turn:
            flags.append(support_check["detail"])
            return _result("CANNOT_DETERMINE", "R4 supporting quote not verified",
                           f"The AI found the claim {verdict} but its supporting quote could not be found in "
                           "the source text after the final reflection turn, so its verdict is treated as unreliable.",
                           flags, needs_reflection=False)
    if support_check is not None and support_check["found"] and not support_check["in_claimed_para"]:
        flags.append(support_check["detail"])

    if judgement["confidence"] < CONFIDENCE_THRESHOLD:
        needs_reflection = True

    # Rules 5 and 6. Pinpoints.
    rule = "Element gate"
    if check_pinpoints:
        status = pinpoint_check.get("status")
        if status == "inconsistent" and verdict == "SUPPORTED":
            verdict, rule = "WRONG_PINPOINT", "R5 evidence outside pinpoint"
            explanation = (f"The proposition is supported, but not at the cited pinpoint. {pinpoint_check['detail']} "
                           f"Correct pinpoint: {_paras(pinpoint_check['correct_paras'])}.")
        elif status == "inconsistent":
            flags.append("Pinpoint also inconsistent: " + pinpoint_check["detail"])
        elif status == "unverifiable":
            flags.append("Pinpoint unverifiable for this version of the source (R6): " + pinpoint_check["detail"])
            rule += " + R6 page pinpoint unverifiable"

    return _result(verdict, rule, explanation, flags, needs_reflection=needs_reflection)


def _paras(paras: list) -> str:
    return format_paras(paras)


def _result(verdict, rule, explanation, flags, needs_reflection=False, deterministic=False) -> dict:
    return {"verdict": verdict, "rule": rule, "explanation": explanation, "flags": flags,
            "needs_reflection": needs_reflection, "deterministic": deterministic}

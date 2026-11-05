"""
STEP 9 - REFLECT AND LOOP (Lab 2 evaluate_objective + application.py pattern)

For each citation unit we run a loop, exactly like Lab 2's application.py:

    while not complete:
        turn += 1
        plan queries -> search (step 6) -> judge (step 7) -> grounding checks (step 8)
        reviewer AI: is the verdict well founded?   (strict JSON {verdict_well_founded: bool})
        append everything to the WORK RECORD (fed back into the next turn)
        hard stop after max_turns

Another turn is taken if ANY of these hold:
  * the reviewer says the verdict is not well founded;
  * the AI's supporting quote failed the computer check (rule 4);
  * the AI's overall verdict contradicts its own element findings;
  * the AI's confidence is below the threshold.

If a CODE rule has already decided the verdict (source not provided, unreadable,
quotation not in the source), the loop stops: no amount of reflection can make a
missing quotation appear, so another AI call would only cost money.
"""

import streamlit as st

from reusable_code.ai_client import ask_ai
from reusable_code.grounding import (check_pinpoint, check_quoted_phrases, check_supporting_quote, format_paras)
from reusable_code.judge_claim import build_judge_prompt, judge_claim
from reusable_code.retrieve_evidence import (_plan_queries, _search, fetch_pinpoint_evidence, fetch_quote_evidence,
                                             merge_evidence)
from reusable_code.suggest_fix import deterministic_fix, suggest_fix
from reusable_code.verdict_rules import VERDICT_NAMES, decide_verdict

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict_well_founded": {"type": "boolean"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "search_next": {"type": "array", "items": {"type": "string"}},
        "reviewer_verdict": {"type": "string", "enum": VERDICT_NAMES},
    },
    "required": ["verdict_well_founded", "problems", "search_next", "reviewer_verdict"],
    "additionalProperties": False,
}

REVIEW_INSTRUCTIONS = """You are a second, sceptical reviewer of a citation check. Be strict: return
verdict_well_founded = true only if the verdict clearly follows from the excerpts and the computer checks.
Return false if: the computer could not find the AI's supporting quote in the source; the verdict does not follow
from the element findings; the excerpts lack a passage needed to decide (for example the court's own conclusion on
the point, or the surrounding text that shows who is speaking); or the attribution or strength analysis looks wrong.
Judge only from the excerpts - never from your own memory of the case. If false, list the problems and suggest what
to search for next (short phrases). reviewer_verdict is the verdict you would give."""


def evaluate_verdict(unit, judge_prompt_summary: str, judgement: dict, decision: dict, support: dict,
                     pinpoint_check: dict, effort: str = "medium") -> dict:
    """The AI stop condition (Lab 2 evaluate_objective)."""
    prompt = (f"{judge_prompt_summary}\n\n"
              f"FIRST REVIEWER'S FINDINGS: {judgement}\n\n"
              f"COMPUTER CHECKS:\n"
              f"- supporting quote: {support['detail']}\n"
              f"- pinpoint: {pinpoint_check['detail']}\n"
              f"- verdict after code rules: {decision['verdict']} ({decision['rule']})\n"
              f"- flags: {decision['flags'] or 'none'}")
    return ask_ai(prompt, instructions=REVIEW_INSTRUCTIONS, effort=effort, schema=REVIEW_SCHEMA,
                  step=f"9 reflect (fn {unit['unit_id']})")


def _short(source: dict) -> str:
    return (source.get("identity") or {}).get("short_title") or source["source_id"]


def _evidence_summary(evidence: list) -> str:
    return ", ".join(f"[{e['para']}]" + (f"(part {e['sub_index'] + 1})" if e["sub_index"] is not None else "")
                     for e in evidence)


def verify_unit(unit: dict, claim: dict, sources_by_id: dict, all_sources: list, collection, settings: dict) -> dict:
    """Run steps 6-10 for one citation unit and return its result record."""
    source = sources_by_id.get(unit["matched_source_id"])
    record = base_record(unit, claim, source)
    label = f"fn {unit['unit_id']}: {unit['citation_text'][:70]}"

    # Rules 1 and 2 need no AI at all.
    if source is None or not source.get("readable", True):
        decision = decide_verdict(unit, source, [], None, None, {}, settings["check_pinpoints"])
        record.update(_decision_fields(decision, None))
        record["fix"] = deterministic_fix(record, source)
        with st.status(f"{label} → {decision['verdict']}", state="complete", expanded=False):
            st.write(decision["explanation"])
        return record

    short, phrases = _short(source), unit["unit_quoted_phrases"]
    quote_results = check_quoted_phrases(phrases, source, all_sources)
    record["quote_results"] = quote_results
    pinpoint_evidence = fetch_pinpoint_evidence(source, unit["pinpoint"], phrases)
    quote_evidence = fetch_quote_evidence(source, quote_results, unit["pinpoint"])
    search_hits, work_record, turn, complete = [], "", 0, False

    with st.status(label, expanded=True) as status:
        if quote_results:
            st.write("**Quote check (code):** " + "; ".join(
                f"‘{q['phrase'][:50]}’ {'✓ ' + format_paras(q['paras'][:6]) if q['found'] else '✗ not in source'}"
                for q in quote_results))

        while not complete:
            turn += 1
            final_turn = turn >= settings["max_turns"]

            # Step 6: plan queries and search this source only.
            status.update(label=f"{label} - turn {turn}: planning searches...")
            queries = _plan_queries(unit, claim, source["label"], work_record)
            search_hits += _search(collection, queries, source["source_id"])
            evidence = merge_evidence(pinpoint_evidence, quote_evidence, search_hits)
            st.write(f"**Turn {turn} queries:** " + " · ".join(queries))
            st.write("**Evidence:** " + _evidence_summary(evidence) + "  ·  search distances: " +
                     ", ".join(f"[{h['para']}] {h['distance']}" for h in search_hits[-8:]))

            # Step 7: judge.
            status.update(label=f"{label} - turn {turn}: judging...")
            judgement = judge_claim(unit, claim, short, source["label"], evidence, quote_results, work_record,
                                    effort=settings["effort"])

            # Step 8: grounding checks in code.
            support = check_supporting_quote(judgement["supporting_quote"], source, judgement["found_at_para"])
            evidence_paras = support["paras"] if support["found"] else []
            pinpoint_check = check_pinpoint(unit["pinpoint"], source, evidence_paras, quote_results)
            decision = decide_verdict(unit, source, quote_results, judgement, support, pinpoint_check,
                                      settings["check_pinpoints"], final_turn=final_turn)
            if decision["deterministic"]:
                # Rule 3 settled it: a quotation that isn't in the source has no location to check.
                pinpoint_check = {"status": "not_checked", "correct_paras": [],
                                  "detail": "Not checked: the quotation is not in the cited source, so its location is moot."}
            st.write(f"**AI elements:** content {judgement['content_match']}, attribution "
                     f"{judgement['attribution_match']} ({judgement['who_is_speaking']}), strength "
                     f"{judgement['strength_match']}, at [{judgement['found_at_para']}], confidence "
                     f"{judgement['confidence']:.2f} → AI verdict {judgement['verdict']}")
            st.write(f"**Grounding (code):** supporting quote {'✓' if support['found'] else '✗'} · pinpoint "
                     f"{pinpoint_check['status']} → **{decision['verdict']}** ({decision['rule']})")

            # Step 9: reflect (unless a code rule has already settled it).
            if decision["deterministic"]:
                review = None
                complete = True
                stop_reason = "a code rule decides the verdict, so reflection cannot change it"
            else:
                status.update(label=f"{label} - turn {turn}: reflecting...")
                context = build_judge_prompt(unit, claim, short, source["label"], evidence, quote_results, work_record)
                review = evaluate_verdict(unit, context, judgement, decision, support, pinpoint_check,
                                          effort=settings["effort"])
                complete = review["verdict_well_founded"] and not decision["needs_reflection"]
                stop_reason = "reviewer satisfied" if complete else "reviewer or checks not satisfied"
                st.write(f"**Reviewer:** {'✓ well founded' if review['verdict_well_founded'] else '✗ ' + '; '.join(review['problems'])}")
            if final_turn and not complete:
                complete = True
                stop_reason = f"hard stop after {settings['max_turns']} turn(s)"
                if review and not review["verdict_well_founded"]:
                    decision["flags"].append("Reviewer still not satisfied after the final turn: "
                                             + "; ".join(review["problems"]))

            record["turns"].append({"turn": turn, "queries": queries, "evidence": _evidence_summary(evidence),
                                    "ai_verdict": judgement["verdict"], "verdict": decision["verdict"],
                                    "confidence": judgement["confidence"], "supporting_quote_found": support["found"],
                                    "review": review, "stop_reason": stop_reason if complete else "continue"})
            # Lab 2 work record: what was searched, found and concluded, fed into the next turn.
            work_record += (f"\nTurn {turn}: queried {queries}; evidence {_evidence_summary(evidence)}; "
                            f"verdict {decision['verdict']} (AI said {judgement['verdict']}, confidence "
                            f"{judgement['confidence']:.2f}); supporting quote "
                            f"{'verified' if support['found'] else 'NOT found in source'}; "
                            + (f"reviewer: {'satisfied' if review['verdict_well_founded'] else review['problems']}"
                               f"; search next: {review['search_next']}" if review else "no review needed") + "\n")

        status.update(label=f"{label} → {decision['verdict']} ({turn} turn{'s' if turn > 1 else ''})",
                      state="complete", expanded=False)

    record.update(_decision_fields(decision, judgement))
    record.update({"support_check": support, "pinpoint_check": pinpoint_check, "work_record": work_record.strip(),
                   "evidence": [{**e, "text": e["text"][:3000]} for e in evidence]})

    # Step 10: suggest a fix for anything not SUPPORTED.
    if record["verdict"] != "SUPPORTED" and settings["suggest_fixes"]:
        if record["verdict"] == "CANNOT_DETERMINE":
            record["fix"] = deterministic_fix(record, source)
        else:
            record["fix"] = suggest_fix(record, source, effort="low")
    return record


def base_record(unit: dict, claim: dict, source) -> dict:
    """The result record for one citation unit, before any checking."""
    return {
        "unit_id": unit["unit_id"], "fn": unit["fn"], "citation_text": unit["citation_text"],
        "full_citation": unit["full_citation"], "source_id": unit["matched_source_id"],
        "source_label": source["label"] if source else "not provided",
        "pinpoint_text": unit["pinpoint_text"], "pinpoint_type": unit["pinpoint"]["type"],
        "proposition": unit["unit_proposition"] or claim.get("proposition", ""),
        "sentence": claim.get("sentence", ""), "attributed_to": claim.get("attributed_to", ""),
        "authorial_comment": claim.get("authorial_comment", ""),
        "quoted_phrases": unit["unit_quoted_phrases"], "match_reason": unit.get("match_reason", ""),
        "resolution_notes": unit.get("enforcement", []) + unit.get("warnings", []),
        "quote_results": [], "judgement": None, "support_check": None, "pinpoint_check": {},
        "evidence": [], "turns": [], "work_record": "", "fix": None,
        "verdict": "CANNOT_DETERMINE", "rule": "", "explanation": "", "flags": [], "confidence": None,
    }


def failed_record(unit: dict, claim: dict, source, message: str) -> dict:
    """If an AI call fails for one citation, record it and let the run carry on."""
    record = base_record(unit, claim, source)
    record.update({"rule": "AI step failed", "explanation": f"This citation could not be checked: {message}",
                   "flags": ["Re-run to try again."]})
    return record


def _decision_fields(decision: dict, judgement) -> dict:
    return {"verdict": decision["verdict"], "rule": decision["rule"], "explanation": decision["explanation"],
            "flags": decision["flags"], "judgement": judgement,
            "confidence": judgement["confidence"] if judgement else None}

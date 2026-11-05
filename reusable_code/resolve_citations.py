"""
STEP 5 - RESOLVE CITATIONS (deterministic rules + 1 AI call)

Turns each footnote into one or more CITATION UNITS (one per source cited),
resolves AGLC short forms, parses the pinpoint and matches each unit to an
uploaded source - or to NOT_PROVIDED.

Citation logic encoded here (AGLC 4th ed):
  * "Ibid" refers to the source in the IMMEDIATELY PRECEDING footnote, and
    is only unambiguous if that footnote cites a single source (AGLC 1.4.3).
    "Ibid" alone keeps the same pinpoint; "Ibid [6]" changes it.
  * "(n X)" is a subsequent reference to the source first cited in footnote
    X (AGLC 1.4.1), identified by the short title before "(n X)".
  * Sources in one footnote are separated by semicolons - but so are
    parallel citations of ONE case ("(2007) 16 VR 269; [2007] VSCA 102"),
    which is why splitting is left to the AI.
  * A case is the same case only if the PARTIES match. "(2020) 62 VR 491"
    and "[2020] VSCA 212" are the same Brown v The Queen; but "R v Verdins"
    is NOT the uploaded Brown judgment, however much Brown discusses Verdins.

Order of work:
  1. Code annotates every Ibid / (n X) with the footnote it points to.
  2. One AI call splits footnotes into units and proposes matches. The match
     must be one of the uploaded source ids or NOT_PROVIDED (enforced by an
     enum in the JSON schema, so it cannot invent an id).
  3. Code then ENFORCES the rules, overriding the AI where they apply:
     Ibid/(n X) take the source of the footnote they point to; a citation
     containing a source's exact medium-neutral citation matches it; and a
     match whose party names do not overlap is refused.
"""

import re

from reusable_code.ai_client import ask_ai
from reusable_code.grounding import normalise, paragraph_pinpoints_in, parse_pinpoint

NOT_PROVIDED = "NOT_PROVIDED"
IBID = re.compile(r"^\s*ibid\b", re.I)
CROSS_REF = re.compile(r"\(n\s*(\d{1,4})\)")

# Words that do not identify a party (so they cannot show two cases are the same).
NON_DISTINCTIVE = set("""
r v vs the queen king regina rex crown state of and dpp director public prosecutions prosecution cth nsw vic qld wa sa tas nt
act pty ltd limited inc co corp re ex parte in matter on appeal from a an for attorney general attorneygeneral commonwealth
australia his her majesty minister department secretary commissioner police pseudonym anor ors another others applicant
respondent appellant plaintiff defendant no court case hma people united states
""".split())


def party_words(case_name: str) -> set:
    """Distinctive party words: 'Brown v The Queen' -> {'brown'}."""
    words = re.findall(r"[a-z]+", normalise(case_name))
    return {w for w in words if len(w) >= 3 and w not in NON_DISTINCTIVE}


# ---------------------------------------------------------------------------
# 1. Deterministic annotation
# ---------------------------------------------------------------------------
def annotate_footnotes(footnotes: list) -> list:
    """Add 'points_to' notes for Ibid and (n X), so the AI sees what they mean."""
    by_fn = {f["fn"]: f for f in footnotes}
    annotated = []
    for f in footnotes:
        notes = []
        if IBID.match(f["text"]) and f["fn"] - 1 in by_fn:
            notes.append(f"'Ibid' refers to footnote {f['fn'] - 1}: {by_fn[f['fn'] - 1]['text']}")
        for m in CROSS_REF.finditer(f["text"]):
            x = int(m.group(1))
            if x in by_fn:
                notes.append(f"'(n {x})' refers to the source first cited in footnote {x}: {by_fn[x]['text']}")
            else:
                notes.append(f"'(n {x})' refers to footnote {x}, which is not in this excerpt.")
        annotated.append({**f, "notes": notes})
    return annotated


# ---------------------------------------------------------------------------
# 2. The AI call
# ---------------------------------------------------------------------------
def _schema(source_ids: list) -> dict:
    unit = {
        "type": "object",
        "properties": {
            "citation_text": {"type": "string", "description": "This unit's part of the footnote, verbatim"},
            "reference_type": {"type": "string", "enum": ["full", "ibid", "cross_reference", "short_title"]},
            "refers_to_fn": {"type": ["integer", "null"]},
            "source_kind": {"type": "string", "enum": ["case", "legislation", "secondary", "other"]},
            "case_name": {"type": "string", "description": "Parties of the case cited ('' if not a case)"},
            "full_citation": {"type": "string", "description": "The full citation with Ibid / (n X) / short titles resolved"},
            "pinpoint_text": {"type": "string", "description": "The pinpoint exactly as written, e.g. '[6], [29]' or '493'; '' if none"},
            "matched_source_id": {"type": "string", "enum": source_ids + [NOT_PROVIDED]},
            "match_reason": {"type": "string"},
            "unit_proposition": {"type": "string", "description": "The part of the claim this unit is cited for"},
            "unit_quoted_phrases": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["citation_text", "reference_type", "refers_to_fn", "source_kind", "case_name", "full_citation",
                     "pinpoint_text", "matched_source_id", "match_reason", "unit_proposition", "unit_quoted_phrases"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"footnotes": {"type": "array", "items": {
            "type": "object",
            "properties": {"fn": {"type": "integer"}, "units": {"type": "array", "items": unit}},
            "required": ["fn", "units"], "additionalProperties": False}}},
        "required": ["footnotes"],
        "additionalProperties": False,
    }


RESOLVE_INSTRUCTIONS = f"""You resolve legal citations in footnotes written in AGLC style and match them to uploaded source files.

For each footnote, split it into citation units: one unit per distinct source. Semicolons usually separate sources,
but parallel citations of the SAME case (e.g. "(2007) 16 VR 269; [2007] VSCA 102") are ONE unit.
For each unit:
- reference_type: "ibid" (Ibid), "cross_reference" (Short title (n X)), "short_title" (a short title alone), or "full".
- refers_to_fn: the footnote an Ibid or (n X) points to, else null. Use the notes provided.
- full_citation: the complete citation, resolving Ibid / (n X) / short titles from the earlier footnote.
- pinpoint_text: this unit's own pinpoint, verbatim. For "Ibid" with no pinpoint, repeat the earlier pinpoint.
- matched_source_id: the uploaded file that IS this source, or "{NOT_PROVIDED}".
  A file matches only if it is the same document: the same parties (case name) and the same decision.
  Treat a reported citation and a medium-neutral citation as the same case when the parties, year and court agree
  (e.g. "(2020) 62 VR 491" and "[2020] VSCA 212" for the same Brown v The Queen: the Victorian Reports report VSCA decisions).
  NEVER match a case to an uploaded file just because that file discusses, quotes or cites it. If the case is not
  itself uploaded, use "{NOT_PROVIDED}". Articles, books and statutes are "{NOT_PROVIDED}" unless that exact work was uploaded.
- match_reason: one sentence explaining the match (or why there is none).
- unit_proposition: the part of the footnote's claim that this unit is cited for (for a single-source footnote, the whole claim).
- unit_quoted_phrases: which of the claim's quoted phrases belong to this unit's source (copy them exactly).
Do not judge whether the claims are correct."""


def _prompt(annotated: list, claims_by_fn: dict, sources: list) -> str:
    source_lines = []
    for s in sources:
        ident = s.get("identity") or {}
        reported = ", ".join(ident.get("reported_citations") or []) or "none printed"
        source_lines.append(
            f"- source_id: {s['source_id']} | file: {s['filename']} | case: {ident.get('case_name', '?')} | "
            f"medium-neutral citation: {ident.get('medium_neutral_citation') or '?'} | reported: {reported} | "
            f"court: {ident.get('court', '?')} | date: {ident.get('decision_date', '?')} | "
            f"readable: {'yes' if s['readable'] else 'NO (scanned)'}")
    footnote_lines = []
    for f in annotated:
        claim = claims_by_fn.get(f["fn"], {})
        footnote_lines.append(
            f"Footnote {f['fn']}: {f['text']}\n"
            + "".join(f"   NOTE: {n}\n" for n in f["notes"])
            + f"   Claim it supports: {claim.get('proposition', f['span'])}\n"
            + f"   Quoted phrases in the claim: {claim.get('quoted_phrases', [])}")
    return "Uploaded sources:\n" + "\n".join(source_lines) + "\n\nFootnotes:\n" + "\n".join(footnote_lines)


# ---------------------------------------------------------------------------
# 3. Deterministic enforcement
# ---------------------------------------------------------------------------
def _mnc_match(citation_text: str, sources: list):
    """A citation that contains a source's exact medium-neutral citation IS that source."""
    text = normalise(citation_text)
    for s in sources:
        mnc = (s.get("identity") or {}).get("medium_neutral_citation")
        if mnc and normalise(mnc) and normalise(mnc) in text:
            return s["source_id"]
    return None


def _set_source(unit: dict, source_id: str, why: str):
    if unit["matched_source_id"] != source_id:
        unit["enforcement"].append(f"{why} (the AI had matched {unit['matched_source_id']}).")
    unit["matched_source_id"] = source_id


def enforce_rules(footnotes_out: list, sources: list) -> list:
    by_id = {s["source_id"]: s for s in sources}
    units_by_fn = {}   # earlier footnotes, already resolved (we work in footnote order)
    for f in footnotes_out:
        for unit in f["units"]:
            # The reference type is decided by the text itself, not by the AI.
            cross_ref = CROSS_REF.search(unit["citation_text"])
            if IBID.match(unit["citation_text"]):
                unit["reference_type"], unit["refers_to_fn"] = "ibid", f["fn"] - 1
            elif cross_ref:
                unit["reference_type"], unit["refers_to_fn"] = "cross_reference", int(cross_ref.group(1))
            elif unit["reference_type"] in ("ibid", "cross_reference"):
                unit["reference_type"] = "full"   # the AI said Ibid / (n X) but the text has neither
            ref, target = unit["reference_type"], unit.get("refers_to_fn")

            # Ibid -> the single source of the immediately preceding footnote.
            if ref == "ibid":
                previous = units_by_fn.get(f["fn"] - 1, [])
                if len(previous) == 1:
                    prev = previous[0]
                    _set_source(unit, prev["matched_source_id"], f"Ibid rule: footnote {f['fn'] - 1} cites one source")
                    unit["full_citation"] = prev["full_citation"]
                    if not unit["pinpoint_text"].strip():
                        unit["pinpoint_text"] = prev["pinpoint_text"]
                        unit["enforcement"].append("Ibid without a pinpoint keeps the previous pinpoint.")
                elif len(previous) > 1:
                    unit["warnings"].append(f"Ambiguous 'Ibid': footnote {f['fn'] - 1} cites more than one source "
                                            "(AGLC 1.4.3 says Ibid should not be used here).")

            # (n X) -> the unit in footnote X whose short title / party name matches.
            elif ref == "cross_reference" and target in units_by_fn:
                words = party_words(unit["citation_text"].split("(n")[0])
                candidates = [u for u in units_by_fn[target]
                              if words & party_words(u["case_name"] + " " + u["full_citation"])]
                if len(units_by_fn[target]) == 1 and not candidates:
                    candidates = units_by_fn[target]
                if candidates:
                    _set_source(unit, candidates[0]["matched_source_id"], f"(n {target}) rule: footnote {target}")
                    if "(n" in unit["full_citation"] or not unit["full_citation"].strip():
                        unit["full_citation"] = candidates[0]["full_citation"]
                else:
                    unit["warnings"].append(f"Could not tell which source in footnote {target} '(n {target})' means.")

            # Exact medium-neutral citation in the footnote text -> certain match.
            else:
                exact = _mnc_match(unit["citation_text"], sources)
                if exact:
                    _set_source(unit, exact, "Medium-neutral citation matches the uploaded file exactly")

            # Party-name guard: a case is only the same case if the parties overlap.
            matched = by_id.get(unit["matched_source_id"])
            if matched and ref in ("full", "short_title") and unit["source_kind"] == "case":
                cited_words = party_words(unit["case_name"] or unit["citation_text"])
                file_words = party_words((matched.get("identity") or {}).get("case_name", "") + " " + matched["filename"])
                if cited_words and not cited_words & file_words:
                    unit["enforcement"].append(
                        f"Party-name guard: '{unit['case_name']}' does not share a party name with "
                        f"'{matched.get('label', matched['source_id'])}', so the match was refused. A judgment that "
                        "discusses a case is not that case.")
                    unit["matched_source_id"] = NOT_PROVIDED

            # Pinpoint cross-check: paragraph pinpoints written in the footnote are read by code too.
            written = paragraph_pinpoints_in(unit["citation_text"])
            if written and parse_pinpoint(written)["paras"] != parse_pinpoint(unit["pinpoint_text"])["paras"]:
                unit["enforcement"].append(f"Pinpoint read from the footnote text by code: {written} "
                                           f"(the AI had '{unit['pinpoint_text']}').")
                unit["pinpoint_text"] = written
            unit["pinpoint"] = parse_pinpoint(unit["pinpoint_text"])
        units_by_fn[f["fn"]] = f["units"]
    return footnotes_out


def resolve_citations(footnotes: list, claims: list, sources: list, effort: str = "medium") -> list:
    """
    Returns a flat list of citation units:
        {"unit_id": "4.1", "fn", "citation_text", "full_citation", "matched_source_id", "pinpoint": {...},
         "unit_proposition", "unit_quoted_phrases", "match_reason", "enforcement": [...], "warnings": [...], ...}
    """
    annotated = annotate_footnotes(footnotes)
    claims_by_fn = {c["fn"]: c for c in claims}
    readable_ids = [s["source_id"] for s in sources]
    result = ask_ai(_prompt(annotated, claims_by_fn, sources), instructions=RESOLVE_INSTRUCTIONS, effort=effort,
                    schema=_schema(readable_ids), step="5 resolve citations")
    ai_by_fn = {f["fn"]: f["units"] for f in result["footnotes"]}

    # Make sure every footnote has at least one unit, in footnote order.
    footnotes_out = []
    for f in footnotes:
        units = ai_by_fn.get(f["fn"]) or [{
            "citation_text": f["text"], "reference_type": "ibid" if IBID.match(f["text"]) else "full",
            "refers_to_fn": None, "source_kind": "other", "case_name": "", "full_citation": f["text"],
            "pinpoint_text": "", "matched_source_id": NOT_PROVIDED, "match_reason": "The AI did not return this footnote.",
            "unit_proposition": claims_by_fn.get(f["fn"], {}).get("proposition", f["span"]), "unit_quoted_phrases": []}]
        for unit in units:
            unit.setdefault("enforcement", [])
            unit.setdefault("warnings", [])
        footnotes_out.append({"fn": f["fn"], "units": units})

    footnotes_out = enforce_rules(footnotes_out, sources)

    flat = []
    for f in footnotes_out:
        claim = claims_by_fn.get(f["fn"], {})
        claim_phrases = claim.get("quoted_phrases", [])
        for k, unit in enumerate(f["units"], start=1):
            # Quoted phrases: a single-source footnote owns all of them; otherwise keep the AI's split,
            # but only phrases that really are in the claim.
            if len(f["units"]) == 1:
                unit["unit_quoted_phrases"] = list(claim_phrases)
            else:
                claim_norm = [normalise(q) for q in claim_phrases]
                unit["unit_quoted_phrases"] = [claim_phrases[claim_norm.index(normalise(q))]
                                               for q in unit["unit_quoted_phrases"] if normalise(q) in claim_norm]
            flat.append({"unit_id": f"{f['fn']}.{k}", "fn": f["fn"], "n_units_in_footnote": len(f["units"]), **unit})
    return flat

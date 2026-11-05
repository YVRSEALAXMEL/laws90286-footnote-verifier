"""
STEP 8 - DETERMINISTIC GROUNDING CHECKS (pure Python, no AI)

Lab 2's grounding idea was: "if the retrieved text is not in the answer,
FAIL and repeat". Here that becomes three checks done in code, so they
cannot be talked round by a persuasive model:

  (i)   QUOTE CHECK: every phrase the draft puts in quotation marks must
        appear in the cited source. If it doesn't, the quote is fabricated
        or altered. If it appears in ANOTHER uploaded source instead, we
        say which one (the quote has been attributed to the wrong case).

  (ii)  SUPPORTING-QUOTE CHECK: the passage the AI says supports the claim
        must really be in the source, and in the paragraph the AI says.
        If not, the AI's verdict is treated as unreliable.

  (iii) PINPOINT CHECK: is the evidence at (or within one paragraph of)
        the cited paragraph pinpoint? A PAGE pinpoint (e.g. "493" in the
        Victorian Reports) cannot be checked against a copy that only has
        paragraph numbers, so it is marked "unverifiable", not "wrong".

Matching is done on NORMALISED text, so that harmless differences in
typography do not cause false alarms: case, curly vs straight quotes,
line-break hyphenation, runs of whitespace and footnote markers like "[27]".
"""

import re
import unicodedata

PINPOINT_TOLERANCE = 1   # "within one paragraph" of the cited pinpoint counts as correct

# ---------------------------------------------------------------------------
# Normalising text for comparison
# ---------------------------------------------------------------------------
_QUOTE_MARKS = "‘’‚‛“”„‟'\"`´′″«»"
_HYPHENS = "-­‐‑"            # hyphen, soft hyphen, unicode hyphens
_DASHES = "‒–—―"        # figure, en, em, horizontal bar


def normalise(text: str) -> str:
    """
    Make text comparable:
        "Impairment is the key’.  [24]"  ->  "impairment is the key."
        "anti-\\nsocial" and "anti-social" -> "antisocial"
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\[\s*\d{1,4}\s*\]", " ", text)        # footnote / paragraph markers
    text = text.translate({ord(c): None for c in _QUOTE_MARKS})
    text = text.translate({ord(c): " " for c in _DASHES})
    text = re.sub(rf"\s*[{_HYPHENS}]\s*", "", text)     # join hyphenated words
    text = text.replace("…", "...")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?)])", r"\1", text)       # "key ." -> "key."
    return text.strip().lower()


def quote_fragments(phrase: str) -> list:
    """
    Split a quotation into the fragments that must each appear verbatim.
    Ellipses (omitted words) and square-bracket alterations ("[t]he") are
    legitimate edits under AGLC 1.5, so they become gaps, not text to match.
    """
    pieces = re.split(r"\.\s?\.\s?\.|…|\[[^\]]*\]", phrase)
    fragments = []
    for piece in pieces:
        piece = normalise(piece).strip(" .,;:")
        if len(piece) >= 2:
            fragments.append(piece)
    return fragments


def _contains(haystack_normalised: str, fragment: str) -> bool:
    """Whole-word containment: 'total' must not match inside 'totality'."""
    pattern = r"(?<![a-z0-9])" + re.escape(fragment) + r"(?![a-z0-9])"
    return re.search(pattern, haystack_normalised) is not None


def phrase_in_text(phrase: str, text: str) -> bool:
    fragments = quote_fragments(phrase)
    if not fragments:
        return False
    haystack = normalise(text)
    return all(_contains(haystack, f) for f in fragments)


def locate_phrase(phrase: str, source: dict) -> dict:
    """
    Where does a phrase occur in a source?
        {"paras": [111], "in_front_matter": False}
    All fragments of a quotation must be found inside the same paragraph.
    """
    paras = [p["para"] for p in source.get("paragraphs", []) if phrase_in_text(phrase, p["text"])]
    in_front = bool(source.get("front_matter")) and phrase_in_text(phrase, source["front_matter"])
    return {"paras": paras, "in_front_matter": in_front}


# ---------------------------------------------------------------------------
# (i) Quote check
# ---------------------------------------------------------------------------
def check_quoted_phrases(phrases: list, source: dict, other_sources: list) -> list:
    """
    For each quoted phrase in the draft:
        {"phrase", "found", "paras", "in_front_matter_only", "found_elsewhere": [...]}
    found_elsewhere lists OTHER uploaded sources containing the phrase.
    """
    results = []
    for phrase in phrases:
        where = locate_phrase(phrase, source)
        found = bool(where["paras"])
        elsewhere = []
        if not found:
            for other in other_sources:
                if other["source_id"] == source["source_id"] or not other.get("readable", True):
                    continue
                other_where = locate_phrase(phrase, other)
                if other_where["paras"]:
                    elsewhere.append({"source_id": other["source_id"], "label": other.get("label", other["source_id"]),
                                      "paras": other_where["paras"][:5]})
        results.append({
            "phrase": phrase,
            "found": found,
            "paras": where["paras"],
            "in_front_matter_only": (not found) and where["in_front_matter"],
            "found_elsewhere": elsewhere,
        })
    return results


# ---------------------------------------------------------------------------
# (ii) Supporting-quote check
# ---------------------------------------------------------------------------
def check_supporting_quote(quote: str, source: dict, claimed_para) -> dict:
    """
    Is the AI's supporting quote really in the source, and in the paragraph it claims?
        {"found": bool, "paras": [...], "in_claimed_para": bool, "detail": str}
    """
    if not quote or not quote.strip():
        return {"found": False, "paras": [], "in_claimed_para": False,
                "detail": "The AI gave no supporting quote."}
    paras = locate_phrase(quote, source)["paras"]
    in_claimed = claimed_para in paras
    if not paras:
        detail = "The AI's supporting quote does NOT appear in the source (possible hallucination)."
    elif in_claimed:
        detail = f"Supporting quote found at [{claimed_para}], as the AI said."
    else:
        detail = (f"Supporting quote found, but at {format_paras(paras)} rather than "
                  f"[{claimed_para}] as the AI said; the located paragraph is used.")
    return {"found": bool(paras), "paras": paras, "in_claimed_para": in_claimed, "detail": detail}


# ---------------------------------------------------------------------------
# (iii) Pinpoints
# ---------------------------------------------------------------------------
_DASH_CLASS = r"[-‐-―]"


def _looks_like_year(number_text: str) -> bool:
    return len(number_text) == 4 and 1800 <= int(number_text) <= 2100


def paragraph_pinpoints_in(citation_text: str) -> str:
    """
    The paragraph pinpoints written in a footnote, found by code, e.g.
    'Kapua v R [2023] NSWCCA 14, [111] (‘Kapua’).' -> '[111]'. Used to
    cross-check the pinpoint the AI read from the footnote.
    """
    found = re.findall(rf"\[\d{{1,4}}\](?:\s*{_DASH_CLASS}\s*\[\d{{1,4}}\])?", citation_text or "")
    return ", ".join(f for f in found if not _looks_like_year(re.match(r"\[(\d+)", f).group(1)))


def _expand_range(start: int, end: int, cap: int = 60) -> list:
    if end < start:
        end = start
    return list(range(start, min(end, start + cap) + 1))


def parse_pinpoint(text: str) -> dict:
    """
    Parse an AGLC pinpoint into a structure we can check.
        "[1]–[3]"                -> paragraph [1, 2, 3]
        "[6], [56]–[58]"         -> paragraph [6, 56, 57, 58]
        "493"                    -> page [493]
        "193–4"                  -> page [193, 194]   (abbreviated range)
        "507–8 [61]–[62]"        -> paragraph [61, 62] (pages kept too)
    """
    text = (text or "").strip()
    paras = []
    for m in re.finditer(rf"\[(\d{{1,4}})\](?:\s*{_DASH_CLASS}\s*\[(\d{{1,4}})\])?", text):
        if _looks_like_year(m.group(1)):
            continue                     # "[2023]" in "[2023] NSWCCA 14" is a year, not a paragraph
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        paras.extend(_expand_range(start, end))
    # Pages: numbers NOT inside square brackets.
    without_paras = re.sub(r"\[[^\]]*\]", " ", text)
    pages = []
    for m in re.finditer(rf"(?<![\d(])(\d{{1,5}})(?:\s*{_DASH_CLASS}\s*(\d{{1,5}}))?(?![\d)])", without_paras):
        start = int(m.group(1))
        end_text = m.group(2)
        if end_text:
            # "193–4" means 193–194: borrow the missing leading digits.
            if len(end_text) < len(m.group(1)):
                end_text = m.group(1)[: len(m.group(1)) - len(end_text)] + end_text
            pages.extend(_expand_range(start, int(end_text)))
        else:
            pages.append(start)
    if paras:
        kind = "paragraph"
    elif pages:
        kind = "page"
    else:
        kind = "none"
    return {"type": kind, "paras": sorted(set(paras)), "pages": sorted(set(pages)), "raw": text}


def format_paras(paras: list) -> str:
    """[42, 43, 44, 52] -> '[42]–[44], [52]'"""
    paras = sorted(set(p for p in paras if p))
    if not paras:
        return "(none)"
    groups, start, prev = [], paras[0], paras[0]
    for p in paras[1:] + [None]:
        if p is not None and p == prev + 1:
            prev = p
            continue
        groups.append(f"[{start}]" if start == prev else f"[{start}]–[{prev}]")
        if p is not None:
            start = prev = p
    return ", ".join(groups)


def within_pinpoint(para: int, cited: list, tolerance: int = PINPOINT_TOLERANCE) -> bool:
    return any(abs(para - c) <= tolerance for c in cited)


def check_pinpoint(pinpoint: dict, source: dict, evidence_paras: list, quote_results: list) -> dict:
    """
    Compare the cited pinpoint with where the evidence actually is.

    evidence_paras: paragraphs where the AI's (verified) supporting quote was found.
    quote_results:  output of check_quoted_phrases (where the draft's own quotes are).

    Returns {"status", "detail", "correct_paras"} with status one of:
        "consistent"   evidence is at (or within 1 paragraph of) the pinpoint
        "inconsistent" evidence is found, but elsewhere -> WRONG_PINPOINT
        "unverifiable" a page pinpoint against a paragraph-numbered copy
        "no_pinpoint"  the citation has no pinpoint
        "not_checked"  nothing located that could confirm or refute it
    """
    kind = pinpoint.get("type", "none")
    if kind == "none":
        return {"status": "no_pinpoint", "detail": "No pinpoint given.", "correct_paras": []}
    if kind == "page":
        if source.get("numbering_style") in ("N.", "[N]"):
            detail = (f"Page pinpoint {pinpoint['raw']} cannot be checked: the uploaded copy has paragraph "
                      "numbers but no report page numbers (unverifiable for this version of the source).")
        else:
            detail = f"Page pinpoint {pinpoint['raw']} cannot be checked against this copy."
        return {"status": "unverifiable", "detail": detail, "correct_paras": []}

    cited = pinpoint["paras"]
    problems, located = [], set()
    # Every quoted phrase that IS in the source must sit within the pinpoint window.
    for q in quote_results:
        if not q["found"]:
            continue
        located.update(q["paras"])
        if not any(within_pinpoint(p, cited) for p in q["paras"]):
            problems.append(f"‘{q['phrase']}’ is at {format_paras(q['paras'])}")
    # The supporting passage (if verified) should also be within the window.
    if evidence_paras:
        located.update(evidence_paras)
        if not quote_results or not any(q["found"] for q in quote_results):
            if not any(within_pinpoint(p, cited) for p in evidence_paras):
                problems.append(f"the supporting passage is at {format_paras(evidence_paras)}")

    if not located:
        return {"status": "not_checked", "detail": "No located evidence to compare with the pinpoint.",
                "correct_paras": []}
    if problems:
        outside = sorted(p for p in located if not within_pinpoint(p, cited))
        return {"status": "inconsistent",
                "detail": f"Cited {format_paras(cited)}, but " + "; ".join(problems) + ".",
                "correct_paras": outside or sorted(located)}
    return {"status": "consistent",
            "detail": f"Evidence located within {format_paras(cited)} (±{PINPOINT_TOLERANCE}).",
            "correct_paras": []}

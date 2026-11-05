"""
Deterministic tests for steps 1, 2 and 8 (no API calls, no API key needed).

Run from the repository root:
    python -m pytest -q
"""

import io
from pathlib import Path

import pytest

from reusable_code.chunking import (MAX_CHUNK_TOKENS, chunk_source, count_tokens,
                                    get_token_counter, rejoin_sub_chunks)
from reusable_code.docx_footnotes import extract_draft, extract_quoted_phrases
from reusable_code.grounding import (check_pinpoint, check_quoted_phrases, check_supporting_quote,
                                     format_paras, locate_phrase, normalise, parse_pinpoint,
                                     phrase_in_text, quote_fragments)
from reusable_code.pdf_ingest import ingest_pdf, paragraph_lookup

ARTEFACTS = Path(__file__).resolve().parent.parent / "test_artefacts"
SOURCES = {
    "brown": "Brown_v_The_Queen_2020_VSCA_212.pdf",
    "kapua": "Kapua_v_R_2023_NSWCCA_14.pdf",
    "lojovic": "R_v_Lojovic_2025_ONCA_319.pdf",
}


def _draft(name):
    return extract_draft((ARTEFACTS / "drafts" / name).read_bytes())


@pytest.fixture(scope="module")
def sources():
    """Ingest each source PDF once for the whole module (Brown takes ~1 second)."""
    return {key: ingest_pdf((ARTEFACTS / "sources" / name).read_bytes(), name)
            for key, name in SOURCES.items()}


# ---------------------------------------------------------------------------
# Step 1: the draft
# ---------------------------------------------------------------------------
def test_tc1_has_seven_footnotes_numbered_by_order():
    footnotes = _draft("TC1_accurate_citations.docx")["footnotes"]
    assert len(footnotes) == 7
    assert [f["fn"] for f in footnotes] == [1, 2, 3, 4, 5, 6, 7]
    # Word's internal ids start at 20; the visible numbers are 1-7.
    assert footnotes[0]["word_id"] == "20"
    assert footnotes[1]["text"] == "Ibid [6], [29]."
    assert footnotes[3]["text"].startswith("Brown (n 1)")


def test_tc2_and_tc3_footnote_counts():
    assert len(_draft("TC2_seeded_errors.docx")["footnotes"]) == 6
    assert len(_draft("TC3_limitations.docx")["footnotes"]) == 4


def test_body_text_has_markers_and_spans():
    draft = _draft("TC1_accurate_citations.docx")
    assert "[^1]" in draft["body_text"] and "[^7]" in draft["body_text"]
    fn5 = draft["footnotes"][4]
    assert fn5["span"].startswith("New South Wales has taken a similar line.")
    assert fn5["quoted_phrases"] == ["No such requirement operated."]


def test_apostrophes_are_not_quotes():
    text = "the Director’s concession about O’Neill, ‘could not be sustained’, and Strawson’s ‘invitation’."
    assert extract_quoted_phrases(text) == ["could not be sustained", "invitation"]


def test_single_word_quotes():
    fn6 = _draft("TC1_accurate_citations.docx")["footnotes"][5]
    assert fn6["quoted_phrases"] == ["worse", "total"]


# ---------------------------------------------------------------------------
# Step 2: the sources
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key,style,last", [("brown", "N.", 89), ("kapua", "N.", 143), ("lojovic", "[N]", 71)])
def test_paragraph_sequences_are_complete(sources, key, style, last):
    source = sources[key]
    assert source["readable"]
    assert source["numbering_style"] == style
    assert [p["para"] for p in source["paragraphs"]] == list(range(1, last + 1))


def test_known_passages_are_in_the_right_paragraphs(sources):
    kapua = paragraph_lookup(sources["kapua"])
    lojovic = paragraph_lookup(sources["lojovic"])
    brown = paragraph_lookup(sources["brown"])
    assert "No such requirement operated" in kapua[111]["text"]
    assert "worse" in lojovic[43]["text"]
    assert "Impairment is the key" in brown[52]["text"]


def test_fabricated_quote_is_not_in_brown(sources):
    assert locate_phrase("ordinarily operate to reduce", sources["brown"])["paras"] == []
    assert not any("ordinarily operate to reduce" in p["text"] for p in sources["brown"]["paragraphs"])


def test_boilerplate_is_removed(sources):
    for source in sources.values():
        for p in source["paragraphs"]:
            assert "BarNet publication information" not in p["text"]
            assert "(CanLII)" not in p["text"]


def test_kapua_headnote_is_front_matter_not_paragraph_1(sources):
    kapua = paragraph_lookup(sources["kapua"])
    assert kapua[1]["text"].startswith("SIMPSON AJA:")
    assert "This headnote is not to be read as part of the judgment" in sources["kapua"]["front_matter"]


def test_repeated_jade_page_numbers_are_not_in_text(sources):
    brown = paragraph_lookup(sources["brown"])
    assert not brown[2]["text"].startswith("2.")
    assert brown[2]["text"].startswith("Following paragraph cited by:")


def test_scanned_pdf_is_flagged_unreadable():
    """Simulate a Tsiaras-style scan: a PDF page that is only an image."""
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (1240, 1754), "white")
    ImageDraw.Draw(image).text((100, 100), "R v Tsiaras [1996] 1 VR 398 (scanned)", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, "PDF")
    source = ingest_pdf(buffer.getvalue(), "Tsiaras_scan.pdf")
    assert source["readable"] is False
    assert source["paragraphs"] == []
    assert "scanned" in source["warnings"][0]


# ---------------------------------------------------------------------------
# Step 2: sub-chunking
# ---------------------------------------------------------------------------
def test_no_chunk_exceeds_the_cap(sources):
    print("token counter:", get_token_counter()[1])
    for source in sources.values():
        for c in chunk_source(source):
            assert count_tokens(c["text"]) <= MAX_CHUNK_TOKENS, c["id"]


def test_oversized_paragraphs_are_split_and_offsets_are_exact(sources):
    for key, para in [("brown", 89), ("brown", 6), ("kapua", 111)]:
        source = sources[key]
        text = paragraph_lookup(source)[para]["text"]
        pieces = [c for c in chunk_source(source) if c["metadata"]["para"] == para]
        assert len(pieces) > 1, f"{key} [{para}] should be split"
        for c in pieces:
            m = c["metadata"]
            assert text[m["char_start"]:m["char_end"]] == c["text"]
            assert m["source_id"] == source["source_id"] and m["page"] >= 1
        # Re-joining the sub-chunks gives back the whole paragraph.
        assert normalise(rejoin_sub_chunks(pieces)) == normalise(text)


def test_short_paragraphs_stay_whole(sources):
    chunks = [c for c in chunk_source(sources["kapua"]) if c["metadata"]["para"] == 113]
    assert len(chunks) == 1 and chunks[0]["metadata"]["sub_index"] == 0


# ---------------------------------------------------------------------------
# Optional Jade "cited by" exclusion (default OFF)
# ---------------------------------------------------------------------------
def test_cited_by_exclusion_toggle():
    pdf = (ARTEFACTS / "sources" / SOURCES["brown"]).read_bytes()
    off = ingest_pdf(pdf, SOURCES["brown"])
    on = ingest_pdf(pdf, SOURCES["brown"], exclude_cited_by=True)
    assert [p["para"] for p in on["paragraphs"]] == list(range(1, 90))
    # Off: the TC3(1) words are (wrongly) filed under Brown [6]. On: they are gone.
    assert 6 in locate_phrase("exceptional", off)["paras"]
    assert locate_phrase("exceptional", on)["paras"] == []
    # The judgment's own text survives.
    for phrase, para in [("strong causal links to the offending", 2), ("Impairment is the key", 52),
                         ("only in a case of some severity", 68), ("should no longer be followed", 6)]:
        assert para in locate_phrase(phrase, on)["paras"]


# ---------------------------------------------------------------------------
# Step 8: grounding helpers
# ---------------------------------------------------------------------------
def test_normalise_handles_typography():
    assert normalise("‘Impairment is the key’.  [24]") == "impairment is the key."
    assert normalise("anti-\nsocial") == normalise("anti-social") == "antisocial"
    assert phrase_in_text("diagnosis is not the key. Impairment is the key",
                          "As he put it, ‘the starting point is that diagnosis is not the key.  Impairment is the key’.")


def test_quote_fragments_respect_ellipses_and_brackets():
    assert quote_fragments("whether of such disorders … could not be sustained") == [
        "whether of such disorders", "could not be sustained"]
    assert quote_fragments("[t]he judge erred") == ["he judge erred"]


def test_whole_word_matching():
    assert phrase_in_text("total", "his responsibility was total.")
    assert not phrase_in_text("total", "the totality principle")


def test_wrong_source_quote_is_traced_to_kapua(sources):
    brown, others = sources["brown"], list(sources.values())
    [result] = check_quoted_phrases(["a degree of planning, system, and persistence"], brown, others)
    assert result["found"] is False
    assert result["found_elsewhere"][0]["source_id"] == sources["kapua"]["source_id"]
    assert result["found_elsewhere"][0]["paras"] == [113]


def test_supporting_quote_check(sources):
    kapua = sources["kapua"]
    good = check_supporting_quote("No such requirement operated.", kapua, 111)
    assert good["found"] and good["in_claimed_para"]
    moved = check_supporting_quote("No such requirement operated.", kapua, 113)
    assert moved["found"] and not moved["in_claimed_para"] and moved["paras"] == [111]
    made_up = check_supporting_quote("must never be taken into account", kapua, 111)
    assert not made_up["found"]


@pytest.mark.parametrize("text,kind,paras,pages", [
    ("[1]–[3]", "paragraph", [1, 2, 3], []),
    ("[6], [56]–[58], [68]–[69]", "paragraph", [6, 56, 57, 58, 68, 69], []),
    ("493", "page", [], [493]),
    ("187, 193–4", "page", [], [187, 193, 194]),
    ("", "none", [], []),
])
def test_parse_pinpoint(text, kind, paras, pages):
    parsed = parse_pinpoint(text)
    assert (parsed["type"], parsed["paras"], parsed["pages"]) == (kind, paras, pages)


def test_pinpoint_checks(sources):
    kapua, brown = sources["kapua"], sources["brown"]
    # TC2 fn3: quote is at [111], citation says [113] -> inconsistent, correct para 111.
    quotes = check_quoted_phrases(["No such requirement operated."], kapua, [])
    result = check_pinpoint(parse_pinpoint("[113]"), kapua, [111], quotes)
    assert result["status"] == "inconsistent" and result["correct_paras"] == [111]
    # TC1 fn5: same quote, correct pinpoint.
    assert check_pinpoint(parse_pinpoint("[111]"), kapua, [111], quotes)["status"] == "consistent"
    # TC3 fn3: a VR page pinpoint against a paragraph-numbered copy.
    quotes = check_quoted_phrases(["strong causal links to the offending"], brown, [])
    assert check_pinpoint(parse_pinpoint("493"), brown, [2], quotes)["status"] == "unverifiable"


def test_format_paras():
    assert format_paras([44, 42, 43, 52]) == "[42]–[44], [52]"


def test_years_in_citations_are_not_paragraph_pinpoints():
    from reusable_code.grounding import paragraph_pinpoints_in
    assert parse_pinpoint("[2023] NSWCCA 14, [111]")["paras"] == [111]
    assert paragraph_pinpoints_in("Kapua v R [2023] NSWCCA 14, [111] (‘Kapua’).") == "[111]"
    assert paragraph_pinpoints_in("Brown v The Queen (2020) 62 VR 491, [1]–[3] (Maxwell P)") == "[1]–[3]"
    assert paragraph_pinpoints_in("Brown (n 1) 493.") == ""

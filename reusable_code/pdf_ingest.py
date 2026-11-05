"""
STEP 2 - INGEST SOURCES (no AI)

PDF -> text per page -> paragraph-numbered segments.

Why paragraphs? Australian (and Canadian) pinpoints cite judgments by
PARAGRAPH number, not page: "Kapua [111]". So the paragraph is the unit we
store, search and check against. (Oversized paragraphs are later split into
sub-chunks by chunking.py, but every sub-chunk keeps its paragraph number.)

Design decisions, each learned from the test sources:

1. pypdf "layout" mode. In the default mode, Jade's margin paragraph
   numbers come out in a clump at the top of the page ("2. 3. 4. 5."),
   detached from their text. Layout mode keeps each number on the same
   line as its paragraph. It also skips the rotated CanLII watermark.

2. Publisher boilerplate is removed line by line (Jade's "BarNet
   publication information..." footer, CanLII's "Page: N" header and
   "2025 ONCA 319 (CanLII)" watermark).

3. Paragraph numbers in two formats: "111." (Jade, NSW, Vic) and "[42]"
   (CanLII). A number is only accepted if it CONTINUES THE RUNNING SEQUENCE
   (the next paragraph after 41 must be 42). This rejects numbered lists
   inside paragraphs, table rows, and paragraph numbers of OTHER cases
   quoted in the text (e.g. "139." from a later judgment quoted by Jade).

4. Numbering can restart (Kapua's headnote has its own "1. 2. 3." before
   the judgment begins at "1."). We try every "1." as a possible start and
   keep the longest unbroken chain; if two chains tie, the later one wins
   (the judgment comes after the headnote).

5. A PDF with (almost) no text layer is a scan. We flag it as unreadable
   instead of failing, so the verdict can say CANNOT_DETERMINE and explain.
"""

import io
import re
from collections import Counter

from pypdf import PdfReader

# Lines that are publisher furniture, not judgment text.
BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*BarNet publication information\b.*$", re.I),   # Jade footer
    re.compile(r"^\s*BarNet Jade\s+jade\.io\s*$", re.I),             # Jade cover
    re.compile(r"^\s*\d{4}\s+[A-Z]{2,8}\s+\d+\s+\(CanLII\)\s*$"),     # CanLII watermark
    re.compile(r"^\s*Page:\s*\d+\s*$"),                              # CanLII header
]

# Paragraph-number candidates. Group 1 = leading spaces, 2 = number, 3 = rest.
DOT_NUMBER = re.compile(r"^(\s*)(\d{1,4})\.(?:\s+(.*))?$")       # "111.  The ..."
BRACKET_NUMBER = re.compile(r"^(\s*)\[(\d{1,4})\](?:\s+(.*))?$")   # "[42]  In his ..."

# A real paragraph number sits at (or near) the left margin. Numbers indented
# further than this are inside block quotes or publisher annotations.
MAX_NUMBER_INDENT = 12

# When unwrapping lines into running text, an indentation change of at least
# this many columns starts a new line (block quote, footnote, annotation).
BLOCK_INDENT_JUMP = 5

# Fewer than this many non-space characters per page on average = a scan.
MIN_CHARS_PER_PAGE = 40

# Jade's "who later cited this paragraph" annotations (see strip_jade_cited_by).
CITED_BY_HEADER = re.compile(r"^\s*(?:\d{1,4}\.\s+)?Following paragraph cited by:\s*$", re.I)
TRAILING_INDEX_HEADER = re.compile(r"^\s*Cited by:\s*$", re.I)


# ---------------------------------------------------------------------------
# Reading pages
# ---------------------------------------------------------------------------
def extract_page_texts(pdf_bytes: bytes) -> list:
    """Return one string per page, using pypdf's layout mode when possible."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except Exception:
            # Some unusual PDFs break layout mode; plain mode is better than nothing.
            text = page.extract_text() or ""
        pages.append(text)
    return pages


def is_boilerplate(line: str) -> bool:
    return any(p.match(line) for p in BOILERPLATE_PATTERNS)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _page_lines(page_texts: list) -> list:
    """Flatten pages into a list of {"page", "indent", "raw"} lines, boilerplate removed."""
    lines = []
    for page_number, text in enumerate(page_texts, start=1):
        for raw in text.replace("\t", "    ").replace(" ", " ").splitlines():
            raw = raw.rstrip()
            if not raw.strip() or is_boilerplate(raw):
                continue
            lines.append({"page": page_number, "indent": _indent(raw), "raw": raw})
    return lines


def is_readable(page_texts: list) -> bool:
    """A PDF is 'readable' if it has a real text layer (not just a scan)."""
    if not page_texts:
        return False
    chars = sum(len(re.sub(r"\s", "", t)) for t in page_texts)
    return chars / len(page_texts) >= MIN_CHARS_PER_PAGE


# ---------------------------------------------------------------------------
# Paragraph numbers
# ---------------------------------------------------------------------------
def _candidates(lines: list, pattern) -> list:
    """Every line that LOOKS like it starts a numbered paragraph: (line_index, number)."""
    found = []
    for i, line in enumerate(lines):
        m = pattern.match(line["raw"])
        if m and len(m.group(1)) <= MAX_NUMBER_INDENT:
            found.append((i, int(m.group(2))))
    return found


def _chain_from(candidates: list, start: int) -> list:
    """Follow the running sequence 1, 2, 3... from candidates[start]; skip everything else."""
    chain = [candidates[start]]
    for line_index, number in candidates[start + 1:]:
        if number == chain[-1][1] + 1:
            chain.append((line_index, number))
    return chain


def _best_chain(candidates: list) -> list:
    """Longest unbroken 1..N chain; on a tie the LATER start wins (headnote before judgment)."""
    best = []
    for start, (_, number) in enumerate(candidates):
        if number != 1:
            continue
        chain = _chain_from(candidates, start)
        if len(chain) >= len(best):
            best = chain
    return best


def detect_paragraph_starts(lines: list) -> tuple:
    """
    Decide the numbering style and where each numbered paragraph starts.
    Returns (style, chain) where style is "N." / "[N]" / "none" and chain is
    a list of (line_index, paragraph_number).
    """
    dot_chain = _best_chain(_candidates(lines, DOT_NUMBER))
    bracket_chain = _best_chain(_candidates(lines, BRACKET_NUMBER))
    if not dot_chain and not bracket_chain:
        return "none", []
    if len(bracket_chain) > len(dot_chain):
        return "[N]", bracket_chain
    return "N.", dot_chain


# ---------------------------------------------------------------------------
# Turning lines into paragraph text
# ---------------------------------------------------------------------------
def _strip_number(raw: str, style: str) -> tuple:
    """Split a paragraph's first line into (text after the number, column where that text starts)."""
    pattern = DOT_NUMBER if style == "N." else BRACKET_NUMBER
    m = pattern.match(raw)
    rest = (m.group(3) or "") if m else raw.strip()
    text_column = raw.find(rest) if rest else len(raw)
    return rest, text_column


def _collapse(text: str) -> str:
    """CanLII justifies text with runs of spaces; collapse them to one."""
    return re.sub(r" {2,}", " ", text.strip())


def _join_lines(items: list) -> str:
    """
    Unwrap PDF lines into running text. items = [(text, [indent columns]), ...].
    Lines at a similar indentation are joined with a space (one block of
    prose); an indentation jump starts a new line (block quote, footnote,
    annotation). A hyphen at a line end is joined without a space.
    """
    out = ""
    previous_columns = None
    for text, columns in items:
        text = _collapse(text)
        if not text:
            continue
        if previous_columns is None:
            out = text
        else:
            gap = min(abs(a - b) for a in previous_columns for b in columns)
            if gap >= BLOCK_INDENT_JUMP:
                out += "\n" + text
            elif re.search(r"[A-Za-z]-$", out) and text[:1].isalpha():
                out += text            # "anti-" + "social" -> "anti-social"
            else:
                out += " " + text
        previous_columns = columns
    return out


def _paragraph_text(lines: list, first: int, last: int, number: int, style: str) -> str:
    """Build one paragraph's text from lines[first:last]."""
    items = []
    for i in range(first, last):
        raw = lines[i]["raw"]
        if i == first:
            rest, text_column = _strip_number(raw, style)
            # The first line's text may hang off the number (CanLII) or be
            # indented to match the body (Jade), so either column can match.
            items.append((rest, [lines[i]["indent"], text_column]))
            continue
        # Jade repeats the current paragraph number at the top of a new page
        # ("6." on its own line, or "2.  Following paragraph cited by:").
        # That number is a continuation marker, not text, so remove it.
        if raw.strip() in (f"{number}.", f"[{number}]"):
            continue
        repeat = (DOT_NUMBER if style == "N." else BRACKET_NUMBER).match(raw)
        if repeat and int(repeat.group(2)) == number and len(repeat.group(1)) <= MAX_NUMBER_INDENT:
            rest, text_column = _strip_number(raw, style)
            items.append((rest, [lines[i]["indent"], text_column]))
            continue
        items.append((raw, [lines[i]["indent"]]))
    return _join_lines(items)


# ---------------------------------------------------------------------------
# Optional: remove Jade "cited by" annotations (default OFF)
# ---------------------------------------------------------------------------
def _body_indent(lines: list) -> int:
    """The most common indentation of text lines = the body-text margin."""
    counts = Counter(line["indent"] for line in lines)
    return counts.most_common(1)[0][0] if counts else 0


def strip_jade_cited_by(lines: list) -> tuple:
    """
    Remove BarNet Jade's citation annotations, which are not part of the judgment:

      (a) inline blocks that start with "N.  Following paragraph cited by:" and
          list (and quote) LATER cases that cited paragraph N; and
      (b) the trailing "Cited by:" index after the end of the judgment.

    An inline block ends at the first line back at the body-text margin (the
    paragraph's own text resumes there). The paragraph number on the header
    line is kept, so paragraph detection is unaffected.

    Returns (kept_lines, removed_line_count). This is a heuristic based on
    Jade's layout; it is OFF by default (see README, Test Case 3).
    """
    # (b) Cut the trailing index: everything from the LAST "Cited by:" line.
    cut = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if TRAILING_INDEX_HEADER.match(lines[i]["raw"]):
            cut = i
            break
    body_margin = _body_indent(lines[:cut])
    kept, removed, inside_block = [], len(lines) - cut, False
    for line in lines[:cut]:
        raw = line["raw"]
        if CITED_BY_HEADER.match(raw):
            inside_block = True
            removed += 1
            number = DOT_NUMBER.match(raw)
            if number:  # keep "6." so paragraph 6 still starts here
                kept.append({**line, "raw": f"{number.group(1)}{number.group(2)}."})
            continue
        if inside_block:
            is_number_only = re.fullmatch(r"\s*\d{1,4}\.\s*", raw) is not None
            back_at_margin = line["indent"] <= body_margin + 2 and not is_number_only
            starts_paragraph = DOT_NUMBER.match(raw) and line["indent"] <= MAX_NUMBER_INDENT and not is_number_only
            if back_at_margin or starts_paragraph:
                inside_block = False
            else:
                removed += 1
                continue
        kept.append(line)
    return kept, removed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def make_source_id(filename: str) -> str:
    """A safe id for Chroma metadata, e.g. 'Brown_v_The_Queen_2020_VSCA_212'."""
    stem = re.sub(r"\.pdf$", "", filename, flags=re.I)
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_") or "source"


def ingest_pdf(pdf_bytes: bytes, filename: str, exclude_cited_by: bool = False) -> dict:
    """
    Read one source PDF and return:
        {
          "source_id", "filename", "n_pages", "readable", "numbering_style",
          "paragraphs": [{"para": 1, "page": 3, "text": "..."}, ...],
          "front_matter": text before paragraph 1 (cover, catchwords, headnote),
          "first_pages_text": cleaned text of the first two pages (for step 3),
          "warnings": [...]
        }
    """
    warnings = []
    page_texts = extract_page_texts(pdf_bytes)
    result = {
        "source_id": make_source_id(filename),
        "filename": filename,
        "n_pages": len(page_texts),
        "readable": is_readable(page_texts),
        "numbering_style": "none",
        "paragraphs": [],
        "front_matter": "",
        "first_pages_text": "",
        "excluded_cited_by_lines": 0,
        "warnings": warnings,
    }
    if not result["readable"]:
        warnings.append(
            "Almost no text could be extracted - this looks like a scanned (image-only) PDF. "
            "Upload a text-searchable copy (e.g. from Jade, AustLII or CanLII) or run OCR first."
        )
        return result

    lines = _page_lines(page_texts)
    first_pages = [l for l in lines if l["page"] <= 2]
    result["first_pages_text"] = "\n".join(_collapse(l["raw"]) for l in first_pages)[:6000]

    if exclude_cited_by:
        lines, removed = strip_jade_cited_by(lines)
        result["excluded_cited_by_lines"] = removed

    style, chain = detect_paragraph_starts(lines)
    result["numbering_style"] = style
    if not chain:
        warnings.append("No numbered paragraphs were detected, so paragraph pinpoints cannot be checked.")
        # Fall back to one 'paragraph' per page so the text is still searchable.
        for page_number in sorted({l["page"] for l in lines}):
            page_lines = [(l["raw"], [l["indent"]]) for l in lines if l["page"] == page_number]
            result["paragraphs"].append({"para": 0, "page": page_number, "text": _join_lines(page_lines)})
        return result

    # Front matter = everything before paragraph 1.
    first_line = chain[0][0]
    result["front_matter"] = _join_lines([(l["raw"], [l["indent"]]) for l in lines[:first_line]])

    # Each paragraph runs to the start of the next one. The last paragraph
    # runs to the end of the document (so any trailing material attaches to it).
    for k, (line_index, number) in enumerate(chain):
        end = chain[k + 1][0] if k + 1 < len(chain) else len(lines)
        text = _paragraph_text(lines, line_index, end, number, style)
        result["paragraphs"].append({"para": number, "page": lines[line_index]["page"], "text": text})

    return result


def paragraph_lookup(source: dict) -> dict:
    """{paragraph number: paragraph dict} for quick pinpoint fetches."""
    return {p["para"]: p for p in source["paragraphs"] if p["para"]}


# ---------------------------------------------------------------------------
# Small helpers for the preview (deterministic, no AI)
# ---------------------------------------------------------------------------
MNC_PATTERN = re.compile(r"\[(\d{4})\]\s+([A-Z][A-Za-z]{1,9})\s+(\d{1,5})")   # [2020] VSCA 212
CANLII_PATTERN = re.compile(r"\b(\d{4})\s+(ON|BC|AB|SK|MB|QC|NS|NB|NL|PE|YK|NT|NU)?(CA|SC|SCC|CJ|QB|KB)\s+(\d{1,5})\b")


def guess_citation(source: dict) -> str:
    """First medium-neutral citation on the first pages, as a quick check for the user."""
    text = source.get("first_pages_text", "")
    m = MNC_PATTERN.search(text)
    if m:
        return f"[{m.group(1)}] {m.group(2)} {m.group(3)}"
    m = CANLII_PATTERN.search(text)
    if m:
        return m.group(0)
    return ""

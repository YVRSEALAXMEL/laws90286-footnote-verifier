"""
STEP 1 - INGEST THE DRAFT (no AI)

python-docx does not expose footnotes, so we open the .docx as what it
really is (a zip file of XML parts) and read two parts directly:

    word/document.xml   the body text. Each footnote is anchored by a
                        <w:footnoteReference w:id="20"/> element.
    word/footnotes.xml  the footnote texts, keyed by the same w:id.

Important legal-citation detail: the w:id is an internal Word number
(our test drafts start at 20). The number the reader sees, and the number
used in cross-references such as "Brown (n 1)", is the ORDER in which the
footnotes appear in the body. So we number footnotes 1, 2, 3... by order
of appearance and keep the Word id only for reference.

Footnotes with ids -1 and 0 (type "separator" / "continuationSeparator")
are Word's horizontal separator lines, not real footnotes. We ignore them.
"""

import io
import re
import zipfile

from lxml import etree

# The WordprocessingML namespace. Every tag in the XML is prefixed "w:".
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}

# Footnote "types" that are Word furniture, not real footnotes.
SEPARATOR_TYPES = {"separator", "continuationSeparator", "continuationNotice"}

# How a footnote marker is written into the extracted body text.
# Example: "...materially reduced the sentence'.[^1] A five-member bench..."
MARKER = "[^{n}]"


def _tag(element) -> str:
    """Return a tag name without its namespace, e.g. '{...}t' -> 't'."""
    return etree.QName(element).localname


def _read_part(docx_bytes: bytes, part_name: str):
    """Open the .docx zip and parse one XML part (None if it is missing)."""
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as z:
        if part_name not in z.namelist():
            return None
        return etree.fromstring(z.read(part_name))


def _paragraph_pieces(paragraph):
    """
    Walk one <w:p> in document order and yield its pieces:
        ("text", "some words")   ordinary text
        ("fn", "20")             a footnote reference with its Word id

    We skip deleted text in tracked changes (w:delText) and field codes
    (w:instrText), because neither is part of the visible text.
    """
    for el in paragraph.iter():
        name = _tag(el)
        if name == "t":
            yield ("text", el.text or "")
        elif name == "tab":
            yield ("text", "\t")
        elif name in ("br", "cr"):
            yield ("text", "\n")
        elif name == "noBreakHyphen":
            yield ("text", "-")
        elif name == "footnoteReference":
            yield ("fn", el.get(f"{{{W}}}id"))


def _footnote_texts(footnotes_root) -> dict:
    """Map Word footnote id -> footnote text (separators skipped)."""
    texts = {}
    if footnotes_root is None:
        return texts
    for fn in footnotes_root.findall("w:footnote", NS):
        fn_id = fn.get(f"{{{W}}}id")
        fn_type = fn.get(f"{{{W}}}type")
        if fn_type in SEPARATOR_TYPES or fn_id in ("-1", "0"):
            continue
        paragraphs = []
        for p in fn.findall(".//w:p", NS):
            # <w:footnoteRef/> is the little superscript number at the start
            # of the footnote. It has no text, so joining the w:t pieces is enough.
            paragraphs.append("".join(text for kind, text in _paragraph_pieces(p) if kind == "text"))
        texts[fn_id] = _clean_spaces(" ".join(paragraphs))
    return texts


def _clean_spaces(text: str) -> str:
    """Collapse runs of spaces/tabs and trim, keeping the words unchanged."""
    return re.sub(r"[ \t ]+", " ", text).strip()


def extract_draft(docx_bytes: bytes) -> dict:
    """
    Read a .docx and return:
        {
          "body_text":  body paragraphs joined by blank lines, with [^n] markers,
          "paragraphs": list of body paragraph strings (with markers),
          "footnotes":  [{"fn": 1, "word_id": "20", "text": "...",
                          "paragraph_index": 0, "span": "text the footnote supports",
                          "quoted_phrases": [...]}, ...]
        }

    "span" is the text from the previous footnote marker (or the start of the
    paragraph, whichever is later) up to this footnote's marker. Under AGLC a
    footnote supports the text that precedes it, so this is the passage the
    footnote is responsible for.
    """
    document_root = _read_part(docx_bytes, "word/document.xml")
    if document_root is None:
        raise ValueError("This file has no word/document.xml - is it really a .docx?")
    fn_texts = _footnote_texts(_read_part(docx_bytes, "word/footnotes.xml"))

    body = document_root.find("w:body", NS)
    paragraphs, footnotes = [], []
    display_number = 0

    for p_index, p in enumerate(body.iter(f"{{{W}}}p")):
        text_so_far = ""   # the paragraph text built up so far (with markers)
        span_start = 0     # where the current footnote's span begins
        for kind, value in _paragraph_pieces(p):
            if kind == "text":
                text_so_far += value
                continue
            # A footnote reference: give it the next visible number.
            if value not in fn_texts:
                continue  # a reference to a separator or a missing footnote
            display_number += 1
            span = _clean_spaces(text_so_far[span_start:])
            # Remove any earlier markers that fall inside the span.
            span = _clean_spaces(re.sub(r"\[\^\d+\]", "", span))
            footnotes.append({
                "fn": display_number,
                "word_id": value,
                "text": fn_texts[value],
                "paragraph_index": len(paragraphs),
                "span": span,
                "quoted_phrases": extract_quoted_phrases(span),
            })
            text_so_far += MARKER.format(n=display_number)
            span_start = len(text_so_far)
        # Empty paragraphs are kept (as "") so that paragraph_index values
        # on the footnotes still line up with this list.
        paragraphs.append(_clean_spaces(text_so_far))

    body_text = "\n\n".join(p for p in paragraphs if p)
    return {"body_text": body_text, "paragraphs": paragraphs, "footnotes": footnotes}


# ---------------------------------------------------------------------------
# Quoted phrases (deterministic)
# ---------------------------------------------------------------------------
# Legal writing uses single curly quotes for quotations (AGLC 1.5) and the
# same character (’) as an apostrophe: O’Neill, Director’s, Strawson’s.
# A closing quote is therefore a ’ that is NOT followed by a letter, and an
# opening quote is a ‘ that is not preceded by a letter.
_QUOTE_PATTERNS = [
    re.compile(r"(?<![A-Za-z])‘(.+?)’(?![A-Za-z])", re.S),   # ‘single curly’
    re.compile(r"“(.+?)”", re.S),                            # “double curly”
    re.compile(r'(?<![A-Za-z])"(.+?)"(?![A-Za-z])', re.S),    # "straight double"
    re.compile(r"(?<![A-Za-z])'(.+?)'(?![A-Za-z])", re.S),    # 'straight single'
]


def extract_quoted_phrases(text: str) -> list:
    """
    Return the quoted phrases in a passage, in order, without the quote marks.

    >>> extract_quoted_phrases("held it ‘worse’ and ‘total’.")
    ['worse', 'total']
    """
    found = []
    for pattern in _QUOTE_PATTERNS:
        for match in pattern.finditer(text):
            phrase = _clean_spaces(match.group(1))
            if phrase:
                found.append((match.start(), phrase))
    # Sort by position and drop duplicates (keep the first occurrence).
    seen, phrases = set(), []
    for _, phrase in sorted(found):
        if phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)
    return phrases

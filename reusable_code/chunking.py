"""
SUB-CHUNKING OVERSIZED PARAGRAPHS (part of step 2, no AI)

The PARAGRAPH stays the primary unit, because pinpoints are paragraph
numbers. Most judgment paragraphs are short (about 70-150 tokens), but a few
are far too long for good semantic retrieval:

    Kapua [111]  ~820 tokens (mostly block quotes from other cases)
    Brown [6]    ~1,700 tokens (swollen by Jade "cited by" blocks)
    Brown [89]   ~10,600 tokens (Jade's trailing citation index attaches to
                 the last paragraph) - this is OVER the ~8,191-token input
                 limit of text-embedding-3-large, so it cannot be embedded
                 at all without splitting.

So ONLY paragraphs over MAX_CHUNK_TOKENS are split, using semchunk
(MIT licence, by Isaacus / Umar Butler). semchunk splits at natural
boundaries - newlines first, then sentence ends, then words - and with
offsets=True it tells us exactly where each piece came from:

    paragraph_text[char_start:char_end] == piece      (guaranteed)

Every sub-chunk keeps {source_id, para, page} and adds sub_index and
char_start/char_end, so a semantic hit always maps back to its paragraph
pinpoint and an exact location inside it.

We do NOT use semchunk's AI-chunking mode (chunking_model=...): it needs a
separate Isaacus API key that our users won't have.
"""

import functools
import logging

import semchunk

MAX_CHUNK_TOKENS = 400     # paragraphs longer than this are sub-chunked
CHUNK_OVERLAP = 0.1        # 10% overlap between neighbouring sub-chunks

log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_token_counter():
    """
    Return (counter, description). text-embedding-3-large uses the
    cl100k_base encoding. tiktoken downloads it on first use; if that fails
    (no internet), fall back to a rough word-based estimate and log a warning.
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding, "tiktoken cl100k_base"
    except Exception as error:  # network failure, missing package, etc.
        log.warning("tiktoken unavailable (%s); using a word-based token estimate.", error)
        return (lambda text: int(len(text.split()) * 1.35)), "word-based estimate (tiktoken unavailable)"


def count_tokens(text: str) -> int:
    counter, _ = get_token_counter()
    if callable(counter):
        return counter(text)
    return len(counter.encode(text, disallowed_special=()))


@functools.lru_cache(maxsize=1)
def get_chunker():
    """Build the semchunk chunker once (it is reused for every paragraph)."""
    counter, _ = get_token_counter()
    return semchunk.chunkerify(counter, chunk_size=MAX_CHUNK_TOKENS)


def chunk_id(source_id: str, para: int, sub_index: int) -> str:
    return f"{source_id}::p{para}::s{sub_index}"


def chunk_source(source: dict) -> list:
    """
    Turn a source's paragraphs into chunks ready for Chroma.
    Short paragraphs become one chunk; long ones become several sub-chunks.
    """
    chunks = []
    for paragraph in source["paragraphs"]:
        text = paragraph["text"]
        if not text.strip():
            continue
        if count_tokens(text) <= MAX_CHUNK_TOKENS:
            pieces, offsets = [text], [(0, len(text))]
        else:
            pieces, offsets = get_chunker()(text, offsets=True, overlap=CHUNK_OVERLAP)
        # Pages are numbered from 1, so para 0 (a page-based fallback) still gets a unique id.
        id_number = paragraph["para"] if paragraph["para"] else f"page{paragraph['page']}"
        for sub_index, (piece, (start, end)) in enumerate(zip(pieces, offsets)):
            chunks.append({
                "id": chunk_id(source["source_id"], id_number, sub_index),
                "text": piece,
                "metadata": {
                    "source_id": source["source_id"],
                    "para": paragraph["para"],
                    "page": paragraph["page"],
                    "sub_index": sub_index,
                    "n_sub_chunks": len(pieces),
                    "char_start": start,
                    "char_end": end,
                },
            })
    return chunks


def rejoin_sub_chunks(chunks: list) -> str:
    """
    Rebuild a paragraph from its sub-chunks using their character offsets.
    Overlapping text is only kept once. Any gap between chunks (semchunk
    drops the whitespace it splits on) becomes a single space.
    """
    ordered = sorted(chunks, key=lambda c: c["metadata"]["char_start"])
    text, covered_to = "", 0
    for c in ordered:
        start, end = c["metadata"]["char_start"], c["metadata"]["char_end"]
        if end <= covered_to:
            continue                      # fully inside what we already have
        if start > covered_to and text:
            text += " "                   # whitespace semchunk split on
        text += c["text"][max(0, covered_to - start):]
        covered_to = end
    return text

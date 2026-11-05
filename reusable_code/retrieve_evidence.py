"""
STEP 6 - RETRIEVE EVIDENCE for one citation unit (1 AI call per turn)

Three channels, merged and de-duplicated:

  (a) PINPOINT FETCH (no AI): the cited paragraphs, as WHOLE paragraphs,
      plus one neighbour either side (a pinpoint that is one paragraph out
      still counts as correct). Fetched straight from the parsed source, so
      a sub-chunked paragraph comes back whole.
  (b) QUOTE MATCH (no AI): the paragraphs where the draft's quoted phrases
      actually occur. This is what finds 'No such requirement operated' at
      [111] when the draft cites [113].
  (c) SEMANTIC SEARCH (Lab 2 _plan_queries -> _search): the AI proposes up
      to 3 short queries, and Chroma searches them FILTERED TO THE CITED
      SOURCE ONLY. Later turns see the work record and must try new queries.

Very long paragraphs (only Brown [89], which carries Jade's 26,000-token
citation index) are not sent whole: we send the sub-chunks that mention the
quoted words, or the first sub-chunks, and say so in the excerpt label.
"""

import functools

from reusable_code.ai_client import ask_ai
from reusable_code.chunking import chunk_source
from reusable_code.grounding import phrase_in_text, within_pinpoint
from reusable_code.pdf_ingest import paragraph_lookup
from reusable_code.vector_store import search

MAX_QUERIES = 3          # as in Lab 2
N_RESULTS = 4            # chunks returned per query
NEIGHBOURS = 1           # paragraphs either side of a pinpoint
MAX_PINPOINT_PARAS = 15  # cost control for long pinpoint ranges
MAX_QUOTE_PARAS = 4      # per quoted phrase
MAX_PARAGRAPH_CHARS = 8000

PLAN_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}

PLAN_INSTRUCTIONS = f"""You are a research agent checking a footnote in a law paper. You can search ONE judgment
(the cited source) with semantic search over its paragraphs. Propose the next search queries (at most {MAX_QUERIES}).
Each query should be a short, specific phrase in the language a judge would use, aimed at the passage that would
SUPPORT or REFUTE the proposition - including the court's own conclusion on the point, and any passage where the
words are attributed to someone else (the trial judge, counsel, an expert, another case).
Do NOT repeat queries already in the work record; target what the reviewer said was missing."""


@functools.lru_cache(maxsize=32)
def _sub_chunks(source_id: str, para: int, text: str) -> tuple:
    """Sub-chunks of one long paragraph (cached: Brown [89] takes a moment to split)."""
    return tuple(chunk_source({"source_id": source_id, "paragraphs": [{"para": para, "page": 0, "text": text}]}))


def _excerpt(source: dict, para: int, text: str, phrases: list) -> list:
    """Whole paragraph if it is a sensible size, otherwise its most relevant sub-chunks."""
    if len(text) <= MAX_PARAGRAPH_CHARS:
        return [{"para": para, "sub_index": None, "text": text, "note": ""}]
    pieces = list(_sub_chunks(source["source_id"], para, text))
    relevant = [c for c in pieces if any(phrase_in_text(q, c["text"]) for q in phrases)] or pieces[:2]
    total = len(pieces)
    return [{"para": para, "sub_index": c["metadata"]["sub_index"], "text": c["text"],
             "note": f"part {c['metadata']['sub_index'] + 1} of {total}; paragraph too long to show whole"}
            for c in relevant[:3]]


def fetch_pinpoint_evidence(source: dict, pinpoint: dict, phrases: list) -> list:
    """Channel (a): cited paragraphs + neighbours, as whole paragraphs."""
    if pinpoint.get("type") != "paragraph":
        return []
    lookup = paragraph_lookup(source)
    cited = pinpoint["paras"][:MAX_PINPOINT_PARAS]
    wanted = sorted({p + d for p in cited for d in range(-NEIGHBOURS, NEIGHBOURS + 1)} & set(lookup))
    evidence = []
    for para in wanted:
        origin = "cited pinpoint" if para in cited else "neighbour of pinpoint"
        for item in _excerpt(source, para, lookup[para]["text"], phrases):
            evidence.append({**item, "origin": origin, "distance": None, "query": ""})
    return evidence


def fetch_quote_evidence(source: dict, quote_results: list, pinpoint: dict) -> list:
    """Channel (b): paragraphs containing the draft's quoted phrases (nearest the pinpoint first)."""
    lookup = paragraph_lookup(source)
    cited = pinpoint.get("paras") or []
    evidence = []
    for q in quote_results:
        if not q["found"]:
            continue
        paras = sorted(q["paras"], key=lambda p: min([abs(p - c) for c in cited] or [0]))
        for para in [p for p in paras if p in lookup][:MAX_QUOTE_PARAS]:   # (no lookup for unnumbered text)
            for item in _excerpt(source, para, lookup[para]["text"], [q["phrase"]]):
                evidence.append({**item, "origin": f"contains the quoted words ‘{q['phrase'][:40]}’",
                                 "distance": None, "query": ""})
    return evidence


def _plan_queries(unit: dict, claim: dict, source_label: str, work_record: str, effort: str = "low") -> list:
    """Channel (c), THINK: the AI decides what to search for (Lab 2 pattern)."""
    prompt = (f"Cited source: {source_label}\n"
              f"Citation: {unit['citation_text']}\n"
              f"Proposition the source is cited for: {unit['unit_proposition'] or claim.get('proposition', '')}\n"
              f"Draft attributes it to: {claim.get('attributed_to', 'the source')}\n"
              f"Quoted phrases: {unit['unit_quoted_phrases']}\n\n"
              f"Work record so far:\n{work_record or '(nothing yet - this is the first turn)'}")
    queries = ask_ai(prompt, instructions=PLAN_INSTRUCTIONS, effort=effort, schema=PLAN_SCHEMA,
                     step=f"6 plan queries (fn {unit['unit_id']})")["queries"]
    return [q.strip() for q in queries if q.strip()][:MAX_QUERIES]


def _search(collection, queries: list, source_id: str) -> list:
    """Channel (c), ACT: run each query against Chroma, filtered to the cited source; dedupe by id."""
    seen, hits = set(), []
    for query in queries:
        for hit in search(collection, query, source_id, n_results=N_RESULTS):
            if hit["id"] in seen:
                continue
            seen.add(hit["id"])
            meta = hit["meta"]
            note = (f"part {meta['sub_index'] + 1} of {meta['n_sub_chunks']}" if meta["n_sub_chunks"] > 1 else "")
            hits.append({"para": meta["para"], "sub_index": meta["sub_index"] if meta["n_sub_chunks"] > 1 else None,
                         "text": hit["text"], "note": note, "origin": "semantic search",
                         "distance": hit["distance"], "query": query})
    return hits


def merge_evidence(*groups) -> list:
    """
    Merge evidence channels. A whole paragraph beats any sub-chunk of it;
    duplicates are dropped; the result is in paragraph order (as a judge reads).
    """
    whole, parts = {}, {}
    for group in groups:
        for item in group:
            if item["sub_index"] is None:
                if item["para"] not in whole:
                    whole[item["para"]] = item
            else:
                parts.setdefault((item["para"], item["sub_index"]), item)
    merged = list(whole.values()) + [p for (para, _), p in parts.items() if para not in whole]
    return sorted(merged, key=lambda e: (e["para"], e["sub_index"] or 0))


def evidence_label(source_short: str, item: dict, cited: list) -> str:
    """The label the judge must cite, e.g. '[Kapua [111] | cited pinpoint]'."""
    extra = [item["origin"]]
    if item["note"]:
        extra.append(item["note"])
    if item["distance"] is not None:
        extra.append(f"distance {item['distance']}")
    if cited and item["origin"] == "semantic search" and not within_pinpoint(item["para"], cited):
        extra.append("outside the cited pinpoint")
    return f"[{source_short} [{item['para']}] | " + "; ".join(extra) + "]"

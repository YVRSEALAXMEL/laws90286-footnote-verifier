"""
RUN ORCHESTRATION - runs steps 3 to 11 in order, each in its own st.status
box (as in Lab 2), and returns everything the results page needs.

Steps 1 and 2 (parsing the .docx and the PDFs) already ran when the files
were loaded, so the user could see the ingestion preview first.
"""

import hashlib
import time

import streamlit as st

from reusable_code.ai_client import AIError, reset_call_log
from reusable_code.api_key import current_api_key
from reusable_code.chunking import chunk_source, get_token_counter
from reusable_code.extract_claims import extract_claims
from reusable_code.identify_source import identify_source, source_label
from reusable_code.reflect import failed_record, verify_unit
from reusable_code.report import summarise_report
from reusable_code.resolve_citations import NOT_PROVIDED, resolve_citations
from reusable_code.vector_store import make_embedding_function, new_collection, store_chunks


def _fingerprint(sources: list, api_key: str) -> str:
    """Identifies a set of parsed sources, so the evidence database is only rebuilt when they change."""
    h = hashlib.sha256(api_key.encode())
    for s in sources:
        h.update(s["source_id"].encode())
        h.update(str(len(s["paragraphs"])).encode())
        h.update(str(s["excluded_cited_by_lines"]).encode())
        h.update("".join(p["text"][:200] for p in s["paragraphs"][:5]).encode())
    return h.hexdigest()


def get_collection(sources: list):
    """
    Build (or reuse) this session's Chroma collection. Re-running the same files,
    e.g. running a test case twice, reuses the embeddings instead of paying again.
    Returns (collection, number_of_chunks, rebuilt?).
    """
    key = current_api_key()
    fingerprint = _fingerprint(sources, key)
    cached = st.session_state.get("vector_cache")
    if cached and cached["fingerprint"] == fingerprint:
        return cached["collection"], cached["n_chunks"], False
    collection = new_collection(make_embedding_function(key),
                                old_name=cached["collection"].name if cached else None)
    chunks = [c for s in sources if s["readable"] for c in chunk_source(s)]
    store_chunks(collection, chunks)
    st.session_state.vector_cache = {"fingerprint": fingerprint, "collection": collection, "n_chunks": len(chunks)}
    return collection, len(chunks), True


def run_verification(draft_name: str, draft: dict, sources: list, settings: dict) -> dict:
    reset_call_log()
    started = time.time()

    # Step 3: identify each source (1 AI call per PDF).
    with st.status("Step 3 · Identifying the uploaded sources...", expanded=True) as status:
        for s in sources:
            s["identity"] = identify_source(s)
            s["label"] = source_label(s["identity"], s)
            ident = s["identity"]
            st.write(f"**{s['filename']}** → {s['label']} · {ident['court']} · {ident['decision_date']} · "
                     f"reported: {', '.join(ident['reported_citations']) or 'none printed'}")
        status.update(label=f"Step 3 · Identified {len(sources)} source(s)", state="complete", expanded=False)

    # Evidence database (embeddings, not counted as AI calls).
    with st.status("Building the evidence database (Chroma + text-embedding-3-large)...") as status:
        collection, n_chunks, rebuilt = get_collection(sources)
        status.update(label=f"Evidence database: {n_chunks} chunks "
                            f"({'embedded now' if rebuilt else 'reused from the previous run'}; "
                            f"tokens counted with {get_token_counter()[1]})",
                      state="complete", expanded=False)

    # Step 4: extract claims (1 AI call).
    footnotes = draft["footnotes"][: settings["max_footnotes"]]
    with st.status("Step 4 · Extracting the claim behind each footnote...") as status:
        claims = extract_claims(draft, footnotes)
        for c in claims:
            st.write(f"**fn {c['fn']}** ({c['attributed_to']}): {c['proposition']}")
        status.update(label=f"Step 4 · Extracted {len(claims)} claim(s)", state="complete", expanded=False)

    # Step 5: resolve citations (1 AI call + deterministic rules).
    with st.status("Step 5 · Resolving citations (Ibid, (n X), parallel citations)...") as status:
        units = resolve_citations(footnotes, claims, sources)
        for u in units:
            target = next((s["label"] for s in sources if s["source_id"] == u["matched_source_id"]), "not provided")
            st.write(f"**fn {u['unit_id']}** {u['citation_text']} → **{target}**, pinpoint "
                     f"{u['pinpoint_text'] or '(none)'} ({u['pinpoint']['type']})"
                     + (f"  \n   ↳ {' '.join(u['enforcement'] + u['warnings'])}" if u['enforcement'] or u['warnings'] else ""))
        checkable = sum(1 for u in units if u["matched_source_id"] != NOT_PROVIDED)
        status.update(label=f"Step 5 · {len(units)} citation unit(s); {checkable} matched to an uploaded source",
                      state="complete", expanded=False)

    # Steps 6-10: each unit in its own reflection loop.
    st.markdown("**Steps 6–10 · Checking each citation** (retrieve → judge → grounding checks → reflect → fix)")
    sources_by_id = {s["source_id"]: s for s in sources}
    claims_by_fn = {c["fn"]: c for c in claims}
    results = []
    for unit in units:
        claim = claims_by_fn.get(unit["fn"], {})
        try:
            results.append(verify_unit(unit, claim, sources_by_id, sources, collection, settings))
        except AIError as error:
            if "API key" in str(error):
                raise                     # no point carrying on with a rejected key
            st.error(f"fn {unit['unit_id']}: {error}")
            results.append(failed_record(unit, claim, sources_by_id.get(unit["matched_source_id"]), str(error)))

    # Step 11: summary (1 AI call).
    with st.status("Step 11 · Writing the summary...") as status:
        summary = summarise_report(results)
        status.update(label="Step 11 · Summary written", state="complete", expanded=False)

    return {
        "draft_name": draft_name,
        "sources": [{"filename": s["filename"], "label": s["label"], "identity": s["identity"],
                     "readable": s["readable"], "numbering_style": s["numbering_style"],
                     "n_paragraphs": len(s["paragraphs"]), "warnings": s["warnings"]} for s in sources],
        "claims": claims,
        "units": units,
        "results": results,
        "summary": summary,
        "settings": dict(settings),
        "ai_call_count": st.session_state.ai_call_count,
        "embedding_requests": st.session_state.embedding_requests,
        "call_log": list(st.session_state.ai_call_log),
        "runtime_seconds": time.time() - started,
        "n_footnotes_in_draft": len(draft["footnotes"]),
    }

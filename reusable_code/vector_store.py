"""
EVIDENCE DATABASE - Chroma, as in Lab 2, with three changes for this app:

  1. chromadb.EphemeralClient(): in memory only. Nothing is written to disk
     and nothing can be committed to GitHub by accident.
  2. The OpenAI embedding function is built from the USER'S session key,
     not from os.environ (which won't exist on Streamlit Cloud).
  3. One collection per run, with a unique name. All EphemeralClient()
     instances in one Python process share the same store, so on Streamlit
     Cloud (one process, many users) a fixed name like "knowledge_base"
     would mix one user's sources with another's.

Every stored chunk has metadata {source_id, para, page, sub_index, ...} so a
search can be FILTERED to one source: where={"source_id": ...}. That filter
is what stops the tool from "finding" Brown's words in Kapua (TC2 fn 5).
"""

import uuid

import chromadb
from chromadb.utils import embedding_functions

from reusable_code.ai_client import log_embedding_requests

EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 64          # chunks per embedding request


def make_embedding_function(api_key: str):
    return embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=EMBEDDING_MODEL)


def new_collection(embedding_function, old_name: str | None = None):
    """Create a fresh, uniquely named collection (and drop the previous one for this session)."""
    client = chromadb.EphemeralClient()
    if old_name:
        try:
            client.delete_collection(old_name)
        except Exception:
            pass  # already gone
    name = f"fnv-{uuid.uuid4().hex[:16]}"
    return client.create_collection(name=name, embedding_function=embedding_function,
                                    configuration={"hnsw": {"space": "cosine"}})


def store_chunks(collection, chunks: list) -> int:
    """Embed and store chunks in batches. Returns the number of embedding requests made."""
    requests = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        collection.add(ids=[c["id"] for c in batch],
                       documents=[c["text"] for c in batch],
                       metadatas=[c["metadata"] for c in batch])
        requests += 1
    log_embedding_requests(requests)
    return requests


def search(collection, query: str, source_id: str, n_results: int = 4) -> list:
    """
    Semantic search restricted to ONE source. Chroma returns lists of lists
    (one inner list per query text); we send one query, so we take [0].
    """
    result = collection.query(query_texts=[query], n_results=n_results, where={"source_id": source_id})
    log_embedding_requests(1)
    hits = []
    for chunk_id, text, meta, distance in zip(result["ids"][0], result["documents"][0],
                                              result["metadatas"][0], result["distances"][0]):
        hits.append({"id": chunk_id, "text": text, "meta": meta, "distance": round(float(distance), 3),
                     "query": query})
    return hits

"""The Chroma vector store.

One persistent store holds several collections, named strategy__backend, so a
single deployment can serve more than one index and the evaluation can compare
them without rebuilding anything.

Every collection records the embedding backend it was built with. A query is
embedded through that recorded backend, never through a default - embedding a
query with a different model than the index does not raise, it just returns
plausible-looking nonsense.
"""

from __future__ import annotations

import datetime as _dt
import threading
from dataclasses import dataclass

from chromadb import PersistentClient
from chromadb.config import Settings

from .chunking import Chunk
from .config import DB_PATH
from .embeddings import embed, embed_query, get_backend

_client = None

# Chroma's Rust bindings are not safe to call concurrently from several threads:
# under the parallel evaluation, overlapping queries raise
# "'RustBindingsAPI' object has no attribute 'bindings'". Reads are fast (tens of
# milliseconds), so serialising them costs little and removes the failure mode.
_lock = threading.Lock()


def client():
    global _client
    with _lock:
        if _client is None:
            DB_PATH.mkdir(parents=True, exist_ok=True)
            _client = PersistentClient(
                path=str(DB_PATH), settings=Settings(anonymized_telemetry=False)
            )
        return _client


def collection_name(strategy: str, backend: str) -> str:
    return f"{strategy}__{backend}"


@dataclass(frozen=True)
class Retrieved:
    page_content: str
    metadata: dict
    distance: float = 0.0

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")


def build(chunks: list[Chunk], strategy: str, backend: str, progress=None) -> int:
    """Embed chunks and (re)create the collection. Returns the vector count."""
    name = collection_name(strategy, backend)
    chroma = client()
    if name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(name)

    texts = [c.page_content for c in chunks]
    vectors = embed(texts, backend, progress=progress)

    spec = get_backend(backend)
    collection = chroma.create_collection(
        name,
        metadata={
            "strategy": strategy,
            "embedding_backend": backend,
            "embedding_model": spec.model,
            "dimensions": spec.dimensions,
            "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        },
    )
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=vectors,
        documents=texts,
        metadatas=[c.metadata for c in chunks],
    )
    return collection.count()


def open_collection(strategy: str, backend: str):
    name = collection_name(strategy, backend)
    existing = [c.name for c in client().list_collections()]
    if name not in existing:
        raise FileNotFoundError(
            f"No index named {name!r}. Built indexes: {existing or 'none'}. "
            f"Run:  python ingest.py --strategy {strategy} --embeddings {backend}"
        )
    return client().get_collection(name)


def available_indexes() -> list[str]:
    return sorted(c.name for c in client().list_collections())


def query(collection, question: str, k: int) -> list[Retrieved]:
    """Embed the question through the collection's own backend, then search."""
    backend = collection.metadata["embedding_backend"]
    vector = embed_query(question, backend)  # outside the lock: no Chroma involved
    with _lock:
        results = collection.query(query_embeddings=[vector], n_results=k)
    return [
        Retrieved(page_content=doc, metadata=meta, distance=dist)
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]

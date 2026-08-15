"""Build a vector index from the knowledge base.

    python ingest.py --strategy llm       --embeddings gemini
    python ingest.py --strategy recursive --embeddings gemini

Chunking is cached to chunks/<strategy>.json, so building the same chunks
against a second embedding backend costs only the embedding calls - the LLM
chunking pass is not repeated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from rag.chunking import STRATEGIES, Chunk, llm_chunks, recursive_chunks
from rag.config import ROOT
from rag.documents import fetch_documents
from rag.embeddings import BACKENDS, backend_available, get_backend
from rag.providers import DEFAULT_MODEL
from rag.store import build

CHUNK_CACHE = ROOT / "chunks"


def _bar(done: int, total: int, label: str) -> None:
    width = 30
    filled = int(width * done / total) if total else width
    sys.stdout.write(f"\r  {label} [{'#' * filled}{'.' * (width - filled)}] {done}/{total}")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")


def load_or_make_chunks(strategy: str, model: str, refresh: bool) -> list[Chunk]:
    CHUNK_CACHE.mkdir(exist_ok=True)
    cache_file = CHUNK_CACHE / f"{strategy}.json"

    if cache_file.exists() and not refresh:
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        print(f"  reusing {len(raw)} cached chunks from {cache_file.relative_to(ROOT)}")
        return [Chunk(**c) for c in raw]

    documents = fetch_documents()
    print(f"  loaded {len(documents)} documents")

    start = time.perf_counter()
    if strategy == "recursive":
        chunks = recursive_chunks(documents)
    else:
        print(f"  chunking with {model}")
        chunks = llm_chunks(documents, model, progress=lambda d, t: _bar(d, t, "chunking "))
    print(f"  produced {len(chunks)} chunks in {time.perf_counter() - start:.1f}s")

    cache_file.write_text(
        json.dumps([c.__dict__ for c in chunks], indent=1), encoding="utf-8"
    )
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=STRATEGIES, default="llm")
    parser.add_argument("--embeddings", choices=sorted(BACKENDS), default="local")
    parser.add_argument("--chunk-model", default=DEFAULT_MODEL)
    parser.add_argument("--refresh", action="store_true", help="re-chunk, ignoring the cache")
    parser.add_argument(
        "--paid",
        action="store_true",
        help="allow a paid embedding backend. Without it, a paid backend aborts.",
    )
    args = parser.parse_args()

    # Every index is already built and committed, so re-running this is normally
    # a mistake rather than a need. A paid backend has to be asked for by name.
    if get_backend(args.embeddings).cost_per_1m_tokens > 0 and not args.paid:
        print(
            f"Refusing to run: --embeddings {args.embeddings} is a paid backend "
            f"({get_backend(args.embeddings).model}).\n"
            "The index it builds is already committed under vectorstore/.\n"
            "Use --embeddings local, or pass --paid if you really want to rebuild it."
        )
        return 2

    if not backend_available(args.embeddings):
        spec = get_backend(args.embeddings)
        print(f"{spec.key_env} is not set - cannot use the {args.embeddings} embedding backend.")
        return 1

    print(f"Building index: strategy={args.strategy} embeddings={args.embeddings}")
    chunks = load_or_make_chunks(args.strategy, args.chunk_model, args.refresh)

    characters = sum(len(c.page_content) for c in chunks)
    spec = get_backend(args.embeddings)
    estimate = characters / 4 / 1_000_000 * spec.cost_per_1m_tokens
    print(f"  embedding {len(chunks)} chunks with {spec.model} (~${estimate:.3f})")

    start = time.perf_counter()
    count = build(chunks, args.strategy, args.embeddings, progress=lambda d, t: _bar(d, t, "embedding"))
    print(f"  {count} vectors stored in {time.perf_counter() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

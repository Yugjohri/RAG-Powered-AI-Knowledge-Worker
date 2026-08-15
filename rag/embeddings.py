"""Embedding backends.

  local   all-MiniLM-L6-v2 via ONNX      384 dims   no key, no quota, no cost
  openai  text-embedding-3-large        3072 dims   paid, $0.13 / 1M tokens
  gemini  gemini-embedding-001          3072 dims   free tier, 1000 requests/DAY

`local` is what the public demo ships. It is the only one of the three that puts
no credential on the server at all, and the only one with no ceiling a visitor
could exhaust. It arrives with chromadb - onnxruntime, no torch - and the model
is fetched once on first use and then cached.

The gemini quota is per request, not per token, and litellm issues one request
per input string: embedding this knowledge base once costs ~900 of the 1000
daily requests, which is why it builds indexes far too slowly to be the default.
Measured: 877 chunks took 15+ minutes and still hit the cap on gemini, and
28 seconds on openai.

Which backend an index was built with is recorded in the Chroma collection's
metadata, so a query can never be embedded with a different model than the
index - that mismatch does not raise, it silently returns nonsense.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import host_key
from .scrub import scrub_exception

litellm.suppress_debug_info = True


@dataclass(frozen=True)
class EmbeddingBackend:
    name: str
    model: str
    key_env: str  # empty means the backend needs no credential
    dimensions: int
    batch_size: int
    cost_per_1m_tokens: float


BACKENDS: dict[str, EmbeddingBackend] = {
    "local": EmbeddingBackend("local", "all-MiniLM-L6-v2 (onnx)", "", 384, 64, 0.0),
    "openai": EmbeddingBackend(
        "openai", "text-embedding-3-large", "OPENAI_API_KEY", 3072, 256, 0.13
    ),
    "gemini": EmbeddingBackend(
        "gemini", "gemini/gemini-embedding-001", "GOOGLE_API_KEY", 3072, 16, 0.0
    ),
}

DEFAULT_BACKEND = "local"

_onnx = None


def _onnx_model():
    """chromadb's bundled MiniLM. Loaded lazily: the first call fetches ~80 MB."""
    global _onnx
    if _onnx is None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        _onnx = ONNXMiniLM_L6_V2()
    return _onnx


def get_backend(name: str) -> EmbeddingBackend:
    if name not in BACKENDS:
        raise ValueError(f"Unknown embedding backend {name!r}; have {sorted(BACKENDS)}")
    return BACKENDS[name]


def backend_available(name: str) -> bool:
    backend = get_backend(name)
    return not backend.key_env or host_key(backend.key_env) is not None


@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=2, min=15, max=120),
    reraise=True,
)
def _embed_batch(backend: EmbeddingBackend, texts: list[str]) -> list[list[float]]:
    if backend.name == "local":
        # ONNXMiniLM returns numpy arrays; Chroma wants plain Python floats.
        return [[float(x) for x in vector] for vector in _onnx_model()(texts)]
    response = litellm.embedding(model=backend.model, input=texts)
    return [item["embedding"] for item in response.data]


def embed(texts: list[str], backend_name: str, progress=None) -> list[list[float]]:
    """Embed texts with the named backend, batched to that provider's limit."""
    backend = get_backend(backend_name)
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, backend.batch_size):
        batch = texts[start : start + backend.batch_size]
        try:
            vectors.extend(_embed_batch(backend, batch))
        except Exception as exc:  # noqa: BLE001 - message must be credential-free
            raise RuntimeError(
                f"Embedding failed on batch starting at {start}: {scrub_exception(exc)}"
            ) from None
        if progress:
            progress(min(start + backend.batch_size, total), total)
        if backend.name == "gemini" and start + backend.batch_size < total:
            # The free tier's ceiling is tokens per minute. Pace to roughly
            # 20k tokens/min so the retry path is the exception, not the norm.
            time.sleep(max(0.0, sum(len(t) for t in batch) / 4 / 20_000 * 60))
    return vectors


def embed_query(text: str, backend_name: str) -> list[float]:
    return embed([text], backend_name)[0]

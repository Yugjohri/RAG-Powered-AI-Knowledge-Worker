"""Two chunking strategies, so the evaluation can measure what chunking is worth.

  recursive  Split on paragraph/line/word boundaries at a fixed size. The
             standard baseline; costs nothing and takes milliseconds.

  llm        Ask a model to divide each document into semantically coherent
             chunks, each carrying a generated headline and summary in front of
             the original text. Costs an LLM call per document, and the headline
             and summary are what make short queries match - the retrieval
             target stops being raw prose and becomes prose plus an index card.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import litellm
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from .documents import Document
from .scrub import scrub_exception

litellm.suppress_debug_info = True

# ~1200 characters is about 200 words, which makes the prompt's "about 50 words
# of overlap" the ~25% it claims to be. The original notebook used 100, which
# asked for chunks smaller than the overlap it also asked for.
AVERAGE_CHUNK_SIZE = 1200
RECURSIVE_CHUNK_SIZE = 500
RECURSIVE_OVERLAP = 200

WORKERS = 4


@dataclass(frozen=True)
class Chunk:
    page_content: str
    source: str
    doc_type: str

    @property
    def metadata(self) -> dict:
        return {"source": self.source, "type": self.doc_type}


# --------------------------------------------------------------------------
# recursive character splitting - the baseline
# --------------------------------------------------------------------------

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_recursive(text: str, size: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []

    separator = next((s for s in separators if s and s in text), "")
    if not separator:
        return [text[i : i + size] for i in range(0, len(text), size)]

    rest = separators[separators.index(separator) + 1 :]
    pieces, current = [], ""
    for part in text.split(separator):
        candidate = part if not current else current + separator + part
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                pieces.append(current)
            if len(part) > size:
                pieces.extend(_split_recursive(part, size, rest))
                current = ""
            else:
                current = part
    if current.strip():
        pieces.append(current)
    return pieces


def recursive_chunks(documents: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        pieces = _split_recursive(document.text, RECURSIVE_CHUNK_SIZE, _SEPARATORS)
        for index, piece in enumerate(pieces):
            # Re-attach the tail of the previous piece as overlap.
            if index and RECURSIVE_OVERLAP:
                piece = pieces[index - 1][-RECURSIVE_OVERLAP:] + piece
            chunks.append(Chunk(piece.strip(), document.source, document.doc_type))
    return chunks


# --------------------------------------------------------------------------
# LLM semantic chunking
# --------------------------------------------------------------------------


class _Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )


class _Chunks(BaseModel):
    chunks: list[_Chunk]


def _make_prompt(document: Document) -> str:
    how_many = (len(document.text) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document.doc_type}
The document has been retrieved from: {document.source}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document.text}

Respond with the chunks.
"""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    reraise=True,
)
def _chunk_document(document: Document, model: str) -> list[Chunk]:
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": _make_prompt(document)}],
        response_format=_Chunks,
    )
    parsed = _Chunks.model_validate_json(response.choices[0].message.content)
    return [
        Chunk(
            page_content=f"{c.headline}\n\n{c.summary}\n\n{c.original_text}",
            source=document.source,
            doc_type=document.doc_type,
        )
        for c in parsed.chunks
    ]


def llm_chunks(documents: list[Document], model: str, progress=None) -> list[Chunk]:
    chunks: list[Chunk] = []
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_chunk_document, d, model): d for d in documents}
        for future in as_completed(futures):
            document = futures[future]
            try:
                chunks.extend(future.result())
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Chunking failed for {document.source}: {scrub_exception(exc)}"
                ) from None
            done += 1
            if progress:
                progress(done, len(documents))
    return chunks


STRATEGIES = ("recursive", "llm")

"""Loading the knowledge base off disk."""

from __future__ import annotations

from dataclasses import dataclass

from .config import KNOWLEDGE_BASE_PATH


@dataclass(frozen=True)
class Document:
    doc_type: str
    source: str
    text: str


def fetch_documents(root=None) -> list[Document]:
    """Every .md under knowledge-base/, tagged with its top-level folder name.

    The source is stored relative to the knowledge base so it is identical on
    Windows and in the Linux container, and so no absolute path from the
    author's machine leaks into the deployed app's citations.
    """
    root = root or KNOWLEDGE_BASE_PATH
    documents: list[Document] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for file in sorted(folder.rglob("*.md")):
            documents.append(
                Document(
                    doc_type=folder.name,
                    source=file.relative_to(root).as_posix(),
                    text=file.read_text(encoding="utf-8"),
                )
            )
    return documents

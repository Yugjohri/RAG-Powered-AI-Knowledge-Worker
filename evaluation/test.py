"""The test set: 150 questions with keywords, reference answers and categories."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

TEST_FILE = Path(__file__).parent / "tests.jsonl"


class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")


def load_tests(path: str | Path | None = None) -> list[TestQuestion]:
    """Load test questions from a JSONL file."""
    file = Path(path) if path else TEST_FILE
    with open(file, "r", encoding="utf-8") as f:
        return [TestQuestion(**json.loads(line)) for line in f if line.strip()]

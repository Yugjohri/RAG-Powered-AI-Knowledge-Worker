"""Print the committed evaluation results as a markdown table.

Used to keep the README's numbers identical to the ones the app renders - they
come from the same JSON files, so they cannot drift apart.

    python scripts/results_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import results

def short_model(config) -> str:
    """Which model did the rewriting and re-ranking.

    Blank for the basic pipeline, which makes no LLM call during retrieval at
    all - printing the configured model there would imply it influenced a number
    it cannot have influenced.

    Otherwise the last path segment of the litellm id: 'groq/openai/gpt-oss-120b'
    is too wide for a column, and the provider is implied by the model name.
    """
    if config.get("pipeline") == "basic":
        return "-"
    return str(config.get("model", "?")).split("/")[-1]


COLUMNS = [
    ("Pipeline", lambda c, r, a: c.get("pipeline", "?")),
    ("Chunking", lambda c, r, a: c.get("strategy", "?")),
    ("Embeddings", lambda c, r, a: c.get("embeddings", "?")),
    # Named because the rows were not all measured on the same one: Groq's free
    # quota ran out mid-project, so later rows were rewritten and re-ranked by
    # Gemini. Hiding that would make the rows look more comparable than they are.
    ("Rewrite/rerank", lambda c, r, a: short_model(c)),
    ("MRR", lambda c, r, a: f"{r['mrr']:.3f}" if r else "-"),
    ("nDCG", lambda c, r, a: f"{r['ndcg']:.3f}" if r else "-"),
    ("Coverage", lambda c, r, a: f"{r['keyword_coverage']:.1f}%" if r else "-"),
    ("Accuracy", lambda c, r, a: f"{a['accuracy']:.2f}" if a else "-"),
    ("Complete", lambda c, r, a: f"{a['completeness']:.2f}" if a else "-"),
    ("Relevant", lambda c, r, a: f"{a['relevance']:.2f}" if a else "-"),
    (
        "Median s",
        lambda c, r, a: f"{(a or r or {}).get('median_seconds', 0):.2f}",
    ),
    ("n", lambda c, r, a: str((a or r or {}).get("n", 0))),
]


def main() -> int:
    data = results.all_results()
    if not data:
        print("No results in results/ - run: python -m evaluation.eval")
        return 1

    rows = []
    for name in sorted(data):
        payload = data[name]
        config = payload.get("config", {})
        retrieval = payload.get("retrieval")
        answers = payload.get("answers")
        rows.append([fn(config, retrieval, answers) for _, fn in COLUMNS])

    headers = [h for h, _ in COLUMNS]
    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))
    ]

    def line(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    print(line(headers))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print(line(row))

    n = {p.get("questions") for p in data.values()}
    print(f"\nQuestions per configuration: {sorted(n)}")
    judges = {p.get("judge") for p in data.values() if p.get("answers")}
    if judges:
        print(f"Answer judge: {', '.join(sorted(j for j in judges if j))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

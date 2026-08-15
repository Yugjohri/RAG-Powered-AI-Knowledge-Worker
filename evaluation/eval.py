"""Score a RAG configuration against the 150-question test set.

    # free: local embeddings, retrieval only - no LLM call, deterministic
    python -m evaluation.eval --embeddings local --pipeline basic --retrieval-only

    # free: the deployed configuration, retrieval only
    python -m evaluation.eval --strategy llm --embeddings local --pipeline advanced \
                              --retrieval-only

    # paid, and refused unless --paid is passed: anything using OpenAI, whether
    # as the embedding backend or as the answer judge
    python -m evaluation.eval --embeddings openai --pipeline basic --retrieval-only --paid
    python -m evaluation.eval --answers-only --judge gpt-4.1-mini --paid

Retrieval metrics (MRR, nDCG, keyword coverage) need no judge and depend only on
the index, so the embedding and chunking comparisons are run on those. Answer
metrics are graded by a larger model than the one that produced the answer.

Results are merged into results/<name>.json, so --retrieval-only and
--answers-only can be run separately without either discarding the other.
Everything is measured at the FULL retrieval budget: the shared demo key's token
ceiling is a property of that key, not of the system being measured.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import litellm
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from evaluation.test import TestQuestion, load_tests
from rag.config import RESULTS_PATH
from rag.embeddings import BACKENDS
from rag.pipeline import ADVANCED, BASIC, FULL, answer_question, fetch_context
from rag.providers import BY_ID, DEMO, DEFAULT_MODEL
from rag.scrub import scrub_exception

litellm.suppress_debug_info = True

# The judge needs throughput more than it needs to be free: Groq's free tier
# rate-limited 29 of 150 gradings, which silently shrinks the sample. A larger
# model than the one being graded, with headroom to actually finish the set.
JUDGE_MODEL = "gpt-4.1-mini"
WORKERS = 4


# --------------------------------------------------------------------------
# retrieval metrics
# --------------------------------------------------------------------------


@dataclass
class RetrievalScore:
    mrr: float
    ndcg: float
    keywords_found: int
    total_keywords: int
    keyword_coverage: float
    seconds: float


def reciprocal_rank(keyword: str, docs) -> float:
    needle = keyword.lower()
    for rank, doc in enumerate(docs, start=1):
        if needle in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg(keyword: str, docs, k: int) -> float:
    needle = keyword.lower()
    relevances = [1 if needle in d.page_content.lower() else 0 for d in docs[:k]]
    ideal = _dcg(sorted(relevances, reverse=True))
    return _dcg(relevances) / ideal if ideal > 0 else 0.0


def score_retrieval(test: TestQuestion, k: int = 10, **kwargs) -> RetrievalScore:
    start = time.perf_counter()
    docs = fetch_context(test.question, budget=FULL, **kwargs)
    elapsed = time.perf_counter() - start

    ranks = [reciprocal_rank(kw, docs) for kw in test.keywords]
    ndcgs = [ndcg(kw, docs, k) for kw in test.keywords]
    found = sum(1 for r in ranks if r > 0)
    total = len(test.keywords)

    return RetrievalScore(
        mrr=statistics.fmean(ranks) if ranks else 0.0,
        ndcg=statistics.fmean(ndcgs) if ndcgs else 0.0,
        keywords_found=found,
        total_keywords=total,
        keyword_coverage=found / total * 100 if total else 0.0,
        seconds=elapsed,
    )


# --------------------------------------------------------------------------
# answer metrics - LLM as judge
# --------------------------------------------------------------------------


class AnswerVerdict(BaseModel):
    feedback: str = Field(description="Concise feedback comparing the answer to the reference")
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong - any wrong answer must score 1) to 5 (perfectly accurate). An acceptable answer scores 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (missing key information) to 5 (all information from the reference answer is present). Only answer 5 if ALL of it is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (off-topic) to 5 (directly addresses the question and adds nothing extra)."
    )


@dataclass
class AnswerScore:
    accuracy: float
    completeness: float
    relevance: float
    seconds: float
    feedback: str = ""


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=60), reraise=True)
def judge(test: TestQuestion, generated: str, judge_model: str) -> AnswerVerdict:
    messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated}

Reference Answer:
{test.reference_answer}

Evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 for perfect answers.
2. Completeness: How thoroughly does it cover all the information in the reference answer?
3. Relevance: How directly does it answer the specific question, without padding?

Give detailed feedback and scores from 1 (very poor) to 5 (ideal) for each. If the answer is wrong, accuracy must be 1.""",
        },
    ]
    response = litellm.completion(
        model=judge_model, messages=messages, response_format=AnswerVerdict
    )
    return AnswerVerdict.model_validate_json(response.choices[0].message.content)


def score_answer(test: TestQuestion, judge_model: str, **kwargs) -> tuple[AnswerScore, str]:
    start = time.perf_counter()
    answer = answer_question(test.question, budget=FULL, **kwargs)
    elapsed = time.perf_counter() - start
    verdict = judge(test, answer.text, judge_model)
    return (
        AnswerScore(
            accuracy=verdict.accuracy,
            completeness=verdict.completeness,
            relevance=verdict.relevance,
            seconds=elapsed,
            feedback=verdict.feedback,
        ),
        answer.text,
    )


# --------------------------------------------------------------------------
# runners
# --------------------------------------------------------------------------


def _run_parallel(fn, tests, label, workers=WORKERS):
    results, failures = {}, []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, t): i for i, t in enumerate(tests)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append((tests[index].question, scrub_exception(exc)))
            done += 1
            sys.stdout.write(f"\r  {label}: {done}/{len(tests)}   ")
            sys.stdout.flush()
    sys.stdout.write("\n")
    if failures:
        print(f"  {len(failures)} failed, e.g. {failures[0][1][:160]}")
    return [results[i] for i in sorted(results)], [tests[i] for i in sorted(results)]


def evaluate_retrieval(tests, workers=WORKERS, **kwargs) -> dict:
    scores, kept = _run_parallel(
        lambda t: score_retrieval(t, **kwargs), tests, "retrieval", workers
    )
    by_category = defaultdict(list)
    for test, score in zip(kept, scores):
        by_category[test.category].append(score.mrr)
    return {
        "n": len(scores),
        "mrr": statistics.fmean(s.mrr for s in scores),
        "ndcg": statistics.fmean(s.ndcg for s in scores),
        "keyword_coverage": statistics.fmean(s.keyword_coverage for s in scores),
        "median_seconds": statistics.median(s.seconds for s in scores),
        "mrr_by_category": {c: statistics.fmean(v) for c, v in sorted(by_category.items())},
    }


def evaluate_answers(tests, judge_model=JUDGE_MODEL, workers=WORKERS, **kwargs) -> dict:
    scores, kept = _run_parallel(
        lambda t: score_answer(t, judge_model, **kwargs)[0], tests, "answers  ", workers
    )
    by_category = defaultdict(list)
    for test, score in zip(kept, scores):
        by_category[test.category].append(score.accuracy)
    return {
        "n": len(scores),
        "accuracy": statistics.fmean(s.accuracy for s in scores),
        "completeness": statistics.fmean(s.completeness for s in scores),
        "relevance": statistics.fmean(s.relevance for s in scores),
        "median_seconds": statistics.median(s.seconds for s in scores),
        "accuracy_by_category": {c: statistics.fmean(v) for c, v in sorted(by_category.items())},
    }


def save(name: str, payload: dict) -> None:
    """Merge into any existing result for this configuration.

    --retrieval-only and --answers-only each compute one half, so writing the
    whole file would silently discard the half this run did not measure.
    """
    RESULTS_PATH.mkdir(exist_ok=True)
    file = RESULTS_PATH / f"{name}.json"
    merged = {}
    if file.exists():
        try:
            merged = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    merged.update(payload)
    file.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"  -> {file.relative_to(RESULTS_PATH.parent)}")


def paid_spend_refusal(args) -> str:
    """Refuse to start a run that would bill a paid provider.

    The committed results were produced with a paid judge and, for two rows, a
    paid embedding backend, so those names stay as the defaults that document
    what was measured. But re-running the suite is not something that should be
    possible to do by reflex - a full pass is a few dollars, and the numbers are
    already in results/. Anything paid now has to be asked for by name.
    """
    if args.paid:
        return ""
    paid = []
    if BACKENDS[args.embeddings].cost_per_1m_tokens > 0:
        paid.append(f"--embeddings {args.embeddings} ({BACKENDS[args.embeddings].model})")
    for flag, model_id in (("--model", args.model), ("--judge", args.judge)):
        model = BY_ID.get(model_id)
        if model is not None and model.tier != DEMO:
            paid.append(f"{flag} {model_id}")
        elif model is None and not model_id.startswith(("groq/", "gemini/", "ollama/")):
            # Not in the registry and not on a known free provider - assume paid.
            paid.append(f"{flag} {model_id} (unrecognised provider, assumed paid)")
    if args.retrieval_only:
        paid = [p for p in paid if not p.startswith("--judge")]  # no judge call happens
    if not paid:
        return ""
    return (
        "Refusing to run: this would spend on a paid provider.\n  "
        + "\n  ".join(paid)
        + "\n\nThe published numbers for these are already in results/ and README.md.\n"
        "Free equivalents: --embeddings local, --model groq/openai/gpt-oss-120b.\n"
        "If you really do want to pay for this run, pass --paid."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="llm")
    parser.add_argument("--embeddings", default="local")
    parser.add_argument("--pipeline", default=ADVANCED, choices=[BASIC, ADVANCED])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--judge", default=JUDGE_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N questions")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--answers-only", action="store_true")
    parser.add_argument("--name", default="", help="output file stem")
    parser.add_argument(
        "--paid",
        action="store_true",
        help="allow a paid provider. Without it, any paid model or embedding backend aborts.",
    )
    args = parser.parse_args()

    if refusal := paid_spend_refusal(args):
        print(refusal, file=sys.stderr)
        return 2

    tests = load_tests()
    if args.limit:
        tests = tests[: args.limit]

    config = {
        "strategy": args.strategy,
        "embeddings": args.embeddings,
        "pipeline": args.pipeline,
        "model": args.model,
    }
    name = args.name or f"{args.pipeline}__{args.strategy}__{args.embeddings}"
    print(f"Evaluating {name} on {len(tests)} questions")

    payload = {
        "config": config,
        "judge": args.judge,
        "questions": len(tests),
        "budget": FULL.name,
    }

    if not args.answers_only:
        payload["retrieval"] = evaluate_retrieval(tests, workers=args.workers, **config)
        r = payload["retrieval"]
        print(
            f"  MRR {r['mrr']:.3f}   nDCG {r['ndcg']:.3f}   "
            f"coverage {r['keyword_coverage']:.1f}%   median {r['median_seconds']:.2f}s"
        )

    if not args.retrieval_only:
        payload["answers"] = evaluate_answers(
            tests, judge_model=args.judge, workers=args.workers, **config
        )
        a = payload["answers"]
        print(
            f"  accuracy {a['accuracy']:.2f}/5   completeness {a['completeness']:.2f}/5   "
            f"relevance {a['relevance']:.2f}/5   median {a['median_seconds']:.2f}s"
        )

    save(name, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

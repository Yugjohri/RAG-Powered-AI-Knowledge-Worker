"""The two RAG pipelines the app and the evaluation both use.

BASIC     embed the question -> top-k by cosine distance -> answer.
ADVANCED  rewrite the question into a keyword-shaped query, retrieve for both
          the original and the rewrite, merge, have a model re-rank the union,
          keep the top few -> answer.

The advanced path costs two extra LLM calls per question. Whether that buys
anything is measured, not assumed - see evaluation/.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import litellm
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import EMBEDDINGS, STRATEGY
from .scrub import scrub_exception
from .store import Retrieved, open_collection, query

litellm.suppress_debug_info = True

BASIC = "basic"
ADVANCED = "advanced"

BASIC_K = 8


@dataclass(frozen=True)
class Budget:
    """How deep to retrieve, given what the request is allowed to spend.

    Groq's free tier allows 8000 tokens per minute, and re-ranking 40 full
    chunks exceeds that in a single call - so the shared demo path retrieves
    fewer candidates and shows the re-ranker a truncated view of each. LLM-built
    chunks lead with a headline and a summary, which is most of what ranking
    needs, so the truncation costs less than it sounds.

    That constraint belongs to the shared free key, not to the system. A visitor
    on their own key gets the full-depth pipeline, and so does the offline
    evaluation - the published numbers describe the real thing, not the
    rate-limited one.
    """

    name: str
    retrieval_k: int
    final_k: int
    rerank_chars: int | None  # None means send the whole chunk


FULL = Budget("full", retrieval_k=20, final_k=10, rerank_chars=None)
FREE_TIER = Budget("free-tier", retrieval_k=12, final_k=8, rerank_chars=420)


def budget_for(own_key: bool) -> Budget:
    """Visitor's own key -> full depth. Shared demo key -> rate-limit-safe path.

    Takes whether the VISITOR supplied a key, not the key that was resolved -
    a demo-tier request also ends up holding a key (the host's), and that one
    is precisely the one with the token ceiling.
    """
    return FULL if own_key else FREE_TIER


SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""

RERANK_SYSTEM_PROMPT = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


@dataclass
class Answer:
    text: str
    chunks: list[Retrieved]
    timings: dict[str, float] = field(default_factory=dict)
    rewritten_query: str | None = None
    budget: str = FULL.name

    @property
    def total_seconds(self) -> float:
        return sum(self.timings.values())


class PipelineError(RuntimeError):
    pass


class ProviderRateLimited(PipelineError):
    """The provider refused for rate-limit reasons, after the retries gave up.

    Separate from PipelineError because it is the one failure another provider
    can fix: the caller can retry the same question on a different free model
    instead of showing the visitor an error.
    """


@retry(
    retry=retry_if_exception_type(litellm.RateLimitError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=16),
    reraise=True,
)
def _call(model: str, messages: list[dict], api_key: str | None, **kwargs):
    """Every LLM call in the pipeline goes through here, so the rate-limit
    backoff applies to all three stages - including the final answer, which
    previously had no retry and simply failed.

    Three attempts, not six. A longer backoff used to be the only defence
    against a rate limit, and it cost a measured 204 seconds before giving up -
    far past the point a visitor has left. The app now fails over to a free
    model on a different provider instead, so this retry only needs to cover a
    brief per-minute spike: ~12 seconds of waiting, then hand over to someone
    whose quota is not exhausted.
    """
    return litellm.completion(model=model, messages=messages, api_key=api_key, **kwargs)


def _complete(model: str, messages: list[dict], api_key: str | None = None, **kwargs):
    try:
        return _call(model, messages, api_key, **kwargs)
    except litellm.RateLimitError:
        # The provider's own message names the organisation and its exact token
        # budget. None of that helps whoever is reading the page, so it is
        # replaced rather than forwarded - and it does not claim to know which
        # tier the caller is on, because a visitor's own key lands here too.
        raise ProviderRateLimited(
            "The model provider is rate limiting this request. Wait a few seconds and ask "
            "again, or use a different model."
        ) from None
    except Exception as exc:  # noqa: BLE001 - the message reaches a browser
        raise PipelineError(scrub_exception(exc)) from None


# These stage-level retries exist for transient faults - a malformed response, a
# dropped connection. NOT for rate limits: _call already backs those off, so
# retrying here too multiplies the delay before failover can start, which cost a
# measured 29 seconds instead of 12.
@retry(
    retry=retry_if_not_exception_type(ProviderRateLimited),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def rewrite_query(question: str, model: str, history=None, api_key=None) -> str:
    prompt = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history or []}

And this is the user's current question:
{question}

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = _complete(model, [{"role": "system", "content": prompt}], api_key)
    return (response.choices[0].message.content or question).strip()


@retry(
    retry=retry_if_not_exception_type(ProviderRateLimited),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def rerank(
    question: str,
    chunks: list[Retrieved],
    model: str,
    api_key=None,
    rerank_chars: int | None = None,
) -> list[Retrieved]:
    user_prompt = (
        f"The user has asked the following question:\n\n{question}\n\n"
        "Order all the chunks of text by relevance to the question, from most relevant to least "
        "relevant. Include all the chunk ids you are provided with, reranked.\n\nHere are the chunks:\n\n"
    )
    for index, chunk in enumerate(chunks, start=1):
        excerpt = chunk.page_content
        if rerank_chars and len(excerpt) > rerank_chars:
            excerpt = excerpt[:rerank_chars] + "..."
        user_prompt += f"# CHUNK ID: {index}:\n\n{excerpt}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."

    response = _complete(
        model,
        [
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        api_key,
        response_format=RankOrder,
    )
    order = RankOrder.model_validate_json(response.choices[0].message.content).order

    # A model can drop, repeat or invent an id. Keep what is valid, in the order
    # given, then append anything it forgot so no chunk is silently lost.
    seen, ranked = set(), []
    for i in order:
        if 1 <= i <= len(chunks) and i not in seen:
            seen.add(i)
            ranked.append(chunks[i - 1])
    ranked.extend(chunks[i] for i in range(len(chunks)) if i + 1 not in seen)
    return ranked


def _merge(primary: list[Retrieved], extra: list[Retrieved]) -> list[Retrieved]:
    merged = list(primary)
    known = {c.page_content for c in primary}
    for chunk in extra:
        if chunk.page_content not in known:
            merged.append(chunk)
            known.add(chunk.page_content)
    return merged


def _context_block(chunks: list[Retrieved]) -> str:
    return "\n\n".join(f"Extract from {c.source}:\n{c.page_content}" for c in chunks)


def answer_question(
    question: str,
    history: list[dict] | None = None,
    *,
    # Defaulted from config, never hardcoded here: a caller that omits these
    # would otherwise silently query a different index than the one deployed,
    # and get a confidently wrong answer out of a perfectly working pipeline.
    strategy: str = STRATEGY,
    embeddings: str = EMBEDDINGS,
    pipeline: str = ADVANCED,
    model: str,
    api_key: str | None = None,
    budget: Budget | None = None,
) -> Answer:
    history = history or []
    budget = budget or FREE_TIER  # the safe default; callers opt into FULL
    collection = open_collection(strategy, embeddings)
    timings: dict[str, float] = {}
    rewritten = None

    if pipeline == BASIC:
        t = time.perf_counter()
        chunks = query(collection, question, BASIC_K)
        timings["retrieve"] = time.perf_counter() - t
    else:
        t = time.perf_counter()
        rewritten = rewrite_query(question, model, history, api_key)
        timings["rewrite"] = time.perf_counter() - t

        t = time.perf_counter()
        chunks = _merge(
            query(collection, question, budget.retrieval_k),
            query(collection, rewritten, budget.retrieval_k),
        )
        timings["retrieve"] = time.perf_counter() - t

        t = time.perf_counter()
        chunks = rerank(question, chunks, model, api_key, budget.rerank_chars)[: budget.final_k]
        timings["rerank"] = time.perf_counter() - t

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT.format(context=_context_block(chunks))}]
        + history
        + [{"role": "user", "content": question}]
    )
    t = time.perf_counter()
    response = _complete(model, messages, api_key)
    timings["answer"] = time.perf_counter() - t

    return Answer(
        text=response.choices[0].message.content or "",
        chunks=chunks,
        timings=timings,
        rewritten_query=rewritten,
        budget=budget.name,
    )


def answer_with_fallback(
    question: str,
    history: list[dict] | None = None,
    *,
    model: str,
    resolve,
    fallbacks: list[str] | None = None,
    **kwargs,
) -> tuple[Answer, str | None]:
    """Answer, moving to another free provider if this one is rate limited.

    Returns (answer, served_by) where served_by is None when the requested model
    answered, and the substitute's id when it did not - callers are expected to
    tell the user, because silently swapping the model they picked is a lie.

    `resolve` is a callable taking a model id and returning its API key, so the
    key policy stays in providers.py and this module never learns about tiers.
    Lives here rather than in the UI so the app, the deployment simulation and
    any script all get the same behaviour.
    """
    attempts = [model] + list(fallbacks or [])
    last: ProviderRateLimited | None = None

    for index, candidate in enumerate(attempts):
        try:
            answer = answer_question(
                question, history, model=candidate, api_key=resolve(candidate), **kwargs
            )
        except ProviderRateLimited as exc:
            last = exc
            continue
        return answer, (candidate if index else None)

    raise last if last else PipelineError("No model was available to answer.")


def fetch_context(
    question: str,
    *,
    strategy: str = STRATEGY,
    embeddings: str = EMBEDDINGS,
    pipeline: str = ADVANCED,
    model: str | None = None,
    api_key: str | None = None,
    budget: Budget | None = None,
) -> list[Retrieved]:
    """Retrieval only - what the retrieval metrics score.

    Defaults to the full-depth budget: the evaluation measures the system, not
    the shared free key's token ceiling.
    """
    budget = budget or FULL
    collection = open_collection(strategy, embeddings)
    if pipeline == BASIC:
        return query(collection, question, BASIC_K)
    rewritten = rewrite_query(question, model, None, api_key)
    chunks = _merge(
        query(collection, question, budget.retrieval_k),
        query(collection, rewritten, budget.retrieval_k),
    )
    return rerank(question, chunks, model, api_key, budget.rerank_chars)[: budget.final_k]

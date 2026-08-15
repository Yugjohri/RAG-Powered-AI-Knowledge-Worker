# Benchmarks

150 questions with keywords and written reference answers
(`evaluation/tests.jsonl`). Retrieval metrics — MRR, nDCG, keyword coverage —
depend only on the index, so they need no judge and no randomness. Answer scores
are graded 1–5 by `gpt-4.1-mini`, a larger model than the `gpt-4.1-nano` that
produced the answers. All rows measured at full retrieval depth.

Two columns deserve as much attention as the scores.

**Rewrite/rerank** — the `basic` rows make no LLM call during retrieval at all,
so they depend only on the index and are exactly reproducible. The `advanced`
rows do, and they were not all measured on the same model: Groq's free quota ran
out partway through, so later rows were rewritten and re-ranked by Gemini. Rows
sharing a model are comparable; rows that do not, are not.

**n** — the number of questions actually scored. Small `n` means questions were
lost to rate limiting, not to retrieval failures, and a metric over 62 questions
deserves less weight than the same metric over 150. Nothing is dropped here for
looking bad; every run that completed is in the table.

| Pipeline | Chunking  | Embeddings | Rewrite/rerank        | MRR   | nDCG  | Coverage | Accuracy | Complete | Relevant | Median s | n   |
|----------|-----------|------------|-----------------------|-------|-------|----------|----------|----------|----------|----------|-----|
| advanced | recursive | local      | gemini-3.5-flash-lite | 0.905 | 0.895 | 93.6%    | –        | –        | –        | 2.40     | 110 |
| advanced | recursive | local      | gemini-3.5-flash-lite | 0.942 | 0.922 | 96.7%    | –        | –        | –        | 2.08     | 62  |
| advanced | llm       | local      | gemini-3.5-flash-lite | 0.877 | 0.873 | 93.1%    | –        | –        | –        | 2.28     | 70  |
| advanced | llm       | local      | gpt-4.1-nano          | 0.843 | 0.825 | 92.1%    | 4.15     | 3.77     | 4.67     | 2.64     | 150 |
| advanced | llm       | openai     | gpt-4.1-nano          | 0.845 | 0.824 | 91.7%    | 4.25     | 3.80     | 4.64     | 6.54     | 135 |
| basic    | llm       | local      | –                     | 0.775 | 0.778 | 92.1%    | –        | –        | –        | 0.09     | 146 |
| basic    | llm       | openai     | –                     | 0.826 | 0.817 | 92.5%    | –        | –        | –        | 0.46     | 148 |
| basic    | recursive | local      | –                     | 0.774 | 0.789 | 91.9%    | –        | –        | –        | 0.09     | 149 |
| basic    | recursive | openai     | –                     | 0.828 | 0.827 | 93.3%    | –        | –        | –        | 0.47     | 149 |

## Re-ranking closes the gap between a free local embedder and a paid API

This is the result that decided the deployment. On simple top-k retrieval,
OpenAI's 3072-dimensional `text-embedding-3-large` beats a 384-dimensional
MiniLM running in-process by a clear margin — 0.826 against 0.775 MRR. Add query
rewriting and LLM re-ranking and the two become indistinguishable: **0.845
against 0.843**.

The free embedder is not a compromise the hosting forced on the project. It is
within noise of the paid one, at zero cost and with no key on the server, and the
retrieval stage runs in 0.09 s instead of 0.46 s because there is no network
round trip. The paid index is still built and committed so the comparison can be
re-run rather than taken on trust.

## The advanced pipeline earns its two extra API calls — on the weaker index

It lifts local embeddings from 0.775 to 0.843 MRR (+8.8%), and OpenAI embeddings
from 0.826 to only 0.845 (+2.3%). Re-ranking is largely compensating for what
the cheaper embedding model missed. That is a more useful way to describe it
than "re-ranking improves retrieval".

## LLM chunking does not just fail to help — it buries facts

76 documents became 311 semantic chunks against 877 from a character splitter,
and on aggregate the difference is nil: 0.775 vs 0.774 local, 0.826 vs 0.828
OpenAI. I expected the semantic chunks to win, and the averages said it was a tie.

Driving the deployed app said otherwise. Asked *"Who won the prestigious IIOTY
award in 2023?"* — the first question anyone would try — the LLM-chunked index
answered **"I don't have that information."** The answer is Maxine Thompson and
it is in the knowledge base twice.

The cause is the headline the chunker generates. It filed the award under a chunk
titled *"Compensation and Recognition"*, summarised as salary history: the award
is one line at the end of 1,285 characters that are otherwise a table of numbers.

| | LLM chunks | Recursive chunks |
|---|---|---|
| rank in raw vector search | **16** | **2** |
| position of the fact in its chunk | char 1,268 and 727 | char 36 and 181 |
| visible in the 420-char re-rank excerpt | **no** | yes |

The embedding is dominated by the headline and the salary table, so the chunk
ranks 16th for a question about an award — below the 12 candidates the free-tier
budget retrieves. And because the fact sits at character 1,268, the re-ranker
never sees it even when it is retrieved, so it cannot rescue it.

### The head-to-head, and why it is weaker than it looks

So I ran the two chunkers head to head at advanced depth — same re-ranking
model, same worker count, same session. Under identical conditions recursive
beats LLM chunking on every retrieval metric:

| | LLM chunks | Recursive chunks |
|---|---|---|
| MRR | 0.877 | **0.942** |
| nDCG | 0.873 | **0.922** |
| keyword coverage | 93.1% | **96.7%** |

**That comparison needs a caveat stated plainly.** Both arms lost more than half
the test set to rate limiting — n=70 and n=62 out of 150 — because by then Groq's
daily quota was gone and Gemini's was going. A 60-question sample is thin, and if
the dropped questions were not dropped at random the gap could be an artefact. I
do not think they were, since a rate limit depends on when a request lands rather
than what it asks, but that is an argument rather than a measurement.

What makes the conclusion trustworthy is not this table alone. Three independent
lines point the same way: the mechanism (rank 2 versus 16, the fact at character
36 versus 1,268), the head-to-head above, and the app answering the question
correctly on one index and wrongly on the other. The `basic` pair — the only rows
sharing every condition at full sample — shows a dead tie, which is exactly why
the aggregate missed this for so long.

The deployed row is the 0.905/n=110 one: a larger sample than the head-to-head,
measured on the configuration that actually ships.

## Where retrieval is weak is more interesting than the average

The test set labels each question by type, and the average hides a wide spread.
Both columns below are the recursive/local index — the only difference is
whether the advanced pipeline ran:

| Category | basic (n=149) | deployed: advanced (n=110) |
|---|---|---|
| temporal | 0.826 | **1.000** |
| numerical | 0.775 | **1.000** |
| direct_fact | 0.894 | **0.958** |
| relationship | 0.675 | **0.865** |
| comparative | 0.736 | **0.833** |
| spanning | 0.489 | **0.764** |
| holistic | 0.549 | **0.643** |

Two things fall out of this. Re-ranking helps *most* where plain similarity does
worst — `spanning` gains +0.275 and `relationship` +0.190, while `direct_fact`,
already near the ceiling, gains 0.064. And even after that lift, `holistic` and
`spanning` questions — the ones whose answer is distributed across several
documents — remain the weakest categories by a distance. That is the real
limitation of chunk-level retrieval here: re-ranking reorders a list of chunks,
and no reordering assembles an answer that no single chunk contains.

## Free-tier ceilings, and the design they forced

Three limits were discovered by hitting them, and all three changed the
architecture.

**One free provider is a single point of failure.** Benchmarking exhausted
Groq's per-minute token budget and the demo stopped answering — the retry backoff
spent a measured **204 seconds** discovering that, well past the point any
visitor has closed the tab. The retry dropped from six attempts to three, because
a long backoff is a poor substitute for a second provider, and a rate-limited
free model now fails over to a free model on a *different* provider. End to end
that turned a 289-second failure into a 35-second answer.

**Google's embedding API allows 1000 requests per day, and litellm sends one
request per input string.** Embedding this knowledge base once costs ~900 of
them. Measured: 877 chunks took over 15 minutes and still hit the cap, against 28
seconds on OpenAI. A free API was therefore not a viable embedding backend at all
— which is what pushed embeddings in-process, and that turned out better than
either API: no key, no quota, no per-query cost.

**Groq's free tier allows 8000 tokens per minute.** The original re-ranking step
sent 40 full chunks in one prompt, exceeding a whole minute's budget in a single
call. The fix is a `Budget`: the shared demo key retrieves 12 candidates and
re-ranks on 420-character excerpts, while your own key gets 20 candidates
re-ranked on complete chunks. The published numbers are all at full depth — the
constraint belongs to the shared key, not to the system.

## Reproducing these

```bash
# retrieval metrics - deterministic, no API call, nothing to pay
python -m evaluation.eval --strategy recursive --embeddings local --pipeline basic --retrieval-only
python scripts/results_table.py
```

The retrieval half costs nothing: local embeddings run in-process and the metrics
need no judge. The answer scores do cost money — they were graded by
`gpt-4.1-mini` — so `evaluation/eval.py` refuses to start any run touching a paid
provider unless `--paid` is passed explicitly. The committed `results/` are what
those runs produced, so the paid half never needs repeating to read the table.

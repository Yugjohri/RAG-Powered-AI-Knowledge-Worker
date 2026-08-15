# RAG-Powered AI Knowledge Worker

[![Try it live](https://img.shields.io/badge/Try_it_live-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://rag-powered-ai-knowledge-worker-xyotc8knhafumxdsw3w9zf.streamlit.app/)

**Live demo:** https://rag-powered-ai-knowledge-worker-xyotc8knhafumxdsw3w9zf.streamlit.app/

A retrieval-augmented assistant over a knowledge base of 76 internal documents —
contracts, employee records, product sheets — that **shows you its own
retrieval**. Every answer sits next to the query the system rewrote to search
with, the chunks it pulled, the order the re-ranker put them in, and how long
each stage took. You can check the grounding instead of trusting it.

Try *"Who won the prestigious IIOTY award in 2023?"* It runs on free-tier keys, so
it is rate limited, and sleeps after 12 quiet hours — first visit takes ~30s.

## How it works

```
knowledge-base/       76 markdown documents
   |
   |  chunking.py     recursive 500-char split, or an LLM asked to split
   |                  semantically and prepend a headline + summary
   v
chunks/*.json         cached, so re-embedding never re-chunks
   |
   |  embeddings.py   local ONNX MiniLM (384d) | text-embedding-3-large (3072d)
   v
vectorstore/          Chroma, one collection per strategy x backend
   |
   |  pipeline.py     BASIC     embed -> top-k
   |                  ADVANCED  rewrite -> retrieve x2 -> merge -> rerank -> top-k
   v
streamlit_app.py      the UI: answer + the retrieval behind it
```

Each collection records its embedding backend in its own metadata, and queries
are embedded through that recorded backend. Embedding a query with a different
model than the index does not raise — it silently returns plausible nonsense — so
it is the one thing here made impossible rather than documented.

## Running it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY - free, and the only one needed

python ingest.py --strategy recursive --embeddings local   # build the index
streamlit run streamlit_app.py                             # http://127.0.0.1:8501
```

`RAGWorker.ipynb` is the workbench: chunk-length distributions, the embedding
space via t-SNE, and the two chunking strategies side by side. It imports `rag/`
rather than duplicating it, so it cannot drift from the app.

## Nobody can spend my money

The demo is public and runs on my keys, so the constraint was making that safe
structurally, not trusting a spend limit to catch it afterwards.

- **Embeddings use no API at all.** The index is built with `all-MiniLM-L6-v2`
  in-process through ONNX — the model chromadb already bundles. No embedding key
  on the server, so nothing to leak and no per-query cost.
- **Paid models refuse before the key is read.** Models are tiered `demo` (free
  providers only) or `byok`; `resolve_key()` raises for a `byok` model with no
  visitor key, so the host key is never fetched, never mind spent.
- **Bring your own key removes every limit.** A key pasted into the UI is used
  for one request and dropped — not logged, stored, or written to disk.
- **Credentials cannot reach the browser.** Provider SDKs put the failing key in
  exception messages, so every error is regex-scrubbed to `<redacted>`.
- **One provider is a single point of failure**, so a rate-limited free model
  fails over to a *different* provider, and says so rather than switching
  silently.

## Benchmarks

150 questions with keywords and written reference answers
(`evaluation/tests.jsonl`), all at full retrieval depth. Retrieval metrics — MRR,
nDCG, coverage — depend only on the index, so they need no judge and no
randomness. Answer scores are graded 1–5 by `gpt-4.1-mini`, a larger model than
the `gpt-4.1-nano` that produced the answers.

Two columns deserve as much attention as the scores. **Rewrite/rerank**: `basic`
rows make no LLM call during retrieval, so they depend only on the index and are
exactly reproducible; `advanced` rows do, and were not all measured on the same
model, because Groq's free quota ran out partway through — rows sharing a model
are comparable, rows that do not are not. **n**: questions actually scored; a
small `n` means questions were lost to rate limiting, not to retrieval failures.
Nothing is dropped here for looking bad.

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

### Re-ranking closes the gap between a free local embedder and a paid API

This is the result that decided the deployment. On simple top-k retrieval,
OpenAI's 3072-dimensional `text-embedding-3-large` beats a 384-dimensional
MiniLM running in-process by a clear margin — 0.826 against 0.775 MRR. Add query
rewriting and LLM re-ranking and the two become indistinguishable: **0.845
against 0.843**.

The free embedder is not a compromise the hosting forced on the project: it is
within noise of the paid one, at zero cost, with no key on the server, and it
retrieves in 0.09 s instead of 0.46 s because there is no network round trip.
So the advanced pipeline earns its two extra API calls, but mostly on the weaker
index — +8.8% MRR on local embeddings against +2.3% on OpenAI. Re-ranking is
largely compensating for what the cheaper embedding model missed, which is more
useful than "re-ranking improves retrieval".

### LLM chunking does not just fail to help — it buries facts

76 documents became 311 semantic chunks against 877 from a character splitter,
and on aggregate the difference is nil: 0.775 vs 0.774 local, 0.826 vs 0.828
OpenAI. I expected the semantic chunks to win; the averages said it was a tie.

Driving the deployed app said otherwise. Asked *"Who won the prestigious IIOTY
award in 2023?"* — the first question anyone would try — the LLM-chunked index
answered **"I don't have that information."** The answer is Maxine Thompson, and
it is in the knowledge base twice. The chunker had filed the award under a chunk
titled *"Compensation and Recognition"* and summarised it as salary history: the
award is one line at the end of 1,285 characters that are otherwise a table of
numbers.

| | LLM chunks | Recursive chunks |
|---|---|---|
| rank in raw vector search | **16** | **2** |
| position of the fact in its chunk | char 1,268 and 727 | char 36 and 181 |
| visible in the 420-char re-rank excerpt | **no** | yes |

The embedding is dominated by the headline and the salary table, putting the
chunk 16th — below the 12 candidates the free-tier budget retrieves — and the
fact sits past the excerpt the re-ranker reads, so re-ranking cannot rescue it.
Head to head at advanced depth, same model and workers and session, recursive
wins on everything: MRR **0.942 vs 0.877**, nDCG **0.922 vs 0.873**, coverage
**96.7% vs 93.1%**.

**That head-to-head needs a caveat stated plainly.** Both arms lost more than
half the test set to rate limiting — n=70 and n=62 — so if the dropped questions
were not dropped at random, the gap could be an artefact. I do not think they
were, since a rate limit depends on when a request lands rather than what it
asks, but that is an argument rather than a measurement. What makes the
conclusion trustworthy is that three things agree: the mechanism above, the
head-to-head, and the app answering correctly on one index and wrongly on the
other. The deployed row is the 0.905/n=110 one — larger sample, and it is what
actually ships.

### Where retrieval is weak is more interesting than the average

Both columns are the recursive/local index; only the pipeline differs.

| Category | basic (n=149) | deployed: advanced (n=110) |
|---|---|---|
| temporal | 0.826 | **1.000** |
| numerical | 0.775 | **1.000** |
| direct_fact | 0.894 | **0.958** |
| relationship | 0.675 | **0.865** |
| comparative | 0.736 | **0.833** |
| spanning | 0.489 | **0.764** |
| holistic | 0.549 | **0.643** |

Re-ranking helps most where plain similarity does worst: `spanning` gains +0.275
and `relationship` +0.190, while `direct_fact`, already near the ceiling, gains
0.064. Yet `holistic` and `spanning` — answers spread across several documents —
stay weakest even after that lift. That is the real limit of chunk-level
retrieval: reordering chunks cannot assemble an answer no single chunk contains.

### Free-tier ceilings, and the design they forced

**One free provider is a single point of failure.** Benchmarking exhausted Groq's
per-minute budget and the demo stopped answering — the retry backoff spent a
measured 204 seconds discovering that. Retries dropped from six attempts to three
and a rate-limited model now fails over: a 289-second failure became a 35-second
answer.

**Google's embedding API allows 1000 requests/day, and litellm sends one per
input string** — ~900 to embed this base once. 877 chunks took over 15 minutes
and still hit the cap, against 28 seconds on OpenAI. That is what pushed
embeddings in-process: no key, no quota, no per-query cost.

**Groq's free tier allows 8000 tokens/minute**, and re-ranking 40 full chunks
exceeds a minute's budget in one call. Hence a `Budget`: the shared key gets 12
candidates re-ranked on 420-char excerpts, your own key gets 20 and complete
chunks. Published numbers are all at full depth — the constraint belongs to the
shared key, not the system.

### Reproducing these

```bash
python -m evaluation.eval --strategy recursive --embeddings local --pipeline basic --retrieval-only
python scripts/results_table.py
```

Free and deterministic. Answer scores do cost money, so `evaluation/eval.py`
refuses any run touching a paid provider unless `--paid` is passed.

---

See **[DEPLOY.md](DEPLOY.md)** for hosting and the pre-publish checks, and
**[LEGACY.md](LEGACY.md)** for the earlier LangChain versions. Insurellm is
fictional and the knowledge base synthetic; the test set, results and index are
all committed, so every number here can be re-derived from this repository.

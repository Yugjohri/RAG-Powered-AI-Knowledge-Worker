[![Try it live](https://img.shields.io/badge/Try_it_live-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](DEMO_URL_PENDING)

# RAG-Powered AI Knowledge Worker

A retrieval-augmented assistant over a private knowledge base of 76 internal
documents, which **shows you its own retrieval**. Every answer sits next to the
query the system rewrote to search with, the chunks it pulled, the order the
re-ranker put them in, and how long each stage took. You can check the grounding
instead of trusting it.

The part worth reading, though, is not the chat. It is that **every design
choice in it was settled by measurement.** Whether LLM-written chunks beat a
character splitter, whether a 3072-dimensional paid embedding model beats a
384-dimensional one running locally, whether query rewriting and re-ranking earn
their two extra API calls — all of it is scored over a committed 150-question
test set with reference answers, and the results are in the table below and in
the app's Benchmarks tab. Several of the answers were not what I expected.

## Measured, on this machine

150 questions with keywords and written reference answers (`evaluation/tests.jsonl`).
Answer scores are graded 1–5 by `gpt-4.1-mini`, a larger model than the
`gpt-4.1-nano` that produced the answers. All rows measured at full retrieval
depth. `n` is the number of questions actually scored, which is lower than 150
where a provider rate-limited some of them.

The **Rewrite/rerank** column matters for reading the rest. The `basic` rows make
no LLM call during retrieval at all, so their scores depend only on the index and
are exactly reproducible. The `advanced` rows do, and they were not all measured
on the same model — Groq's free quota ran out partway through the project, so
later rows were rewritten and re-ranked by Gemini instead. Rows sharing a model
are directly comparable; rows that do not, are not, and the column is there so
that is visible rather than buried in a footnote.

| Pipeline | Chunking  | Embeddings | Rewrite/rerank | MRR   | nDCG  | Coverage | Accuracy | Complete | Relevant | Median s | n   |
|----------|-----------|------------|----------------|-------|-------|----------|----------|----------|----------|----------|-----|
| advanced | recursive | local      | gemini-flash-lite | 0.905 | 0.895 | 93.6% | –        | –        | –        | 2.40     | 110 |
| advanced | llm       | local      | gpt-4.1-nano   | 0.843 | 0.825 | 92.1%    | 4.15     | 3.77     | 4.67     | 2.64     | 150 |
| advanced | llm       | openai     | gpt-4.1-nano   | 0.845 | 0.824 | 91.7%    | 4.25     | 3.80     | 4.64     | 6.54     | 135 |
| basic    | llm       | local      | –              | 0.775 | 0.778 | 92.1%    | –        | –        | –        | 0.09     | 146 |
| basic    | llm       | openai     | –              | 0.826 | 0.817 | 92.5%    | –        | –        | –        | 0.46     | 148 |
| basic    | recursive | local      | –              | 0.774 | 0.789 | 91.9%    | –        | –        | –        | 0.09     | 149 |
| basic    | recursive | openai     | –              | 0.828 | 0.827 | 93.3%    | –        | –        | –        | 0.47     | 149 |

## What the numbers say

**Re-ranking closes the gap between a free local embedder and a paid API.** This
is the result that decided the deployment. On simple top-k retrieval, OpenAI's
3072-dimensional `text-embedding-3-large` beats a 384-dimensional MiniLM running
in-process by a clear margin — 0.826 against 0.775 MRR. Add query rewriting and
LLM re-ranking and the two become indistinguishable: **0.845 against 0.843**.

The free embedder is not a compromise the hosting forced on the project. It is
within noise of the paid one, at zero cost and with no key on the server, and
the retrieval stage runs in 0.09 s instead of 0.46 s because there is no network
round trip. The paid index is still built and committed so the comparison can be
re-run rather than taken on trust.

**The advanced pipeline earns its two extra API calls** — but only on the weaker
index. It lifts local embeddings from 0.775 to 0.843 MRR (+8.8%), and OpenAI
embeddings from 0.826 to only 0.845 (+2.3%). Re-ranking is largely compensating
for what the cheaper embedding model missed. That is a more useful way to
describe it than "re-ranking improves retrieval".

**LLM chunking does not just fail to help — it buries facts.** 76 documents
became 311 semantic chunks against 877 from a character splitter, and on
aggregate the difference is nil: 0.775 vs 0.774 local, 0.826 vs 0.828 OpenAI. I
expected the semantic chunks to win, and the averages said it was a tie.

Driving the deployed app said otherwise. Asked *"Who won the prestigious IIOTY
award in 2023?"* — the first question anyone would try — the LLM-chunked index
answered **"I don't have that information."** The answer is Maxine Thompson and
it is in the knowledge base twice.

The cause is the headline the chunker generates. It filed the award under a
chunk titled *"Compensation and Recognition"*, summarised as salary history: the
award is one line at the end of 1,285 characters that are otherwise a table of
numbers. Two failures follow from that:

| | LLM chunks | Recursive chunks |
|---|---|---|
| rank in raw vector search | **16** | **2** |
| position of the fact in its chunk | char 1,268 and 727 | char 36 and 181 |
| visible in the 420-char re-rank excerpt | **no** | yes |

The embedding is dominated by the headline and the salary table, so the chunk
ranks 16th for a question about an award — below the 12 candidates the free-tier
budget retrieves. And because the fact sits at character 1,268, the re-ranker
never sees it even when it is retrieved, so it cannot rescue it.

That is why the deployed index uses recursive chunks. The aggregate scores could
not distinguish the two; one question from the UI could. It is a good argument
for driving the thing you built rather than only reading its dashboard.

The top row of the table — recursive chunks at advanced depth, 0.905 MRR — is
the deployed configuration, and it is the best number here. **Read it with two
caveats.** It was re-ranked by a different model from the two rows below it,
because Groq's free quota was exhausted by then, and its `n` is 110 rather than
150 because rate limiting cost 40 questions. Both make it a weaker comparison
than a difference of 0.905 against 0.843 looks. It is reported at the top of the
table because it is what the app serves, not because it is the cleanest
measurement in it — the clean one is the `basic` pair, which shares every
condition and shows a dead tie.

**Where retrieval is weak is more interesting than the average.** By question
category, the deployed configuration scores 1.000 MRR on temporal questions,
0.917 on relationship and 0.889 on numerical — but 0.604 on `spanning`
questions, the ones whose answer is distributed across several documents. That
is the real limitation of chunk-level retrieval here, and no amount of
re-ranking a chunk list fixes it.

## Nobody can spend my money

The demo is public and runs on my keys, so the interesting constraint was making
that safe rather than trusting a spend limit to catch it afterwards.

**Embeddings do not use an API at all.** The shipped index is built with
`all-MiniLM-L6-v2` running in-process through ONNX — the model chromadb already
bundles, so it costs no extra dependency and no torch. There is no embedding key
on the server, which means there is nothing to leak and no per-query cost. This
was not the original plan; see the free-tier section below for why it became one.

**Paid models refuse before the key is read.** Models are tiered `demo` (free
providers only) or `byok`. `resolve_key()` raises for a `byok` model with no
visitor-supplied key, so the host key is never fetched, never mind spent — the
check is structural, not a limit inspected after the fact:

```python
if model.tier != DEMO:
    raise KeyPolicyError(...)          # host key never read
key = host_key_for(model)              # only reachable for free-tier models
_charge_host_quota(session)            # and only then, rate limited
```

**Bring your own key removes every limit.** A key pasted into the UI is used for
exactly one request and then dropped — not logged, not stored, not written to
disk. It also switches the request to the full-depth pipeline and skips the rate
limiter, because that request is no longer spending the shared quota.

**Credentials cannot reach the browser.** Provider SDKs put the failing key
inside exception messages, so every error is passed through a regex scrub before
it is rendered. Verified against real SDK error text:

```
AuthenticationError: Incorrect API key provided: <redacted>
headers={'Authorization': 'Bearer <redacted>'}
OpenAI(api_key='<redacted>')
```

**Public mode is derived, not configured.** Streamlit Cloud checks the repository
out under `/mount/src`, so the app takes its hardened path from the path it was
imported from rather than from a variable someone has to remember to set. In
public mode it does not read a `.env` at all, and a platform secret always wins
because `load_dotenv` runs with `override=False`.

## Free-tier ceilings, and the design they forced

Three limits were discovered by hitting them, and all three changed the
architecture.

**One free provider is a single point of failure.** Benchmarking exhausted
Groq's per-minute token budget, and the demo simply stopped answering — the
retry backoff spent a measured **204 seconds** discovering that, which is well
past the point any visitor has closed the tab. Two changes came out of it. The
retry dropped from six attempts to three, because a long backoff is a poor
substitute for a second provider. And a rate-limited free model now fails over
to a free model on a *different* provider — same provider means the same
exhausted quota, so those are skipped. The visitor is told which model actually
answered rather than being switched silently, and a visitor on their own key
gets no failover at all: their key, their model, their choice. End to end that
turned a 289-second failure into a 35-second answer.

**Google's embedding API allows 1000 requests per day, and litellm sends one
request per input string.** Embedding this knowledge base once costs ~900 of
them. Measured: 877 chunks took over 15 minutes and still hit the cap, against
28 seconds on OpenAI. A free API was therefore not a viable embedding backend at
all — which is what pushed embeddings in-process, and that turned out better
than either API: no key, no quota, no per-query cost.

**Groq's free tier allows 8000 tokens per minute.** The original re-ranking step
sent 40 full chunks in one prompt, which exceeds a whole minute's budget in a
single call, so it failed outright. The fix is a `Budget`: the shared demo key
retrieves 12 candidates and re-ranks on 420-character excerpts, while your own
key gets 20 candidates re-ranked on complete chunks. LLM-written chunks lead
with a generated headline and summary, so an excerpt is most of what a ranker
needs anyway. The published benchmark numbers are all measured at full depth —
the constraint belongs to the shared key, not to the system.

## How it works

```
knowledge-base/          76 markdown documents
   |
   |  chunking.py     recursive 500-char split, or an LLM asked to split
   |                  semantically and prepend a headline + summary
   v
chunks/*.json         cached, so re-embedding never re-chunks
   |
   |  embeddings.py   local ONNX MiniLM (384d) | text-embedding-3-large (3072d)
   v
vectorstore/          Chroma, one collection per strategy x backend,
                      each recording the backend it was built with
   |
   |  pipeline.py     BASIC     embed -> top-k
   |                  ADVANCED  rewrite -> retrieve x2 -> merge -> rerank -> top-k
   v
streamlit_app.py      the UI: answer + the retrieval behind it
```

A collection stores its embedding backend in its own metadata, and queries are
embedded through that recorded backend. Embedding a query with a different model
than the index does not raise — it silently returns plausible nonsense — so this
is the one thing in the system that is made impossible rather than documented.

## Running it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY - free, and the only one needed

python ingest.py --strategy recursive --embeddings local   # build the index
streamlit run streamlit_app.py                             # http://127.0.0.1:8501
```

Reproduce the benchmarks:

```bash
# retrieval metrics - deterministic, no API call, nothing to pay
python -m evaluation.eval --strategy llm --embeddings local --pipeline advanced --retrieval-only
python scripts/results_table.py
```

The retrieval half of the table costs nothing: local embeddings run in-process
and MRR, nDCG and coverage need no judge. The answer scores do cost money — they
were graded by `gpt-4.1-mini` — so `evaluation/eval.py` refuses to start any run
that touches a paid provider unless `--paid` is passed explicitly. The committed
`results/` are what those runs produced, so the paid half does not need repeating
to read the table.

`RAGWorker.ipynb` is the workbench: chunk-length distributions, the embedding
space projected with t-SNE, and the basic/advanced pipelines run side by side on
the query that motivated re-ranking. It imports `rag/` rather than duplicating
it, so it cannot drift from what is deployed.

## Deployment

Streamlit Community Cloud, chosen by elimination against measured numbers. The
app holds a Chroma index in memory and answers in seconds, so it needs a
persistent process — which rules out serverless. Runtime dependencies measure
389 MB against the 250 MB unzipped cap on both Vercel and Netlify, and one
question makes five sequential API calls, which does not fit Netlify's
10-second function timeout. Measured resident memory is **443 MB** once the ONNX
model has loaded, which is 87% of Render's 512 MB free tier before a single
visitor arrives — survivable in principle, but with no headroom for concurrent
requests. Streamlit's 1 GB leaves room to actually serve people.

Hugging Face Spaces was the original target and this code was written for it,
until creating the Space returned `402 Payment Required`: Gradio and Docker
Spaces now need a PRO subscription, and only static Spaces remain free. That is
what moved the UI from Gradio to Streamlit — a platform decision, not a taste
one, and the reason `streamlit_app.py` exists.

Streamlit Cloud deploys the whole repository, so the "upload an explicit file
list" safeguard does not apply and something equivalent replaces it.
`scripts/check_repo_safe.py` asks git what is tracked — only tracked files are
published — and fails on a tracked secrets file or on any file containing
something shaped like an API key, reporting file and line without ever printing
the value. `scripts/simulate_cloud.py` then copies only `git ls-files` into a
temp directory and boots the app there with an environment rebuilt from an
allowlist rather than `os.environ` with keys deleted, so a newly added provider
cannot silently ride along. Full steps and the verification checklist are in
[DEPLOY.md](DEPLOY.md).

---

Insurellm is a fictional company and the knowledge base is synthetic. The test
set, the results, and the index are all committed, so every number here can be
re-derived from this repository.

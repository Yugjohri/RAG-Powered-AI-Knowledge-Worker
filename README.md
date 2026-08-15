# RAG-Powered AI Knowledge Worker

[![Try it live](https://img.shields.io/badge/Try_it_live-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://rag-powered-ai-knowledge-worker-xyotc8knhafumxdsw3w9zf.streamlit.app/)

**Live demo:** https://rag-powered-ai-knowledge-worker-xyotc8knhafumxdsw3w9zf.streamlit.app/

A retrieval-augmented assistant over a knowledge base of 76 internal documents —
contracts, employee records, product sheets — that **shows you its own
retrieval**. Every answer sits next to the query the system rewrote to search
with, the chunks it pulled, the order the re-ranker put them in, and how long
each stage took. You can check the grounding instead of trusting it.

Try *"Who won the prestigious IIOTY award in 2023?"* The demo runs on free-tier
keys, so it is rate limited and sleeps after 12 quiet hours — the first visit
after that takes about 30 seconds to wake.

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
is the one thing in the system made impossible rather than documented.

## Running it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY - free, and the only one needed

python ingest.py --strategy recursive --embeddings local   # build the index
streamlit run streamlit_app.py                             # http://127.0.0.1:8501
```

`RAGWorker.ipynb` is the workbench: chunk-length distributions, the embedding
space projected with t-SNE, and the two chunking strategies run side by side. It
imports `rag/` rather than duplicating it, so it cannot drift from the app.

## Nobody can spend my money

The demo is public and runs on my keys, so the constraint was making that safe
structurally rather than trusting a spend limit to catch it afterwards.

- **Embeddings use no API at all.** The shipped index is built with
  `all-MiniLM-L6-v2` running in-process through ONNX — the model chromadb
  already bundles. No embedding key on the server, so nothing to leak and no
  per-query cost.
- **Paid models refuse before the key is read.** Models are tiered `demo` (free
  providers only) or `byok`. `resolve_key()` raises for a `byok` model with no
  visitor key, so the host key is never fetched, never mind spent.
- **Bring your own key removes every limit.** A key pasted into the UI is used
  for one request and dropped — not logged, not stored, not written to disk.
- **Credentials cannot reach the browser.** Provider SDKs put the failing key
  inside exception messages, so every error is regex-scrubbed to `<redacted>`
  before rendering.
- **One provider is a single point of failure**, so a rate-limited free model
  fails over to a free model on a *different* provider, and says so rather than
  switching silently.

## The measurements

Every design choice here was settled by scoring it over a committed 150-question
test set — whether LLM-written chunks beat a character splitter, whether a paid
3072-dimensional embedding model beats a free 384-dimensional one, whether query
rewriting and re-ranking earn their two extra API calls.

The headline: **re-ranking closes the gap between the free local embedder and
the paid API**, which is why the deployment needs no embedding key at all. And
**LLM chunking does not help — it buries facts**, which one question from the UI
exposed after the aggregate scores had called it a tie.

Full table, method, and the honest caveats are in **[BENCHMARKS.md](BENCHMARKS.md)**.

## More

- **[BENCHMARKS.md](BENCHMARKS.md)** — the results table and what it means
- **[DEPLOY.md](DEPLOY.md)** — hosting, secrets, and the pre-publish checks
- **[LEGACY.md](LEGACY.md)** — the earlier LangChain versions, kept for reference

---

Insurellm is a fictional company and the knowledge base is synthetic. The test
set, the results, and the index are all committed, so every number can be
re-derived from this repository.

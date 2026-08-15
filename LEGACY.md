# Superseded code, kept on purpose

Three earlier versions of this project are still in the repository:
`implementation/`, `pro_implementation/`, and `evaluator.py`. **None of them
run any more.** They are kept because the path from the first to the last is
the interesting part of the project, and deleting them would hide it.

If you are looking for the working code, it is `rag/`, `streamlit_app.py`,
`ingest.py` and `evaluation/`. Nothing below is imported by any of them.

## What each one was

### `implementation/` — the LangChain version

`ingest.py` and `answer.py`, built on LangChain: `DirectoryLoader`,
`RecursiveCharacterTextSplitter`, `langchain_chroma`, `ChatOpenAI`. Chunking was
a 500-character recursive split; embedding was `text-embedding-3-large`;
answering was a single retrieve-then-generate pass with no rewriting and no
re-ranking.

It worked, but every component was reached through a framework wrapper, which
made it hard to see what was actually being sent to the model — and the
embedding backend was hardcoded to a paid API.

### `pro_implementation/` — LangChain removed

The same pipeline rewritten against the underlying libraries directly: the
`openai` SDK, `chromadb`'s `PersistentClient`, `litellm`, and `pydantic` for
structured output. This is where query rewriting and LLM re-ranking first
appeared, and where chunking became a model-written semantic split rather than a
character count.

Dropping the framework is what made the rest of the project possible. Swapping
the embedding backend, recording which backend built an index, scrubbing keys
out of provider errors and tiering models by who pays are all changes that
happen at exactly the layer LangChain was covering.

### `evaluator.py` — the Gradio evaluation UI

A Gradio app for browsing evaluation results. It is broken twice over: it
imports `evaluate_all_retrieval` and `evaluate_all_answers` from
`evaluation.eval`, which were renamed to `evaluate_retrieval` and
`evaluate_answers` during the rewrite, and its Gradio UI code predates the
Gradio 6 API changes. Its job is now the **Benchmarks** tab in
`streamlit_app.py`, which renders the same committed `results/*.json`.

## Why they do not run

`requirements.txt` installs what the current app needs, which no longer includes
LangChain:

| Package | Needed by | Installed |
|---|---|---|
| `langchain_community` | `implementation/ingest.py` | no |
| `langchain_chroma` | both `implementation/` files | no |
| `langchain_openai` | `implementation/answer.py` | no |
| `langchain_huggingface` | `implementation/answer.py` | no |

`pro_implementation/` is closer to runnable — its dependencies are all still
installed — but it reads the old `vector_db/` index, which is gitignored and not
built by `ingest.py` any more, and it hardcodes `text-embedding-3-large`, so
running it would spend money on an OpenAI key that the current project
deliberately never needs.

To run either of the old versions you would need to install the LangChain
packages and rebuild their index. That is deliberate: they are reference
material, not an alternative entry point.

## What replaced what

| Old | New |
|---|---|
| `implementation/ingest.py`, `pro_implementation/ingest.py` | `ingest.py` + `rag/chunking.py` + `rag/embeddings.py` + `rag/store.py` |
| `implementation/answer.py`, `pro_implementation/answer.py` | `rag/pipeline.py` |
| `evaluator.py` | the Benchmarks tab in `streamlit_app.py` |
| — | `rag/providers.py`, `rag/scrub.py`, `rag/config.py` — the key policy, credential scrubbing and public-mode detection, none of which existed before |

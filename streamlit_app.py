"""Insurellm knowledge worker - a RAG chat over a private document store.

Run locally:            streamlit run streamlit_app.py
On Streamlit Cloud:     the platform runs this file directly. Secrets set in the
                        app's settings arrive as environment variables, which is
                        where rag.config looks for them.

Streamlit re-runs this whole script top to bottom on every interaction, so the
expensive things - the Chroma client, the ONNX embedding model - must be cached
or the app reloads ~400 MB of model on every keystroke. That is what
@st.cache_resource on the retrieval side is for; the index itself is cached
inside rag.store.
"""

from __future__ import annotations

import html
import sys
import uuid

# chromadb needs sqlite >= 3.35 and some Linux images ship an older one. The
# pysqlite3 wheel is installed on Linux only (see requirements.txt), so this is
# a no-op locally on Windows and a rescue on the host.
try:  # pragma: no cover - platform dependent
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

st.set_page_config(
    page_title="Insurellm Knowledge Worker",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

from rag import results
from rag.config import EMBEDDINGS, STRATEGY
from rag.embeddings import backend_available
from rag.pipeline import (
    ADVANCED,
    BASIC,
    FREE_TIER,
    FULL,
    PipelineError,
    answer_with_fallback,
    budget_for,
)
from rag.providers import (
    DEFAULT_MODEL,
    KeyPolicyError,
    RateLimited,
    choices,
    fallbacks_for,
    resolve_key,
)
from rag.scrub import scrub_exception
from rag.store import available_indexes
from styles import CSS

PIPELINE_LABELS = {
    "Advanced - rewrite, dual retrieve, rerank": ADVANCED,
    "Basic - single retrieve, top 8": BASIC,
}

EXAMPLES = [
    "Who won the prestigious IIOTY award in 2023?",
    "Which customers signed contracts for Carllm, and what are they worth?",
    "What does Insurellm's engineering team look like, and who leads it?",
    "Which employee went to Manchester University?",
]

REPO_URL = "https://github.com/Yugjohri/RAG-Powered-AI-Knowledge-Worker"


def write(markup: str) -> None:
    """Render our own HTML.

    st.html, not st.markdown(unsafe_allow_html=True): the latter runs the string
    through the markdown parser first, so a retrieved chunk containing a line
    like "## Other HR Notes" was rendered as a page heading inside the panel.
    st.html inserts the markup as-is. Everything interpolated into it is escaped
    at the call site.
    """
    st.html(markup)


# --------------------------------------------------------------------------
# scoreboard - measured numbers only
# --------------------------------------------------------------------------


def _tile(value: str, unit: str, name: str, detail: str, tone: str = "plain") -> str:
    unit_html = f'<span class="unit">{html.escape(unit)}</span>' if unit else ""
    return (
        f'<div class="score {tone}"><div class="value">{html.escape(value)}{unit_html}</div>'
        f'<div class="name">{html.escape(name)}</div>'
        f'<div class="detail">{html.escape(detail)}</div></div>'
    )


def scoreboard_html() -> str:
    data = results.load(results.HEADLINE)
    if not data:
        tiles = _tile("--", "", "not yet measured", "run: python -m evaluation.eval")
        return f'<div class="scoreboard">{tiles}</div>'

    retrieval = data.get("retrieval", {})
    answers = data.get("answers", {})
    n = data.get("questions", 0)
    tiles = []

    if retrieval:
        mrr = retrieval["mrr"]
        tiles.append(
            _tile(
                f"{mrr:.2f}",
                "MRR",
                "retrieval rank",
                f"{retrieval.get('n', n)} held-out questions",
                results.band(mrr, 0.85, 0.7),
            )
        )
        coverage = retrieval["keyword_coverage"]
        tiles.append(
            _tile(
                f"{coverage:.0f}",
                "%",
                "keyword coverage",
                "expected facts present in context",
                results.band(coverage, 90, 75),
            )
        )
    if answers:
        accuracy = answers["accuracy"]
        tiles.append(
            _tile(
                f"{accuracy:.2f}",
                "/ 5",
                "answer accuracy",
                "graded against reference answers",
                results.band(accuracy, 4.5, 4.0),
            )
        )
        tiles.append(_tile(f"{answers['median_seconds']:.1f}", "s", "median answer", "end to end"))
    return f'<div class="scoreboard">{"".join(tiles)}</div>'


def _cell(value, fmt: str, good: float, ok: float) -> str:
    if value is None:
        return '<td class="">-</td>'
    return f'<td class="{results.band(value, good, ok)}">{format(value, fmt)}</td>'


def benchmarks_html() -> str:
    data = results.all_results()
    if not data:
        return (
            '<div class="empty">No results committed yet. Run '
            "<code>python -m evaluation.eval</code> to produce them.</div>"
        )

    rows = []
    for name, payload in sorted(data.items()):
        retrieval = payload.get("retrieval") or {}
        answers = payload.get("answers") or {}
        config = payload.get("config", {})
        live = name == results.HEADLINE
        tag = '<span class="tag live">live demo</span>' if live else ""
        pretty = (
            f"{config.get('pipeline', '?')} &middot; {config.get('strategy', '?')} chunks "
            f"&middot; {config.get('embeddings', '?')} embeddings"
        )
        rows.append(
            f'<tr><td class="config">{pretty}{tag}</td>'
            + _cell(retrieval.get("mrr"), ".3f", 0.85, 0.7)
            + _cell(retrieval.get("ndcg"), ".3f", 0.85, 0.7)
            + _cell(retrieval.get("keyword_coverage"), ".1f", 90, 75)
            + _cell(answers.get("accuracy"), ".2f", 4.5, 4.0)
            + _cell(answers.get("completeness"), ".2f", 4.5, 4.0)
            + _cell(answers.get("relevance"), ".2f", 4.5, 4.0)
            + f'<td>{(answers.get("median_seconds") or retrieval.get("median_seconds") or 0):.1f}</td>'
            # the number actually scored, which is not the number requested
            # if a provider rate-limited some of them
            + f'<td>{(answers or retrieval or {}).get("n", payload.get("questions", 0))}</td></tr>'
        )

    header = (
        "<tr><th>Configuration</th><th>MRR</th><th>nDCG</th><th>Coverage %</th>"
        "<th>Accuracy</th><th>Complete</th><th>Relevant</th><th>Median s</th><th>n</th></tr>"
    )
    # Only a run that actually produced answers has a judge worth naming.
    judge = next((p.get("judge") for p in data.values() if p.get("answers") and p.get("judge")), "")
    return (
        '<div class="bench-wrap"><table class="bench"><thead>'
        f"{header}</thead><tbody>{''.join(rows)}</tbody></table></div>"
        '<div class="bench-note">Retrieval metrics are deterministic given an index: MRR and '
        "nDCG are computed over the keywords each question is expected to surface, and coverage "
        "is the share of those keywords present anywhere in the retrieved context."
        + (
            " Answer scores are graded 1-5 against written reference answers by "
            f"{html.escape(str(judge))}, deliberately a different model family from the one "
            "being graded."
            if judge
            else ""
        )
        + "</div>"
    )


# --------------------------------------------------------------------------
# retrieved context
# --------------------------------------------------------------------------


def context_html(answer=None) -> str:
    if answer is None:
        return (
            '<div class="empty">Ask a question. The chunks the model was given, '
            "and the order it ranked them in, appear here.</div>"
        )

    parts = []
    if answer.rewritten_query:
        parts.append(
            '<div class="rewrite"><b>Knowledge-base query</b><br>'
            f"{html.escape(answer.rewritten_query)}</div>"
        )
    for index, chunk in enumerate(answer.chunks, start=1):
        parts.append(
            '<div class="chunk"><div class="head">'
            f'<span class="src">{html.escape(chunk.source)}</span>'
            f'<span class="rank">rank {index}</span></div>'
            f'<div class="body">{html.escape(chunk.page_content.strip())}</div></div>'
        )
    if answer.budget:
        depth = FULL if answer.budget == FULL.name else FREE_TIER
        parts.append(
            f'<div class="rewrite"><b>Retrieval depth</b><br>{html.escape(answer.budget)} - '
            f"{depth.retrieval_k} candidates per query, top {depth.final_k} kept"
            + ("" if depth.rerank_chars is None else f", re-ranked on {depth.rerank_chars}-char excerpts")
            + "</div>"
        )
    if answer.timings:
        rows = "".join(
            f'<span class="k">{html.escape(k)}</span>'
            f'<span class="n">{v:.2f}</span><span class="u">s</span>'
            for k, v in answer.timings.items()
        )
        rows += (
            '<span class="k"><b>total</b></span>'
            f'<span class="n"><b>{answer.total_seconds:.2f}</b></span><span class="u">s</span>'
        )
        parts.append(f'<div class="timings">{rows}</div>')
    return f'<div class="context-scroll">{"".join(parts)}</div>'


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

state = st.session_state
state.setdefault("history", [])       # [{role, content}], what the model sees
state.setdefault("answer", None)      # the last Answer, for the context panel
state.setdefault("error", "")
state.setdefault("session", uuid.uuid4().hex)  # rate-limit identity, per browser session
state.setdefault("pending", None)     # a question submitted but not yet answered
state.setdefault("served_by", "")     # set when a fallback model answered instead


def ask(question: str) -> None:
    """Queue a question. The answer is produced on the rerun this triggers, so
    the user's message paints immediately instead of after the model returns."""
    question = (question or "").strip()
    if question:
        state.pending = question


# --------------------------------------------------------------------------
# ui
# --------------------------------------------------------------------------

write(f"<style>{CSS}</style>")

write(
    '<div class="masthead">'
    '<h1>Insurellm <span class="accent">Knowledge Worker</span></h1>'
    '<p class="sub">Retrieval-augmented answers over 76 internal documents - '
    "contracts, employee records, product sheets. Every answer is grounded in "
    "chunks retrieved from the store, and the retrieved chunks are shown next to "
    "it so the grounding can be checked rather than trusted.</p></div>"
)
write(scoreboard_html())

model_choices = choices() or [("(no model configured)", DEFAULT_MODEL)]
labels = [label for label, _ in model_choices]
ids = {label: model_id for label, model_id in model_choices}

with st.sidebar:
    st.markdown("### Settings")
    model_label = st.selectbox("Model", labels, index=0)
    model_id = ids[model_label]
    pipeline_label = st.radio("Retrieval pipeline", list(PIPELINE_LABELS), index=0)
    pipeline = PIPELINE_LABELS[pipeline_label]

    st.markdown("---")
    st.markdown("### Use your own API key")
    byok = st.text_input(
        "API key",
        type="password",
        placeholder="Used for this request only",
        label_visibility="collapsed",
    )
    write(
        '<div class="notice">Models marked <b>(need your own key)</b> are paid '
        "APIs and will never run on the host's key. A key pasted here is used for "
        "one request and then discarded - not written to disk, not logged, not kept "
        "between requests.<br><br>It also removes the shared-demo limits: your "
        f"request runs the <b>full-depth</b> pipeline ({FULL.retrieval_k} candidates "
        f"per query, top {FULL.final_k}, re-ranked on complete chunks) instead of the "
        f"reduced path the free shared key has to use ({FREE_TIER.retrieval_k}/"
        f"{FREE_TIER.final_k}), and it is not rate limited.</div>"
    )
    st.markdown("---")
    if st.button("Clear conversation"):
        state.history, state.answer, state.error = [], None, ""
        st.rerun()

# Startup problems are worth saying out loud rather than failing at question time.
problems = []
if not available_indexes():
    problems.append(
        "No vector index found. Build one with: python ingest.py --strategy llm --embeddings local"
    )
if not backend_available(EMBEDDINGS):
    problems.append(
        f"The {EMBEDDINGS} embedding key is not configured, so queries cannot be embedded."
    )
if not choices():
    problems.append("No model is reachable: no free-tier key is configured on this host.")
for problem in problems:
    write(f'<div class="notice err">{html.escape(problem)}</div>')

def answer_pending(question: str) -> None:
    """Run the pipeline and fold the result into state.

    Called while the left column is being rendered, so the spinner and the
    answer land in the transcript in script order - no second rerun, and no
    writing into a column from outside its block.

    If the chosen free model is rate limited, the same question is retried on a
    free model from a different provider rather than shown as an error. One
    provider's quota running out is not a reason for the demo to be down.
    """
    state.error = ""
    state.served_by = ""
    own_key = bool((byok or "").strip())
    try:
        answer, served_by = answer_with_fallback(
            question,
            list(state.history),
            model=model_id,
            resolve=lambda candidate: resolve_key(candidate, byok, state.session),
            # A visitor on their own key gets no substitution: their key, their
            # model, their choice.
            fallbacks=[] if own_key else fallbacks_for(model_id),
            strategy=STRATEGY,
            embeddings=EMBEDDINGS,
            pipeline=pipeline,
            budget=budget_for(own_key),
        )
    except (KeyPolicyError, RateLimited, PipelineError) as exc:
        state.error = str(exc)
        return
    except Exception as exc:  # noqa: BLE001 - nothing uncaught may reach the browser
        state.error = scrub_exception(exc)
        return
    state.served_by = served_by or ""

    # A failed question is deliberately NOT kept: it would otherwise poison the
    # history sent with the next call, which is charged for every token of it.
    state.history = state.history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer.text},
    ]
    state.answer = answer


# chat_input pins itself to the bottom of the viewport wherever it is called, so
# reading it here - before the tabs are drawn - means a submitted question is
# answered on this same run instead of needing a rerun to be noticed.
if prompt := st.chat_input("Ask anything about Insurellm..."):
    ask(prompt)

ask_tab, bench_tab = st.tabs(["Ask", "Benchmarks"])

with ask_tab:
    left, right = st.columns([3, 2], gap="large")

    with left:
        write('<div class="panel-title gen"><span class="dot"></span>Conversation</div>')
        for message in state.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if state.pending:
            question, state.pending = state.pending, None
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving and answering..."):
                    answer_pending(question)
                if state.error:
                    write(f'<div class="notice err">{html.escape(state.error)}</div>')
                else:
                    if state.served_by:
                        # Never swap the model silently - the visitor picked one.
                        write(
                            '<div class="notice warn">The selected model was rate limited, '
                            f"so this answer came from <b>{html.escape(state.served_by)}</b> "
                            "instead.</div>"
                        )
                    st.markdown(state.history[-1]["content"])
        elif state.error:
            write(f'<div class="notice err">{html.escape(state.error)}</div>')

        if not state.history and not state.error:
            st.markdown("**Try one**")
            for example in EXAMPLES:
                st.button(
                    example, key=f"ex-{example}", on_click=ask, args=(example,),
                    use_container_width=True,
                )

    with right:
        # Rendered after the answer was computed above, so it shows this turn's
        # chunks rather than the previous turn's.
        write('<div class="panel-title ret"><span class="dot"></span>Retrieved context</div>')
        write(context_html(state.answer))

with bench_tab:
    write(
        '<div class="panel-title ret"><span class="dot"></span>'
        "Measured over the 150-question test set</div>"
    )
    write(benchmarks_html())

write(
    '<div class="footnote">Insurellm is a fictional company; the knowledge base is '
    "synthetic. Scores above are measured over the committed 150-question test set - "
    f'see the <a href="{REPO_URL}">README</a> for the method.</div>'
)

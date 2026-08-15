"""Paths, environment loading, and the public/local mode switch.

Import this before anything that reads an API key.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_PATH = ROOT / "knowledge-base"
DB_PATH = ROOT / "vectorstore"
EVAL_PATH = ROOT / "evaluation"
RESULTS_PATH = ROOT / "results"

# Public mode is derived from the environment, never configured, so the hardened
# path cannot be forgotten by failing to set a variable. Each signal is one the
# platform sets itself:
#
#   /mount/src/...  Streamlit Community Cloud checks the repo out there, so the
#                   path this file was imported from is the giveaway.
#   SPACE_ID        set on every Hugging Face Space (kept so the app still
#                   hardens correctly if it is ever run on one).
#
# RAG_PUBLIC exists only to force the hardened path on locally, for testing it.
SPACE_ID = os.environ.get("SPACE_ID")
ON_STREAMLIT_CLOUD = str(ROOT).replace("\\", "/").startswith("/mount/src")
IS_PUBLIC = bool(SPACE_ID) or ON_STREAMLIT_CLOUD or os.environ.get("RAG_PUBLIC") == "1"

# override=False so a platform secret always beats a stray .env on disk.
# In public mode we do not read a .env at all: the only key source is the
# platform's secret store.
if not IS_PUBLIC:
    load_dotenv(ROOT / ".env", override=False)


# The deployed configuration, in one place. The UI serves it, and the scoreboard
# reads the results file named after it - so the tiles cannot end up describing a
# configuration other than the one answering the questions.
STRATEGY = os.environ.get("RAG_STRATEGY", "recursive")
EMBEDDINGS = os.environ.get("RAG_EMBEDDINGS", "local")
PIPELINE = "advanced"
HEADLINE_RESULT = f"{PIPELINE}__{STRATEGY}__{EMBEDDINGS}"


def host_key(env_var: str) -> str | None:
    """The host's own key for a provider, or None if it is not configured."""
    value = os.environ.get(env_var)
    return value.strip() if value and value.strip() else None


# Bind 0.0.0.0 inside a hosting container, loopback on a laptop.
SERVER_NAME = "0.0.0.0" if IS_PUBLIC else "127.0.0.1"

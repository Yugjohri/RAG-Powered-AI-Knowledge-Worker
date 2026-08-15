"""Read the committed evaluation results.

The scoreboard in the UI is rendered from these files. If a file is missing the
tile says so, rather than showing a number that was never measured.
"""

from __future__ import annotations

import json
from functools import cache

from .config import HEADLINE_RESULT, RESULTS_PATH

# Named after the configuration the app actually serves, so the scoreboard and
# the "live demo" tag in the benchmark table follow it automatically.
HEADLINE = HEADLINE_RESULT


@cache
def load(name: str) -> dict | None:
    file = RESULTS_PATH / f"{name}.json"
    if not file.exists():
        return None
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def all_results() -> dict[str, dict]:
    if not RESULTS_PATH.exists():
        return {}
    out = {}
    for file in sorted(RESULTS_PATH.glob("*.json")):
        data = load(file.stem)
        if data:
            out[file.stem] = data
    return out


def band(value: float, good: float, ok: float) -> str:
    if value >= good:
        return "good"
    if value >= ok:
        return "ok"
    return "poor"

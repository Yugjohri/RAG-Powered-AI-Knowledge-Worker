"""Model registry, key policy, and rate limiting.

Key policy, in priority order per request:

  1. A key the visitor typed into the UI ("bring your own key"). Used for that
     one request and then dropped - never logged, never persisted, never
     written to disk.
  2. Otherwise the host's own key, but ONLY for models in the DEMO tier.

The DEMO tier contains free-tier providers exclusively. That is the whole
enforcement: a visitor cannot select a paid model without supplying their own
key, because the check happens before the host key is ever read - not by
inspecting a spend limit afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import litellm

from .config import host_key

litellm.suppress_debug_info = True

DEMO = "demo"  # free tier - the host's key may serve these
BYOK = "byok"  # paid - the visitor must supply their own key


@dataclass(frozen=True)
class Provider:
    key_env: str
    label: str
    signup: str


PROVIDERS: dict[str, Provider] = {
    "groq": Provider("GROQ_API_KEY", "Groq (free tier)", "https://console.groq.com/keys"),
    "gemini": Provider("GOOGLE_API_KEY", "Google (free tier)", "https://aistudio.google.com/apikey"),
    "openai": Provider("OPENAI_API_KEY", "OpenAI", "https://platform.openai.com/api-keys"),
    "deepseek": Provider("DEEPSEEK_API_KEY", "DeepSeek", "https://platform.deepseek.com/api_keys"),
}


@dataclass(frozen=True)
class Model:
    id: str  # litellm model id
    provider: str
    display: str
    note: str = ""
    tier: str = BYOK  # default to the safe side: never spend the host's money


# Verified against the live APIs on 2026-08-14. gemini-2.5-flash and
# gemini-2.5-pro were in the registry but return 404 "no longer available", so
# they are not listed here.
MODELS: list[Model] = [
    # -- demo tier: free providers, safe to serve with the host's key --------
    Model("groq/openai/gpt-oss-120b", "groq", "GPT-OSS 120B", "free - fast, the demo default", DEMO),
    Model("groq/openai/gpt-oss-20b", "groq", "GPT-OSS 20B", "free - smaller, faster", DEMO),
    # Lite before Flash deliberately: both are free, and measured latency on a
    # full pipeline was 0.7s against 18-85s. Order here also sets failover order.
    Model("gemini/gemini-3.5-flash-lite", "gemini", "Gemini 3.5 Flash Lite", "free tier", DEMO),
    Model("gemini/gemini-3.5-flash", "gemini", "Gemini 3.5 Flash", "free tier", DEMO),
    # -- paid: visitor's own key only ----------------------------------------
    Model("gpt-5", "openai", "GPT-5", "frontier", BYOK),
    Model("gpt-5-mini", "openai", "GPT-5 Mini", "", BYOK),
    Model("gpt-5-nano", "openai", "GPT-5 Nano", "", BYOK),
    Model("gpt-4.1-nano", "openai", "GPT-4.1 Nano", "", BYOK),
    Model("deepseek/deepseek-chat", "deepseek", "DeepSeek Chat", "", BYOK),
]

BY_ID = {m.id: m for m in MODELS}

DEFAULT_MODEL = "groq/openai/gpt-oss-120b"


class KeyPolicyError(RuntimeError):
    """Raised when a request would spend the host's money."""


class RateLimited(RuntimeError):
    """Raised when a session has used up its share of the host's free tier."""


def host_key_for(model: Model) -> str | None:
    return host_key(PROVIDERS[model.provider].key_env)


def available(model: Model) -> bool:
    """A model is shown only if it can actually be used.

    DEMO models need the host key present. BYOK models are always usable
    because the visitor supplies the key - so they are always offered.
    """
    if model.tier == DEMO:
        return host_key_for(model) is not None
    return True


def label_for(model: Model) -> str:
    """Dropdown label.

    The warning goes FIRST: a narrow <select> truncates the tail, and a
    trailing marker is exactly the part that disappears.
    """
    if model.tier == BYOK:
        return f"(need your own key) {model.display}"
    note = f" - {model.note}" if model.note else ""
    return f"{model.display}{note}"


def choices() -> list[tuple[str, str]]:
    """(label, model_id) pairs for the model picker, demo tier first."""
    usable = [m for m in MODELS if available(m)]
    usable.sort(key=lambda m: (m.tier != DEMO,))
    return [(label_for(m), m.id) for m in usable]


def fallbacks_for(model_id: str) -> list[str]:
    """Free models to try if `model_id` is rate limited, best first.

    Only DEMO-tier models on a DIFFERENT provider are offered: a second model
    on the same provider shares the same exhausted quota, so failing over to it
    just spends more time discovering the same limit. A visitor on their own key
    gets no fallback - their key, their model, their choice.
    """
    original = BY_ID.get(model_id)
    if original is None or original.tier != DEMO:
        return []
    return [
        m.id
        for m in MODELS
        if m.tier == DEMO and m.provider != original.provider and available(m)
    ]


# --------------------------------------------------------------------------
# rate limiting - applies only to requests that spend the host's free tier
# --------------------------------------------------------------------------

WINDOW_SECONDS = 60 * 60
MAX_HOST_CALLS_PER_WINDOW = 40

_session_calls: dict[str, list[float]] = {}


def _charge_host_quota(session: str) -> None:
    now = time.time()
    hits = [t for t in _session_calls.get(session, []) if now - t < WINDOW_SECONDS]
    if len(hits) >= MAX_HOST_CALLS_PER_WINDOW:
        wait_min = int((WINDOW_SECONDS - (now - hits[0])) / 60) + 1
        raise RateLimited(
            f"This demo allows {MAX_HOST_CALLS_PER_WINDOW} questions per hour on the shared "
            f"free-tier key. Try again in about {wait_min} minutes, or paste your own API key "
            f"in the box below to remove the limit."
        )
    hits.append(now)
    _session_calls[session] = hits


def resolve_key(model_id: str, byok: str | None, session: str) -> str:
    """Return the key to use for this request, enforcing the policy first.

    byok is used for this call only. It is never stored or logged.
    """
    model = BY_ID.get(model_id)
    if model is None:
        raise KeyPolicyError(f"Unknown model: {model_id}")

    byok = (byok or "").strip()
    if byok:
        return byok  # visitor pays for their own request; no rate limit

    if model.tier != DEMO:
        provider = PROVIDERS[model.provider]
        raise KeyPolicyError(
            f"{model.display} runs on {provider.label}, which is a paid API. This demo only "
            f"spends the host's key on free-tier models. Paste your own {provider.label} key "
            f"below to use it, or pick one of the free models. Keys: {provider.signup}"
        )

    key = host_key_for(model)
    if key is None:
        raise KeyPolicyError(f"{model.display} is not configured on this deployment.")

    _charge_host_quota(session)
    return key

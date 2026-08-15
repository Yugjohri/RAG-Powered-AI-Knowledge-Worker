"""Refuse to publish a repository that carries a secret.

Streamlit Community Cloud deploys the whole GitHub repository, so there is no
upload list to police - whatever git tracks becomes public. This script is the
replacement for that check, and it asks git what is tracked rather than walking
the filesystem, because a file is only published if git knows about it.

    python scripts/check_repo_safe.py

Exit code 0 means safe to push. Anything else means do not push.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files that must never be tracked, whatever .gitignore currently says.
FORBIDDEN_NAMES = {".env", ".env.local", ".env.production", "secrets.toml"}
FORBIDDEN_SUFFIXES = {".pem", ".key", ".pfx"}

# Credential shapes, not values. Ordered longest-prefix-first so a key that
# starts with a shorter prefix is still reported under its real provider.
SECRET_PATTERNS = [
    ("Anthropic", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI project", re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("Groq", re.compile(r"gsk_[A-Za-z0-9]{20,}")),
    ("Google", re.compile(r"AIza[A-Za-z0-9_\-]{30,}")),
    ("xAI", re.compile(r"xai-[A-Za-z0-9]{20,}")),
    ("HuggingFace", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    ("Replicate", re.compile(r"r8_[A-Za-z0-9]{20,}")),
    ("DeepSeek", re.compile(r"sk-[0-9a-f]{32}")),
]

# Documentation has to be able to say "sk-..." without tripping the scan.
PLACEHOLDER = re.compile(r"(x{6,}|\.\.\.|your[-_ ]?key|<[^>]+>|example|placeholder)", re.I)


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [p for p in out.stdout.split("\0") if p]


def main() -> int:
    try:
        files = tracked_files()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Could not ask git what is tracked: {exc}")
        return 1

    problems: list[str] = []

    for path in files:
        name = Path(path).name
        if name in FORBIDDEN_NAMES or Path(path).suffix in FORBIDDEN_SUFFIXES:
            problems.append(f"{path} is tracked by git and must never be published")

    for path in files:
        full = ROOT / path
        try:
            text = full.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue  # binary, or the index - nothing to read
        for provider, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                context = text[max(0, match.start() - 60) : match.end() + 20]
                if PLACEHOLDER.search(context):
                    continue  # documented shape, not a real key
                # The value itself is never printed.
                problems.append(f"{path}:{line} looks like a {provider} key")

    if problems:
        print("UNSAFE TO PUBLISH:\n")
        for problem in problems:
            print(f"  {problem}")
        print(f"\n{len(problems)} problem(s). Fix these before pushing.")
        return 1

    print(f"Safe to publish: {len(files)} tracked files, no secrets found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

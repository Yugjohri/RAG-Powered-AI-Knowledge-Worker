"""Boot the app the way Streamlit Community Cloud will, before pushing anything.

Streamlit Cloud clones the GitHub repository and runs it, so the files it gets
are exactly the files git tracks - which is what this copies, via `git ls-files`.
Anything untracked (a .env, a local index, a scratch script) is absent here for
the same reason it will be absent on the host.

The child process gets an environment rebuilt from an allowlist, not os.environ
with keys deleted. A denylist silently fails the moment a new provider is added,
and that failure is a leaked credential.

    python scripts/simulate_cloud.py            # boot and answer one question
    python scripts/simulate_cloud.py --serve    # boot and stay up on :8502
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

# Read the local .env here, in the PARENT, purely to stand in for the platform's
# secret store: the demo keys are then handed to the child as environment
# variables, which is exactly how Streamlit Cloud delivers its secrets. The .env
# file itself is never copied into the simulated deployment.
load_dotenv(ROOT / ".env", override=False)

# Only these reach the child. Everything else - including OPENAI_API_KEY - is
# left behind, which is exactly the deployed situation.
ALLOWED_ENV = [
    # the only credentials the deployment gets
    "GROQ_API_KEY",
    "GOOGLE_API_KEY",
    # machine plumbing, none of it secret. HOME and friends are required:
    # chromadb resolves Path.home() at import time to locate its ONNX model
    # cache, and raises "Could not determine home directory" without them.
    "PATH",
    "PATHEXT",
    "COMSPEC",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "LOCALAPPDATA",
    "APPDATA",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
]

PROBE = """
import os, sys
sys.path.insert(0, os.getcwd())
assert os.environ.get("RAG_PUBLIC") == "1", "RAG_PUBLIC not set"
assert not os.path.exists(".env"), ".env reached the simulated deployment"
assert not os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY leaked into the child env"

from rag.config import IS_PUBLIC
assert IS_PUBLIC, "app did not take its public path"
print("BOOT OK   public mode:", IS_PUBLIC)

from rag.store import available_indexes
from rag.providers import choices, resolve_key, KeyPolicyError
print("BOOT OK   indexes:", available_indexes())
print("BOOT OK   models:", [c[1] for c in choices()])

# The paid tier must refuse before any host key is read.
try:
    resolve_key("gpt-5", None, "sim")
    print("POLICY FAIL   gpt-5 was allowed to run on the host key")
    raise SystemExit(1)
except KeyPolicyError as e:
    print("POLICY OK     gpt-5 refused:", str(e)[:70])

# Answer through the same helper the app uses, failover included, so this
# proves the deployed path rather than a simplified version of it.
from rag.pipeline import answer_with_fallback
from rag.providers import fallbacks_for
model = next((c[1] for c in choices() if not c[0].startswith("(need")), None)
if model is None:
    print("SKIP      no free model configured, cannot answer")
    raise SystemExit(0)
a, served_by = answer_with_fallback(
    "Who won the IIOTY award in 2023?",
    model=model,
    resolve=lambda m: resolve_key(m, None, "sim"),
    fallbacks=fallbacks_for(model),
)
print("ANSWER OK", round(a.total_seconds, 2), "s  chunks:", len(a.chunks),
      "" if served_by is None else f"(failed over to {served_by})")
print("ANSWER   ", a.text[:200].replace(chr(10), " "))
print("CORRECT  " if "maxine" in a.text.lower() else "WRONG    ",
      "expected Maxine Thompson")
"""


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [p for p in out.stdout.split("\0") if p]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    files = tracked_files()
    if not any(f == "streamlit_app.py" for f in files):
        print(
            "streamlit_app.py is not tracked by git yet, so the deployment would not\n"
            "contain the app. Stage the new files first (git add -A), then re-run."
        )
        return 1

    with tempfile.TemporaryDirectory(prefix="cloud-sim-") as tmp:
        target = Path(tmp)
        for relative in files:
            source = ROOT / relative
            if not source.exists():
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        # .env.example is published on purpose: it is the template, with every
        # value blank. Every other .env* is a real secrets file.
        strays = [
            str(p.relative_to(target))
            for p in list(target.rglob(".env*")) + list(target.rglob("secrets.toml"))
            if p.name != ".env.example"
        ]
        if strays:
            print(f"FAIL: a secrets file is tracked and reached the simulation: {strays}")
            return 1

        env = {k: os.environ[k] for k in ALLOWED_ENV if k in os.environ}
        env["RAG_PUBLIC"] = "1"

        print(f"Simulating in {target}")
        print(f"  {len(files)} tracked files, env: {sorted(env)}")

        if args.serve:
            command = [
                sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
                "--server.port", "8502", "--server.headless", "true",
            ]
        else:
            command = [sys.executable, "-c", PROBE]

        result = subprocess.run(command, cwd=target, env=env, text=True)
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())

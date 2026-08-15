# Deploying to Streamlit Community Cloud

Free, no credit card, 1 GB of RAM. The app measures **443 MB** resident once the
ONNX embedding model is loaded, so it fits with room to spare.

## Why not Hugging Face Spaces

It was the original target, and the code was written for it. Creating a new
Gradio or Docker Space now returns:

```
402 Payment Required
Static Spaces are free for everyone, but hosting Gradio and Docker Spaces
on free cpu-basic requires a PRO subscription.
```

Static Spaces stay free but cannot run Python, so they cannot serve this app.
Render's free tier caps at 512 MB, which leaves 69 MB of headroom over this
app's 443 MB idle footprint - enough to boot, not enough to serve concurrent
requests with confidence. Vercel and Netlify cap an unzipped deployment at
250 MB against 389 MB of runtime dependencies. Streamlit Community Cloud is the
free host with room to actually run it.

## Before you push

Streamlit Cloud deploys **the whole GitHub repository**. There is no upload list
to police - whatever git tracks becomes public. Two checks replace that:

```bash
python scripts/check_repo_safe.py    # nothing secret is tracked
python scripts/simulate_cloud.py     # the tracked files alone can boot the app
```

`check_repo_safe.py` asks git what is tracked rather than walking the disk,
because only tracked files are published. It fails on a tracked `.env` or
`secrets.toml`, and on any file containing something shaped like an API key -
reporting the file and line without ever printing the value.

`simulate_cloud.py` copies only `git ls-files` into a temp directory and boots
the app there with an environment rebuilt from an allowlist. That proves the
deployment does not depend on an untracked file, and that `OPENAI_API_KEY` is
absent from the process that serves the public.

## Deploying

1. Push the repository to GitHub, public.

2. Go to https://share.streamlit.io and sign in with GitHub.

3. **New app** -> **Deploy a public app from GitHub**, then:

   - Repository: `Yugjohri/RAG-Powered-AI-Knowledge-Worker`
   - Branch: `main`
   - Main file path: `streamlit_app.py`

4. Open **Advanced settings** before the first deploy and paste the secrets in
   TOML form. Streamlit exposes these to the process as environment variables,
   which is where `rag/config.py` reads them from:

   ```toml
   GROQ_API_KEY = "..."
   GOOGLE_API_KEY = "..."
   ```

   `OPENAI_API_KEY` is deliberately **not** in that list. No paid key is ever
   deployed - paid models refuse before a host key is read, and a visitor who
   wants one supplies their own.

5. Deploy. The first build installs dependencies and takes a few minutes.

## Verifying, once it says Running

1. **The app answers.** Ask "Who won the prestigious IIOTY award in 2023?" and
   confirm the answer names Maxine Thompson and the retrieval panel shows the
   chunks behind it.

2. **A paid model refuses.** Pick GPT-5 in the sidebar with the key box empty.
   It must reply that the model needs your own key. If it answers instead, the
   host key is reachable and the deployment must come down.

3. **The rate limiter engages.** Ask more than 40 questions in an hour on the
   shared key. The 41st must be refused with the wait time, not served.

4. **No key reaches the page.** Paste an obviously invalid key, ask anything,
   and confirm the error says `<redacted>` rather than echoing what you typed.

## Notes

- The app sleeps after 12 hours without traffic and wakes on the next visit,
  taking roughly 30 seconds to come back. That is the free tier's behaviour, not
  a fault.
- Logs are in the app's **Manage app** panel, bottom right of the running app.
- Rebooting from that panel picks up new secrets; editing secrets alone does not
  always restart the process.

"""The stylesheet, injected into the page as a <style> block.

One committed palette, painted explicitly on every surface, so the result does
not depend on which Streamlit theme happens to be active. `.streamlit/config.toml`
sets the same colours as the base theme, which stops the page from flashing
light before this CSS lands.

Colour is meaning, not decoration:
    cyan    retrieval - the vector search half of the system
    violet  generation - the LLM half
    green / amber / red   a measured score, good / acceptable / poor

Streamlit's own DOM is targeted through data-testid attributes rather than the
generated class names, which are not stable across releases.
"""

CSS = """
:root {
  --ink:        #e8eef5;
  --ink-dim:    #93a3b5;
  --ink-faint:  #62748a;

  --bg:         #0a0e13;
  --surface:    #121a23;
  --surface-2:  #18222d;
  --surface-3:  #1f2b38;
  --line:       #26333f;

  --retrieval:  #4cc9f0;
  --generation: #a78bfa;
  --good:       #3ddc97;
  --ok:         #ffb454;
  --poor:       #ff6b6b;

  --r-lg: 20px;
  --r-md: 14px;
  --r-pill: 999px;
  color-scheme: dark;
}

/* ---- ground ---------------------------------------------------------- */

.stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--ink) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { max-width: 1400px !important; padding-top: 2rem !important; }

/* Light stacked from above: the page is not flat, it is lit. Fixed, behind
   everything, and pointer-events:none so it never eats a click. */
[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background:
    radial-gradient(900px 420px at 18% -8%, rgba(76,201,240,.10), transparent 70%),
    radial-gradient(760px 380px at 84% -4%, rgba(167,139,250,.10), transparent 70%);
}
.block-container { position: relative; z-index: 1; }

/* Streamlit's chrome carries no meaning here. */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
  display: none !important;
}

/* ---- sidebar --------------------------------------------------------- */

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] * { color: var(--ink-dim); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] label { color: var(--ink) !important; }

/* ---- tabs ------------------------------------------------------------ */

[data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--line) !important;
  gap: 4px !important;
}
button[data-baseweb="tab"] {
  background: transparent !important;
  color: var(--ink-faint) !important;
  font-weight: 600 !important;
  font-size: .92rem !important;
  padding: 10px 16px !important;
}
button[data-baseweb="tab"]:hover { color: var(--ink-dim) !important; background: var(--surface) !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--ink) !important; }
button[data-baseweb="tab"][aria-selected="true"] * { color: var(--ink) !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {
  background: var(--retrieval) !important;
}

/* ---- masthead -------------------------------------------------------- */

.masthead { padding: 8px 4px 8px; }
.masthead h1 {
  font-size: 2.1rem; font-weight: 700; letter-spacing: -.02em;
  margin: 0 0 6px; color: var(--ink);
}
.masthead .sub { color: var(--ink-dim); font-size: 1rem; margin: 0; max-width: 70ch; }
.masthead .accent {
  display: inline-block;   /* an inline span has a fragmented background box,
                              which renders background-clip:text invisible */
  background: linear-gradient(96deg, var(--retrieval), var(--generation));
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
}

/* ---- the payoff: measured scores, large ------------------------------ */

.scoreboard {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 14px; margin: 18px 0 26px;
}
.score {
  background: linear-gradient(180deg, var(--surface-2), var(--surface));
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  padding: 20px 22px;
  box-shadow: 0 1px 0 rgba(255,255,255,.04) inset, 0 10px 26px -18px rgba(0,0,0,.9);
}
.score .value {
  font-size: 2.6rem; font-weight: 700; line-height: 1.05; letter-spacing: -.03em;
  display: flex; align-items: baseline; gap: .28em;
}
.score .unit { font-size: .95rem; font-weight: 600; color: var(--ink-faint); }
.score .name {
  margin-top: 8px; font-size: .74rem; font-weight: 700;
  letter-spacing: .10em; text-transform: uppercase; color: var(--ink-dim);
}
.score .detail { margin-top: 3px; font-size: .82rem; color: var(--ink-faint); }
.score.good  .value { color: var(--good); }
.score.ok    .value { color: var(--ok); }
.score.poor  .value { color: var(--poor); }
.score.plain .value { color: var(--ink); }

/* ---- panels ---------------------------------------------------------- */

.panel-title {
  display: flex; align-items: center; gap: 9px;
  font-size: .76rem; font-weight: 700; letter-spacing: .10em;
  text-transform: uppercase; color: var(--ink-dim); margin: 2px 0 12px;
}
.panel-title .dot { width: 8px; height: 8px; border-radius: 50%; display: block; }
.panel-title.gen .dot { background: var(--generation); box-shadow: 0 0 10px var(--generation); }
.panel-title.ret .dot { background: var(--retrieval);  box-shadow: 0 0 10px var(--retrieval); }

/* ---- chat ------------------------------------------------------------ */

[data-testid="stChatMessage"] {
  background: var(--surface-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-md) !important;
  margin-bottom: 10px !important;
}
[data-testid="stChatMessage"] * { color: var(--ink) !important; }

[data-testid="stChatInput"] {
  background: var(--surface-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-pill) !important;
}
[data-testid="stChatInput"] textarea { color: var(--ink) !important; }
[data-testid="stChatInput"]:focus-within {
  border-color: var(--retrieval) !important;
  box-shadow: 0 0 0 3px rgba(76,201,240,.16) !important;
}

/* ---- buttons --------------------------------------------------------- */

.stButton button {
  border-radius: var(--r-pill) !important;
  border: 1px solid var(--line) !important;
  background: var(--surface-3) !important;
  color: var(--ink) !important;
  font-weight: 600 !important;
  transition: border-color .15s ease, background .15s ease;
}
.stButton button:hover { border-color: var(--retrieval) !important; color: var(--ink) !important; }
.stButton button[kind="primary"] {
  background: linear-gradient(96deg, var(--retrieval), var(--generation)) !important;
  border: none !important; color: #08111a !important; font-weight: 700 !important;
}

/* ---- form controls --------------------------------------------------- */

[data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea {
  background: var(--surface-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-md) !important;
  color: var(--ink) !important;
}
[data-baseweb="popover"] li { background: var(--surface-2) !important; color: var(--ink) !important; }
[data-baseweb="popover"] li:hover { background: var(--surface-3) !important; }
.stRadio label, .stSelectbox label, .stTextInput label { color: var(--ink-dim) !important; }

/* ---- retrieved context ----------------------------------------------- */

.context-scroll { max-height: 620px; overflow-y: auto; padding-right: 6px; }
.chunk {
  background: var(--surface-2);
  border: 1px solid var(--line);
  border-left: 3px solid var(--retrieval);
  border-radius: var(--r-md);
  padding: 12px 14px; margin-bottom: 10px;
}
.chunk .head {
  display: flex; justify-content: space-between; align-items: baseline;
  gap: 12px; margin-bottom: 7px;
}
.chunk .src {
  font-size: .78rem; font-weight: 700; color: var(--retrieval);
  overflow-wrap: anywhere;
}
.chunk .rank {
  font-size: .68rem; font-weight: 700; letter-spacing: .08em; color: var(--ink-faint);
  white-space: nowrap; text-transform: uppercase;
}
.chunk .body {
  font-size: .87rem; line-height: 1.5; color: var(--ink-dim);
  white-space: pre-wrap; max-height: 150px; overflow: hidden;
}
.rewrite {
  background: var(--surface-3); border: 1px dashed var(--line);
  border-radius: var(--r-md); padding: 11px 14px; margin-bottom: 12px;
  font-size: .85rem; color: var(--ink-dim);
}
.rewrite b { color: var(--generation); font-weight: 700; }
.empty { color: var(--ink-faint); font-size: .9rem; padding: 26px 4px; text-align: center; }

/* ---- timings: number and unit in separate columns, so the decimal
        points line up instead of jittering ---------------------------- */

.timings { display: grid; grid-template-columns: 1fr auto auto; gap: 4px 8px; margin-top: 4px; }
.timings .k { font-size: .78rem; color: var(--ink-faint); }
.timings .n { font-size: .78rem; color: var(--ink); text-align: right; font-variant-numeric: tabular-nums; }
.timings .u { font-size: .78rem; color: var(--ink-faint); text-align: left; }

/* ---- notices --------------------------------------------------------- */

.notice {
  border-radius: var(--r-md); padding: 11px 14px; font-size: .86rem;
  border: 1px solid var(--line); background: var(--surface-2); color: var(--ink-dim);
  margin-bottom: 10px;
}
.notice.warn { border-color: rgba(255,180,84,.45); color: var(--ok); }
.notice.err  { border-color: rgba(255,107,107,.45); color: var(--poor); }

/* ---- benchmark table ------------------------------------------------- */

.bench-wrap { overflow-x: auto; }
table.bench { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .88rem; }
table.bench th {
  text-align: right; padding: 9px 12px; white-space: nowrap;
  font-size: .7rem; letter-spacing: .09em; text-transform: uppercase;
  color: var(--ink-dim); border-bottom: 1px solid var(--line);
}
table.bench th:first-child, table.bench td:first-child { text-align: left; }
table.bench td {
  padding: 11px 12px; text-align: right; white-space: nowrap;
  border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums;
  color: var(--ink);
}
table.bench tr:last-child td { border-bottom: none; }
table.bench tbody tr:hover td { background: var(--surface-2); }
table.bench .config { color: var(--ink); font-weight: 600; }
table.bench .tag {
  display: inline-block; margin-left: 8px; padding: 2px 9px;
  border-radius: var(--r-pill); font-size: .66rem; font-weight: 700;
  letter-spacing: .06em; text-transform: uppercase;
  background: var(--surface-3); color: var(--ink-dim); border: 1px solid var(--line);
}
table.bench .tag.live { color: var(--good); border-color: rgba(61,220,151,.45); }
/* `table.bench td` is (0,1,2) and would otherwise outrank a bare `td.good`
   at (0,1,1), so the colour never landed. Qualified and forced. */
table.bench td.good { color: var(--good) !important; }
table.bench td.ok   { color: var(--ok)   !important; }
table.bench td.poor { color: var(--poor) !important; }
.bench-note { color: var(--ink-faint); font-size: .84rem; margin: 14px 2px 4px; max-width: 78ch; }

.footnote { color: var(--ink-faint); font-size: .8rem; padding: 18px 4px 30px; }
.footnote a { color: var(--retrieval); text-decoration: none; }
"""

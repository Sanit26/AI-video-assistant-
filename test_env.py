import streamlit as st
import time
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LUMEN — Meeting Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --cream:   #f5f0e8;
    --warm:    #ede5d4;
    --paper:   #faf7f2;
    --ink:     #1a1510;
    --ink-2:   #3d352a;
    --ink-3:   #6b5e4f;
    --rust:    #c0392b;
    --rust-2:  #e05c3a;
    --gold:    #b8860b;
    --gold-lt: #d4a017;
    --line:    #d4c9b8;
    --shadow:  rgba(26,21,16,0.08);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--paper) !important;
    color: var(--ink) !important;
}

.stApp { background: var(--paper) !important; }

/* Subtle paper texture */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23noise)' opacity='0.025'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 9999;
    opacity: 0.4;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--ink) !important;
    border-right: none !important;
    padding: 0 !important;
}

[data-testid="stSidebar"] * {
    color: var(--cream) !important;
}

[data-testid="stSidebar"] label {
    color: var(--ink-3) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(245,240,232,0.07) !important;
    border: 1px solid rgba(245,240,232,0.12) !important;
    color: var(--cream) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 4px !important;
}

[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: var(--rust-2) !important;
    box-shadow: 0 0 0 2px rgba(192,57,43,0.2) !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: var(--rust) !important;
    color: var(--cream) !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.5rem !important;
    transition: background 0.2s !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--rust-2) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: 'Playfair Display', serif !important;
    color: var(--ink) !important;
}

/* ── Hero ── */
.lumen-hero {
    padding: 3rem 0 2rem;
    border-bottom: 1px solid var(--line);
    margin-bottom: 2.5rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
}

.lumen-wordmark {
    font-family: 'Playfair Display', serif;
    font-size: clamp(3rem, 7vw, 5.5rem);
    font-weight: 900;
    color: var(--ink);
    letter-spacing: -0.03em;
    line-height: 0.9;
    position: relative;
}

.lumen-wordmark span {
    color: var(--rust);
}

.lumen-tagline {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--ink-3);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    line-height: 1.6;
    text-align: right;
}

.edition-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--rust);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border: 1px solid var(--rust);
    padding: 0.2rem 0.5rem;
    display: inline-block;
    margin-bottom: 1rem;
}

/* ── Cards ── */
.ink-card {
    background: var(--cream);
    border: 1px solid var(--line);
    border-top: 3px solid var(--ink);
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    position: relative;
}

.ink-card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.ink-card-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--line);
}

.ink-card-body {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    line-height: 1.75;
    color: var(--ink-2);
}

/* Title card special */
.title-card {
    background: var(--ink);
    border: none;
    padding: 2rem 2rem 1.75rem;
    margin-bottom: 2rem;
}

.title-card .headline {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.4rem, 3.5vw, 2.2rem);
    font-weight: 700;
    color: var(--cream);
    line-height: 1.25;
    margin-bottom: 0.75rem;
}

.title-card-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--ink-3);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* ── Pipeline status (sidebar) ── */
.pipeline-wrap {
    margin-top: 1.5rem;
}

.pipe-step {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(245,240,232,0.07);
    font-size: 0.78rem;
    font-family: 'DM Sans', sans-serif;
    color: rgba(245,240,232,0.6);
    transition: color 0.3s;
}

.pipe-step.done   { color: rgba(245,240,232,0.95); }
.pipe-step.active { color: var(--rust-2); }

.pipe-icon {
    font-size: 0.9rem;
    width: 20px;
    text-align: center;
}

.pipe-bar {
    margin-left: auto;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: rgba(245,240,232,0.12);
    flex-shrink: 0;
}

.pipe-bar.done   { background: #6fcf97; }
.pipe-bar.active { background: var(--rust-2); animation: blink 1s infinite; }

@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* ── Section divider ── */
.section-rule {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2rem 0 1.5rem;
}

.section-rule-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--ink);
    white-space: nowrap;
}

.section-rule-line {
    flex: 1;
    height: 1px;
    background: var(--line);
}

/* ── Transcript ── */
.transcript-pane {
    background: var(--warm);
    border: 1px solid var(--line);
    border-left: 4px solid var(--gold);
    padding: 1.25rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.9;
    max-height: 320px;
    overflow-y: auto;
    color: var(--ink-2);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Chat ── */
.chat-pane {
    background: var(--cream);
    border: 1px solid var(--line);
    border-bottom: 3px solid var(--ink);
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-turn {
    margin-bottom: 1.25rem;
}

.chat-who {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

.chat-who.user { color: var(--rust); }
.chat-who.bot  { color: var(--gold); }

.chat-text {
    font-size: 0.875rem;
    line-height: 1.65;
    color: var(--ink-2);
    padding: 0.65rem 1rem;
    border-left: 2px solid var(--line);
}

.chat-text.user { border-color: var(--rust); background: rgba(192,57,43,0.04); }
.chat-text.bot  { border-color: var(--gold); background: rgba(184,134,11,0.04); }

/* ── Inputs main area ── */
.stTextInput > div > div > input {
    background: var(--cream) !important;
    border: 1px solid var(--line) !important;
    border-bottom: 2px solid var(--ink) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 0.7rem 1rem !important;
}

.stTextInput > div > div > input:focus {
    border-bottom-color: var(--rust) !important;
    box-shadow: none !important;
}

.stButton > button {
    background: var(--ink) !important;
    color: var(--cream) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.5rem !important;
    transition: background 0.15s !important;
}

.stButton > button:hover {
    background: var(--rust) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Secondary */
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--ink-3) !important;
    border: 1px solid var(--line) !important;
}

.stButton > button[kind="secondary"]:hover {
    background: var(--warm) !important;
    color: var(--ink) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--warm) !important;
    border: 1px solid var(--line) !important;
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--ink-2) !important;
    letter-spacing: 0.08em !important;
}

/* ── Misc ── */
hr { border: none !important; border-top: 1px solid var(--line) !important; margin: 1.5rem 0 !important; }
.stProgress > div > div > div { background: var(--rust) !important; }
.stSpinner > div { border-top-color: var(--rust) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--ink-2) !important; }
label { color: var(--ink-3) !important; font-size: 0.75rem !important; letter-spacing: 0.08em !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--warm); }
::-webkit-scrollbar-thumb { background: var(--line); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--ink-3); }

/* Empty state illustration */
.empty-state {
    padding: 5rem 2rem;
    text-align: center;
    border: 1px dashed var(--line);
    background: var(--cream);
}

.empty-numeral {
    font-family: 'Playfair Display', serif;
    font-size: 6rem;
    font-weight: 900;
    color: var(--line);
    line-height: 1;
    margin-bottom: 1rem;
    letter-spacing: -0.05em;
}

.empty-headline {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 0.5rem;
}

.empty-body {
    font-size: 0.875rem;
    color: var(--ink-3);
    line-height: 1.7;
    max-width: 360px;
    margin: 0 auto 2rem;
}

.caps-tag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border: 1px solid var(--line);
    color: var(--ink-3);
    margin: 0 0.25rem;
}

.caps-tag.accent {
    border-color: var(--rust);
    color: var(--rust);
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def pipe_class(key):
    s = st.session_state.pipeline_steps.get(key, "pending")
    return s  # "active" | "done" | "pending"

def render_pipeline_sidebar():
    steps = [
        ("audio",      "🔊", "Audio Processing"),
        ("transcript", "📝", "Transcription"),
        ("title",      "◈",  "Title Generation"),
        ("summary",    "◉",  "Summarisation"),
        ("extract",    "◎",  "Extraction"),
        ("rag",        "◆",  "RAG Engine"),
    ]
    html = '<div class="pipeline-wrap">'
    for key, icon, label in steps:
        cls = pipe_class(key)
        html += f"""
        <div class="pipe-step {cls}">
            <span class="pipe-icon">{icon}</span>
            <span>{label}</span>
            <div class="pipe-bar {cls}"></div>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:2rem 1.5rem 1.5rem">
        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.25em;text-transform:uppercase;color:rgba(107,94,79,0.7);margin-bottom:0.4rem">Meeting Intelligence</div>
        <div style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:900;color:#f5f0e8;letter-spacing:-0.02em;line-height:1">LUMEN</div>
        <div style="width:40px;height:2px;background:#c0392b;margin-top:0.75rem"></div>
    </div>
    <hr style="border-color:rgba(245,240,232,0.08)!important;margin:0 0 1.5rem!important">
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding:0 1.5rem">', unsafe_allow_html=True)
    source = st.text_input("Source", placeholder="YouTube URL or file path…")
    language = st.selectbox("Language", ["english", "hinglish"], index=0)
    run_btn = st.button("◈  Run Analysis", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.pipeline_done or st.session_state.pipeline_steps:
        st.markdown('<div style="padding:0 1.5rem"><hr style="border-color:rgba(245,240,232,0.08)!important;margin:1.25rem 0!important"></div>', unsafe_allow_html=True)
        st.markdown('<div style="padding:0 1.5rem">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;letter-spacing:0.2em;text-transform:uppercase;color:rgba(107,94,79,0.6);margin-bottom:0.5rem">Pipeline</div>', unsafe_allow_html=True)
        render_pipeline_sidebar()
        st.markdown('</div>', unsafe_allow_html=True)

# ─── Main ────────────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="lumen-hero">
    <div>
        <div class="edition-tag">◈ Intelligence Platform</div>
        <div class="lumen-wordmark">LUMEN<span>.</span></div>
    </div>
    <div class="lumen-tagline">
        Transcribe · Summarise<br>
        Extract · Converse<br>
        <span style="color:#c0392b">↳ Your meetings, decoded.</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Run pipeline ─────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}
        progress_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        try:
            with progress_placeholder.container():
                st.info("Pipeline running — monitor progress in the sidebar.")

            update_step("audio", "active"); chunks = process_input(source); update_step("audio", "done")
            update_step("transcript", "active"); transcript = transcribe_all(chunks, language); update_step("transcript", "done")
            update_step("title", "active"); title = generate_title(transcript); update_step("title", "done")
            update_step("summary", "active"); summary = summarize(transcript); update_step("summary", "done")
            update_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions    = extract_key_decisions(transcript)
            questions    = extract_questions(transcript)
            update_step("extract", "done")
            update_step("rag", "active"); rag_chain = build_rag_chain(transcript); update_step("rag", "done")

            st.session_state.result = {
                "title": title, "transcript": transcript, "summary": summary,
                "action_items": action_items, "key_decisions": decisions,
                "open_questions": questions, "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            progress_placeholder.success("✓ Analysis complete.")
            time.sleep(0.4)
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            for k in ["audio","transcript","title","summary","extract","rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_placeholder.error(f"Error: {e}")

# ── Results ───────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Title banner
    st.markdown(f"""
    <div class="title-card">
        <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.2em;text-transform:uppercase;color:rgba(107,94,79,0.5);margin-bottom:0.65rem">◈ Session Title</div>
        <div class="headline">{r['title']}</div>
        <div class="title-card-meta">Lumen Intelligence Report &nbsp;·&nbsp; Analysis Complete</div>
    </div>
    """, unsafe_allow_html=True)

    # Summary + Transcript
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="ink-card">
            <div class="ink-card-label">◉ Summary</div>
            <div class="ink-card-body">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        with st.expander("▸ Full Transcript", expanded=False):
            st.markdown(f'<div class="transcript-pane">{r["transcript"]}</div>', unsafe_allow_html=True)

    # 3-column extraction
    st.markdown("""
    <div class="section-rule">
        <div class="section-rule-line"></div>
        <div class="section-rule-text">Extracted Intelligence</div>
        <div class="section-rule-line"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="ink-card" style="border-top-color:#c0392b">
            <div class="ink-card-label">✓ Action Items</div>
            <div class="ink-card-body">{r['action_items']}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="ink-card" style="border-top-color:#b8860b">
            <div class="ink-card-label">◆ Key Decisions</div>
            <div class="ink-card-body">{r['key_decisions']}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="ink-card" style="border-top-color:#1a1510">
            <div class="ink-card-label">? Open Questions</div>
            <div class="ink-card-body">{r['open_questions']}</div>
        </div>""", unsafe_allow_html=True)

    # ── RAG Chat ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-rule">
        <div class="section-rule-line"></div>
        <div class="section-rule-text">Interrogate the Meeting</div>
        <div class="section-rule-line"></div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.chat_history:
        chat_html = '<div class="chat-pane">'
        for msg in st.session_state.chat_history:
            role_cls = "user" if msg["role"] == "user" else "bot"
            who_label = "You" if msg["role"] == "user" else "Lumen"
            chat_html += f"""
            <div class="chat-turn">
                <div class="chat-who {role_cls}">{who_label}</div>
                <div class="chat-text {role_cls}">{msg['content']}</div>
            </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ink-card" style="text-align:center;padding:2.5rem;border-top-color:var(--gold)">
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:var(--ink);margin-bottom:0.4rem">Ask anything about your meeting</div>
            <div style="font-size:0.82rem;color:var(--ink-3)">What were the key risks raised? Who owns each action item?</div>
        </div>""", unsafe_allow_html=True)

    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        user_input = st.text_input("Ask Lumen", placeholder="What decisions were deferred?", label_visibility="collapsed")
    with chat_col2:
        send_btn = st.button("Ask →", use_container_width=True)

    if send_btn and user_input.strip():
        with st.spinner("Thinking…"):
            answer = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user",      "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Empty state
    st.markdown("""
    <div class="empty-state">
        <div class="empty-numeral">◈</div>
        <div class="empty-headline">Ready for Analysis</div>
        <div class="empty-body">
            Paste a YouTube URL or local file path into the sidebar, select your language, and run the pipeline.
        </div>
        <span class="caps-tag accent">Transcription</span>
        <span class="caps-tag">Summarisation</span>
        <span class="caps-tag">RAG Chat</span>
        <span class="caps-tag">Extraction</span>
    </div>
    """, unsafe_allow_html=True)
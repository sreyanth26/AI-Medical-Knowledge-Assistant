"""
AI Medical Knowledge Assistant — Production Frontend v2.0
Dark theme with full UX: validation, loading states, error messages,
document management, upload feedback, and polished interactions.

Run:
    streamlit run frontend/app.py
"""

import os
import time
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT_UPLOAD = 60
REQUEST_TIMEOUT_ASK    = 120

st.set_page_config(
    page_title="MedRAG · AI Medical Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"], .stApp { background-color:#0a0e1a!important; color:#c8d6e5!important; font-family:'Space Grotesk',sans-serif!important; }
[data-testid="stSidebar"] { background:#0d1120!important; border-right:1px solid #1e2d45!important; }
[data-testid="stSidebar"] * { color:#a8bdd0!important; }

.med-header { background:linear-gradient(120deg,#0d1f35 0%,#0a1628 40%,#091524 100%); border:1px solid #1a3a5c; border-radius:16px; padding:2.2rem 2.5rem; margin-bottom:1.8rem; position:relative; overflow:hidden; }
.med-header::before { content:''; position:absolute; top:-60px; right:-60px; width:220px; height:220px; background:radial-gradient(circle,rgba(0,180,220,.12) 0%,transparent 70%); border-radius:50%; }
.med-header h1 { font-family:'Playfair Display',serif!important; font-size:2.1rem!important; font-weight:700!important; color:#e8f4ff!important; margin:0 0 .5rem 0!important; }
.med-header p { color:#6a93b8!important; font-size:.9rem!important; margin:0!important; }
.badge { display:inline-block; background:rgba(0,180,220,.15); border:1px solid rgba(0,180,220,.3); color:#00c8e0!important; font-family:'JetBrains Mono',monospace; font-size:.7rem; padding:.2rem .7rem; border-radius:20px; margin-right:.4rem; margin-top:.8rem; }

.answer-box { background:#0d1a2e; border:1px solid #1a3a5c; border-left:4px solid #00a8c8; border-radius:0 12px 12px 0; padding:1.5rem 1.8rem; margin:1rem 0; font-size:.96rem; line-height:1.8; color:#c8daea!important; }

.source-card { background:#0d1627; border:1px solid #1a2e45; border-radius:10px; padding:1rem 1.3rem; margin:.5rem 0; transition:border-color .2s,transform .15s; }
.source-card:hover { border-color:#00a8c8; transform:translateX(3px); }
.doc-name { font-family:'JetBrains Mono',monospace; font-size:.78rem; color:#00c8e0!important; font-weight:600; letter-spacing:.05em; }
.snippet { color:#7a9bb8!important; font-size:.855rem; margin-top:.45rem; line-height:1.65; }

.disclaimer { background:#1a1500; border:1px solid #3a2e00; border-left:4px solid #f0a500; border-radius:0 10px 10px 0; padding:.85rem 1.2rem; font-size:.83rem; color:#c8a840!important; margin-top:1.5rem; }

.stat-pill { display:inline-block; background:#0d1f35; border:1px solid #1a3a5c; color:#4a9abb!important; font-family:'JetBrains Mono',monospace; font-size:.74rem; padding:.25rem .85rem; border-radius:20px; margin:.3rem .2rem; }

.error-box { background:#1a0808; border:1px solid #5a1a1a; border-left:4px solid #e05050; border-radius:0 10px 10px 0; padding:.9rem 1.2rem; color:#e8a0a0!important; font-size:.88rem; margin:.5rem 0; }
.warn-box  { background:#1a1200; border:1px solid #4a3200; border-left:4px solid #f0a500; border-radius:0 10px 10px 0; padding:.9rem 1.2rem; color:#d4aa60!important; font-size:.88rem; margin:.5rem 0; }
.info-box  { background:#071220; border:1px solid #0a2a4a; border-left:4px solid #2080c0; border-radius:0 10px 10px 0; padding:.9rem 1.2rem; color:#80b8e0!important; font-size:.88rem; margin:.5rem 0; }
.success-box { background:#071a0e; border:1px solid #0a4a20; border-left:4px solid #30c060; border-radius:0 10px 10px 0; padding:.9rem 1.2rem; color:#80e0a0!important; font-size:.88rem; margin:.5rem 0; }

.doc-item { background:#0d1627; border:1px solid #1a2e45; border-radius:8px; padding:.7rem 1rem; margin:.35rem 0; display:flex; justify-content:space-between; align-items:center; }
.doc-item .doc-title { font-family:'JetBrains Mono',monospace; font-size:.75rem; color:#00c8e0!important; }
.doc-item .doc-meta  { font-size:.72rem; color:#4a6a88!important; }

.stButton>button { background:linear-gradient(135deg,#003a5c,#005a7a)!important; color:#a0d8ef!important; border:1px solid #005a7a!important; border-radius:8px!important; font-family:'Space Grotesk',sans-serif!important; font-weight:600!important; font-size:.88rem!important; transition:all .2s!important; }
.stButton>button:hover { background:linear-gradient(135deg,#004a72,#007a9a)!important; border-color:#00a8c8!important; color:#e0f4ff!important; transform:translateY(-1px)!important; box-shadow:0 4px 16px rgba(0,168,200,.25)!important; }

.stTextInput>div>div>input { background:#0d1627!important; border:1px solid #1a3a5c!important; border-radius:8px!important; color:#c8daea!important; font-family:'Space Grotesk',sans-serif!important; }
.stTextInput>div>div>input:focus { border-color:#00a8c8!important; box-shadow:0 0 0 2px rgba(0,168,200,.15)!important; }
.stTextInput>div>div>input::placeholder { color:#3a5a78!important; }

.stTabs [data-baseweb="tab-list"] { background:transparent!important; border-bottom:1px solid #1a2e45!important; }
.stTabs [data-baseweb="tab"] { background:transparent!important; color:#4a6a88!important; font-family:'Space Grotesk',sans-serif!important; }
.stTabs [aria-selected="true"] { color:#00c8e0!important; border-bottom-color:#00c8e0!important; background:transparent!important; }

div[data-testid="stFileUploader"] { border:2px dashed #1a3a5c!important; border-radius:12px!important; background:#080e1a!important; }
.section-label { font-family:'JetBrains Mono',monospace; font-size:.72rem; color:#3a6a88!important; text-transform:uppercase; letter-spacing:.1em; margin-bottom:.5rem; }
.char-count { font-family:'JetBrains Mono',monospace; font-size:.7rem; text-align:right; margin-top:.2rem; }
.char-count.ok   { color:#3a8a5a!important; }
.char-count.warn { color:#f0a500!important; }
.char-count.over { color:#e05050!important; }
.chat-user { background:#0d1f35; border:1px solid #1a3a5c; border-radius:12px 12px 4px 12px; padding:.9rem 1.2rem; margin:.5rem 0 .5rem 3rem; color:#c0d8f0!important; }
.chat-bot  { background:#091524; border:1px solid #152840; border-left:3px solid #00a8c8; border-radius:4px 12px 12px 12px; padding:.9rem 1.2rem; margin:.5rem 3rem .5rem 0; color:#a8c8e0!important; line-height:1.7; }
.pipe-step { background:#0d1627; border:1px solid #1a2e45; border-radius:8px; padding:.7rem 1.2rem; font-size:.88rem; color:#8ab8d0!important; margin-bottom:2px; }
.pipe-num  { font-family:'JetBrains Mono',monospace; color:#00c8e0!important; font-size:.75rem; margin-right:.5rem; }
.pipe-arrow { text-align:center; color:#1a3a5c!important; line-height:1.2; margin:1px 0; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#080e1a; }
::-webkit-scrollbar-thumb { background:#1a3a5c; border-radius:3px; }
hr { border-color:#1a2e45!important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# API Helpers (with proper error handling)
# ──────────────────────────────────────────────────────────────────

def _api_get(path: str, timeout: int = 5):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Run: `cd backend` → `uvicorn api:app --reload`"
    except requests.exceptions.Timeout:
        return None, f"Request timed out after {timeout}s."
    except requests.exceptions.HTTPError as e:
        return None, f"Server error {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def _api_post(path: str, json_body: dict = None, files: dict = None, timeout: int = 30):
    try:
        kwargs = {"timeout": timeout}
        if json_body:
            kwargs["json"] = json_body
        if files:
            kwargs["files"] = files
        r = requests.post(f"{API_BASE}{path}", **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Is it running?"
    except requests.exceptions.Timeout:
        return None, f"Request timed out ({timeout}s). The LLM may be slow — try again."
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", e.response.text[:300])
        except Exception:
            detail = e.response.text[:300]
        return None, f"Error {e.response.status_code}: {detail}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def _api_delete(path: str, timeout: int = 10):
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", e.response.text[:200])
        except Exception:
            detail = e.response.text[:200]
        return None, f"Error {e.response.status_code}: {detail}"
    except Exception as e:
        return None, str(e)


def validate_question(q: str) -> str | None:
    """Return error string or None if valid."""
    q = q.strip()
    if not q:
        return "Please enter a question."
    if len(q) < 5:
        return "Question is too short. Please be more specific."
    if len(q) > 1000:
        return f"Question is too long ({len(q)}/1000 chars). Please shorten it."
    return None


# ──────────────────────────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────────────────────────

for key, default in [
    ("chat_history",  []),
    ("indexed_docs",  {}),
    ("last_error",    None),
    ("question_val_error", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ──────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)

    status_data, status_err = _api_get("/status")
    if status_data:
        st.success("✅ Backend Online")
        st.caption(f"🤖 `{status_data.get('llm_model','unknown')}`  •  v{status_data.get('version','?')}")
        doc_count = status_data.get("document_count", 0)
        if doc_count > 0:
            st.caption(f"📚 {doc_count} document(s) indexed")
        else:
            st.caption("📂 No documents indexed yet")
    else:
        st.error("❌ Backend Offline")
        st.caption(status_err or "Unknown error")

    st.markdown("---")

    # ── Upload Section ──────────────────────────────────────────

    st.markdown('<div class="section-label">Upload Documents</div>', unsafe_allow_html=True)
    st.caption("PDF, TXT, or MD • Max 50 MB each")

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="file_uploader",
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_docs]
        already   = [f for f in uploaded_files if f.name in st.session_state.indexed_docs]

        if already:
            for f in already:
                st.markdown(
                    f'<div class="info-box">ℹ️ <code>{f.name}</code> already indexed.</div>',
                    unsafe_allow_html=True,
                )

        if new_files:
            if st.button(f"⚡ Index {len(new_files)} File(s)", use_container_width=True):
                progress = st.progress(0, text="Preparing...")
                for i, f in enumerate(new_files):
                    progress.progress((i) / len(new_files), text=f"Indexing {f.name}...")
                    content = f.read()

                    # Client-side file validation
                    if len(content) == 0:
                        st.markdown(
                            f'<div class="error-box">❌ <code>{f.name}</code> is empty. Skipped.</div>',
                            unsafe_allow_html=True,
                        )
                        continue

                    size_mb = len(content) / (1024 * 1024)
                    if size_mb > 50:
                        st.markdown(
                            f'<div class="error-box">❌ <code>{f.name}</code> is {size_mb:.1f} MB — exceeds 50 MB limit.</div>',
                            unsafe_allow_html=True,
                        )
                        continue

                    ext = f.name.rsplit(".", 1)[-1].lower()
                    mime = "application/pdf" if ext == "pdf" else "text/plain"

                    result, err = _api_post(
                        "/upload",
                        files={"file": (f.name, content, mime)},
                        timeout=REQUEST_TIMEOUT_UPLOAD,
                    )

                    if err:
                        st.markdown(
                            f'<div class="error-box">❌ <b>{f.name}</b>: {err}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        chunks = result.get("chunks_indexed", "?")
                        doc_id = result.get("doc_id", "?")
                        st.session_state.indexed_docs[f.name] = {
                            "doc_id": doc_id,
                            "chunks": chunks,
                            "size_kb": result.get("file_size_kb", 0),
                        }
                        st.markdown(
                            f'<div class="success-box">✅ <b>{f.name}</b> — {chunks} chunks indexed</div>',
                            unsafe_allow_html=True,
                        )

                progress.progress(1.0, text="Done!")
                time.sleep(0.5)
                progress.empty()
                st.rerun()

    # ── Indexed Documents List ──────────────────────────────────

    if st.session_state.indexed_docs:
        st.markdown('<div class="section-label" style="margin-top:.8rem">Indexed Files</div>', unsafe_allow_html=True)
        for name, meta in list(st.session_state.indexed_docs.items()):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(
                    f'<div class="doc-item">'
                    f'<span class="doc-title">📄 {name[:28]}</span>'
                    f'<span class="doc-meta">{meta["chunks"]} chunks · {meta["size_kb"]:.1f} KB</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_b:
                if st.button("🗑", key=f"del_{name}", help=f"Remove {name}"):
                    doc_id = meta.get("doc_id")
                    if doc_id:
                        _, err = _api_delete(f"/documents/{doc_id}")
                        if err:
                            st.error(f"Delete failed: {err}")
                        else:
                            del st.session_state.indexed_docs[name]
                            st.rerun()

    st.markdown("---")

    # ── Settings ──────────────────────────────────────────────

    st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
    top_k         = st.slider("Top-K chunks", 1, 10, 5, help="Document sections retrieved per query")
    show_sources  = st.toggle("Show citations",      value=True)
    show_snippets = st.toggle("Show text snippets",  value=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("🔄 Clear Index", use_container_width=True):
            _, err = _api_delete("/index")
            if err:
                st.error(err)
            else:
                st.session_state.indexed_docs = {}
                st.success("Index cleared.")
                st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-label">Quick Queries</div>', unsafe_allow_html=True)
    EXAMPLES = [
        "Treatment for dengue fever?",
        "Side effects of paracetamol?",
        "Amoxicillin dosage for adults?",
        "Symptoms of malaria?",
        "How to manage hypertension?",
        "Insulin types and usage?",
        "CURB-65 score for pneumonia?",
        "Metformin contraindications?",
    ]
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex_{ex[:16]}", use_container_width=True):
            st.session_state["prefill_q"] = ex
            st.rerun()


# ──────────────────────────────────────────────────────────────────
# Main Panel
# ──────────────────────────────────────────────────────────────────

st.markdown("""
    <div class="med-header">
        <h1>🧬 MedRAG Assistant</h1>
        <p>AI-powered medical knowledge retrieval · Answers grounded in your documents · Zero hallucination</p>
        <div style="margin-top:.8rem">
            <span class="badge">RAG</span><span class="badge">FAISS</span>
            <span class="badge">Llama 3</span><span class="badge">LangChain</span>
            <span class="badge">Open-Source</span><span class="badge">v2.0</span>
        </div>
    </div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔬 Ask a Question", "📜 Chat History", "⚙️ How It Works"])

# ──────────────────────────────────────────────────────────────────
# Tab 1: Q&A
# ──────────────────────────────────────────────────────────────────
with tab1:

    # Guard: no documents indexed
    if not st.session_state.indexed_docs and (status_data and not status_data.get("documents_indexed")):
        st.markdown("""
            <div class="warn-box">
            ⚠️ <b>No documents indexed.</b> Upload medical PDFs or TXT files in the sidebar first,
            then click <b>⚡ Index Documents</b>. You can use the sample files in the
            <code>data/</code> folder to get started.
            </div>
        """, unsafe_allow_html=True)

    # Question input
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        default_q = st.session_state.pop("prefill_q", "")
        question  = st.text_input(
            "q", value=default_q,
            placeholder="Ask a medical question — e.g. What is the treatment for dengue fever?",
            label_visibility="collapsed",
            max_chars=1000,
        )
        # Live char counter
        q_len = len(question)
        css_class = "ok" if q_len <= 800 else ("warn" if q_len <= 1000 else "over")
        if q_len > 0:
            st.markdown(
                f'<div class="char-count {css_class}">{q_len}/1000</div>',
                unsafe_allow_html=True,
            )

    with col_btn:
        st.markdown("<div style='margin-top:0.35rem'></div>", unsafe_allow_html=True)
        ask_btn = st.button("🔍 Ask", use_container_width=True)

    # Inline validation error
    if st.session_state.question_val_error:
        st.markdown(
            f'<div class="error-box">⚠️ {st.session_state.question_val_error}</div>',
            unsafe_allow_html=True,
        )
        st.session_state.question_val_error = None

    if ask_btn:
        val_err = validate_question(question)
        if val_err:
            st.session_state.question_val_error = val_err
            st.rerun()
        else:
            with st.spinner("🔎 Retrieving relevant sections and generating answer..."):
                t0     = time.perf_counter()
                result, err = _api_post(
                    "/ask",
                    json_body={"question": question.strip(), "top_k": top_k},
                    timeout=REQUEST_TIMEOUT_ASK,
                )
                elapsed = time.perf_counter() - t0

            if err:
                st.markdown(
                    f'<div class="error-box">❌ <b>Error:</b> {err}</div>',
                    unsafe_allow_html=True,
                )
            else:
                answer  = result.get("answer", "No answer returned.")
                sources = result.get("sources", [])
                chunks  = result.get("retrieved_chunks", 0)
                model   = result.get("model_used", "unknown")
                rid     = result.get("request_id", "")

                st.session_state.chat_history.append({
                    "question": question.strip(),
                    "answer":   answer,
                    "sources":  sources,
                })

                st.markdown('<div class="section-label" style="margin-top:1rem">Answer</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                st.markdown(
                    f'<span class="stat-pill">📄 {chunks} chunks</span>'
                    f'<span class="stat-pill">🤖 {model}</span>'
                    f'<span class="stat-pill">⏱ {elapsed:.1f}s</span>'
                    f'<span class="stat-pill">🔑 top-{top_k}</span>'
                    + (f'<span class="stat-pill">🆔 {rid}</span>' if rid else ""),
                    unsafe_allow_html=True,
                )

                if show_sources and sources:
                    st.markdown('<div class="section-label" style="margin-top:1.2rem">Source Citations</div>', unsafe_allow_html=True)
                    for src in sources:
                        snip = f'<div class="snippet">{src["snippet"][:220]}…</div>' if show_snippets else ""
                        st.markdown(f"""
                            <div class="source-card">
                                <div class="doc-name">📄 {src['document']} &nbsp;·&nbsp; chunk #{src['chunk_index']}</div>
                                {snip}
                            </div>
                        """, unsafe_allow_html=True)

                st.markdown("""
                    <div class="disclaimer">⚠️ <strong>Medical Disclaimer:</strong>
                    This assistant provides informational guidance based on uploaded documents only.
                    It does <em>not</em> replace professional medical advice, diagnosis, or treatment.
                    Always consult a qualified healthcare professional for medical decisions.
                    </div>
                """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# Tab 2: Chat History
# ──────────────────────────────────────────────────────────────────
with tab2:
    if not st.session_state.chat_history:
        st.markdown('<div class="info-box">ℹ️ No history yet. Ask a question in the first tab.</div>', unsafe_allow_html=True)
    else:
        st.caption(f"{len(st.session_state.chat_history)} conversation(s) this session")
        for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Q{len(st.session_state.chat_history)-i+1}: {entry['question'][:70]}…", expanded=(i == 1)):
                st.markdown(f'<div class="chat-user">🧑 {entry["question"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-bot">🤖 {entry["answer"]}</div>', unsafe_allow_html=True)
                if entry.get("sources"):
                    srcs = " · ".join(f"`{s['document']}`" for s in entry["sources"][:3])
                    st.caption(f"Sources: {srcs}")

# ──────────────────────────────────────────────────────────────────
# Tab 3: How It Works
# ──────────────────────────────────────────────────────────────────
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🧠 RAG Pipeline")
        steps = [
            "User submits a question",
            "Embedding model vectorises the query",
            "FAISS searches for similar document chunks",
            "Top-K relevant sections retrieved",
            "LLM generates a grounded, cited answer",
            "Answer + source citations returned",
        ]
        html = ""
        for i, s in enumerate(steps, 1):
            html += f'<div class="pipe-step"><span class="pipe-num">0{i}</span>{s}</div>'
            if i < len(steps):
                html += '<div class="pipe-arrow">↓</div>'
        st.markdown(html, unsafe_allow_html=True)

    with col_b:
        st.markdown("#### 🛠️ Tech Stack")
        st.markdown("""
| Layer | Technology |
|---|---|
| RAG Framework | LangChain |
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| LLM | Llama 3 / Mistral |
| LLM Runtime | Ollama (local) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Testing | pytest + httpx |
| Language | Python 3.11 |
""")

    st.markdown("---")
    st.markdown("#### 🚀 Startup (Windows PowerShell)")
    st.code("""# Terminal 1 — Ollama LLM
ollama pull llama3
ollama serve

# Terminal 2 — Backend API
cd AI-Medical-Knowledge-Assistant\\backend
uvicorn api:app --reload

# Terminal 3 — Frontend
cd AI-Medical-Knowledge-Assistant
streamlit run frontend/app.py

# Run Tests
cd AI-Medical-Knowledge-Assistant
pip install pytest httpx
pytest tests/test_medrag.py -v
""", language="powershell")

    st.markdown("---")
    st.markdown("#### 📊 API Endpoints")
    st.markdown("""
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe |
| GET | `/status` | System status |
| POST | `/upload` | Upload & index a document |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{id}` | Remove a document |
| DELETE | `/index` | Clear all documents |
| POST | `/ask` | Ask a medical question |
| POST | `/summarize` | Summarize medical text |
""")

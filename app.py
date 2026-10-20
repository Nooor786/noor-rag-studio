from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Any

import streamlit as st

from rag_engine import DocumentIndex, answer_with_ollama, load_embedding_model, load_reranker


APP_TITLE = "NooR RAG Studio"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
SAMPLE_PDF = Path(__file__).parent / "documents" / "sample.pdf"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def cached_embedding_model():
    return load_embedding_model()


@st.cache_resource(show_spinner=False)
def cached_reranker():
    return load_reranker()

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root {
        --ink:#f1f5ff; --muted:#a7b4d0; --line:rgba(139,157,255,.22);
        --accent:#8b5cf6; --accent2:#22d3ee; --panel:rgba(20,29,55,.82);
      }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
      .stApp {
        color:var(--ink);
        background:
          radial-gradient(ellipse at 10% 4%,rgba(99,102,241,.22),transparent 36%),
          radial-gradient(ellipse at 92% 18%,rgba(34,211,238,.12),transparent 32%),
          linear-gradient(135deg,#080d1c 0%,#10162c 55%,#0b1020 100%);
      }
      [data-testid="stAppViewContainer"] > .main { position:relative; z-index:1; }
      .block-container { max-width:1450px; padding-top:2rem; padding-bottom:3rem; }
      [data-testid="stSidebar"] {
        background:rgba(11,17,35,.97); border-right:1px solid var(--line);
      }
      [data-testid="stSidebar"] * { color:#eef2ff; }
      [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background:rgba(255,255,255,.05); border-color:rgba(139,157,255,.35);
      }

/* Knowledge Sources - Upload button */
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #6D5DFB, #8B5CF6) !important;
    color: #FFFFFF !important;
    border: 1px solid #A78BFA !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(139, 92, 246, 0.25);
}

/* Hover effect */
[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #0891B2, #22D3EE) !important;
    color: #08111F !important;
    border-color: #67E8F9 !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(34, 211, 238, 0.35);
}

/* Keyboard focus */
[data-testid="stFileUploader"] button:focus-visible {
    outline: 2px solid #67E8F9 !important;
    outline-offset: 3px;
}

      .hero {
        position:relative; overflow:hidden; padding:1.8rem 1.9rem; border-radius:24px;
        color:#fff;
        background:
          radial-gradient(circle at 88% 18%,rgba(34,211,238,.18),transparent 30%),
          linear-gradient(120deg,rgba(23,29,53,.98),rgba(42,39,101,.97) 54%,rgba(78,54,143,.94));
        border:1px solid rgba(139,157,255,.30);
        box-shadow:0 18px 45px rgba(0,0,0,.24),inset 0 1px 0 rgba(255,255,255,.06);
        margin-bottom:1.2rem; animation:heroIn .55s ease both;
      }
      .hero:after {
        content:""; position:absolute; inset:-50% -80%; pointer-events:none;
        background:linear-gradient(110deg,transparent 42%,rgba(255,255,255,.055) 50%,transparent 58%);
        transform:translateX(-35%); animation:heroShimmer 12s ease-in-out infinite;
      }
      .hero-kicker { font-size:.76rem; text-transform:uppercase; letter-spacing:.18em; color:#c7d2fe; font-weight:800; }
      .hero h1 {
        font-family:'Manrope',sans-serif; font-size:clamp(1.8rem,3vw,2.65rem);
        line-height:1.12; margin:.5rem 0 .65rem; color:#fff; font-weight:800; letter-spacing:-.035em;
      }
      .hero p { margin:0; color:#dce6ff; max-width:760px; font-size:1rem; }
      .pill {
        display:inline-block; padding:.32rem .7rem; border-radius:999px;
        background:rgba(255,255,255,.09); border:1px solid rgba(196,181,253,.28);
        color:#e9e7ff; font-size:.78rem; margin:.9rem .35rem 0 0;
      }
      .section-label { color:#a5b4fc; font-weight:800; font-size:.75rem; text-transform:uppercase; letter-spacing:.13em; margin-bottom:.25rem; }
      .panel {
        background:var(--panel); color:#f1f5ff; border:1px solid var(--line); border-radius:18px;
        padding:1rem 1.15rem; box-shadow:0 12px 30px rgba(0,0,0,.15);
        transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease;
      }
      .panel:hover { transform:translateY(-3px); border-color:rgba(34,211,238,.42); box-shadow:0 16px 34px rgba(34,211,238,.07),0 12px 28px rgba(0,0,0,.18); }
      .small-muted { color:#a7b4d0; font-size:.88rem; }
      .source-card {
        background:rgba(17,26,50,.92); color:#e7edff; border:1px solid rgba(139,157,255,.22);
        border-left:4px solid #8b5cf6; border-radius:12px; padding:.8rem 1rem; margin:.4rem 0 .65rem;
        overflow-wrap:anywhere;
      }
      .source-meta { font-size:.78rem; color:#a7b4d0; margin-bottom:.4rem; }
      h1,h2,h3,h4 { color:#f1f5ff !important; letter-spacing:-.025em; }
      p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color:#d1daf0; }
      [data-testid="stMetric"] {
        background:var(--panel); color:#f1f5ff; padding:.9rem 1rem;
        border:1px solid var(--line); border-radius:16px;
        box-shadow:0 10px 28px rgba(0,0,0,.14);
        transition:transform .22s ease,border-color .22s ease;
      }
      [data-testid="stMetric"]:hover { transform:translateY(-3px); border-color:rgba(34,211,238,.42); }
      [data-testid="stMetricValue"] { color:#c4b5fd !important; font-weight:800; }
      [data-testid="stMetricLabel"] { color:#a7b4d0 !important; }
      .stButton>button, .stDownloadButton>button {
        color:#fff !important; background:linear-gradient(115deg,#6d5dfb,#8b5cf6 55%,#4f8cff);
        border:1px solid rgba(196,181,253,.32); border-radius:12px; font-weight:700;
        box-shadow:0 6px 20px rgba(109,93,251,.22);
        transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease;
      }
      .stButton>button:hover, .stDownloadButton>button:hover {
        color:#fff !important; transform:translateY(-2px); border-color:rgba(103,232,249,.65);
        box-shadow:0 9px 28px rgba(109,93,251,.34),0 0 18px rgba(34,211,238,.09);
      }
      .stTextInput input, .stNumberInput input, .stTextArea textarea, [data-baseweb="select"] > div {
        color:#f8faff !important; background:rgba(15,23,46,.96) !important;
        border:1px solid rgba(139,157,255,.32) !important; border-radius:12px !important;
      }
      .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
        border-color:#22d3ee !important; box-shadow:0 0 0 3px rgba(34,211,238,.12) !important;
      }
      .stTextInput input::placeholder, .stTextArea textarea::placeholder { color:#8190b2 !important; opacity:1; }
      [data-testid="stFileUploader"] {
        background:rgba(18,27,51,.72); border:1px dashed rgba(139,157,255,.48);
        border-radius:16px; padding:12px; transition:border-color .2s ease,background .2s ease;
      }
      [data-testid="stFileUploader"]:hover { border-color:#22d3ee; background:rgba(34,211,238,.045); }
      [data-testid="stFileUploader"] section { background:transparent !important; }
      [data-testid="stFileUploader"] small { color:#a7b4d0 !important; }
      [data-testid="stExpander"] {
        background:rgba(20,29,55,.76); border:1px solid var(--line); border-radius:14px;
      }
      [data-testid="stExpander"] summary { color:#e2e8ff !important; }
      [data-testid="stChatMessage"] {
        background:rgba(20,29,55,.78); border:1px solid var(--line); border-radius:16px;
        padding:.9rem 1rem; margin-bottom:.75rem; animation:messageIn .32s ease both;
      }
      .stTabs [data-baseweb="tab-list"] { gap:.4rem; }
      .stTabs [data-baseweb="tab"] { color:#a7b4d0 !important; border-radius:10px 10px 0 0; padding:.7rem 1rem; }
      .stTabs [data-baseweb="tab"][aria-selected="true"] { color:#c4b5fd !important; border-bottom:2px solid #8b5cf6 !important; }
      [data-testid="stAlert"] { background:rgba(20,29,55,.9); border:1px solid var(--line); border-radius:12px; }
      hr { border-color:rgba(139,157,255,.18); }
      footer { visibility:hidden; }
      @keyframes heroIn { from { opacity:0; transform:translateY(9px); } to { opacity:1; transform:translateY(0); } }
      @keyframes messageIn { from { opacity:0; transform:translateY(7px); } to { opacity:1; transform:translateY(0); } }
      @keyframes heroShimmer { 0%,65% { transform:translateX(-35%); } 100% { transform:translateX(35%); } }
      @media (prefers-reduced-motion:reduce) { *,*::before,*::after { animation:none !important; transition:none !important; } }
      @media (max-width:768px) { .block-container { padding:1rem; } .hero { padding:1.25rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "index": None,
        "index_signature": None,
        "index_info": [],
        "last_sources": [],
        "last_answer": "",
        "last_question": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

with st.sidebar:
    st.markdown("<div style='font-size:.75rem;letter-spacing:.18em;font-weight:800;color:#a5b4fc'>✦ NooR RAG STUDIO</div>", unsafe_allow_html=True)
    st.markdown("## Your document copilot")
    st.caption("Private, local-first retrieval augmented generation")
    st.markdown("---")
    st.markdown("### 01 · Knowledge sources")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text-based PDFs work best. Scanned PDFs need OCR, which is not included in this version.",
    )
    use_sample = st.checkbox(
        "Include documents/sample.pdf if present",
        value=(SAMPLE_PDF.exists() and not uploaded_files),
        help="Useful for continuing your existing project. You can uncheck this to use only uploads.",
    )
    st.markdown("---")
    st.markdown("### 02 · Retrieval controls")
    top_k = st.slider("Sources to retrieve", min_value=2, max_value=8, value=4)
    chunk_size = st.slider("Chunk size (words)", min_value=80, max_value=220, value=140, step=20)
    chunk_overlap = st.slider("Chunk overlap (words)", min_value=10, max_value=60, value=35, step=5)
    use_reranker = st.toggle("Cross-encoder reranking", value=True, help="Adds a second-stage relevance model after hybrid retrieval. First use may download model files.")
    st.markdown("---")
    st.markdown("### 03 · Local model")
    ollama_url = st.text_input("Ollama server URL", value=DEFAULT_OLLAMA_URL).strip().rstrip("/")
    ollama_model = st.text_input("Ollama model", value=DEFAULT_MODEL).strip()
    st.caption("Default: llama3.2:3b on localhost")
    rebuild = st.button("↻ Rebuild document index", use_container_width=True)
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

# Collect selected documents in memory; no upload is sent to a third-party service by this app.
documents: list[tuple[str, bytes]] = []
if use_sample and SAMPLE_PDF.exists():
    try:
        documents.append((SAMPLE_PDF.name, SAMPLE_PDF.read_bytes()))
    except OSError as exc:
        st.sidebar.error(f"Could not read sample PDF: {exc}")
for uploaded in uploaded_files or []:
    documents.append((uploaded.name, uploaded.getvalue()))

if documents:
    signature_hash = hashlib.sha256()
    signature_hash.update(str((chunk_size, chunk_overlap, use_reranker)).encode())
    for name, data in documents:
        signature_hash.update(name.encode("utf-8", errors="ignore"))
        signature_hash.update(hashlib.sha256(data).digest())
    signature = signature_hash.hexdigest()
    if signature != st.session_state.index_signature or rebuild:
        with st.spinner("Preparing your knowledge base · extracting, chunking and embedding PDFs..."):
            try:
                embedding_model = cached_embedding_model()
                reranker = cached_reranker() if use_reranker else None
                st.session_state.index = DocumentIndex.from_pdfs(
                    documents=documents,
                    embedding_model=embedding_model,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap,
                    reranker=reranker,
                )
                st.session_state.index_signature = signature
                st.session_state.index_info = st.session_state.index.document_summary
                st.session_state.messages = []
                st.session_state.last_sources = []
                st.success("Knowledge base indexed successfully.")
            except Exception as exc:
                st.session_state.index = None
                st.session_state.index_signature = None
                st.error(f"Indexing failed: {exc}")
else:
    st.session_state.index = None
    st.session_state.index_signature = None
    st.session_state.index_info = []

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Research workspace · Hybrid retrieval · Local AI</div>
      <h1>Ask your documents<br>better questions.</h1>
      <p>Search across your PDFs with keyword + semantic retrieval, inspect the evidence, and generate answers with citations from your local LLM.</p>
      <span class="pill">⌕ Hybrid search</span><span class="pill">↗ Source-grounded answers</span><span class="pill">◉ Local Ollama</span>
    </div>
    """,
    unsafe_allow_html=True,
)

index = st.session_state.index
if index is None:
    st.info("Upload a PDF in the left sidebar, or enable the sample PDF option if `documents/sample.pdf` exists.")
    st.markdown("### What you can do here")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='panel'><div class='section-label'>01 · Retrieve</div><h3>Find relevant passages</h3><div class='small-muted'>BM25 keyword matching and vector similarity work together to find evidence.</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><div class='section-label'>02 · Verify</div><h3>Inspect source chunks</h3><div class='small-muted'>See file names, page numbers, ranking scores and the original passage.</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='panel'><div class='section-label'>03 · Ask</div><h3>Chat with your PDFs</h3><div class='small-muted'>Ask follow-up questions and evaluate retrieval with a multi-question test bench.</div></div>", unsafe_allow_html=True)
    st.stop()

# Overview metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("PDF documents", len(index.document_summary))
m2.metric("Indexed chunks", len(index.chunks))
m3.metric("Embedding size", index.embedding_dimension)
m4.metric("Reranker", "Enabled" if use_reranker else "Disabled")

chat_tab, lab_tab, docs_tab = st.tabs(["✦  Chat workspace", "⌕  Retrieval quality lab", "▤  Documents & index"])

with chat_tab:
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        st.markdown("<div class='section-label'>DOCUMENT CHAT</div>", unsafe_allow_html=True)
        st.subheader("Ask a question")
        st.caption("Answers are generated from retrieved passages. Check the source cards before relying on a response.")
        if not st.session_state.messages:
            st.markdown("<div class='panel'><b>Try a starter question</b><br><span class='small-muted'>Pick one to populate the chat input.</span></div>", unsafe_allow_html=True)
            starter_cols = st.columns(2)
            starters = [
                "Summarize the main points in these documents.",
                "Which tools, languages, or technologies are mentioned?",
                "What evidence supports the key claims?",
                "What information is missing or unclear?",
            ]
            for i, starter in enumerate(starters):
                with starter_cols[i % 2]:
                    if st.button(starter, key=f"starter_{i}", use_container_width=True):
                        st.session_state.pending_question = starter
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander(f"View source evidence ({len(message['sources'])})", expanded=False):
                        for source in message["sources"]:
                            st.markdown(
                                f"<div class='source-card'><div class='source-meta'><b>{source['source_id']}</b> · {source['filename']} · Page {source['page']} · Retrieval score {source['score']:.3f}</div>{source['text_html']}</div>",
                                unsafe_allow_html=True,
                            )
        pending = st.session_state.pop("pending_question", "")
        question = st.chat_input("Ask something about your PDFs…")
        if pending and not question:
            question = pending
        if question:
            question = question.strip()
            if question:
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving evidence and asking the local model..."):
                        try:
                            results = index.search(question, top_k=top_k, use_reranker=use_reranker)
                            answer, sources = answer_with_ollama(
                                question=question,
                                results=results,
                                ollama_url=ollama_url,
                                model_name=ollama_model,
                            )
                            st.markdown(answer)
                            with st.expander(f"View source evidence ({len(sources)})", expanded=True):
                                for source in sources:
                                    st.markdown(
                                        f"<div class='source-card'><div class='source-meta'><b>{source['source_id']}</b> · {source['filename']} · Page {source['page']} · Retrieval score {source['score']:.3f}</div>{source['text_html']}</div>",
                                        unsafe_allow_html=True,
                                    )
                            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                            st.session_state.last_sources = sources
                            st.session_state.last_answer = answer
                            st.session_state.last_question = question
                        except Exception as exc:
                            error_text = f"I couldn't complete the request: {exc}"
                            st.error(error_text)
                            st.session_state.messages.append({"role": "assistant", "content": error_text, "sources": []})
    with right:
        st.markdown("<div class='section-label'>RETRIEVAL SNAPSHOT</div>", unsafe_allow_html=True)
        st.subheader("Current knowledge base")
        for info in index.document_summary:
            st.markdown(f"<div class='panel'><b>▤ {info['filename']}</b><br><span class='small-muted'>{info['pages']} pages · {info['chunks']} chunks</span></div>", unsafe_allow_html=True)
        st.markdown("#### Retrieval pipeline")
        for number, title, desc in [
            ("01", "Keyword search", "BM25 finds exact terms and phrases."),
            ("02", "Semantic search", "Embeddings find meaning-related passages."),
            ("03", "Hybrid fusion", "Normalized scores are combined."),
            ("04", "Reranking", "A cross-encoder reorders the best candidates."),
        ]:
            st.markdown(f"<div class='panel' style='margin-bottom:.45rem'><b style='color:#6558f5'>{number}</b> &nbsp; <b>{title}</b><br><span class='small-muted'>{desc}</span></div>", unsafe_allow_html=True)
        if st.session_state.last_sources:
            st.markdown("#### Last answer · sources")
            for src in st.session_state.last_sources:
                st.caption(f"{src['source_id']} · {src['filename']} · p.{src['page']} · {src['score']:.3f}")

with lab_tab:
    st.markdown("<div class='section-label'>RETRIEVAL EVALUATION</div>", unsafe_allow_html=True)
    st.subheader("Test retrieval with multiple questions")
    st.write("This lightweight test bench runs retrieval only, so you can inspect whether relevant passages appear before involving the LLM.")
    default_questions = "\n".join([
        "Summarize the main experience or project described in the documents.",
        "Which programming languages are mentioned?",
        "Which AI tools, frameworks, or libraries are mentioned?",
        "What certifications or training are listed?",
        "What evidence is provided for cybersecurity or blockchain experience?",
    ])
    questions_text = st.text_area("One test question per line", value=default_questions, height=150)
    eval_top_k = st.slider("Results per test question", min_value=1, max_value=5, value=3, key="eval_top_k")
    run_eval = st.button("Run retrieval evaluation", type="primary")
    if run_eval:
        questions = [line.strip() for line in questions_text.splitlines() if line.strip()]
        if not questions:
            st.warning("Enter at least one question.")
        else:
            evaluation_rows = []
            with st.spinner(f"Running retrieval for {len(questions)} questions..."):
                for q in questions:
                    try:
                        found = index.search(q, top_k=eval_top_k, use_reranker=use_reranker)
                        top = found[0] if found else None
                        evaluation_rows.append({
                            "Question": q,
                            "Top source": f"{top['filename']} · p.{top['page']}" if top else "No result",
                            "Top score": round(float(top["score"]), 4) if top else 0.0,
                            "Retrieved": len(found),
                            "Top passage": top["text"][:260] + ("…" if top and len(top["text"]) > 260 else "") if top else "",
                            "Results": found,
                        })
                    except Exception as exc:
                        evaluation_rows.append({"Question": q, "Top source": "Error", "Top score": 0.0, "Retrieved": 0, "Top passage": str(exc), "Results": []})
            st.session_state.eval_rows = evaluation_rows
    for row in st.session_state.get("eval_rows", []):
        with st.expander(f"{row['Question']}  ·  top score {row['Top score']:.3f}", expanded=False):
            st.caption(f"Top source: {row['Top source']} · Results returned: {row['Retrieved']}")
            st.write(row["Top passage"])
            for rank, source in enumerate(row["Results"], start=1):
                st.markdown(f"**{rank}. {source['source_id']} · {source['filename']} · p.{source['page']} · score {source['score']:.3f}**")
                st.write(source["text"])
    st.caption("Tip: record questions with known answers and manually mark whether the expected passage appears in the top-k results. Scores alone do not prove retrieval quality.")

with docs_tab:
    st.markdown("<div class='section-label'>KNOWLEDGE BASE</div>", unsafe_allow_html=True)
    st.subheader("Documents and indexed chunks")
    st.write("Uploaded PDF bytes are processed in this Streamlit session. This app does not upload them to a remote vector database.")
    for info in index.document_summary:
        with st.expander(f"{info['filename']} · {info['pages']} pages · {info['chunks']} chunks", expanded=False):
            st.write(f"**Extracted words:** {info['words']}")
            if info.get("warning"):
                st.warning(info["warning"])
    st.markdown("#### Preview indexed chunks")
    preview_count = min(12, len(index.chunks))
    for chunk in index.chunks[:preview_count]:
        with st.expander(f"{chunk['source_id']} · {chunk['filename']} · page {chunk['page']}", expanded=False):
            st.write(chunk["text"])
    st.caption(f"Showing {preview_count} of {len(index.chunks)} indexed chunks. Rebuild the index after changing chunk settings.")

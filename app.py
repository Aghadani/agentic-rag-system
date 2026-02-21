import streamlit as st
import os
from datetime import datetime
from typing import List, TypedDict

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="Dani Tech · RAG", page_icon="✦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400&display=swap');

* { box-sizing: border-box; }

html, body, .stApp {
    background: #faf8f5 !important;
    font-family: 'DM Sans', sans-serif !important;
}

#MainMenu, footer { visibility: hidden; }

/* Keep the header transparent and clean */
header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}

/* Sidebar collapse/expand tab (shown when sidebar is hidden) */
[data-testid="stSidebarCollapsedControl"] {
    background: #ffffff !important;
    border: 1px solid #ede9e3 !important;
    border-radius: 0 12px 12px 0 !important;
    box-shadow: 4px 0 16px rgba(0,0,0,0.06) !important;
    width: 1.5rem !important;
    padding: 1rem 0.2rem !important;
    overflow: hidden !important;
    /* suppress the raw "keyboard_double_arrow" text */
    font-size: 0 !important;
    color: transparent !important;
}
[data-testid="stSidebarCollapsedControl"]:hover {
    border-color: #d4935a !important;
    box-shadow: 4px 0 20px rgba(212,147,90,0.15) !important;
}
[data-testid="stSidebarCollapsedControl"] svg {
    display: block !important;
    width: 1rem !important;
    height: 1rem !important;
    color: #9e917e !important;
    fill: #9e917e !important;
}
/* hide the material-icon text ligature specifically */
[data-testid="stSidebarCollapsedControl"] span {
    display: none !important;
}

button[kind="header"] {
    background: #ffffff !important;
    border: 1px solid #ede9e3 !important;
    border-radius: 10px !important;
    color: #9e917e !important;
}
button[kind="header"]:hover {
    border-color: #d4935a !important;
    color: #d4935a !important;
}

.block-container {
    padding: 0 2.5rem 4rem !important;
    max-width: 900px !important;
}

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 800px 500px at 10% 0%, rgba(255,200,130,0.18) 0%, transparent 70%),
        radial-gradient(ellipse 600px 400px at 90% 100%, rgba(180,160,255,0.12) 0%, transparent 70%),
        radial-gradient(ellipse 500px 300px at 50% 50%, rgba(255,220,180,0.06) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #ede9e3 !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebar"] > div { padding: 2rem 1.4rem !important; }

/* Always keep sidebar toggle buttons visible */
[data-testid="stSidebar"] button {
    opacity: 1 !important;
    visibility: visible !important;
    display: flex !important;
}
/* The chevron/arrow collapse button inside the sidebar */
[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] {
    background: #faf8f5 !important;
    border: 1px solid #ede9e3 !important;
    border-radius: 8px !important;
    color: #9e917e !important;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"]:hover {
    border-color: #d4935a !important;
    color: #d4935a !important;
}

[data-testid="stSidebar"] [data-testid="stImage"] img {
    border-radius: 50%;
    width: 96px !important;
    height: 96px !important;
    object-fit: cover;
    display: block;
    margin: 0 auto 0.8rem;
    box-shadow: 0 8px 30px rgba(220,150,80,0.25), 0 0 0 3px #fff, 0 0 0 5px rgba(220,150,80,0.2);
}

[data-testid="stSidebar"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: #9e917e !important;
    text-transform: uppercase;
    letter-spacing: 0.08em !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    color: #9e917e !important;
}

[data-testid="stSidebar"] .stTextInput input {
    background: #faf8f5 !important;
    border: 1.5px solid #ede9e3 !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #3d3530 !important;
    padding: 0.55rem 0.8rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #d4935a !important;
    box-shadow: 0 0 0 3px rgba(212,147,90,0.12) !important;
    outline: none !important;
}

.sidebar-brand { text-align: center; margin-bottom: 1.6rem; }
.sidebar-brand-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #3d3530;
    letter-spacing: 0.06em;
}
.sidebar-brand-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65rem;
    color: #b0a090;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 2px;
}

[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid #ede9e3 !important;
    margin: 1.2rem 0 !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, #fffcf8 0%, #fdf5ec 100%) !important;
    border: 1.5px dashed #e0c9a8 !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #d4935a !important;
    box-shadow: 0 4px 20px rgba(212,147,90,0.1) !important;
}
[data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    color: #b0a090 !important;
}
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #d4935a, #c47a3e) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 1rem !important;
    box-shadow: 0 3px 10px rgba(212,147,90,0.3) !important;
}

/* BUTTONS */
.stButton > button {
    background: #ffffff !important;
    color: #6b5e50 !important;
    border: 1.5px solid #ede9e3 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 1.1rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}
.stButton > button:hover {
    border-color: #d4935a !important;
    color: #d4935a !important;
    box-shadow: 0 4px 16px rgba(212,147,90,0.15) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stCheckbox"] span {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important;
    color: #9e917e !important;
}

/* HERO */
.hero { padding: 3.5rem 0 2rem; position: relative; }
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(212,147,90,0.1);
    border: 1px solid rgba(212,147,90,0.25);
    border-radius: 999px;
    padding: 5px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #c47a3e;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-dot {
    width: 6px; height: 6px;
    background: #d4935a;
    border-radius: 50%;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.3; } }

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #2a2220;
    line-height: 1.1;
    margin-bottom: 0.75rem;
    letter-spacing: -0.02em;
}
.hero-title em {
    font-style: italic;
    background: linear-gradient(135deg, #d4935a 0%, #c05fa0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #9e917e;
    line-height: 1.7;
    max-width: 520px;
}

/* STAT CARDS */
.cards-row { display: flex; gap: 12px; margin: 2rem 0 2.5rem; }
.card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #ede9e3;
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, transform 0.2s;
}
.card:hover { box-shadow: 0 8px 30px rgba(0,0,0,0.08); transform: translateY(-2px); }
.card-icon { font-size: 1.3rem; margin-bottom: 0.5rem; }
.card-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.6rem;
    font-weight: 500;
    color: #c4b5a5;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 2px;
}
.card-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #3d3530;
}

/* PIPELINE */
.pipeline-row {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 0 0 2.5rem;
    flex-wrap: wrap;
}
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #fff;
    border: 1px solid #ede9e3;
    border-radius: 999px;
    padding: 6px 14px 6px 10px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    color: #9e917e;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.pipeline-step .step-num {
    width: 20px; height: 20px;
    background: linear-gradient(135deg, #d4935a, #c05fa0);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.58rem;
    font-weight: 700;
    color: #fff;
    flex-shrink: 0;
}
.pipeline-arrow {
    width: 24px; height: 1px;
    background: #ede9e3;
    position: relative;
    flex-shrink: 0;
}
.pipeline-arrow::after {
    content: '';
    position: absolute;
    right: -1px; top: -3px;
    border: 4px solid transparent;
    border-left: 6px solid #ede9e3;
}

/* CHAT SECTION LABEL */
.chat-section-label {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
}
.chat-section-line { flex: 1; height: 1px; background: linear-gradient(90deg, #ede9e3, transparent); }
.chat-section-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #c4b5a5;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    white-space: nowrap;
}

/* CHAT MESSAGES */
[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid #ede9e3 !important;
    border-radius: 18px !important;
    padding: 1.1rem 1.4rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04) !important;
    transition: box-shadow 0.2s !important;
}
[data-testid="stChatMessage"]:hover {
    box-shadow: 0 6px 24px rgba(0,0,0,0.07) !important;
}
[data-testid="stChatMessage"] p {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    color: #3d3530 !important;
    line-height: 1.75 !important;
}
[data-testid="stChatMessage"] li {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    color: #3d3530 !important;
}
[data-testid="stChatMessage"] code {
    background: #faf4ee !important;
    color: #c47a3e !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 5px !important;
    padding: 1px 6px !important;
}
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    font-family: 'Playfair Display', serif !important;
    color: #2a2220 !important;
}

/* CHAT INPUT */
[data-testid="stChatInput"] { padding: 0 !important; }
[data-testid="stChatInput"] > div {
    background: #ffffff !important;
    border: 1.5px solid #ede9e3 !important;
    border-radius: 18px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.07) !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
    overflow: hidden !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: #d4935a !important;
    box-shadow: 0 4px 32px rgba(212,147,90,0.18) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    color: #3d3530 !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #c4b5a5 !important; }
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #d4935a 0%, #c05fa0 100%) !important;
    border-radius: 12px !important;
    border: none !important;
    margin: 6px !important;
    box-shadow: 0 3px 12px rgba(212,147,90,0.3) !important;
    transition: opacity 0.2s !important;
}
[data-testid="stChatInput"] button:hover { opacity: 0.88 !important; }
[data-testid="stChatInput"] button svg { fill: #fff !important; }

/* STATUS */
[data-testid="stStatusWidget"] {
    background: #fffcf9 !important;
    border: 1px solid #f0e6d8 !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 20px rgba(212,147,90,0.08) !important;
}
[data-testid="stStatusWidget"] p, [data-testid="stStatusWidget"] span {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #9e917e !important;
}

/* EXPANDER */
[data-testid="stExpander"] {
    background: #fdf8f3 !important;
    border: 1px solid #ede9e3 !important;
    border-radius: 12px !important;
    margin-top: 0.6rem !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #9e917e !important;
}
[data-testid="stExpander"] p, [data-testid="stExpander"] a {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #c47a3e !important;
}

/* ALERT */
[data-testid="stAlert"] {
    background: #fdf8f3 !important;
    border: 1px solid #f0d8bc !important;
    border-left: 4px solid #d4935a !important;
    border-radius: 12px !important;
    color: #9e917e !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.83rem !important;
}

.side-footer {
    position: fixed;
    bottom: 20px;
    left: 1rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65rem;
    color: #c4b5a5;
    letter-spacing: 0.06em;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #faf8f5; }
::-webkit-scrollbar-thumb { background: #e0cdb8; border-radius: 4px; }

/* ── FIX: Hide "keyboard_double_arrow" text from sidebar toggle ── */
[data-testid="stSidebarCollapsedControl"] span,
[data-testid="stSidebarCollapsedControl"] p,
[data-testid="collapsedControl"] span,
button[data-testid="stBaseButton-headerNoPadding"] span.css-1aehpvj,
[data-testid="stSidebar"] button span[class*="material"] {
    font-size: 0 !important;
    color: transparent !important;
    display: none !important;
}

/* Keep the actual SVG arrow icon visible */
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebar"] button svg {
    display: block !important;
    visibility: visible !important;
    width: 1.1rem !important;
    height: 1.1rem !important;
    color: #9e917e !important;
    fill: #9e917e !important;
}

/* Hide any raw text nodes that render as "keyboard_double_arrow_right/left" */
[data-testid="stSidebarCollapsedControl"] {
    overflow: hidden !important;
    font-size: 0 !important;
}
[data-testid="stSidebarCollapsedControl"] * {
    font-family: inherit !important;
}
/* Material icon font override — make the ligature invisible */
.material-symbols-rounded,
.material-icons {
    font-size: 1.2rem !important;
    color: #9e917e !important;
}
</style>
""", unsafe_allow_html=True)


# ── RESET ──────────────────────────────────────────────────────────────────────
def clear_retriever():
    if "retriever" in st.session_state:
        del st.session_state.retriever
    if "messages" in st.session_state:
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("Dani_Logo.png")
    except Exception:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='sidebar-brand'>
        <div class='sidebar-brand-name'>Dani Tech</div>
        <div class='sidebar-brand-sub'>Agentic Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    groq_key   = st.secrets.get("GROQ_API_KEY")   or st.text_input("Groq API Key",   type="password", placeholder="gsk_···")
    tavily_key = st.secrets.get("TAVILY_API_KEY") or st.text_input("Tavily API Key", type="password", placeholder="tvly-···")

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload PDF", type="pdf",
        on_change=clear_retriever,
        help="Drop your research paper here"
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("↺  Clear"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        show_graph = st.checkbox("Graph", value=False)

    st.markdown("<div class='side-footer'>✦ Built by Dani Tech</div>", unsafe_allow_html=True)


# ── HERO ───────────────────────────────────────────────────────────────────────
doc_ready = "retriever" in st.session_state
pill_text = "System Active · Document Loaded" if doc_ready else "System Active · Awaiting Document"

st.markdown(f"""
<div class='hero'>
    <div class='hero-eyebrow'>
        <div class='hero-dot'></div>
        {pill_text}
    </div>
    <div class='hero-title'>Self‑Reflecting<br><em>Agentic RAG</em></div>
    <div class='hero-desc'>
        Corrective RAG with hallucination grading — your documents meet real-time web intelligence,
        producing answers you can trust.
    </div>
</div>
""", unsafe_allow_html=True)


# ── STAT CARDS ─────────────────────────────────────────────────────────────────
q_count   = len([m for m in st.session_state.messages if m["role"] == "user"])
src_label = "PDF + Web" if doc_ready else "Web Only"

st.markdown(f"""
<div class='cards-row'>
    <div class='card'>
        <div class='card-icon'>🧠</div>
        <div class='card-label'>Model</div>
        <div class='card-value'>Llama 3.3</div>
    </div>
    <div class='card'>
        <div class='card-icon'>📐</div>
        <div class='card-label'>Pipeline</div>
        <div class='card-value'>CRAG</div>
    </div>
    <div class='card'>
        <div class='card-icon'>💬</div>
        <div class='card-label'>Queries</div>
        <div class='card-value'>{q_count}</div>
    </div>
    <div class='card'>
        <div class='card-icon'>🌐</div>
        <div class='card-label'>Source</div>
        <div class='card-value'>{src_label}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── PIPELINE STEPS ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='pipeline-row'>
    <div class='pipeline-step'><div class='step-num'>1</div>Retrieve</div>
    <div class='pipeline-arrow'></div>
    <div class='pipeline-step'><div class='step-num'>2</div>Grade</div>
    <div class='pipeline-arrow'></div>
    <div class='pipeline-step'><div class='step-num'>3</div>Web Search</div>
    <div class='pipeline-arrow'></div>
    <div class='pipeline-step'><div class='step-num'>4</div>Generate</div>
</div>
""", unsafe_allow_html=True)


# ── AGENT SETUP ────────────────────────────────────────────────────────────────
if groq_key and tavily_key:
    os.environ["GROQ_API_KEY"]   = groq_key
    os.environ["TAVILY_API_KEY"] = tavily_key
    llm        = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if uploaded_file and "retriever" not in st.session_state:
        with st.status("✦  Indexing your document…", expanded=True) as status:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100
            ).split_documents(loader.load())
            status.write(f"Creating embeddings for {len(chunks)} chunks…")
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=f"pdf_{int(datetime.now().timestamp())}"
            )
            st.session_state.retriever = vectorstore.as_retriever()
            status.update(label=f"✦  Ready — {uploaded_file.name}", state="complete")

    class GraphState(TypedDict):
        question:      str
        generation:    str
        documents:     List[str]
        links:         List[str]
        search_needed: str

    def retrieve(state):
        docs = st.session_state.retriever.invoke(state["question"]) if "retriever" in st.session_state else []
        return {"documents": [d.page_content for d in docs], "links": []}

    def grade_documents(state):
        if not state["documents"]:
            return {"search_needed": "yes"}
        score = llm.invoke(
            f"Is this context relevant to the question? Answer only yes or no.\n"
            f"Question: {state['question']}\nContext: {state['documents'][0]}"
        ).content.lower()
        return {"search_needed": "no" if "yes" in score else "yes"}

    def web_search(state):
        results = TavilySearchResults(k=3).invoke({"query": state["question"]})
        return {
            "documents": state["documents"] + [r["content"] for r in results],
            "links":     [r["url"] for r in results]
        }

    def generate(state):
        source = "PDF document" if state["search_needed"] == "no" else "web search"
        res = llm.invoke(
            f"Answer the question using the context below. Cite your source as: {source}.\n\n"
            f"Context: {state['documents']}\n\nQuestion: {state['question']}"
        ).content
        return {"generation": res}

    wf = StateGraph(GraphState)
    wf.add_node("retrieve",   retrieve)
    wf.add_node("grade",      grade_documents)
    wf.add_node("web_search", web_search)
    wf.add_node("generate",   generate)
    wf.add_edge(START, "retrieve")
    wf.add_edge("retrieve", "grade")
    wf.add_conditional_edges("grade",
        lambda x: "web" if x["search_needed"] == "yes" else "gen",
        {"web": "web_search", "gen": "generate"})
    wf.add_edge("web_search", "generate")
    wf.add_edge("generate",   END)
    agent = wf.compile()

    if show_graph:
        with st.sidebar:
            st.image(agent.get_graph().draw_mermaid_png())

    # ── CHAT ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='chat-section-label'>
        <div class='chat-section-text'>Conversation</div>
        <div class='chat-section-line'></div>
    </div>
    """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about your document…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            node_labels = {
                "retrieve":   "Retrieving relevant passages…",
                "grade":      "Grading document relevance…",
                "web_search": "Searching the web…",
                "generate":   "Composing your answer…",
            }
            with st.status("✦  Thinking…", expanded=True) as status:
                final_state = {}
                for step in agent.stream({"question": prompt}):
                    for node, output in step.items():
                        status.write(f"→  {node_labels.get(node, node)}")
                        final_state.update(output)
                status.update(label="✦  Done", state="complete")

            st.markdown(final_state["generation"])

            if final_state.get("links"):
                with st.expander("✦  Web Sources"):
                    for link in final_state["links"]:
                        st.write(f"↗  {link}")

        st.session_state.messages.append(
            {"role": "assistant", "content": final_state["generation"]}
        )

else:
    st.markdown("""
    <div style='
        background: #fff;
        border: 1px solid #ede9e3;
        border-left: 4px solid #d4935a;
        border-radius: 16px;
        padding: 2rem 2.2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    '>
        <div style='font-family:"Playfair Display",serif;font-size:1.15rem;color:#3d3530;margin-bottom:6px'>
            Welcome to Dani Tech RAG
        </div>
        <div style='font-family:"DM Sans",sans-serif;font-size:0.83rem;color:#b0a090;line-height:1.65'>
            Enter your <strong style="color:#d4935a">Groq</strong> and
            <strong style="color:#d4935a">Tavily</strong> API keys in the sidebar
            to activate the agent. Optionally upload a PDF to enable document-grounded answers.
        </div>
    </div>
    """, unsafe_allow_html=True)

import streamlit as st
import os
from datetime import datetime
from typing import List, TypedDict

# LangChain / LangGraph Imports
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. --- PAGE CONFIG ---
st.set_page_config(page_title="Dani Tech · Agentic RAG", page_icon="⬡", layout="wide")

# 2. --- CUSTOM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --bg-void:    #07080d;
    --bg-card:    #0d0f1a;
    --bg-raised:  #131626;
    --border:     rgba(255,255,255,0.07);
    --accent-1:   #00e5ff;
    --accent-2:   #7b5ea7;
    --accent-glow:rgba(0,229,255,0.18);
    --text-pri:   #eef2ff;
    --text-sec:   #8892b0;
    --text-dim:   #4a5270;
    --radius:     14px;
    --font-head:  'Syne', sans-serif;
    --font-mono:  'Space Mono', monospace;
}

/* ── GLOBAL RESET ── */
html, body, .stApp {
    background-color: var(--bg-void) !important;
    color: var(--text-pri) !important;
    font-family: var(--font-head) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--accent-2); border-radius: 4px; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

[data-testid="stSidebar"] [data-testid="stImage"] img {
    border-radius: 50%;
    width: 110px !important;
    height: 110px !important;
    object-fit: cover;
    margin: 0 auto 0.5rem;
    display: block;
    border: 2px solid var(--accent-1);
    box-shadow: 0 0 24px var(--accent-glow);
}

/* Sidebar labels & text */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: var(--text-sec) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
}

[data-testid="stSidebar"] .stTextInput input {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    color: var(--accent-1) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 12px var(--accent-glow) !important;
}

/* ── BRAND NAME ── */
.brand-name {
    text-align: center;
    font-family: var(--font-head) !important;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    color: var(--accent-1) !important;
    text-transform: uppercase;
    margin-bottom: 0.15rem;
}
.brand-tagline {
    text-align: center;
    font-family: var(--font-mono) !important;
    font-size: 0.58rem;
    color: var(--text-dim);
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

/* ── DIVIDER ── */
hr {
    border: none;
    border-top: 1px solid var(--border) !important;
    margin: 1rem 0 !important;
}

/* ── MAIN HEADER ── */
.hero-header {
    padding: 2.5rem 0 1rem;
    position: relative;
}
.hero-title {
    font-family: var(--font-head) !important;
    font-size: 2.6rem;
    font-weight: 800;
    color: var(--text-pri) !important;
    line-height: 1.1;
    margin-bottom: 0.4rem;
    letter-spacing: -0.02em;
}
.hero-title span {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem;
    color: var(--text-dim);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,229,255,0.07);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 999px;
    padding: 4px 14px;
    font-family: var(--font-mono) !important;
    font-size: 0.65rem;
    color: var(--accent-1);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.8rem;
}
.pulse-dot {
    width: 6px; height: 6px;
    background: var(--accent-1);
    border-radius: 50%;
    animation: pulse 1.8s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 0.75rem !important;
    padding: 1rem 1.2rem !important;
    transition: border-color 0.2s;
}
[data-testid="stChatMessage"]:hover {
    border-color: rgba(0,229,255,0.15) !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: var(--text-pri) !important;
    font-family: var(--font-head) !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
}
[data-testid="stChatMessage"] code {
    background: var(--bg-raised) !important;
    color: var(--accent-1) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    border-radius: 4px;
    padding: 1px 5px;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 20px var(--accent-glow) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-pri) !important;
    font-family: var(--font-head) !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-dim) !important;
}

/* ── SEND BUTTON ── */
[data-testid="stChatInput"] button {
    background: var(--accent-1) !important;
    border-radius: 8px !important;
    border: none !important;
}
[data-testid="stChatInput"] button svg {
    fill: var(--bg-void) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: var(--bg-raised) !important;
    color: var(--text-sec) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: var(--accent-1) !important;
    color: var(--accent-1) !important;
    box-shadow: 0 0 12px var(--accent-glow) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--bg-raised) !important;
    border: 1px dashed rgba(0,229,255,0.2) !important;
    border-radius: var(--radius) !important;
    padding: 0.8rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-1) !important;
}
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span {
    color: var(--text-dim) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(0,229,255,0.08) !important;
    border: 1px solid rgba(0,229,255,0.2) !important;
    border-radius: 6px !important;
    color: var(--accent-1) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
}

/* ── STATUS / SPINNER ── */
[data-testid="stStatusWidget"] {
    background: var(--bg-card) !important;
    border: 1px solid rgba(0,229,255,0.15) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-sec) !important;
}
[data-testid="stStatusWidget"] p,
[data-testid="stStatusWidget"] span {
    color: var(--text-sec) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-top: 0.5rem !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] p {
    color: var(--text-sec) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
}
[data-testid="stExpander"] a {
    color: var(--accent-1) !important;
    font-size: 0.72rem !important;
    font-family: var(--font-mono) !important;
}

/* ── CHECKBOX ── */
[data-testid="stCheckbox"] span {
    color: var(--text-sec) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
}

/* ── WARNING / INFO ── */
[data-testid="stAlert"] {
    background: rgba(123, 94, 167, 0.12) !important;
    border: 1px solid rgba(123, 94, 167, 0.3) !important;
    border-radius: 10px !important;
    color: var(--text-sec) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}

/* ── FOOTER ── */
.side-footer {
    position: fixed;
    bottom: 18px;
    left: 18px;
    font-family: var(--font-mono) !important;
    font-size: 0.62rem;
    color: var(--text-dim);
    letter-spacing: 0.08em;
}

/* ── STAT CARDS ── */
.stat-row {
    display: flex;
    gap: 12px;
    margin-bottom: 1.8rem;
}
.stat-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
}
.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 4px;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--accent-1);
}

/* ── NODE BADGE ── */
.node-badge {
    display: inline-block;
    background: rgba(0,229,255,0.07);
    border: 1px solid rgba(0,229,255,0.18);
    border-radius: 6px;
    padding: 2px 8px;
    font-family: var(--font-mono) !important;
    font-size: 0.65rem;
    color: var(--accent-1);
    margin-right: 4px;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# 3. --- RESET CALLBACK ---
def clear_retriever():
    if "retriever" in st.session_state:
        del st.session_state.retriever
    if "messages" in st.session_state:
        st.session_state.messages = []

# 4. --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. --- SIDEBAR UI ---
with st.sidebar:
    # Try loading logo, fallback gracefully
    try:
        st.image("Dani_Logo.png")
    except Exception:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown("<div class='brand-name'>DANI TECH</div>", unsafe_allow_html=True)
    st.markdown("<div class='brand-tagline'>Intelligent Document Systems</div>", unsafe_allow_html=True)

    # API Keys
    groq_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    tavily_key = st.secrets.get("TAVILY_API_KEY") or st.text_input("Tavily API Key", type="password", placeholder="tvly-...")

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Research Paper",
        type="pdf",
        on_change=clear_retriever,
        help="PDF documents only"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⟳  Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        show_graph = st.checkbox("Graph", value=False)

    if show_graph:
        pass  # graph drawn below after app compiled

    st.markdown("<div class='side-footer'>Built by DANI TECH ⬡</div>", unsafe_allow_html=True)


# 6. --- MAIN HEADER ---
st.markdown("""
<div class='hero-header'>
    <div class='hero-title'>Agentic <span>Self‑Reflecting</span> RAG</div>
    <div class='hero-sub'>Corrective RAG · Hallucination Grading · Web Augmentation</div>
</div>
""", unsafe_allow_html=True)

# Status pill
doc_loaded = "retriever" in st.session_state
pill_text  = "Document Indexed · Ready" if doc_loaded else "Awaiting Document"
st.markdown(f"""
<div class='status-pill'>
    <div class='pulse-dot'></div>
    {pill_text}
</div>
""", unsafe_allow_html=True)

# Stat cards
msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
st.markdown(f"""
<div class='stat-row'>
    <div class='stat-card'>
        <div class='stat-label'>Pipeline</div>
        <div class='stat-value'>CRAG</div>
    </div>
    <div class='stat-card'>
        <div class='stat-label'>Model</div>
        <div class='stat-value' style='font-size:0.9rem;margin-top:4px;color:#8892b0'>Llama 3.3 · 70B</div>
    </div>
    <div class='stat-card'>
        <div class='stat-label'>Queries</div>
        <div class='stat-value'>{msg_count}</div>
    </div>
    <div class='stat-card'>
        <div class='stat-label'>Source</div>
        <div class='stat-value' style='font-size:0.85rem;margin-top:4px'>{"PDF + Web" if doc_loaded else "Web Only"}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# 7. --- AGENT LOGIC ---
if groq_key and tavily_key:
    os.environ["GROQ_API_KEY"]    = groq_key
    os.environ["TAVILY_API_KEY"]  = tavily_key
    llm        = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Index PDF
    if uploaded_file and "retriever" not in st.session_state:
        with st.status("⬡  Indexing Document...", expanded=True) as status:
            status.write("Parsing PDF pages...")
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100
            ).split_documents(loader.load())
            status.write(f"Creating vector store from {len(chunks)} chunks...")
            collection_name = f"pdf_{int(datetime.now().timestamp())}"
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=collection_name
            )
            st.session_state.retriever = vectorstore.as_retriever()
            status.update(label=f"✓  Indexed: {uploaded_file.name}", state="complete")

    # --- LANGGRAPH SETUP ---
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
            f"Is context relevant to: {state['question']}? Answer yes/no.\nContext: {state['documents'][0]}"
        ).content.lower()
        return {"search_needed": "no" if "yes" in score else "yes"}

    def web_search(state):
        search_tool = TavilySearchResults(k=3)
        results     = search_tool.invoke({"query": state["question"]})
        return {
            "documents": state["documents"] + [r["content"] for r in results],
            "links":     [r["url"] for r in results]
        }

    def generate(state):
        source = "PDF" if state["search_needed"] == "no" else "Web"
        res    = llm.invoke(
            f"Context: {state['documents']}\nQuestion: {state['question']}\nCite source as: {source}"
        ).content
        return {"generation": res}

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve",   retrieve)
    workflow.add_node("grade",      grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate",   generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        lambda x: "web" if x["search_needed"] == "yes" else "gen",
        {"web": "web_search", "gen": "generate"}
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate",   END)
    app_graph = workflow.compile()

    if show_graph:
        with st.sidebar:
            st.image(app_graph.get_graph().draw_mermaid_png())

    # Pipeline nodes reference
    st.markdown("""
    <div style='margin-bottom:1.5rem'>
        <span class='node-badge'>⬡ Retrieve</span>
        <span class='node-badge'>⬡ Grade</span>
        <span class='node-badge'>⬡ Web Search</span>
        <span class='node-badge'>⬡ Generate</span>
    </div>
    """, unsafe_allow_html=True)

    # --- CHAT INTERFACE ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about your document…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("⬡  Agent Reasoning…", expanded=True) as status:
                final_state = {}
                node_labels = {
                    "retrieve":   "⬡ Retrieve · Searching document chunks",
                    "grade":      "⬡ Grade   · Evaluating relevance",
                    "web_search": "⬡ Search  · Querying the web",
                    "generate":   "⬡ Generate· Synthesizing answer",
                }
                for step in app_graph.stream({"question": prompt}):
                    for node, output in step.items():
                        label = node_labels.get(node, f"⬡ {node.replace('_', ' ').title()}")
                        status.write(label)
                        final_state.update(output)
                status.update(label="✓  Done", state="complete")

            st.markdown(final_state["generation"])

            if final_state.get("links"):
                with st.expander("⬡  Web References"):
                    for link in final_state["links"]:
                        st.write(f"→  {link}")

            st.session_state.messages.append(
                {"role": "assistant", "content": final_state["generation"]}
            )

else:
    st.markdown("""
    <div style='
        background: rgba(123,94,167,0.08);
        border: 1px solid rgba(123,94,167,0.25);
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
    '>
        <div style='font-family:"Syne",sans-serif;font-size:1.1rem;color:#8892b0;margin-bottom:0.5rem'>
            API Keys Required
        </div>
        <div style='font-family:"Space Mono",monospace;font-size:0.7rem;color:#4a5270'>
            Enter your Groq and Tavily keys in the sidebar to activate the agent
        </div>
    </div>
    """, unsafe_allow_html=True)

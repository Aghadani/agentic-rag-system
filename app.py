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
st.set_page_config(page_title="Dani Tech Agentic RAG", page_icon="🤖", layout="wide")

# 2. --- BRANDING & VISIBILITY CSS ---
st.markdown(
    """
    <style>
    /* Force main text visibility */
    .main .block-container h1, .main .block-container h2, .main .block-container h3, 
    .main .block-container p, .main .block-container span, .stMarkdown {
        color: #FFFFFF !important;
    }
    /* Circular Logo Style */
    [data-testid="stSidebar"] [data-testid="stImage"] img {
        border-radius: 50%;
        width: 180px !important;
        height: 180px !important;
        object-fit: cover;
        margin: auto;
        display: block;
        border: 3px solid #00d4ff;
    }
    .brand-name { text-align: center; font-size: 24px; font-weight: bold; color: #00d4ff !important; margin-top: 10px; font-family: 'Courier New', monospace; }
    .side-footer { position: fixed; bottom: 20px; left: 20px; font-size: 12px; color: #888; }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. --- RESET CALLBACK ---
def clear_retriever():
    """Wipes the old PDF data from session state so the new one can load."""
    if "retriever" in st.session_state:
        del st.session_state.retriever
    if "messages" in st.session_state:
        st.session_state.messages = []

# 4. --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []

# 5. --- SIDEBAR UI ---
with st.sidebar:
    st.image("Dani_Logo.png")
    st.markdown("<div class='brand-name'>DANI TECH</div>", unsafe_allow_html=True)
    st.divider()

    # API Keys
    groq_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Groq API Key", type="password")
    tavily_key = st.secrets.get("TAVILY_API_KEY") or st.text_input("Tavily API Key", type="password")
    
    st.divider()
    
    # File Uploader with Callback
    uploaded_file = st.file_uploader(
        "Upload Research Paper (PDF)", 
        type="pdf", 
        on_change=clear_retriever  # Trigger reset on new file
    )

    if st.button("Clear All Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    show_graph = st.checkbox("Show Agent Graph")
    st.markdown("<div class='side-footer'>Built by <b>Dani Tech</b> 🛠️</div>", unsafe_allow_html=True)

# 6. --- APP TITLES ---
st.title("🧠 Agentic Self-Reflecting RAG")
st.subheader("Corrective RAG (CRAG) with Hallucination Grading")

# 7. --- AGENT LOGIC ---
if groq_key and tavily_key:
    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["TAVILY_API_KEY"] = tavily_key
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Indexer runs only if retriever is missing
    if uploaded_file and "retriever" not in st.session_state:
        with st.status("🚀 Dani Tech: Analyzing New Document...", expanded=True) as status:
            with open("temp.pdf", "wb") as f: f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(loader.load())
            
            # Use unique collection name based on timestamp to prevent mixing data
            collection_name = f"pdf_{int(datetime.now().timestamp())}"
            vectorstore = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings,
                collection_name=collection_name
            )
            st.session_state.retriever = vectorstore.as_retriever()
            status.update(label=f"✅ Loaded: {uploaded_file.name}", state="complete")

    # --- LANGGRAPH SETUP ---
    class GraphState(TypedDict):
        question: str
        generation: str
        documents: List[str]
        links: List[str]
        search_needed: str

    def retrieve(state):
        docs = st.session_state.retriever.invoke(state["question"]) if "retriever" in st.session_state else []
        return {"documents": [d.page_content for d in docs], "links": []}

    def grade_documents(state):
        if not state["documents"]: return {"search_needed": "yes"}
        score = llm.invoke(f"Is context relevant to: {state['question']}? Answer yes/no.\nContext: {state['documents'][0]}").content.lower()
        return {"search_needed": "no" if "yes" in score else "yes"}

    def web_search(state):
        search_tool = TavilySearchResults(k=3)
        results = search_tool.invoke({"query": state["question"]})
        return {
            "documents": state["documents"] + [r["content"] for r in results],
            "links": [r["url"] for r in results]
        }

    def generate(state):
        source = "PDF" if state["search_needed"] == "no" else "Web"
        res = llm.invoke(f"Context: {state['documents']}\nQuestion: {state['question']}\nCite source as: {source}").content
        return {"generation": res}

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", lambda x: "web" if x["search_needed"] == "yes" else "gen", {"web": "web_search", "gen": "generate"})
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    app = workflow.compile()

    if show_graph: st.sidebar.image(app.get_graph().draw_mermaid_png())

    # --- CHAT INTERFACE ---
    for msg in st.session_state.messages: 
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.status("Dani Tech Agent Reasoning...", expanded=True) as status:
                final_state = {}
                for step in app.stream({"question": prompt}):
                    for node, output in step.items():
                        status.write(f"✔️ {node.replace('_', ' ').title()}")
                        final_state.update(output)
            
            st.markdown(final_state["generation"])
            if final_state.get("links"):
                with st.expander("🔗 Web References"):
                    for link in final_state["links"]: st.write(f"- {link}")
            
            st.session_state.messages.append({"role": "assistant", "content": final_state["generation"]})
else:
    st.warning("Please provide API Keys in the sidebar to begin.")

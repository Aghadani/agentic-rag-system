import streamlit as st
import os
from datetime import datetime
from typing import List, TypedDict, Literal

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
    /* Force main text and headers to be visible white */
    .main .block-container h1, .main .block-container h2, .main .block-container h3, .main .block-container p, .main .block-container span {
        color: #FFFFFF !important;
    }

    /* Sidebar Image Branding */
    [data-testid="stSidebar"] [data-testid="stImage"] img {
        border-radius: 50%;
        width: 180px !important;
        height: 180px !important;
        object-fit: cover;
        margin: auto;
        display: block;
        border: 3px solid #00d4ff;
        box-shadow: 0px 4px 15px rgba(0, 212, 255, 0.3);
    }

    /* Sidebar Brand Text */
    .brand-name { 
        text-align: center; 
        font-size: 24px; 
        font-weight: bold; 
        color: #00d4ff !important; 
        margin-top: 10px; 
        font-family: 'Courier New', monospace; 
    }
    
    .side-footer { position: fixed; bottom: 20px; left: 20px; font-size: 12px; color: #888; }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "upload_log" not in st.session_state: st.session_state.upload_log = []

# 4. --- SIDEBAR UI ---
with st.sidebar:
    st.image("Dani_Logo.png")

    # Auth
    groq_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Groq API Key", type="password")
    tavily_key = st.secrets.get("TAVILY_API_KEY") or st.text_input("Tavily API Key", type="password")
    
    st.divider()
    
    # Upload & Log
    uploaded_file = st.file_uploader("Upload Documents (PDF)", type="pdf")
    if uploaded_file:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if not any(log['id'] == file_id for log in st.session_state.upload_log):
            st.session_state.upload_log.append({
                "id": file_id,
                "name": uploaded_file.name,
                "time": datetime.now().strftime("%H:%M:%S")
            })

    with st.expander("📊 Document Upload Log"):
        for item in st.session_state.upload_log:
            st.caption(f"✅ {item['name']} (at {item['time']})")

    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    show_graph = st.checkbox("Show Agent Graph")
    st.markdown("<div class='side-footer'>Built by <b>Dani Tech</b> 🛠️</div>", unsafe_allow_html=True)

# 5. --- MAIN UI TITLES ---
# Moved outside of any condition to ensure they are always rendered first
st.title("🧠 Agentic Self-Reflecting RAG")
st.subheader("Corrective RAG (CRAG) with Hallucination Grading")

# 6. --- LOGIC ---
if groq_key and tavily_key:
    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["TAVILY_API_KEY"] = tavily_key
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if uploaded_file and "retriever" not in st.session_state:
        with st.status("🚀 Dani Tech Agent Indexing...", expanded=True) as status:
            with open("temp.pdf", "wb") as f: f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100).split_documents(loader.load())
            st.session_state.retriever = Chroma.from_documents(documents=chunks, embedding=embeddings).as_retriever()
            status.update(label="✅ Ready for Queries!", state="complete")

    # (Agent State/Nodes/Graph logic remains same as your previous version...)
    class GraphState(TypedDict):
        question: str
        generation: str
        documents: List[str]
        links: List[str]
        search_needed: str
        retry_count: int

    def retrieve(state):
        docs = st.session_state.retriever.invoke(state["question"]) if "retriever" in st.session_state else []
        return {"documents": [d.page_content for d in docs], "links": []}

    def grade_documents(state):
        if not state["documents"]: return {"search_needed": "yes"}
        score = llm.invoke(f"Is relevant? {state['documents'][0]}. Query: {state['question']}. Answer yes/no.").content.lower()
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
        res = llm.invoke(f"Context: {state['documents']}\nQuestion: {state['question']}\nCite as {source}.").content
        return {"generation": res, "retry_count": state.get("retry_count", 0) + 1}

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve); workflow.add_node("grade", grade_documents)
    workflow.add_node("web_search", web_search); workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve"); workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", lambda x: "web" if x["search_needed"] == "yes" else "gen", {"web": "web_search", "gen": "generate"})
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    app = workflow.compile()

    if show_graph: st.sidebar.image(app.get_graph().draw_mermaid_png())

    # --- CHAT ---
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])

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
                with st.expander("🔗 Web References (Literature Support)"):
                    for link in final_state["links"]: st.write(f"- {link}")
            
            st.session_state.messages.append({"role": "assistant", "content": final_state["generation"]})
else:
    st.warning("Please provide API Keys in the sidebar to start.")




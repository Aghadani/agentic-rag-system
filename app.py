import streamlit as st
import os
from typing import List, TypedDict, Literal

# LangChain / LangGraph Imports
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- PAGE CONFIG ---
st.set_page_config(page_title="Agentic RAG Explorer", page_icon="🤖", layout="wide")
# --- BRANDING & CUSTOM CSS ---
def add_branding():
    st.markdown(
        """
        <style>
        /* Make the sidebar image large and circular */
        [data-testid="stSidebar"] [data-testid="stImage"] img {
            border-radius: 50%;
            width: 180px !important;  /* Increased size */
            height: 180px !important; /* Perfect circle */
            object-fit: cover;
            margin-left: auto;
            margin-right: auto;
            display: block;
            border: 3px solid #00d4ff; /* Tech-blue border */
            box-shadow: 0px 4px 15px rgba(0, 212, 255, 0.3);
        }
        
        /* Centered Brand Name */
        .brand-name {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            margin-top: 10px;
            font-family: 'Courier New', Courier, monospace;
        }
        
        .brand-tagline {
            text-align: center;
            font-size: 14px;
            color: #888;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:
        # 1. Display the Logo
        # Ensure 'Dani_Logo.png' is in your GitHub root folder
        st.image("Dani_Logo.png")
        
        # 2. Display Brand Text
        st.markdown("<div class='brand-name'>DANI TECH</div>", unsafe_allow_html=True)
        st.markdown("<div class='brand-tagline'>Built by Dani Tech</div>", unsafe_allow_html=True)
        st.divider()

# Call the branding function
add_branding()

# --- MAIN UI ---
st.title("🧠 Agentic Self-Reflecting RAG")
st.subheader("Corrective RAG (CRAG) with Hallucination Grading")

# --- SAFE SECRETS LOADER ---
groq_key = None
tavily_key = None

try:
    if "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]
    if "TAVILY_API_KEY" in st.secrets:
        tavily_key = st.secrets["TAVILY_API_KEY"]
except Exception:
    pass

# Fallback to Sidebar if Secrets aren't set
if not groq_key or not tavily_key:
    with st.sidebar:
        st.header("🔑 Authentication")
        if not groq_key:
            groq_key = st.text_input("Groq API Key", type="password")
        if not tavily_key:
            tavily_key = st.text_input("Tavily API Key", type="password")

# --- SIDEBAR: KNOWLEDGE BASE ---
with st.sidebar:
    st.divider()
    st.header("📄 Knowledge Base")
    st.markdown(
        """
        <style>
        .side-footer {
            position: fixed;
            bottom: 20px;
            left: 20px;
            font-size: 12px;
            color: #888;
        }
        </style>
        <div class="side-footer">
            Built by <b>Dani Tech</b> 🛠️
        </div>
        """,
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("Upload a PDF (e.g., Tesla Impact Report)", type="pdf")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
    
    st.divider()
    show_graph = st.checkbox("Show Agent Architecture")

# --- INITIALIZE LLM & MODELS ---
if groq_key and tavily_key:
    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["TAVILY_API_KEY"] = tavily_key
    
    # Using Llama 3 70B for high-quality reasoning
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # --- VECTOR STORE LOGIC ---
    if uploaded_file and "retriever" not in st.session_state:
        with st.status("Indexing Document...", expanded=True):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
            st.session_state.retriever = vectorstore.as_retriever()
            st.success("Indexing Complete!")

    # --- AGENT STATE & UTILS ---
    class GraphState(TypedDict):
        question: str
        generation: str
        documents: List[str]
        search_needed: str
        retry_count: int

    def get_binary_score(text: str) -> str:
        """Standardizes LLM output to yes/no."""
        cleaned = text.lower().strip()
        return "yes" if "yes" in cleaned else "no"

    # --- NODE FUNCTIONS ---
    def retrieve(state):
        question = state["question"]
        if "retriever" in st.session_state:
            docs = st.session_state.retriever.invoke(question)
            return {"documents": [d.page_content for d in docs], "question": question}
        return {"documents": [], "question": question}

    def grade_documents(state):
        question = state["question"]
        docs = state["documents"]
        if not docs: return {"search_needed": "yes", "documents": docs}
        
        grade_prompt = f"Is this document relevant to the question: '{question}'? Context: {docs[0]}. Answer only 'yes' or 'no'."
        response = llm.invoke(grade_prompt)
        score = get_binary_score(response.content)
        return {"search_needed": "no" if score == "yes" else "yes", "documents": docs}

    def web_search(state):
        search_tool = TavilySearchResults(k=3)
        results = search_tool.invoke({"query": state["question"]})
        web_content = "\n".join([r["content"] for r in results])
        return {"documents": state["documents"] + [web_content]}

    def generate(state):
        source = "PDF Document" if state["search_needed"] == "no" else "Live Web Research"
        prompt = f"Facts: {state['documents']} \nQuestion: {state['question']} \nAnswer accurately. End with 'SOURCES: {source}'"
        response = llm.invoke(prompt)
        # Ensure retry_count exists
        current_retry = state.get("retry_count", 0)
        return {"generation": response.content, "retry_count": current_retry + 1}

    # --- GRAPH SETUP ---
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", lambda x: "search" if x["search_needed"] == "yes" else "generate", {"search": "web_search", "generate": "generate"})
    workflow.add_edge("web_search", "generate")
    
    def check_hallucination(state):
        hallucination_prompt = f"Is this answer supported by these facts? Facts: {state['documents']} \nAnswer: {state['generation']}. Answer only 'yes' or 'no'."
        response = llm.invoke(hallucination_prompt)
        score = get_binary_score(response.content)
        if score == "yes" or state.get("retry_count", 0) > 1:
            return "finish"
        return "retry"

    workflow.add_conditional_edges("generate", check_hallucination, {"finish": END, "retry": "generate"})
    app = workflow.compile()

    if show_graph:
        with st.sidebar:
            st.image(app.get_graph().draw_mermaid_png())

    # --- CHAT INTERFACE ---
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.status("Agent Reasoning...", expanded=True) as status:
                final_state = {}
                for step in app.stream({"question": prompt}):
                    for node, output in step.items():
                        status.write(f"✅ Completed: **{node}**")
                        final_state.update(output)
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            st.markdown(final_state["generation"])
            st.session_state.messages.append({"role": "assistant", "content": final_state["generation"]})
else:
    st.warning("Please provide API Keys in the sidebar or App Secrets to begin.")




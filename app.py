import streamlit as st
import os
from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field

# LangChain / LangGraph Imports
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- PAGE CONFIG ---
st.set_page_config(page_title="Agentic RAG Explorer", page_icon="🧠", layout="wide")

st.title("🧠 Agentic Self-Reflecting RAG")
st.subheader("Corrective RAG with Hallucination Grading")

# --- SAFE SECRETS LOADER ---
groq_key = None
tavily_key = None

try:
    # Look for Streamlit Cloud Secrets
    if "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]
    if "TAVILY_API_KEY" in st.secrets:
        tavily_key = st.secrets["TAVILY_API_KEY"]
except Exception:
    pass

# Fallback to Sidebar if Secrets aren't set up yet
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
    uploaded_file = st.file_uploader("Upload a PDF for the Agent to study", type="pdf")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# --- INITIALIZE LLM & MODELS ---
if groq_key and tavily_key:
    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["TAVILY_API_KEY"] = tavily_key
    
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # --- VECTOR STORE LOGIC ---
    if uploaded_file and "retriever" not in st.session_state:
        with st.status("Indexing Document..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
            st.session_state.retriever = vectorstore.as_retriever()
            st.success("Indexing Complete!")

    # --- AGENT DEFINITIONS ---
    class GraphState(TypedDict):
        question: str
        generation: str
        documents: List[str]
        search_needed: str
        retry_count: int

    class GradeRelevance(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    class GradeHallucination(BaseModel):
        binary_score: str = Field(description="Grounded in facts 'yes' or 'no'")

    relevance_grader = llm.with_structured_output(GradeRelevance)
    hallucination_grader = llm.with_structured_output(GradeHallucination)

    # --- NODES ---
    def retrieve(state):
        question = state["question"]
        if "retriever" in st.session_state:
            docs = st.session_state.retriever.invoke(question)
            return {"documents": [d.page_content for d in docs], "question": question, "retry_count": 0}
        return {"documents": [], "question": question, "retry_count": 0}

    def grade_documents(state):
        question = state["question"]
        docs = state["documents"]
        if not docs: return {"search_needed": "yes", "documents": docs}
        score = relevance_grader.invoke(f"Question: {question} \nDoc: {docs[0]}")
        return {"search_needed": "no" if score.binary_score.lower() == "yes" else "yes", "documents": docs}

    def web_search(state):
        search_tool = TavilySearchResults(k=3)
        results = search_tool.invoke({"query": state["question"]})
        web_content = "\n".join([r["content"] for r in results])
        return {"documents": state["documents"] + [web_content]}

    def generate(state):
        source = "PDF" if state["search_needed"] == "no" else "Web Search"
        prompt = f"Facts: {state['documents']} \nQuestion: {state['question']} \nEnd with Source: {source}"
        response = llm.invoke(prompt)
        return {"generation": response.content, "retry_count": state.get("retry_count", 0) + 1}

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
    workflow.add_conditional_edges("generate", lambda x: "finish" if hallucination_grader.invoke(f"Facts: {x['documents']} \nAnswer: {x['generation']}").binary_score.lower() == "yes" or x["retry_count"] > 1 else "retry", {"finish": END, "retry": "generate"})
    app = workflow.compile()

    # --- CHAT ---
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about the PDF or the world..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            with st.status("Agent processing...") as status:
                final_state = app.invoke({"question": prompt})
                status.update(label="Complete!", state="complete")
            st.markdown(final_state["generation"])
            st.session_state.messages.append({"role": "assistant", "content": final_state["generation"]})
else:
    st.warning("Please provide API Keys in the sidebar or App Secrets.")
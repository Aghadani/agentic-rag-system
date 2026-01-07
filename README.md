# 🧠 Agentic Self-Reflecting RAG (Corrective RAG)

An advanced Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **Groq (Llama 3)**, and **Tavily Search**. This project implements a "Self-Corrective" loop that mimics human metacognition to ensure high-accuracy, grounded responses.



## 🚀 Key Features
- **Self-Correction Loop**: Uses a Critic agent to grade retrieved documents for relevance.
- **Dynamic Web Fallback**: If private data (PDF) is insufficient or irrelevant, the agent automatically triggers a Researcher node to query the live web via Tavily API.
- **Hallucination Grader**: A final verification step that compares the generated answer against the source facts to prevent AI "hallucinations."
- **High-Speed Inference**: Powered by Groq's Llama 3 70B for near-instant agentic reasoning.

## 🛠️ Tech Stack
- **Orchestration**: LangGraph (State Machines)
- **LLM**: Groq (Llama-3-70b-8192)
- **Vector Database**: ChromaDB (Running locally)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Search**: Tavily AI
- **UI**: Streamlit

## 🧠 Logic Flow (The Agentic Path)
1. **Retrieve**: Pulls context from the uploaded PDF.
2. **Grade**: The **Critic Agent** analyzes the snippets.
   - *Relevant?* Proceed to generation.
   - *Irrelevant?* Route to **Web Search**.
3. **Generate**: Synthesizes the final answer.
4. **Reflect**: The **Hallucination Grader** checks the answer.
   - *Grounded?* Deliver to user.
   - *Hallucinated?* Re-run the generation loop.



## 📦 Installation & Deployment

### Local Setup
1. Clone the repo: `git clone https://github.com/your-username/agentic-rag-system.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

### Deployment (Streamlit Cloud)
This repo is configured for **Streamlit Community Cloud**. 
1. Push this code to GitHub.
2. Connect your repo to [Streamlit Share](https://share.streamlit.io/).
3. Add your `GROQ_API_KEY` and `TAVILY_API_KEY` to the app secrets or enter them in the UI sidebar.

## 🎓 Research Context
This implementation is based on the **CRAG (Corrective RAG)** paper and explores **Metacognition in LLMs**. It moves beyond "Naive RAG" by adding a decision-making layer that evaluates its own knowledge retrieval.
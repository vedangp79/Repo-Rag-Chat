# 🧠 Repo RAG Chat — Demo

**Live Demo:** https://huggingface.co/spaces/vedangp79/repo-rag-chat
 
**Source Code:** https://github.com/vedangp79/Repo-Rag-Chat  

A lightweight **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about a codebase using **LangChain**, **Chroma**, and **OpenAI**.  
It retrieves relevant code/doc chunks and generates grounded answers with file-path citations.  

> Built by **Vedang Patel** · [LinkedIn](https://www.linkedin.com/in/vedangp79) · [Contact](mailto:vedangp@umich.edu)

---

## ✨ Features
- **RAG over your repo:** splits → embeds → stores → retrieves → generates.
- **File-path citations:** view exact retrieved chunks for each answer.
- **Two UIs:**
  - 🟢 **Gradio** → Interactive demo (used in the live Space).
  - ⚪ **Streamlit** → Original local version with repo tree and file viewer.
- **LLM re-ranking (optional):** improves retrieval relevance.
- **Instant-load demo index:** ships with a small prebuilt `./chroma_index/`.

---

## 🧱 Tech Stack
| Layer | Tool |
|-------|------|
| UI | Gradio (demo) / Streamlit (local) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | `gpt-4o-mini` |
| Vector Store | ChromaDB |
| Framework | LangChain |
| Hosting | Hugging Face Spaces |

---

## 🧪 Try These Questions
- “Which embedding model and vector store are used?”
- “Explain the `summarize_text` function in `docs/sample.py`.”
- “How does the RAG pipeline work end-to-end?”

---

## ▶️ Run Locally
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Set your key (or load from .env)
export OPENAI_API_KEY="sk-....."

# (Re)build the demo index if needed
python ingest.py --source ./docs --persist ./chroma_index --chunk_size 1000 --chunk_overlap 150

# Launch the Gradio demo
python app.py

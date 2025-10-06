# Repo RAG Chat — Demo Corpus

This folder contains a **tiny knowledge base** used by the demo app to showcase **Retrieval-Augmented Generation (RAG)**.

## What the demo does
- Indexes this folder with **ChromaDB** using **OpenAI embeddings** (`text-embedding-3-small`).
- On each query, it **retrieves top-k relevant chunks** and feeds them into an LLM (`gpt-4o-mini`) via a grounded prompt.
- The UI displays **citations**: file paths and chunk IDs.

## Key pieces to know
- `overview.md` — high-level architecture, data flow, and why RAG matters.
- `sample.py` — small example code with docstrings the assistant can explain and reference.

## Examples of great demo questions
Try:
- “What is RAG and how does this repo implement it?”
- “Which embedding model and vector store are used here?”
- “Explain the `summarize_text` function in `sample.py`.”
- “Describe the end-to-end flow from query to answer with citations.”

## Tech summary (for retrieval)
- **RAG**: retrieval + prompt construction + LLM completion
- **Vector store**: ChromaDB (persistent index under `./chroma_index`)
- **Embeddings**: `text-embedding-3-small`
- **LLM**: `gpt-4o-mini`
- **Top-K**: default 4 (configurable)

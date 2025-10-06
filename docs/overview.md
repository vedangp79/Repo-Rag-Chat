# RAG Architecture Overview

This demo implements a **Retrieval-Augmented Generation** workflow designed for small repositories:

1. **Ingestion**
   - Files in `./docs` are chunked (e.g., 1000 tokens, 150 overlap).
   - Each chunk is embedded with `text-embedding-3-small`.
   - Chunks + embeddings are persisted in **ChromaDB** at `./chroma_index`.

2. **Retrieval**
   - At query time, it performs **similarity_search** against the vector store.
   - We over-retrieve (e.g., 3×k) and optionally **re-rank** with a lightweight LLM scoring pass.

3. **Grounded Generation**
   - The top-k chunks are composed into a **context block** with file paths and chunk IDs.
   - The prompt instructs the model to **prefer retrieved content**, cite sources, and say “I don’t know” if evidence is insufficient.
   - The LLM (`gpt-4o-mini`) returns a concise answer plus citations.

## Why RAG?
- **Accuracy**: reduces hallucination by grounding answers in your own data.
- **Freshness**: swap in updated files without retraining a model.
- **Transparency**: citations show *where* information came from.

## Demo Tips
- Ask about the stack: embeddings, vector store, models, chunking.
- Ask to summarize `sample.py` and expect references like `docs/sample.py`.
- Look for **[n] path (chunk m)** style citations in responses.

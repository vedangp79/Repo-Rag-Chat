import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite"] = pysqlite3
except Exception:
    pass

# app.py — Gradio UI for Hugging Face Spaces demo

import os
import gradio as gr

from openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Reuse your helpers if needed (repo tree/read)
from utils_repo import repo_tree, safe_read

CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# Paths: keep them small for demo
PERSIST_DIR = os.environ.get("PERSIST_DIR", "./chroma_index")
REPO_ROOT   = os.environ.get("REPO_ROOT", "..")
TOP_K       = int(os.environ.get("TOP_K", "4"))

client = None
vectordb = None

def init_clients():
    global client, vectordb
    if client is None:
        client = OpenAI()  # uses OPENAI_API_KEY
    if vectordb is None:
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        # This expects that PERSIST_DIR exists (committed) or was built by ingest on startup
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# Optional: a tiny “ingest if missing” hook. Comment out if you committed chroma_index/.
def ensure_index():
    if not os.path.isdir(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        # If you want to auto-build, import and call your ingest here.
        # from ingest import main as ingest_main
        # ingest_main([...])
        # For speed on Spaces, prefer committing a tiny prebuilt index instead.
        pass

SYS_PROMPT = """You are a helpful repo explainer.
Use the retrieved context to answer.
- If the context is mostly code, summarize the main classes, functions, constants, and tables.
- Prefer retrieved content; infer reasonable high-level summaries from structure.
- If nothing is relevant, say "I don't know".
- Cite file paths (and chunk ids if present) from retrieved chunks when possible.
"""

def rerank_with_llm(query, docs, top_n=4):
    scored = []
    for d in docs:
        snippet = d.page_content[:1000]
        p = f"""Score how relevant the snippet is to the query on a 0-10 scale.
Query: {query}
Snippet:
\"\"\"{snippet}\"\"\"
Return ONLY a number."""
        try:
            r = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict scorer. Reply with ONLY a number between 0 and 10."},
                    {"role": "user", "content": p},
                ],
                temperature=0,
            )
            text = r.choices[0].message.content.strip()
            num = ""
            for ch in text:
                if ch.isdigit() or ch in ".-":
                    num += ch
                elif num:
                    break
            score = float(num) if num else 0.0
        except Exception:
            score = 0.0
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_n]]

def answer_once(query, top_k=TOP_K, enable_rerank=False):
    init_clients()
    ensure_index()
    # Over-retrieve then (optionally) rerank
    raw_hits = vectordb.similarity_search(query, k=max(top_k * 3, top_k))
    hits = rerank_with_llm(query, raw_hits, top_n=min(top_k, 6)) if (enable_rerank and raw_hits) else raw_hits[:top_k]

    context = "\n\n".join(
        f"[{i+1}] (path: {d.metadata.get('path','?')}, chunk: {d.metadata.get('chunk_id', '?')})\n{d.page_content}"
        for i, d in enumerate(hits)
    )
    citations = [f"[{i+1}] {d.metadata.get('path','?')} (chunk {d.metadata.get('chunk_id','?')})"
                 for i, d in enumerate(hits)]
    citations_text = "\n".join(citations) if citations else "—"

    prompt = f"""{SYS_PROMPT}

Context:
{context}

Question: {query}

Return a concise answer. If you reference files, include their paths inline (e.g., file.java).
If context is insufficient, say "I don't know"."""
    try:
        r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        )
        answer = r.choices[0].message.content

    except Exception as e:
        answer = f"Error from model: {e}"

    return answer, citations_text, hits

def chat_fn(user_msg, history, top_k, enable_rerank):
    if not user_msg.strip():
        return history, ""
    answer, cites, hits = answer_once(user_msg, top_k=top_k, enable_rerank=enable_rerank)
    history = (history or []) + [(user_msg, answer + "\n\n**Sources**\n" + cites)]
    return history, ""

with gr.Blocks() as demo:
    gr.HTML('''
    <div class="topbar">
    <h1>🧠 Repo RAG Chat — <span style="color:#2563eb;">Demo</span></h1>
    <p style="margin:0;color:#555;font-size:15px;">by <b>Vedang Patel</b></p>
    </div>
    ''')
    with gr.Row():
        top_k = gr.Slider(1, 10, value=TOP_K, step=1, label="Top-K Retrieved")
        enable_rerank = gr.Checkbox(False, label="LLM re-rank retrieved chunks")
    chat = gr.Chatbot(height=420, label="Chat")
    msg = gr.Textbox(placeholder="Ask about this repo…", label="Your question")

    def respond(user_msg, chat_state, k, rr):
        return chat_fn(user_msg, chat_state, int(k), bool(rr))

    msg.submit(respond, [msg, chat, top_k, enable_rerank], [chat, msg])

    gr.HTML(
    """
    <hr style='margin-top:25px;margin-bottom:15px'>
    <div style='text-align:center; font-size:16px; color:#444;'>
        <b>Repo RAG Chat — Demo</b><br>
        Built by <b>Vedang Patel</b><br><br>

        <a href='mailto:vedangp@umich.edu'
        style='text-decoration:none; background-color:#2563eb; color:white; padding:8px 14px; border-radius:6px; font-weight:600; margin-right:10px;'>
        📧 Contact
        </a>

        <a href='https://www.linkedin.com/in/vedangpatel7' target='_blank'
        style='text-decoration:none; background-color:#0a66c2; color:white; padding:8px 14px; border-radius:6px; font-weight:600;'>
        🔗 LinkedIn
        </a>

        <br><br>
        <a href='https://github.com/vedangp79/Repo-Rag-Chat' target='_blank'
        style='text-decoration:none; color:#2563eb; font-size:15px;'>View Source on GitHub</a>
    </div>
    """,
        elem_id="footer",
    )


if __name__ == "__main__":
    demo.launch()

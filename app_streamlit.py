import os
import streamlit as st
from openai import OpenAI

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from utils_repo import repo_tree, safe_read

# Repo selector via YAML (repos.yaml needed)
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

# Models
CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

st.set_page_config(page_title="Repo RAG Chat", page_icon="🧠", layout="wide")
st.title("Repo RAG Chat + Tools")

# ---- Sidebar: repo selector (repos.yaml)
default_persist = "./chroma_index"
default_repo_root = ".."

cfg = {}
if yaml and os.path.exists("repos.yaml"):
    try:
        cfg = yaml.safe_load(open("repos.yaml"))
    except Exception:
        cfg = {}

repos = cfg.get("repos", {}) if isinstance(cfg, dict) else {}

choice = None
if repos:
    choice = st.sidebar.selectbox("Choose configured repo", list(repos.keys()))
    if choice:
        default_repo_root = repos[choice].get("path", default_repo_root)
        default_persist = repos[choice].get("index", default_persist)

persist_dir = st.sidebar.text_input("Chroma index directory", default_persist)
repo_root = st.sidebar.text_input("Repo root (for tools)", default_repo_root)

k = st.sidebar.slider("Top-K retrieved chunks", 1, 10, 4)
enable_rerank = st.sidebar.checkbox("LLM re-rank retrieved chunks", value=False)

sys_prompt = st.sidebar.text_area(
    "System / Grounding Instruction",
    """You are a helpful repo explainer.
Use the retrieved context to answer.
- If the context is mostly code, summarize it: explain the main classes, functions, constants, and database tables you see.
- Prefer retrieved content, but you may infer reasonable high-level summaries from code structure.
- If nothing at all is relevant, say "I don't know".
- Cite file paths (and chunk ids if given) from the retrieved chunks when possible."""
)

# Initialize client
try:
    client = OpenAI()  # this uses OPENAI_API_KEY from env
except Exception as e:
    st.error(f"OpenAI init error: {e}")

# Load vector store (must be ingested already)
try:
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    st.sidebar.success("Vector index loaded.")
except Exception as e:
    st.sidebar.error(f"Chroma load failed: {e}")
    st.stop()

# ---- LLM-based rerank
def rerank_with_llm(client, query, docs, top_n=4):
    """Tiny 2nd-pass scoring using the chat model. Good enough for demos."""
    scored = []
    for d in docs:
        snippet = d.page_content[:1000]
        p = f"""Score how relevant the snippet is to the query on a 0-10 scale.
Query: {query}
Snippet:
\"\"\"{snippet}\"\"\"
Return ONLY a number."""
        try:
            r = client.responses.create(model=CHAT_MODEL, input=p)
            text = r.output_text.strip()
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

# --- Tabs: Chat | Tools ---
tab_chat, tab_tools = st.tabs(["💬 Chat (RAG)", "🛠 Tools (Agent-like)"])

# --- Chat Tab ---
with tab_chat:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, content in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(content)

    user_q = st.chat_input("Ask about this repo…")
    if user_q:
        with st.chat_message("user"):
            st.markdown(user_q)

        # Over-retrieve for broader coverage, then rerank
        raw_hits = vectordb.similarity_search(user_q, k=max(k * 3, k))
        if enable_rerank and raw_hits:
            hits = rerank_with_llm(client, user_q, raw_hits, top_n=min(k, 6))
        else:
            hits = raw_hits[:k]

        # Build context with file path + chunk id (if present)
        context = "\n\n".join(
            f"[{i+1}] (path: {d.metadata.get('path','?')}, chunk: {d.metadata.get('chunk_id', '?')})\n{d.page_content}"
            for i, d in enumerate(hits)
        )

        # Citations footer to append to the answer rendering
        citations = [f"[{i+1}] {d.metadata.get('path','?')} (chunk {d.metadata.get('chunk_id', '?')})"
                     for i, d in enumerate(hits)]
        citations_text = "\n".join(citations) if citations else "—"

        prompt = f"""{sys_prompt}

Context:
{context}

Question: {user_q}

Return a concise answer. If you reference files, include their paths inline (e.g., file.java).
If context is insufficient, say "I don't know"."""
        try:
            resp = client.responses.create(model=CHAT_MODEL, input=prompt)
            answer = resp.output_text
        except Exception as e:
            answer = f"Error from model: {e}"

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.markdown("**Sources**")
            st.code(citations_text)

            with st.expander("Show retrieved chunks"):
                for i, d in enumerate(hits, 1):
                    p = d.metadata.get("path", "?")
                    cid = d.metadata.get("chunk_id", "?")
                    st.markdown(f"**[{i}] {p} (chunk {cid})**")
                    st.code(d.page_content[:1200])

        st.session_state.chat.append(("user", user_q))
        st.session_state.chat.append(("assistant", answer))

with tab_tools:
    st.subheader("Repo Tree")
    if st.button("Show repo tree"):
        tree = repo_tree(repo_root)
        st.code(tree)

    st.subheader("Read File")
    rel_path = st.text_input("Relative path (e.g., src/app.py)")
    if st.button("Read file"):
        content = safe_read(repo_root, rel_path)
        st.code(content)

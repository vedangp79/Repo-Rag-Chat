import sys
try:
    import pysqlite3  
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite"] = pysqlite3
except Exception:
    pass
# --------------------------------------------------------

import argparse
from pathlib import Path
import os

from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from utils_repo import list_files

# Models
EMBED_MODEL = "text-embedding-3-small"

ALLOWED_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".sql",
    ".md", ".txt", ".json", ".yml", ".yaml", ".html", ".css",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".go", ".rs",
}

def is_good_path(rel_path: str) -> bool:
    # Skip AppleDouble, hidden, or paths with bad suffixes
    name = os.path.basename(rel_path)
    if name.startswith("._") or name.startswith("."):
        return False
    _, ext = os.path.splitext(rel_path)
    return ext.lower() in ALLOWED_EXTS

def ingest_folder(source_dir: str, persist_dir: str = "./chroma_index",
                  chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Walk source_dir, create chunks with metadata (path, chunk_id), embed, and persist to Chroma.
    """
    docs = []
    for rel_path, text in list_files(source_dir):
        if not is_good_path(rel_path):
            continue
        docs.append(Document(page_content=text, metadata={"path": rel_path}))

    if not docs:
        print("No documents found. Check extensions/paths.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # Split and attach a chunk_id per original file for better traceability
    chunks = []
    for d in docs:
        sub_chunks = splitter.split_documents([d])
        for i, c in enumerate(sub_chunks):
            c.metadata = dict(c.metadata)  # ensure writable
            c.metadata["chunk_id"] = i
            chunks.append(c)

    print(f"Embedding {len(chunks)} chunks…")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    print(f"✅ Indexed {len(chunks)} chunks from {len(docs)} files into {persist_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to a repo/folder to ingest")
    parser.add_argument("--persist", default="./chroma_index", help="Chroma index dir")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=150)
    args = parser.parse_args()

    # Sanity check
    _ = OpenAI()

    Path(args.persist).mkdir(parents=True, exist_ok=True)
    ingest_folder(args.source, args.persist, args.chunk_size, args.chunk_overlap)

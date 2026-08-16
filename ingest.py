"""
ingest.py — load both Meridian PDFs, chunk them, embed with a local Ollama
model, and store the vectors in ChromaDB (persisted to disk).

Uses pdfplumber rather than pypdf: pdfplumber is layout-aware and keeps
table columns (scorecards, rate tables) intact far more reliably than
pypdf's raw text extraction, which tends to split numbers apart in dense
tables (e.g. turning "48,211" into "4 8 , 211").

Run directly to index everything currently sitting in data/:
    python ingest.py
"""

import os
from dotenv import load_dotenv
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "meridian_supply_chain"

# Ollama model used for embeddings, read from .env. Pull it once with:
#   ollama pull mxbai-embed-large
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")

# Chunking parameters (within the assignment's 800-1200 / 100-200 range).
# 1200/200 keeps most scorecards and rate tables intact inside one chunk -
# smaller sizes were cutting tables in half in the first assignment.
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


def load_pdfs(file_paths):
    """Load a list of PDF paths into LangChain Documents, one per page."""
    all_docs = []
    for path in file_paths:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                all_docs.append(
                    Document(
                        page_content=text,
                        metadata={"source_file": os.path.basename(path), "page": i},
                    )
                )
    return all_docs


def chunk_docs(docs, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Split page-level documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def get_vectorstore():
    """Return a Chroma vector store backed by a local Ollama embedding model.

    Because persist_directory points at a folder on disk, calling this again
    after restarting the app reconnects to the same persisted collection -
    that's what gives you the "survives a restart" requirement for free.

    Both PDFs go into this SAME collection (same collection_name), which is
    what makes cross-document retrieval possible - the assignment requires
    this, not two separate stores.
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def index_files(file_paths, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Full ingest pipeline: load -> chunk -> embed -> store.

    Returns a dict like {"files": 2, "chunks": 96} for the UI/API to show.
    """
    docs = load_pdfs(file_paths)
    chunks = chunk_docs(docs, chunk_size, chunk_overlap)
    vs = get_vectorstore()
    vs.add_documents(chunks)
    return {"files": len(file_paths), "chunks": len(chunks)}


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".pdf")
    ]
    if not paths:
        print(f"No PDFs found in {DATA_DIR}/. Put both Meridian PDFs there first.")
    else:
        # Guard against indexing the same files twice into the same
        # collection (easy mistake: running this AND uploading via the
        # Streamlit sidebar for the same files doubles every chunk and
        # confuses retrieval).
        vs = get_vectorstore()
        existing = vs._collection.count()
        if existing > 0:
            print(
                f"Warning: the collection already has {existing} chunks stored. "
                f"Re-running this will add duplicates of any file already indexed. "
                f"Run 'rmdir /s /q {PERSIST_DIR}' (Windows) first if you want a clean rebuild."
            )
        result = index_files(paths)
        print(f"{result['files']} files processed, {result['chunks']} chunks stored")

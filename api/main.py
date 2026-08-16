"""
api/main.py — optional FastAPI backend (bonus section of the assignment).

Run from the project root with:
    uvicorn api.main:app --reload

Then check the auto docs at http://localhost:8000/docs
"""

import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from ingest import index_files, DATA_DIR, COLLECTION_NAME, EMBED_MODEL, get_vectorstore
from rag import ask as rag_ask, CHAT_MODEL, TOP_K

app = FastAPI(title="Meridian Supply Chain RAG API")


class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = []
    for f in files:
        path = os.path.join(DATA_DIR, f.filename)
        with open(path, "wb") as out:
            shutil.copyfileobj(f.file, out)
        paths.append(path)
    result = index_files(paths)
    return result  # {"files": N, "chunks": N}


@app.post("/ask")
async def ask_endpoint(req: AskRequest):
    result = rag_ask(req.question, top_k=req.top_k)
    return result  # {"answer": "...", "sources": [{"file": ..., "page": ...}]}


@app.get("/stats")
async def stats():
    vs = get_vectorstore()
    count = vs._collection.count()
    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": count,
        "embedding_model": EMBED_MODEL,
        "llm_model": CHAT_MODEL,
    }
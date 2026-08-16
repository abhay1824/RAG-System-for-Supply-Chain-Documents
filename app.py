"""
app.py — Streamlit interface for the Meridian supply chain / procurement
policy RAG assistant.

Run with:
    streamlit run app.py

Make sure Ollama is running and you've pulled the models this project uses:
    ollama pull mxbai-embed-large
    ollama pull llama3.1
"""

import os
import streamlit as st
from ingest import index_files, DATA_DIR, get_vectorstore
from rag import ask, TOP_K

st.set_page_config(page_title="Meridian Supply Chain Assistant", layout="wide")
st.title("📦 Meridian Supply Chain & Procurement Policy Assistant")
st.caption("Runs fully locally with Ollama — no API key, no cloud calls.")

os.makedirs(DATA_DIR, exist_ok=True)

# ---- Sidebar: upload + index ----
with st.sidebar:
    st.header("1. Upload & Index")
    st.caption(
        "Upload the Q1 supply chain review AND the procurement policy "
        "handbook — cross-document questions need both indexed together."
    )
    uploaded = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    if st.button("Index documents", disabled=not uploaded):
        paths = []
        for f in uploaded:
            path = os.path.join(DATA_DIR, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            paths.append(path)

        with st.spinner("Chunking and embedding with Ollama..."):
            result = index_files(paths)

        st.success(f"{result['files']} files processed, {result['chunks']} chunks stored")
        st.caption(
            "Note: re-uploading the same file(s) again will add duplicate "
            "chunks. If you need to re-index, delete chroma_db/ first."
        )

    st.divider()
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        st.caption(f"Currently indexed: **{count} chunks** (persisted to `chroma_db/`)")
    except Exception:
        st.caption("No documents indexed yet.")

# ---- Main: ask ----
st.header("2. Ask a question")
question = st.text_input(
    "Your question",
    placeholder="e.g. Which supplier had the highest spend in Q1, and what was its on-time delivery percentage?",
)

if st.button("Ask", disabled=not question) and question:
    with st.spinner("Retrieving relevant chunks and generating an answer..."):
        try:
            result = ask(question, top_k=TOP_K)
        except Exception as e:
            st.error(
                "Something went wrong calling Ollama. Is it running, and have "
                "you pulled the models this app needs? "
                f"(mxbai-embed-large, llama3.1)\n\nDetails: {e}"
            )
            st.stop()

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    if result["sources"]:
        seen = set()
        for s in result["sources"]:
            key = (s["file"], s["page"])
            if key not in seen:
                st.markdown(f"- **{s['file']}**, page {s['page']}")
                seen.add(key)
    else:
        st.caption("No chunks were retrieved.")

"""
rag.py — retrieve the most relevant chunks (from either or both PDFs), build
a grounded prompt, and call a local Ollama chat model to answer.

Cross-document questions (the assignment's questions 5-9) need chunks from
BOTH the review and the policy handbook to show up in the same retrieval -
that only happens with a high enough top_k. Assignment 1 taught us local
embedding models often need a noticeably higher top_k than the 4-5 suggested
for GPT-4o + OpenAI embeddings, so this defaults higher and is easy to
override for testing.
"""

import os
from langchain_ollama import ChatOllama
from ingest import get_vectorstore

# Ollama model used for answering, read from .env. Pull it once with:
#   ollama pull llama3.1
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1")

# Assignment asks for temperature 0-0.2. Using 0 (not 0.1) because this app
# frequently needs precise numeric threshold comparisons (e.g. "is 88.1%
# above or below the 90% clause") - determinism matters more here than
# slightly more natural phrasing would.
TEMPERATURE = 0

# Higher than the assignment's suggested 4-5: cross-document questions need
# chunks from BOTH PDFs to appear together, and local embedding models are
# less precise than OpenAI's, so a wider net avoids missing half the answer.
# Tune this down if answers get noisy, or up if cross-document questions
# keep missing one of the two documents (check with debug=True).
TOP_K = 10

SYSTEM_PROMPT = """You are a procurement assistant for Meridian Components Pvt. Ltd.
Answer the question ONLY using the context below, which is extracted from two
internal documents: a quarterly supply chain review and a procurement policy
handbook. For every fact you state, name which document it came from (review
or handbook) and the page number. If the context does not contain the answer,
reply exactly: "This information is not available in the uploaded documents."
Do not use outside knowledge. Do not guess or estimate any number, clause, or
policy that is not explicitly present in the context.

Some questions require combining a number or fact from the review with a rule
or clause from the handbook - when both are present in the context, use both
and clearly state which document each part came from. Give a direct, final
answer. Do not narrate your search process or say a chunk "is not provided" -
if something genuinely is not in the context, use the exact refusal sentence
above and stop there.

When a clause specifies a numeric threshold (e.g. "below 90%", "above 500
parts per million"), explicitly work out whether the figure in the question
is above or below that threshold before stating your conclusion - do not
state whether a clause is triggered until you have done this comparison
correctly. Double-check the direction of the comparison (above vs below)
before answering."""


def format_context(docs):
    """Turn retrieved chunks into a labelled context block for the prompt."""
    parts = []
    for d in docs:
        src = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page", "?")
        parts.append(f"[Source: {src}, page {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def ask(question, top_k=TOP_K, debug=False):
    """Retrieve top_k chunks for the question, ask the local LLM, return
    {"answer": str, "sources": [{"file": str, "page": int|str}, ...]}.

    Pass debug=True to print exactly which chunks were retrieved, and from
    which document(s) - the fastest way to tell whether a cross-document
    question failed because retrieval missed one document entirely.
    """
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)

    if debug:
        files_seen = sorted(set(d.metadata.get("source_file", "unknown") for d in docs))
        print(f"\n--- Retrieved {len(docs)} chunks for: {question!r} ---")
        print(f"--- Documents represented: {files_seen} ---")
        for d in docs:
            src = d.metadata.get("source_file", "unknown")
            page = d.metadata.get("page", "?")
            preview = d.page_content[:150].replace("\n", " ")
            print(f"[{src}, page {page}] {preview}...")
        print("--- end retrieved chunks ---\n")

    context = format_context(docs)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    llm = ChatOllama(model=CHAT_MODEL, temperature=TEMPERATURE)
    response = llm.invoke(prompt)

    sources = [
        {"file": d.metadata.get("source_file", "unknown"), "page": d.metadata.get("page", "?")}
        for d in docs
    ]
    return {"answer": response.content, "sources": sources}


if __name__ == "__main__":
    # Quick manual test from the command line:
    #   python rag.py "Which supplier had the highest spend in Q1?"
    #   python rag.py --k 15 "your question"   (override top_k)
    import sys
    args = sys.argv[1:]
    k = TOP_K
    if args[:1] == ["--k"]:
        k = int(args[1])
        args = args[2:]
    q = " ".join(args) or "Which supplier had the highest spend in Q1?"
    result = ask(q, top_k=k, debug=True)
    print("ANSWER:\n", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f" - {s['file']}, page {s['page']}")
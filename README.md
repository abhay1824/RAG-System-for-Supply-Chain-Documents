# Meridian Supply Chain & Procurement Policy RAG (Local, via Ollama)

A RAG assistant that answers buyer questions by combining data from a
quarterly supply chain review with rules from a procurement policy handbook
— both indexed into a single ChromaDB collection so cross-document questions
can be answered in one shot. Runs **entirely locally** using
[Ollama](https://ollama.com) — no OpenAI key, no cloud calls.

## Setup

1. Install [Ollama](https://ollama.com/download) and make sure it's running.
2. Pull the two models this project uses:
   ```bash
   ollama pull mxbai-embed-large
   ollama pull llama3.1
   ```
3. Clone this repo and install dependencies:
   ```bash
   git clone <your-repo-url>
   cd supplychain-rag
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env         # Windows: copy .env.example .env
   ```
5. Put **both** provided PDFs in `data/`:
   - `Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf`
   - `Meridian_Procurement_Policy_Handbook_v4.2.pdf`

## Run

```bash
streamlit run app.py
```

Upload both PDFs (or index via `python ingest.py` if they're already in
`data/`), click **Index documents**, then ask questions.

**Important:** only index each file once — either via `python ingest.py`
or via the Streamlit uploader, not both. Indexing the same files twice
duplicates every chunk and hurts retrieval quality (learned this the hard
way on a previous assignment).

### Optional: FastAPI backend

```bash
uvicorn api.main:app --reload
```
Docs at http://localhost:8000/docs

## Chunk size & top_k choices

- **Chunk size 1200 / overlap 200** — keeps full scorecards and rate
  tables intact inside a single chunk rather than splitting them apart.
- **top_k = 10** (higher than the assignment's suggested 4-5) — cross-document
  questions need chunks from BOTH the review and the handbook to appear in
  the same retrieval. A local embedding model (`mxbai-embed-large`) is less
  precise than OpenAI's `text-embedding-3-small`, so a wider net was needed
  to reliably pull relevant chunks from both documents at once.

## Models used

| Component  | Model                  | Why |
|------------|--------------------------|-----|
| Embeddings | `mxbai-embed-large` (Ollama) | Free, local, better than smaller models at ranking dense/tabular text |
| Answering  | `llama3.1` (Ollama)     | Good instruction-following for grounded Q&A, runs locally |

## Screenshots

<add screenshots: upload/index, a single-document answer, a cross-document answer with both sources shown, and the trap question refused>

## Test questions & answers

<fill in your app's actual answers>

1. Which supplier had the highest spend in Q1, and what was its on-time delivery percentage?
2. How many line stoppages happened in Q1, what was the total downtime, and what caused them?
3. What is the approval authority for a purchase order worth ₹1.4 crore?
4. What are the four supplier classification categories, and what qualifies a supplier as Critical?
5. Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects per million in Q1. Which policy clauses does this trigger, and what exactly must the buyer do? *(cross-document)*
6. The microcontroller supplier is single-source. What does the sourcing policy require, and what is the company already doing about it? *(cross-document)*
7. Microcontrollers are imported with a 46-day lead time. Using the safety-stock policy, how many days of stock should be held for this part? *(cross-document)*
8. Trident Circuit Boards had a defect rate of 640 parts per million. What is the cost consequence under the policy? *(cross-document)*
9. Which suppliers would fall below the B rating band on on-time delivery alone, and what is the escalation path for them? *(cross-document)*
10. Trap question: "What is the annual salary of the Head of Procurement?" — expected: information not available.

## What didn't work well / honest notes

<e.g. which of questions 5-9 the app got wrong and why — check retrieved
chunks with `python rag.py --k <n> "question"` for any that fail, per the
assignment's own debugging hint>

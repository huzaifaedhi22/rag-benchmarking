# RAG Benchmark

Upload your documents and find out which retrieval method works best on your data.

## How it works

1. Upload one or more documents (PDF, TXT, Markdown, or code files)
2. The backend chunks the documents and generates 50 question-answer pairs using an LLM
3. Seven retrieval methods are benchmarked against those questions
4. Results are ranked by nDCG@10

## Retrieval methods

| Method | Description |
|---|---|
| BM25 | Classic keyword-based retrieval |
| Dense | Embedding-based semantic search |
| Hybrid | BM25 + Dense combined |
| ColBERT | Late-interaction dense retrieval |
| Dense + Reranker | Dense retrieval with a cross-encoder reranker |
| Dense + Reasoning Reranker | Dense retrieval with an LLM-based reranker |
| Query Rewriting + Hybrid | Rewrites the query before hybrid retrieval |

## Metrics

- **nDCG@10** — primary ranking metric
- **Recall@10** — fraction of relevant docs retrieved
- **MRR** — mean reciprocal rank
- **Latency** — end-to-end retrieval time

## Stack

- **Frontend** — Next.js, deployed on Vercel
- **Backend** — FastAPI, deployed on Railway
- **LLM** — Groq (question generation + reasoning reranker)
- **Embeddings** — HuggingFace

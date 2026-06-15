from dotenv import load_dotenv
load_dotenv()

from backend.chunker import chunk_files
from backend.golden_set import generate_golden_set
from backend.retrievers.bm25 import BM25Retriever
from backend.retrievers.dense import DenseRetriever
from backend.retrievers.hybrid import HybridRetriever
from backend.retrievers.colbert import ColBERTRetriever
from backend.retrievers.reranker import RerankerRetriever
from backend.retrievers.reasoning_reranker import ReasoningRerankerRetriever
from backend.retrievers.query_rewriting import QueryRewritingRetriever

files = [
    ("polars.txt", b"Polars uses lazy evaluation with the lazy() method. Call collect() to execute the query plan. The execution plan is optimized before running, which eliminates redundant operations and pushes filters as early as possible. This makes it significantly faster than eager evaluation for large datasets where intermediate results would otherwise consume too much memory."),
    ("pandas.txt", b"Pandas uses eager evaluation by default. Every operation executes immediately and returns a new DataFrame. While this makes debugging easier since you can inspect results at each step, it is slower for large datasets because intermediate DataFrames are fully materialized in memory before the next operation begins."),
    ("arrow.txt", b"Apache Arrow provides a columnar memory format that allows data to be shared between different systems without copying. It is used by Polars internally for efficient data processing. The columnar format means that operations on a single column can be performed without loading other columns into memory, which greatly improves cache efficiency."),
]

print("--- chunker ---")
chunks = chunk_files(files)
print(f"chunks produced: {len(chunks)}")

print("\n--- golden set ---")
golden = generate_golden_set(chunks, n=2)
print(f"questions generated: {len(golden)}")
for q in golden:
    print(f"  Q: {q['question']}")
    print(f"     -> {q['relevant_chunk_id']}")

print("\n--- BM25 ---")
bm25 = BM25Retriever()
bm25.index(chunks)
print(bm25.retrieve("lazy evaluation query plan", k=3))

print("\n--- Dense ---")
dense = DenseRetriever()
dense.index(chunks)
print(dense.retrieve("lazy evaluation query plan", k=3))

print("\n--- Hybrid ---")
hybrid = HybridRetriever()
hybrid.index(chunks)
print(hybrid.retrieve("lazy evaluation query plan", k=3))

print("\n--- ColBERT ---")
colbert = ColBERTRetriever()
colbert.index(chunks)
print(colbert.retrieve("lazy evaluation query plan", k=3))

print("\n--- Dense+Reranker ---")
reranker = RerankerRetriever()
reranker.index(chunks)
print(reranker.retrieve("lazy evaluation query plan", k=3))

print("\n--- Dense+ReasoningReranker ---")
reasoning = ReasoningRerankerRetriever()
reasoning.index(chunks)
print(reasoning.retrieve("lazy evaluation query plan", k=3))

print("\n--- QueryRewriting+Hybrid ---")
qr = QueryRewritingRetriever()
qr.index(chunks)
print(qr.retrieve("why is polars faster than pandas", k=3))

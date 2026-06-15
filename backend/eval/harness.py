import time
from ranx import Qrels, Run, evaluate

from ..retrievers.bm25 import BM25Retriever
from ..retrievers.dense import DenseRetriever
from ..retrievers.hybrid import HybridRetriever
from ..retrievers.colbert import ColBERTRetriever
from ..retrievers.reranker import RerankerRetriever
from ..retrievers.reasoning_reranker import ReasoningRerankerRetriever
from ..retrievers.query_rewriting import QueryRewritingRetriever

METRICS = ["ndcg@10", "recall@10", "mrr"]
K = 10


def _build_qrels(golden: list[dict]) -> Qrels:
    return Qrels({
        item["query_id"]: {item["relevant_chunk_id"]: 1}
        for item in golden
    })


def _run_retriever(retriever, chunks: list[dict], golden: list[dict]) -> tuple[Run, float]:
    retriever.index(chunks)

    run_dict: dict[str, dict[str, float]] = {}
    total_ms = 0.0

    for item in golden:
        t0 = time.perf_counter()
        results = retriever.retrieve(item["question"], k=K)
        total_ms += (time.perf_counter() - t0) * 1000

        run_dict[item["query_id"]] = {
            chunk_id: 1.0 / (rank + 1)
            for rank, chunk_id in enumerate(results)
        }

    return Run(run_dict), total_ms / len(golden)


def _make_retrievers():
    return [
        BM25Retriever(),
        DenseRetriever(),
        HybridRetriever(),
        ColBERTRetriever(),
        RerankerRetriever(),
        ReasoningRerankerRetriever(),
        QueryRewritingRetriever(),
    ]


def stream_benchmark(chunks: list[dict], golden: list[dict]):
    qrels = _build_qrels(golden)
    for retriever in _make_retrievers():
        print(f"[harness] Running {retriever.name}...")
        run, latency_ms = _run_retriever(retriever, chunks, golden)
        scores = evaluate(qrels, run, METRICS)
        yield {
            "method": retriever.name,
            "ndcg@10": round(scores["ndcg@10"], 4),
            "recall@10": round(scores["recall@10"], 4),
            "mrr": round(scores["mrr"], 4),
            "latency_ms": round(latency_ms, 1),
        }


def run_benchmark(chunks: list[dict], golden: list[dict]) -> list[dict]:
    results = list(stream_benchmark(chunks, golden))
    return sorted(results, key=lambda x: x["ndcg@10"], reverse=True)

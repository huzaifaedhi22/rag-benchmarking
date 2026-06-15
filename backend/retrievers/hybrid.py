from .base import BaseRetriever
from .bm25 import BM25Retriever
from .dense import DenseRetriever

RRF_K = 60


def _reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=lambda x: scores[x], reverse=True)[:k]


class HybridRetriever(BaseRetriever):
    def __init__(self):
        self._bm25 = BM25Retriever()
        self._dense = DenseRetriever()

    @property
    def name(self) -> str:
        return "Hybrid"

    def index(self, chunks: list[dict]) -> None:
        self._bm25.index(chunks)
        self._dense.index(chunks)

    def retrieve(self, query: str, k: int = 10) -> list[str]:
        bm25_results = self._bm25.retrieve(query, k=k * 2)
        dense_results = self._dense.retrieve(query, k=k * 2)
        return _reciprocal_rank_fusion([bm25_results, dense_results], k=k)

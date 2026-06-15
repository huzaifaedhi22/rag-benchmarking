from sentence_transformers import CrossEncoder
from .base import BaseRetriever
from .dense import DenseRetriever

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_K = 100


class RerankerRetriever(BaseRetriever):
    def __init__(self):
        self._dense = DenseRetriever()
        self._reranker = CrossEncoder(MODEL_NAME)

    @property
    def name(self) -> str:
        return "Dense+Reranker"

    def index(self, chunks: list[dict]) -> None:
        self._chunks = {c["chunk_id"]: c for c in chunks}
        self._dense.index(chunks)

    def retrieve(self, query: str, k: int = 10) -> list[str]:
        candidates = self._dense.retrieve(query, k=min(CANDIDATE_K, len(self._chunks)))
        pairs = [(query, self._chunks[cid]["text"]) for cid in candidates]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:k]]

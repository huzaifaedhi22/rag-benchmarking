import re
import bm25s
from .base import BaseRetriever


def _tokenize(text: str) -> list[str]:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace('_', ' ').replace('.', ' ')
    return text.lower().split()


class BM25Retriever(BaseRetriever):
    @property
    def name(self) -> str:
        return "BM25"

    def index(self, chunks: list[dict]) -> None:
        self._chunks = {c["chunk_id"]: c for c in chunks}
        corpus = [_tokenize(c["text"]) for c in chunks]
        self._retriever = bm25s.BM25()
        self._retriever.index(corpus)
        self._chunk_ids = [c["chunk_id"] for c in chunks]

    def retrieve(self, query: str, k: int = 10) -> list[str]:
        tokens = _tokenize(query)
        results, _ = self._retriever.retrieve([tokens], k=min(k, len(self._chunk_ids)))
        return [self._chunk_ids[i] for i in results[0]]

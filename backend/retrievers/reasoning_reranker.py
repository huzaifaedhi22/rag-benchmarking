import os
from groq import Groq
from .base import BaseRetriever
from .dense import DenseRetriever

MODEL_NAME = "qwen/qwen3-32b"
CANDIDATE_K = 20

PROMPT_TEMPLATE = (
    "Given a query and a document, determine if the document is relevant to the query.\n\n"
    "Query: {query}\n\n"
    "Document: {document}\n\n"
    "Think step by step about whether this document contains information that answers "
    "the query. End your response with either 'true' (relevant) or 'false' (not relevant)."
)


class ReasoningRerankerRetriever(BaseRetriever):
    def __init__(self):
        self._dense = DenseRetriever()
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"])

    @property
    def name(self) -> str:
        return "Dense+ReasoningReranker"

    def index(self, chunks: list[dict]) -> None:
        self._chunks = {c["chunk_id"]: c for c in chunks}
        self._dense.index(chunks)

    def _score(self, query: str, document: str) -> float:
        prompt = PROMPT_TEMPLATE.format(query=query, document=document[:1000])
        try:
            stream = self._client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_completion_tokens=4096,
                top_p=0.95,
                reasoning_effort="default",
                stream=True,
                stop=None,
            )
            text = "".join(
                chunk.choices[0].delta.content or ""
                for chunk in stream
            ).lower()
            last_true = text.rfind("true")
            last_false = text.rfind("false")
            return 1.0 if last_true > last_false else 0.0
        except Exception as e:
            print(f"[reasoning_reranker] API error: {e}")
            return 0.0

    def retrieve(self, query: str, k: int = 10) -> list[str]:
        candidates = self._dense.retrieve(query, k=min(CANDIDATE_K, len(self._chunks)))
        scored = [(cid, self._score(query, self._chunks[cid]["text"])) for cid in candidates]
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:k]]

import os
from groq import Groq
from .base import BaseRetriever
from .hybrid import HybridRetriever


class QueryRewritingRetriever(BaseRetriever):
    def __init__(self):
        self._hybrid = HybridRetriever()
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"])

    @property
    def name(self) -> str:
        return "QueryRewriting+Hybrid"

    def index(self, chunks: list[dict]) -> None:
        self._hybrid.index(chunks)

    def _rewrite(self, query: str) -> str:
        prompt = (
            "Rewrite the following user question into precise technical search terms "
            "that a developer would use to find relevant documentation or source code.\n"
            "Output only the rewritten query, no explanation.\n\n"
            f"Question: {query}\n\n"
            "Rewritten query:"
        )
        response = self._client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=80,
        )
        return response.choices[0].message.content.strip()

    def retrieve(self, query: str, k: int = 10) -> list[str]:
        rewritten = self._rewrite(query)
        return self._hybrid.retrieve(rewritten, k=k)

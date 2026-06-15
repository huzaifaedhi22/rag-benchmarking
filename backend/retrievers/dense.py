import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from .base import BaseRetriever

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class DenseRetriever(BaseRetriever):
    def __init__(self):
        self._model = SentenceTransformer(MODEL_NAME)

    @property
    def name(self) -> str:
        return "Dense"

    def index(self, chunks: list[dict]) -> None:
        self._client = chromadb.Client()
        self._collection = self._client.create_collection(f"chunks_{uuid.uuid4().hex}")

        texts = [c["text"] for c in chunks]
        embeddings = self._model.encode(texts, show_progress_bar=True).tolist()

        self._collection.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=embeddings,
            documents=texts,
        )

    def retrieve(self, query: str, k: int = 10) -> list[str]:
        query_embedding = self._model.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=k,
        )
        return results["ids"][0]

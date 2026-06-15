import tempfile
from pylate import indexes, models, retrieve
from .base import BaseRetriever

MODEL_NAME = "colbert-ir/colbertv2.0"


class ColBERTRetriever(BaseRetriever):
    def __init__(self):
        self._model = models.ColBERT(model_name_or_path=MODEL_NAME)
        self._tmpdir = None

    @property
    def name(self) -> str:
        return "ColBERT"

    def index(self, chunks: list[dict]) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._index = indexes.PLAID(
            index_folder=self._tmpdir,
            index_name="colbert",
            override=True,
        )

        texts = [c["text"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]

        embeddings = self._model.encode(
            texts,
            batch_size=32,
            is_query=False,
            show_progress_bar=True,
        )

        self._index.add_documents(
            documents_ids=chunk_ids,
            documents_embeddings=embeddings,
        )

        self._retriever = retrieve.ColBERT(index=self._index)

    def retrieve(self, query: str, k: int = 10) -> list[str]:
        query_embedding = self._model.encode(
            [query],
            batch_size=1,
            is_query=True,
        )
        results = self._retriever.retrieve(
            queries_embeddings=query_embedding,
            k=k,
        )
        return [doc["id"] for doc in results[0]]

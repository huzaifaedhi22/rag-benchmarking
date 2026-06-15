from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def index(self, chunks: list[dict]) -> None:
        ...

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> list[str]:
        ...

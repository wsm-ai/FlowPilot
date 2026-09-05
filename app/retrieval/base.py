from typing import Protocol

from app.retrieval.models import RetrievalQuery, RetrievalResult


class RetrievalError(Exception):
    """Raised when retrieval infrastructure cannot complete an operation."""


class Retriever(Protocol):
    async def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        ...

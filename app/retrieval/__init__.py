"""Provider-independent retrieval domain primitives."""

from app.retrieval.base import RetrievalError, Retriever
from app.retrieval.in_memory import InMemoryRetriever
from app.retrieval.models import (
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievalQuery,
    RetrievalResult,
)

__all__ = [
    "InMemoryRetriever",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "RetrievalError",
    "RetrievalQuery",
    "RetrievalResult",
    "Retriever",
]

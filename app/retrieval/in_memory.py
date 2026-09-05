import re

from app.retrieval.base import RetrievalError
from app.retrieval.models import KnowledgeChunk, RetrievalQuery, RetrievalResult


_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


class InMemoryRetriever:
    """Deterministic lexical retriever intended for local development."""

    def __init__(self, chunks: list[KnowledgeChunk] | None = None) -> None:
        self._chunks: dict[str, KnowledgeChunk] = {}
        if chunks:
            self.add_chunks(chunks)

    def add_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        incoming_ids = [chunk.chunk_id for chunk in chunks]
        if len(incoming_ids) != len(set(incoming_ids)):
            raise RetrievalError("Duplicate chunk ID in input")
        if any(chunk_id in self._chunks for chunk_id in incoming_ids):
            raise RetrievalError("Chunk ID already exists")

        additions = {
            chunk.chunk_id: chunk.model_copy(deep=True) for chunk in chunks
        }
        self._chunks.update(additions)

    async def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        query_tokens = _tokenize(query.query)
        if not query_tokens:
            return []

        matches: list[tuple[float, KnowledgeChunk]] = []
        for chunk in self._chunks.values():
            if not self._matches_filters(chunk, query.filters):
                continue

            overlap = query_tokens & _tokenize(chunk.content)
            score = len(overlap) / len(query_tokens)
            if score > 0:
                matches.append((score, chunk))

        matches.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            RetrievalResult(
                chunk=chunk.model_copy(deep=True),
                score=score,
                rank=rank,
            )
            for rank, (score, chunk) in enumerate(
                matches[: query.top_k],
                start=1,
            )
        ]

    @staticmethod
    def _matches_filters(
        chunk: KnowledgeChunk,
        filters: dict[str, object],
    ) -> bool:
        for name, expected in filters.items():
            if name == "document_id":
                actual = chunk.document_id
            elif name == "source":
                actual = chunk.source
            else:
                if name not in chunk.metadata:
                    return False
                actual = chunk.metadata[name]
            if actual != expected:
                return False
        return True

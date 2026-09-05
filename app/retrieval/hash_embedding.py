import hashlib

from app.retrieval.embeddings import EmbeddingError, EmbeddingVector


class HashEmbeddingProvider:
    """Deterministic offline vectors for contract and pipeline development."""

    def __init__(self, dimensions: int = 32) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be at least 1")
        self._dimensions = dimensions

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[EmbeddingVector]:
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("embedding text must not be blank")
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> EmbeddingVector:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("embedding text must not be blank")
        return self._embed(text)

    def _embed(self, text: str) -> EmbeddingVector:
        values: list[float] = []
        block = 0
        encoded = text.encode("utf-8")
        while len(values) < self._dimensions:
            digest = hashlib.sha256(
                encoded + block.to_bytes(8, byteorder="big")
            ).digest()
            values.extend((byte / 127.5) - 1.0 for byte in digest)
            block += 1
        return EmbeddingVector(values=values[: self._dimensions])

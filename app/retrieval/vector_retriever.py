from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.models import RetrievalQuery, RetrievalResult
from app.retrieval.vector_store import VectorStore


class VectorRetriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    async def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        query_embedding = await self._embedding_provider.embed_query(query.query)
        return await self._vector_store.search(
            query_embedding,
            top_k=query.top_k,
            filters=query.filters,
        )

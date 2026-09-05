from app.retrieval.ingestion import DocumentIngestionService
from app.retrieval.models import KnowledgeChunk, KnowledgeDocument
from app.retrieval.vector_store import VectorEntry, VectorStore


class KnowledgeBaseIndexer:
    def __init__(
        self,
        ingestion_service: DocumentIngestionService,
        vector_store: VectorStore,
    ) -> None:
        self._ingestion_service = ingestion_service
        self._vector_store = vector_store

    async def index_document(
        self,
        document: KnowledgeDocument,
    ) -> list[KnowledgeChunk]:
        ingested = await self._ingestion_service.ingest(document)
        if not ingested:
            return []

        await self._vector_store.add(
            [
                VectorEntry(chunk=item.chunk, embedding=item.embedding)
                for item in ingested
            ]
        )
        return [item.chunk for item in ingested]

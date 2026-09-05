from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.persistence.checkpoint import async_checkpoint_saver
from app.persistence.sqlite_repository import SQLiteRunRepository
from app.retrieval.chunking import SimpleTextChunker
from app.retrieval.demo_knowledge import bootstrap_demo_knowledge_base
from app.retrieval.hash_embedding import HashEmbeddingProvider
from app.retrieval.in_memory_vector_store import InMemoryVectorStore
from app.retrieval.indexing import KnowledgeBaseIndexer
from app.retrieval.ingestion import DocumentIngestionService
from app.retrieval.vector_retriever import VectorRetriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_repository = SQLiteRunRepository("data/flowpilot.db")
    await run_repository.initialize()
    embedding_provider = HashEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    chunker = SimpleTextChunker()
    ingestion_service = DocumentIngestionService(chunker, embedding_provider)
    indexer = KnowledgeBaseIndexer(ingestion_service, vector_store)
    await bootstrap_demo_knowledge_base(indexer)
    retriever = VectorRetriever(embedding_provider, vector_store)
    async with async_checkpoint_saver("data/checkpoints.sqlite") as checkpointer:
        app.state.run_repository = run_repository
        app.state.checkpointer = checkpointer
        app.state.retriever = retriever
        yield


app = FastAPI(
    title="FlowPilot API",
    description="Enterprise AI Workflow Agent",
    version="0.1.0",
    lifespan=lifespan,
)


app.include_router(health_router)
app.include_router(chat_router)
app.include_router(agent_router)

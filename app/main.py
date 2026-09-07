from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.core.config import get_settings
from app.mcp.application import compose_configured_mcp_tools
from app.persistence.checkpoint import async_checkpoint_saver
from app.persistence.sqlite_repository import SQLiteRunRepository
from app.retrieval.chunking import SimpleTextChunker
from app.retrieval.demo_knowledge import bootstrap_demo_knowledge_base
from app.retrieval.hash_embedding import HashEmbeddingProvider
from app.retrieval.in_memory_vector_store import InMemoryVectorStore
from app.retrieval.indexing import KnowledgeBaseIndexer
from app.retrieval.ingestion import DocumentIngestionService
from app.retrieval.vector_retriever import VectorRetriever
from app.tools.knowledge_base import KnowledgeBaseTool
from app.tools.registry import create_default_tool_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    run_repository = SQLiteRunRepository("data/flowpilot.db")
    await run_repository.initialize()
    embedding_provider = HashEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    chunker = SimpleTextChunker()
    ingestion_service = DocumentIngestionService(chunker, embedding_provider)
    indexer = KnowledgeBaseIndexer(ingestion_service, vector_store)
    await bootstrap_demo_knowledge_base(indexer)
    retriever = VectorRetriever(embedding_provider, vector_store)
    registry = create_default_tool_registry()
    registry.register(KnowledgeBaseTool(retriever))
    async with AsyncExitStack() as exit_stack:
        checkpointer = await exit_stack.enter_async_context(
            async_checkpoint_saver("data/checkpoints.sqlite")
        )
        mcp_composition = await compose_configured_mcp_tools(
            registry,
            settings.mcp_servers,
            exit_stack,
        )
        app.state.run_repository = run_repository
        app.state.checkpointer = checkpointer
        app.state.retriever = retriever
        app.state.tool_registry = registry
        app.state.mcp_approval_required_actions = (
            mcp_composition.approval_required_actions
        )
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

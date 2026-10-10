import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import logging
import os

from fastapi import FastAPI

from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.errors import register_api_exception_handlers
from app.core.config import get_settings
from app.mcp.application import compose_configured_mcp_tools
from app.mcp.server import (
    create_mcp_export_registry,
    flowpilot_mcp_application,
)
from app.persistence.checkpoint import async_checkpoint_saver
from app.persistence.sqlite_repository import SQLiteRunRepository
from app.persistence.side_effect_repository import (
    SQLiteSideEffectExecutionRepository,
)
from app.persistence.trace_repository import SQLiteTraceRepository
from app.observability import NoOpAsyncTraceEmitter, SQLiteTraceEmitter
from app.reliability.side_effects import SideEffectExecutor
from app.retrieval.chunking import SimpleTextChunker
from app.retrieval.demo_knowledge import bootstrap_demo_knowledge_base
from app.retrieval.hash_embedding import HashEmbeddingProvider
from app.retrieval.in_memory_vector_store import InMemoryVectorStore
from app.retrieval.indexing import KnowledgeBaseIndexer
from app.retrieval.ingestion import DocumentIngestionService
from app.retrieval.vector_retriever import VectorRetriever
from app.security import (
    APIKeyAuthenticationMiddleware,
    RequestBodyLimitMiddleware,
    SecurityHeadersMiddleware,
    configure_sensitive_logging,
)
from app.security.tool_authorization import ToolRisk
from app.tools.knowledge_base import KnowledgeBaseTool
from app.tools.registry import create_default_tool_registry


_diagnostic_logger = logging.getLogger("flowpilot.diagnostics")


def _configured_secret_values(settings) -> tuple[str, ...]:
    values = [settings.deepseek_api_key.get_secret_value()]
    if settings.flowpilot_api_key is not None:
        values.append(settings.flowpilot_api_key.get_secret_value())
    values.extend(
        binding.key.get_secret_value() for binding in settings.flowpilot_api_keys
    )
    for server in settings.mcp_servers:
        environment_name = getattr(server, "bearer_token_env", None)
        if environment_name:
            environment_value = os.environ.get(environment_name)
            if environment_value:
                values.append(environment_value)
    return tuple(values)


async def initialize_trace_persistence(
    database_path: str,
) -> tuple[SQLiteTraceRepository | None, SQLiteTraceEmitter | NoOpAsyncTraceEmitter]:
    repository = SQLiteTraceRepository(database_path)
    try:
        await repository.initialize()
    except asyncio.CancelledError:
        raise
    except Exception:
        try:
            _diagnostic_logger.warning("Trace persistence initialization failed")
        except Exception:
            pass
        return None, NoOpAsyncTraceEmitter()
    return repository, SQLiteTraceEmitter(repository)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_sensitive_logging()
    settings = get_settings()
    configure_sensitive_logging(secrets=_configured_secret_values(settings))
    run_repository = SQLiteRunRepository("data/flowpilot.db")
    await run_repository.initialize()
    side_effect_repository = SQLiteSideEffectExecutionRepository(
        "data/flowpilot.db"
    )
    await side_effect_repository.initialize()
    trace_repository, trace_emitter = await initialize_trace_persistence(
        "data/flowpilot.db"
    )
    embedding_provider = HashEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    chunker = SimpleTextChunker()
    ingestion_service = DocumentIngestionService(chunker, embedding_provider)
    indexer = KnowledgeBaseIndexer(ingestion_service, vector_store)
    await bootstrap_demo_knowledge_base(indexer)
    retriever = VectorRetriever(embedding_provider, vector_store)
    registry = create_default_tool_registry()
    registry.register(KnowledgeBaseTool(retriever), risk=ToolRisk.READ_ONLY)
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
        app.state.side_effect_executor = SideEffectExecutor(
            side_effect_repository
        )
        app.state.trace_repository = trace_repository
        app.state.trace_emitter = trace_emitter
        app.state.checkpointer = checkpointer
        app.state.retriever = retriever
        app.state.tool_registry = registry
        app.state.mcp_approval_required_actions = (
            mcp_composition.approval_required_actions
        )
        app.state.mcp_degradations = mcp_composition.degradations
        export_registry = create_mcp_export_registry(
            registry,
            mcp_composition.approval_required_actions,
        )
        flowpilot_mcp_application.reset_server()
        flowpilot_mcp_application.runtime.bind(export_registry)
        exit_stack.callback(flowpilot_mcp_application.runtime.unbind)
        await exit_stack.enter_async_context(
            flowpilot_mcp_application.run_session_manager()
        )
        yield


app = FastAPI(
    title="FlowPilot API",
    description="Enterprise AI Workflow Agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    RequestBodyLimitMiddleware,
    settings_provider=get_settings,
)
app.add_middleware(
    APIKeyAuthenticationMiddleware,
    settings_provider=get_settings,
)
app.add_middleware(SecurityHeadersMiddleware)

register_api_exception_handlers(app)


app.include_router(health_router)
app.include_router(chat_router)
app.include_router(agent_router)
app.mount("/mcp", flowpilot_mcp_application)

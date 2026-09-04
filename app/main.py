from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.persistence.checkpoint import async_checkpoint_saver
from app.persistence.sqlite_repository import SQLiteRunRepository


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_repository = SQLiteRunRepository("data/flowpilot.db")
    await run_repository.initialize()
    async with async_checkpoint_saver("data/checkpoints.sqlite") as checkpointer:
        app.state.run_repository = run_repository
        app.state.checkpointer = checkpointer
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

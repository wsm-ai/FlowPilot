from fastapi import FastAPI

from app.api.health import router as health_router


app = FastAPI(
    title="FlowPilot API",
    description="Enterprise AI Workflow Agent",
    version="0.1.0",
)


app.include_router(health_router)
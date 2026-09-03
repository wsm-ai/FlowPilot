from fastapi import APIRouter


router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """
    Check whether the FlowPilot API service is running.
    """
    return {
        "status": "ok",
        "service": "FlowPilot",
    }
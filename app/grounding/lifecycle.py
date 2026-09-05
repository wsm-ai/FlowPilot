from dataclasses import dataclass
from typing import Literal

from app.grounding.evidence import EvidenceExtractionError
from app.grounding.models import GroundedAnswer
from app.providers.base import LLMProviderError
from app.schemas.planning import ExecutionPlan
from app.services.grounded_answer_service import GroundedAnswerError, GroundedAnswerService


GroundingSynthesisStatus = Literal["not_attempted", "completed", "failed"]


@dataclass(frozen=True, slots=True)
class GroundingSynthesisOutcome:
    status: GroundingSynthesisStatus
    grounded_answer: GroundedAnswer | None = None
    error_type: str | None = None


async def attempt_grounded_synthesis(
    service: GroundedAnswerService | None,
    *,
    goal: str,
    plan: ExecutionPlan,
    step_results: list[dict],
) -> GroundingSynthesisOutcome:
    if service is None:
        return GroundingSynthesisOutcome(status="not_attempted")
    try:
        answer = await service.synthesize(
            goal=goal,
            plan=plan,
            step_results=step_results,
        )
    except (GroundedAnswerError, EvidenceExtractionError, LLMProviderError) as exc:
        return GroundingSynthesisOutcome(
            status="failed",
            error_type=type(exc).__name__,
        )
    return GroundingSynthesisOutcome(
        status="completed",
        grounded_answer=answer,
    )

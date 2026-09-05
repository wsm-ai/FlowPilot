import asyncio
import json

import pytest

from app.grounding.models import Citation, GroundedAnswer
from app.providers.types import LLMResponse
from app.retrieval.models import KnowledgeChunk, RetrievalQuery, RetrievalResult
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.grounded_answer_service import GroundedAnswerError, GroundedAnswerService
from app.services.llm_service import LLMService
from app.services.planned_agent_service import PlannedAgentService
from app.tools.knowledge_base import KnowledgeBaseTool
from app.tools.registry import create_default_tool_registry


def retrieval_plan(goal: str) -> ExecutionPlan:
    return ExecutionPlan(
        goal=goal,
        steps=[
            PlanStep(
                id=1,
                description="Search internal knowledge",
                action="search_knowledge_base",
                arguments={"query": "Enterprise login troubleshooting"},
                requires_approval=False,
            )
        ],
    )


class FakePlanner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan
        self.call_count = 0

    async def create_plan(self, goal: str) -> ExecutionPlan:
        self.call_count += 1
        return self.plan


class FakeRetriever:
    def __init__(self) -> None:
        self.call_count = 0

    async def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        self.call_count += 1
        return [
            RetrievalResult(
                chunk=KnowledgeChunk(
                    chunk_id="LOGIN:chunk:0",
                    document_id="LOGIN",
                    content="Check the enterprise SSO configuration.",
                    source="support-handbook",
                    metadata={"document_title": "Enterprise Login Guide"},
                    position=0,
                ),
                score=0.97,
                rank=1,
            )
        ]


class FakeGroundedAnswerService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = []

    async def synthesize(self, *, goal, plan, step_results):
        self.calls.append(
            {"goal": goal, "plan": plan, "step_results": step_results}
        )
        if self.error is not None:
            raise self.error
        return GroundedAnswer(
            answer="Grounded final answer",
            citations=[
                Citation(
                    citation_id="E1",
                    chunk_id="LOGIN:chunk:0",
                    document_id="LOGIN",
                    source="support-handbook",
                )
            ],
        )


class SynthesisProvider:
    model = "test-model"

    async def complete(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(
                {
                    "answer": "Check SSO.",
                    "support_basis": "knowledge_evidence",
                    "citation_ids": ["E1"],
                }
            )
        )


def build_registry(retriever: FakeRetriever):
    registry = create_default_tool_registry()
    registry.register(KnowledgeBaseTool(retriever))
    return registry


def test_completed_plan_hands_final_tool_evidence_to_synthesis_once():
    goal = "Find login guidance"
    plan = retrieval_plan(goal)
    retriever = FakeRetriever()
    synthesis = FakeGroundedAnswerService()

    result = asyncio.run(
        PlannedAgentService(
            FakePlanner(plan),
            build_registry(retriever),
            grounded_answer_service=synthesis,
        ).run(goal)
    )

    assert result.status == "completed"
    assert result.grounded_answer.answer == "Grounded final answer"
    assert len(synthesis.calls) == 1
    assert synthesis.calls[0]["goal"] == goal
    assert synthesis.calls[0]["plan"] is plan
    assert synthesis.calls[0]["step_results"] == result.step_results
    assert synthesis.calls[0]["step_results"][0]["result"][0]["chunk_id"] == "LOGIN:chunk:0"
    assert retriever.call_count == 1


def test_planned_service_without_synthesizer_remains_backward_compatible():
    goal = "Find login guidance"
    retriever = FakeRetriever()
    result = asyncio.run(
        PlannedAgentService(
            FakePlanner(retrieval_plan(goal)), build_registry(retriever)
        ).run(goal)
    )
    assert result.status == "completed"
    assert result.grounded_answer is None
    assert retriever.call_count == 1


def test_real_grounded_service_maps_citation_from_executed_step_results():
    goal = "Find login guidance"
    retriever = FakeRetriever()
    result = asyncio.run(
        PlannedAgentService(
            FakePlanner(retrieval_plan(goal)),
            build_registry(retriever),
            GroundedAnswerService(LLMService(SynthesisProvider())),
        ).run(goal)
    )
    assert result.grounded_answer.answer == "Check SSO."
    assert result.grounded_answer.citations[0].chunk_id == "LOGIN:chunk:0"
    assert retriever.call_count == 1


def test_synthesis_failure_propagates_without_rerunning_execution():
    goal = "Find login guidance"
    retriever = FakeRetriever()
    synthesis = FakeGroundedAnswerService(GroundedAnswerError("synthesis failed"))
    service = PlannedAgentService(
        FakePlanner(retrieval_plan(goal)),
        build_registry(retriever),
        synthesis,
    )

    with pytest.raises(GroundedAnswerError, match="synthesis failed"):
        asyncio.run(service.run(goal))

    assert len(synthesis.calls) == 1
    assert retriever.call_count == 1

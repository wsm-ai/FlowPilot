import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import json
from tempfile import TemporaryDirectory
from typing import Any

from app.evaluation.models import EvaluationCase, EvaluationCheck
from app.mcp.client import MCPTransientDiscoveryError
from app.mcp.models import MCPRemoteTool, MCPToolResult
from app.mcp.tool_composition import MCPServerClientBinding, compose_mcp_tools
from app.persistence.checkpoint import async_checkpoint_saver
from app.persistence.side_effect_repository import SQLiteSideEffectExecutionRepository
from app.providers.types import LLMResponse
from app.reliability.side_effects import SideEffectExecutor, SideEffectReplayBlockedError
from app.reliability.timeouts import SideEffectOperationTimeoutError, TimeoutPolicy
from app.retrieval.in_memory import InMemoryRetriever
from app.retrieval.models import KnowledgeChunk
from app.services.approval_workflow_service import ApprovalWorkflowService
from app.services.grounded_answer_service import (
    GroundedAnswerService,
    INSUFFICIENT_SUPPORT_ANSWER,
)
from app.services.llm_service import LLMService
from app.services.planner_service import PlannerService, UnavailablePlanActionError
from app.tools.knowledge_base import KnowledgeBaseTool
from app.tools.registry import ToolRegistry, create_default_tool_registry


def _check(name: str, passed: bool) -> EvaluationCheck:
    return EvaluationCheck(
        name=name,
        passed=passed,
        message=None if passed else "Expected agent behavior was not observed",
    )


class _JSONProvider:
    model = "evaluation-fake"

    def __init__(self, payloads: Sequence[dict[str, Any]]) -> None:
        self._payloads = list(payloads)
        self.call_count = 0

    async def complete(self, **_: Any) -> LLMResponse:
        payload = self._payloads[self.call_count]
        self.call_count += 1
        return LLMResponse(content=json.dumps(payload), tool_calls=[])


@dataclass(frozen=True, slots=True)
class _FunctionScenario:
    operation: Callable[[], Awaitable[Sequence[EvaluationCheck]]]

    async def run(self) -> Sequence[EvaluationCheck]:
        return await self.operation()


class _CountingSideEffectTool:
    name = "create_github_issue"
    description = "Create a test-only issue record"
    parameters = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }

    def __init__(self) -> None:
        self.call_count = 0

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        return {"issue_id": "ISSUE-1", "title": arguments.get("title")}


def _side_effect_plan() -> dict[str, Any]:
    return {
        "goal": "Create an issue",
        "steps": [{
            "id": 1,
            "description": "Create the customer issue",
            "action": "create_github_issue",
            "arguments": {"title": "Customer bug"},
            "requires_approval": False,
        }],
    }


def _approval_setup() -> tuple[PlannerService, ToolRegistry, _CountingSideEffectTool]:
    registry = ToolRegistry()
    tool = _CountingSideEffectTool()
    registry.register(tool)
    planner = PlannerService(
        LLMService(_JSONProvider([_side_effect_plan()])),
        registry.definitions(),
        {tool.name},
    )
    return planner, registry, tool


async def _planner_read_feedback() -> Sequence[EvaluationCheck]:
    registry = create_default_tool_registry()
    payload = {
        "goal": "Analyze high-priority feedback",
        "steps": [{
            "id": 1,
            "description": "Read customer feedback",
            "action": "get_customer_feedback",
            "arguments": {"customer_id": "C001", "priority": "high"},
            "requires_approval": True,
        }],
    }
    plan = await PlannerService(
        LLMService(_JSONProvider([payload])), registry.definitions()
    ).create_plan(payload["goal"])
    has_steps = bool(plan.steps)
    step = plan.steps[0] if has_steps else None
    return (
        _check("plan_created", has_steps),
        _check(
            "expected_action_selected",
            step is not None and step.action == "get_customer_feedback",
        ),
        _check(
            "arguments_valid",
            step is not None
            and step.arguments == payload["steps"][0]["arguments"],
        ),
        _check(
            "approval_not_required",
            step is not None and not step.requires_approval,
        ),
    )


async def _planner_reject_unknown() -> Sequence[EvaluationCheck]:
    payload = {
        "goal": "Unsafe request",
        "steps": [{
            "id": 1, "description": "Unsafe action",
            "action": "delete_production_database", "arguments": {},
            "requires_approval": False,
        }],
    }
    rejected = False
    try:
        await PlannerService(
            LLMService(_JSONProvider([payload])),
            create_default_tool_registry().definitions(),
        ).create_plan(payload["goal"])
    except UnavailablePlanActionError:
        rejected = True
    return (_check("unsafe_or_unavailable_action_rejected", rejected),)


async def _run_approval(decision: str | None) -> Sequence[EvaluationCheck]:
    planner, registry, tool = _approval_setup()
    with TemporaryDirectory(prefix="flowpilot-eval-") as directory:
        repository = SQLiteSideEffectExecutionRepository(f"{directory}/ledger.db")
        await repository.initialize()
        executor = SideEffectExecutor(repository)
        async with async_checkpoint_saver(f"{directory}/checkpoint.db") as saver:
            service = ApprovalWorkflowService(planner, registry, saver, side_effect_executor=executor)
            started = await service.start("Create an issue", thread_id="evaluation-thread")
            count_before = tool.call_count
            if decision is None:
                has_step = bool(started.plan.steps)
                action = started.plan.steps[0].action if has_step else None
                return (
                    _check("side_effect_action_selected", action == tool.name),
                    _check("approval_required", started.status == "approval_required"),
                    _check("no_dispatch_before_approval", count_before == 0),
                )
            finished = await service.resume(
                "evaluation-thread", decision, run_id="evaluation-run"
            )
    if decision == "reject":
        return (
            _check("approval_requested", started.status == "approval_required"),
            _check("rejection_respected", finished.status == "rejected"),
            _check("side_effect_not_executed", tool.call_count == 0),
        )
    return (
        _check("approval_requested", started.status == "approval_required"),
        _check("approval_respected", finished.status == "completed"),
        _check("side_effect_executed", bool(finished.step_results)),
        _check("single_dispatch", tool.call_count == 1),
    )


async def _rag_grounded() -> Sequence[EvaluationCheck]:
    retriever = InMemoryRetriever([KnowledgeChunk(
        chunk_id="policy-p1", document_id="policy", position=0,
        content="Priority P1 issues must be acknowledged within 1 hour.",
        source="internal-policy", metadata={"document_title": "Escalation Policy"},
    )])
    evidence = await KnowledgeBaseTool(retriever).execute(
        {"query": "What is the acknowledgement target for P1 issues?"}
    )
    provider = _JSONProvider([{
        "answer": "P1 issues must be acknowledged within 1 hour.",
        "support_basis": "knowledge_evidence", "citation_ids": ["E1"],
    }])
    plan_data = {
        "goal": "Find the P1 target",
        "steps": [{"id": 1, "description": "Search policy", "action": "search_knowledge_base", "arguments": {"query": "P1 acknowledgement target"}, "requires_approval": False}],
    }
    from app.schemas.planning import ExecutionPlan
    plan = ExecutionPlan.model_validate(plan_data)
    answer = await GroundedAnswerService(LLMService(provider)).synthesize(
        goal=plan.goal, plan=plan,
        step_results=[{"step_id": 1, "action": "search_knowledge_base", "result": evidence}],
    )
    return (
        _check("relevant_evidence_retrieved", len(evidence) == 1),
        _check("grounded_answer_created", "1 hour" in answer.answer),
        _check("citation_present", len(answer.citations) == 1),
        _check("citation_valid", bool(answer.citations) and answer.citations[0].chunk_id == "policy-p1"),
    )


async def _rag_insufficient() -> Sequence[EvaluationCheck]:
    from app.schemas.planning import ExecutionPlan
    plan = ExecutionPlan.model_validate({
        "goal": "Find private phone number",
        "steps": [{"id": 1, "description": "Search", "action": "search_knowledge_base", "arguments": {"query": "phone"}, "requires_approval": False}],
    })
    provider = _JSONProvider([])
    answer = await GroundedAnswerService(LLMService(provider)).synthesize(
        goal=plan.goal, plan=plan,
        step_results=[{"step_id": 1, "action": "search_knowledge_base", "result": []}],
    )
    return (
        _check("insufficient_evidence_detected", answer.answer == INSUFFICIENT_SUPPORT_ANSWER),
        _check("no_fabricated_citation", answer.citations == [] and provider.call_count == 0),
    )


class _MCPClient:
    def __init__(self, *, unavailable: bool = False) -> None:
        self.unavailable = unavailable
        self.discovery_count = 0

    async def list_tools(self) -> list[MCPRemoteTool]:
        self.discovery_count += 1
        if self.unavailable:
            raise MCPTransientDiscoveryError("Evaluation MCP unavailable")
        return [MCPRemoteTool(name="create_ticket", description="Create ticket", input_schema={"type": "object", "properties": {}})]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        return MCPToolResult(structured_content={"ticket_id": "T-1"})


async def _mcp_least_privilege() -> Sequence[EvaluationCheck]:
    registry = create_default_tool_registry()
    composition = await compose_mcp_tools(
        registry, [MCPServerClientBinding("support", _MCPClient())]
    )
    has_tool = bool(composition.local_names)
    name = composition.local_names[0] if has_tool else None
    reactive = registry.excluding(composition.approval_required_actions)
    return (
        _check("mcp_tool_registered", name is not None and registry.contains(name)),
        _check("planned_capability_visible", name is not None and any(item["function"]["name"] == name for item in registry.definitions())),
        _check("reactive_capability_hidden", name is not None and not reactive.contains(name)),
        _check("approval_required", name is not None and name in composition.approval_required_actions),
    )


async def _mcp_degraded() -> Sequence[EvaluationCheck]:
    registry = create_default_tool_registry()
    client = _MCPClient(unavailable=True)
    composition = await compose_mcp_tools(
        registry, [MCPServerClientBinding("optional", client, required=False)]
    )
    action = "mcp_optional_create_ticket"
    payload = {"goal": "Create ticket", "steps": [{"id": 1, "description": "Create", "action": action, "arguments": {}, "requires_approval": True}]}
    rejected = False
    try:
        await PlannerService(LLMService(_JSONProvider([payload])), registry.definitions()).create_plan(payload["goal"])
    except UnavailablePlanActionError:
        rejected = True
    return (
        _check("degradation_recorded", len(composition.degradations) == 1),
        _check("degraded_tool_absent", not registry.contains(action)),
        _check("planner_cannot_use_degraded_action", rejected),
    )


async def _ambiguous_side_effect() -> Sequence[EvaluationCheck]:
    calls = 0
    gate = asyncio.Event()
    with TemporaryDirectory(prefix="flowpilot-eval-") as directory:
        repository = SQLiteSideEffectExecutionRepository(f"{directory}/ledger.db")
        await repository.initialize()
        executor = SideEffectExecutor(repository, timeout_policy=TimeoutPolicy(0.01))

        async def operation() -> None:
            nonlocal calls
            calls += 1
            await gate.wait()

        timed_out = False
        try:
            await executor.execute(run_id="run", step_id=1, action="write", arguments={}, operation=operation)
        except SideEffectOperationTimeoutError:
            timed_out = True
        replay_blocked = False
        try:
            await executor.execute(run_id="run", step_id=1, action="write", arguments={}, operation=operation)
        except SideEffectReplayBlockedError:
            replay_blocked = True
    return (
        _check("ambiguous_outcome_detected", timed_out),
        _check("automatic_retry_not_performed", calls == 1),
        _check("replay_blocked", replay_blocked),
    )


def build_default_evaluation_cases() -> tuple[EvaluationCase, ...]:
    specifications = (
        ("planner.read_customer_feedback", "Read customer feedback plan", "Planner selects the registered read-only feedback tool.", _planner_read_feedback),
        ("planner.reject_unknown_action", "Reject unknown planner action", "Planner rejects an action absent from the tool registry.", _planner_reject_unknown),
        ("hitl.requires_approval", "Require approval for side effects", "A side effect pauses before any dispatch.", lambda: _run_approval(None)),
        ("hitl.reject_side_effect", "Respect approval rejection", "A rejected side effect is never dispatched.", lambda: _run_approval("reject")),
        ("hitl.approve_side_effect", "Execute approved side effect", "An approved side effect executes exactly once.", lambda: _run_approval("approve")),
        ("rag.grounded_citation", "Produce a grounded citation", "Retrieved local evidence supports a valid citation.", _rag_grounded),
        ("rag.insufficient_evidence", "Handle insufficient evidence", "Missing evidence does not produce a fabricated citation.", _rag_insufficient),
        ("mcp.reactive_least_privilege", "Restrict reactive MCP capability", "Approval-required MCP tools remain planned-only.", _mcp_least_privilege),
        ("mcp.degraded_capability", "Handle degraded MCP capability", "Unavailable optional MCP tools cannot be planned.", _mcp_degraded),
        ("reliability.ambiguous_side_effect", "Block ambiguous side-effect replay", "A timed-out side effect is not retried or replayed.", _ambiguous_side_effect),
    )
    return tuple(
        EvaluationCase(case_id=case_id, name=name, description=description, scenario=_FunctionScenario(operation))
        for case_id, name, description, operation in specifications
    )

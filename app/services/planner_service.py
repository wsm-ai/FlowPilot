import json
from collections.abc import Callable
from datetime import datetime
from typing import Any, TYPE_CHECKING

from pydantic import ValidationError

from app.providers.base import LLMProviderError
from app.observability import (
    AgentEvent,
    AsyncTraceEmitter,
    EventEmitter,
    EventOutcome,
    EventStage,
    EventType,
    NoOpEventEmitter,
    emit_best_effort,
    utc_now,
)
from app.schemas.planning import ExecutionPlan
from app.services.llm_service import LLMService

if TYPE_CHECKING:
    from app.reliability.failures import FailureCategory


PLANNER_SYSTEM_PROMPT = """You are FlowPilot's task planner.
Break the user's goal into clear, ordered, executable steps.
You cannot execute tools. You can only create a plan.
Return strict JSON with exactly this structure:
{
  "goal": "...",
  "steps": [
    {
      "id": 1,
      "description": "...",
      "action": "snake_case_action",
      "arguments": {},
      "requires_approval": false
    }
  ]
}
Use concise, machine-readable snake_case values for action.
For tool-like actions, provide the structured arguments required to execute the action.
If an action needs no arguments, return an empty object for arguments.
Do not invent execution results. Only create the action and its arguments.
Do not include Markdown fences or text outside the JSON object."""


TOOL_AWARE_PLANNING_RULES = """
Use only actions listed in AVAILABLE TOOLS.
Each action MUST exactly match an available tool name.
Arguments MUST follow that tool's JSON parameters schema.
Do not invent actions.
Use search_knowledge_base when internal evidence is needed, but do not use it
when retrieval is unnecessary.
search_knowledge_base is read-only and requires_approval MUST be false.
Read-only tools do not require approval. Side-effect tools may require approval
according to their policy.
Do not invent execution results or put unknown future retrieval evidence into
plan arguments.
Do not make later step arguments depend on unknown future retrieval results.
AVAILABLE TOOLS:
"""


READ_ONLY_ACTIONS = {
    "get_customer_feedback",
    "search_knowledge_base",
}


class PlanningError(Exception):
    """Raised when an LLM-generated execution plan is invalid."""


class UnavailablePlanActionError(PlanningError):
    """Raised when a plan selects an action absent from the tool catalog."""


class PlannerService:
    def __init__(
        self,
        llm_service: LLMService,
        tool_definitions: list[dict[str, Any]] | None = None,
        approval_required_actions: set[str] | frozenset[str] | None = None,
        event_emitter: EventEmitter | None = None,
        trace_emitter: AsyncTraceEmitter | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._llm_service = llm_service
        self._approval_required_actions = frozenset(
            approval_required_actions or ()
        )
        if self._approval_required_actions & READ_ONLY_ACTIONS:
            raise PlanningError("Conflicting tool approval policy")
        self._tool_catalog, self._allowed_actions = self._build_tool_catalog(
            tool_definitions
        )
        self._event_emitter = event_emitter or NoOpEventEmitter()
        self._trace_emitter = trace_emitter
        self._clock = clock

    async def create_plan(
        self,
        goal: str,
        *,
        trace_id: str | None = None,
        run_id: str | None = None,
        thread_id: str | None = None,
    ) -> ExecutionPlan:
        await self._emit(
            EventType.PLANNER_STARTED,
            EventOutcome.STARTED,
            trace_id=trace_id,
            run_id=run_id,
            thread_id=thread_id,
        )
        try:
            plan = await self._create_plan(goal)
        except Exception as exc:
            await self._emit_failure(
                exc,
                trace_id=trace_id,
                run_id=run_id,
                thread_id=thread_id,
            )
            raise
        await self._emit(
            EventType.PLANNER_COMPLETED,
            EventOutcome.SUCCEEDED,
            trace_id=trace_id,
            run_id=run_id,
            thread_id=thread_id,
        )
        return plan

    async def _create_plan(self, goal: str) -> ExecutionPlan:
        if not goal.strip():
            raise PlanningError("Planning goal must not be blank")

        system_prompt = PLANNER_SYSTEM_PROMPT
        if self._tool_catalog is not None:
            system_prompt = (
                f"{system_prompt}\n{TOOL_AWARE_PLANNING_RULES}"
                f"{self._tool_catalog}"
            )

        response = await self._llm_service.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": goal},
            ]
        )
        if not response.content:
            raise LLMProviderError("The language model returned no plan")

        try:
            plan_data = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise PlanningError("The language model returned invalid plan JSON") from exc

        try:
            plan = ExecutionPlan.model_validate(plan_data)
        except ValidationError as exc:
            raise PlanningError("The language model returned an invalid plan") from exc

        if self._allowed_actions is not None and any(
            step.action not in self._allowed_actions for step in plan.steps
        ):
            raise UnavailablePlanActionError("Plan contains unavailable action")

        for step in plan.steps:
            if step.action in READ_ONLY_ACTIONS:
                step.requires_approval = False
            elif step.action in self._approval_required_actions:
                step.requires_approval = True
        return plan

    async def _emit_failure(
        self,
        exc: Exception,
        *,
        trace_id: str | None,
        run_id: str | None,
        thread_id: str | None,
    ) -> None:
        try:
            from app.reliability.failures import classify_failure

            failure_category = classify_failure(exc).category
        except Exception:
            return
        await self._emit(
            EventType.PLANNER_FAILED,
            EventOutcome.FAILED,
            trace_id=trace_id,
            run_id=run_id,
            thread_id=thread_id,
            failure_category=failure_category,
        )

    async def _emit(
        self,
        event_type: EventType,
        outcome: EventOutcome,
        *,
        trace_id: str | None = None,
        run_id: str | None = None,
        thread_id: str | None = None,
        failure_category: "FailureCategory | None" = None,
    ) -> None:
        try:
            event = AgentEvent(
                event_type=event_type,
                occurred_at=self._clock(),
                stage=EventStage.PLANNER,
                outcome=outcome,
                run_id=run_id,
                thread_id=thread_id,
                failure_category=failure_category,
            )
        except Exception:
            return
        emit_best_effort(self._event_emitter, event)
        if trace_id is not None and self._trace_emitter is not None:
            await self._trace_emitter.emit(trace_id, event)

    @staticmethod
    def _build_tool_catalog(
        tool_definitions: list[dict[str, Any]] | None,
    ) -> tuple[str | None, set[str] | None]:
        if tool_definitions is None:
            return None, None

        try:
            serialized = json.dumps(
                tool_definitions,
                ensure_ascii=False,
                allow_nan=False,
            )
            definitions = json.loads(serialized)
            catalog: list[dict[str, Any]] = []
            allowed_actions: set[str] = set()
            for definition in definitions:
                function = definition["function"]
                name = function["name"]
                description = function["description"]
                parameters = function["parameters"]
                if (
                    definition.get("type") != "function"
                    or not isinstance(name, str)
                    or not name.strip()
                    or not isinstance(description, str)
                    or not isinstance(parameters, dict)
                ):
                    raise ValueError("malformed tool definition")
                catalog.append(
                    {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    }
                )
                allowed_actions.add(name)
            return json.dumps(catalog, ensure_ascii=False), allowed_actions
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanningError("Invalid tool definition") from exc

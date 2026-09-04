from typing import Awaitable, Callable

from langgraph.types import interrupt

from app.graph.state import AgentState
from app.services.planner_service import PlanningError


ApprovalNode = Callable[[AgentState], Awaitable[dict[str, object]]]


def _build_approval_payload(state: AgentState) -> dict[str, object]:
    plan = state.get("plan")
    if plan is None:
        raise PlanningError("Execution plan is required")

    current_step_index = state.get("current_step_index", 0)
    if current_step_index < 0 or current_step_index >= len(plan.steps):
        raise PlanningError("Current step index is invalid")

    current_step = plan.steps[current_step_index]
    if not current_step.requires_approval:
        raise PlanningError("Current plan step does not require approval")

    serialized_step = current_step.model_dump(mode="json")
    return {
        "type": "plan_step_approval",
        "step_id": serialized_step["id"],
        "description": serialized_step["description"],
        "action": serialized_step["action"],
        "arguments": serialized_step["arguments"],
    }


def create_prepare_approval_node() -> ApprovalNode:
    async def prepare_approval_node(state: AgentState) -> dict[str, object]:
        return {
            "pending_approval": _build_approval_payload(state),
            "approval_decision": None,
        }

    return prepare_approval_node


def create_approval_node() -> ApprovalNode:
    async def approval_node(state: AgentState) -> dict[str, object]:
        pending_approval = state.get("pending_approval")
        if not isinstance(pending_approval, dict):
            raise PlanningError("Pending approval is required")

        resume_value = interrupt(pending_approval)

        if not isinstance(resume_value, dict):
            raise PlanningError("Approval response must be an object")
        decision = resume_value.get("decision")
        if decision not in {"approve", "reject"}:
            raise PlanningError("Approval decision must be approve or reject")

        return {
            "pending_approval": None,
            "approval_decision": decision,
            "route": decision,
        }

    return approval_node

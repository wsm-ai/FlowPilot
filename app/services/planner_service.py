import json

from pydantic import ValidationError

from app.providers.base import LLMProviderError
from app.schemas.planning import ExecutionPlan
from app.services.llm_service import LLMService


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


class PlanningError(Exception):
    """Raised when an LLM-generated execution plan is invalid."""


class PlannerService:
    def __init__(self, llm_service: LLMService) -> None:
        self._llm_service = llm_service

    async def create_plan(self, goal: str) -> ExecutionPlan:
        if not goal.strip():
            raise PlanningError("Planning goal must not be blank")

        response = await self._llm_service.complete(
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
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
            return ExecutionPlan.model_validate(plan_data)
        except ValidationError as exc:
            raise PlanningError("The language model returned an invalid plan") from exc

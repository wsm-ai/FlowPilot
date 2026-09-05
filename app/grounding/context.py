from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from app.grounding.models import GroundingEvidence


@dataclass(frozen=True, slots=True)
class GroundingContextPolicy:
    max_evidence_items: int = 8
    max_chars_per_evidence: int = 2_000
    max_total_evidence_chars: int = 8_000
    max_chars_per_workflow_result: int = 2_000
    max_total_workflow_result_chars: int = 6_000

    def __post_init__(self) -> None:
        if any(
            value < 1
            for value in (
                self.max_evidence_items,
                self.max_chars_per_evidence,
                self.max_total_evidence_chars,
                self.max_chars_per_workflow_result,
                self.max_total_workflow_result_chars,
            )
        ):
            raise ValueError("Grounding context limits must be at least 1")


@dataclass(frozen=True, slots=True)
class GroundingContext:
    selected_evidence: list[GroundingEvidence]
    prompt_evidence: list[dict[str, Any]]
    workflow_results: list[dict[str, Any]]
    has_workflow_support: bool


def build_grounding_context(
    evidence: list[GroundingEvidence],
    step_results: list[dict[str, Any]],
    policy: GroundingContextPolicy,
) -> GroundingContext:
    selected_evidence: list[GroundingEvidence] = []
    prompt_evidence: list[dict[str, Any]] = []
    evidence_chars_remaining = policy.max_total_evidence_chars

    for item in evidence[: policy.max_evidence_items]:
        if evidence_chars_remaining == 0:
            break
        allowed = min(
            len(item.content),
            policy.max_chars_per_evidence,
            evidence_chars_remaining,
        )
        content = item.content[:allowed]
        title = item.metadata.get("document_title")
        prompt_item: dict[str, Any] = {
            "citation_id": item.citation_id,
            "chunk_id": item.chunk_id,
            "document_id": item.document_id,
            "source": item.source,
            "content": content,
            "content_truncated": allowed < len(item.content),
            "rank": item.rank,
            "position": item.position,
        }
        if isinstance(title, str):
            prompt_item["title"] = title
        selected_evidence.append(item)
        prompt_evidence.append(prompt_item)
        evidence_chars_remaining -= allowed

    visible_ids = {item.citation_id for item in selected_evidence}
    ids_by_chunk = {
        item.chunk_id: item.citation_id
        for item in evidence
        if item.citation_id in visible_ids
    }
    workflow_results: list[dict[str, Any]] = []
    workflow_chars_remaining = policy.max_total_workflow_result_chars
    has_workflow_support = False

    for step in step_results:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if action == "search_knowledge_base":
            result = step.get("result", [])
            citation_ids = [
                ids_by_chunk[item["chunk_id"]]
                for item in result
                if isinstance(item, dict) and item.get("chunk_id") in ids_by_chunk
            ]
            workflow_results.append(
                {
                    "step_id": step.get("step_id"),
                    "action": action,
                    "citation_ids": list(dict.fromkeys(citation_ids)),
                }
            )
            continue

        compact_step: dict[str, Any] = {
            "step_id": step.get("step_id"),
            "action": action,
        }
        result = step.get("result")
        if not _has_meaningful_result(result):
            compact_step["result"] = deepcopy(result)
            workflow_results.append(compact_step)
            continue

        serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
        allowed = min(
            len(serialized),
            policy.max_chars_per_workflow_result,
            workflow_chars_remaining,
        )
        if allowed == len(serialized):
            compact_step["result"] = deepcopy(result)
        else:
            compact_step["result_preview"] = serialized[:allowed]
            compact_step["result_truncated"] = True
        if allowed > 0:
            has_workflow_support = True
            workflow_chars_remaining -= allowed
        workflow_results.append(compact_step)

    return GroundingContext(
        selected_evidence=selected_evidence,
        prompt_evidence=prompt_evidence,
        workflow_results=workflow_results,
        has_workflow_support=has_workflow_support,
    )


def _has_meaningful_result(result: Any) -> bool:
    if result is None:
        return False
    if isinstance(result, str):
        return bool(result.strip())
    if isinstance(result, (list, dict)):
        return bool(result)
    return isinstance(result, (bool, int, float))

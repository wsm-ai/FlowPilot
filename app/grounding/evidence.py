from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from app.grounding.models import GroundingEvidence


class EvidenceExtractionError(Exception):
    """Raised when recorded knowledge evidence is malformed."""


def extract_grounding_evidence(
    step_results: list[dict[str, Any]],
) -> list[GroundingEvidence]:
    evidence: list[GroundingEvidence] = []
    seen_chunk_ids: set[str] = set()

    for step in step_results:
        if not isinstance(step, dict) or step.get("action") != "search_knowledge_base":
            continue

        result = step.get("result")
        if not isinstance(result, list):
            raise EvidenceExtractionError("Invalid knowledge evidence")

        for item in result:
            if not isinstance(item, dict):
                raise EvidenceExtractionError("Invalid knowledge evidence")
            try:
                chunk_id = item["chunk_id"]
                if not isinstance(chunk_id, str):
                    raise TypeError("chunk_id must be a string")
                if chunk_id in seen_chunk_ids:
                    continue
                record = GroundingEvidence(
                    citation_id=f"E{len(evidence) + 1}",
                    chunk_id=chunk_id,
                    document_id=item["document_id"],
                    content=item["content"],
                    source=item["source"],
                    score=item["score"],
                    rank=item.get("rank"),
                    position=item.get("position"),
                    metadata=deepcopy(item.get("metadata", {})),
                )
            except (KeyError, TypeError, ValidationError) as exc:
                raise EvidenceExtractionError("Invalid knowledge evidence") from exc
            seen_chunk_ids.add(chunk_id)
            evidence.append(record)

    return evidence

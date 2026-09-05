import json
import math

import pytest

from app.grounding.evidence import EvidenceExtractionError, extract_grounding_evidence


def item(chunk_id: str, **changes):
    value = {
        "chunk_id": chunk_id,
        "document_id": f"DOC-{chunk_id}",
        "content": f"Evidence {chunk_id}",
        "source": "handbook",
        "score": 0.9,
        "rank": 1,
        "position": 0,
        "metadata": {"nested": {"value": "original"}},
    }
    value.update(changes)
    return value


def kb_step(*records):
    return {"step_id": 1, "action": "search_knowledge_base", "result": list(records)}


def test_extracts_only_kb_results_with_stable_global_order():
    results = [
        {"step_id": 1, "action": "get_customer_feedback", "result": [item("IGNORED")]},
        kb_step(item("CH-1"), item("CH-2")),
        {"step_id": 3, "action": "search_knowledge_base", "result": [item("CH-3")]},
    ]

    evidence = extract_grounding_evidence(results)

    assert [record.citation_id for record in evidence] == ["E1", "E2", "E3"]
    assert [record.chunk_id for record in evidence] == ["CH-1", "CH-2", "CH-3"]
    json.dumps([record.model_dump() for record in evidence], allow_nan=False)


def test_duplicate_chunk_uses_first_occurrence_and_empty_results_are_valid():
    evidence = extract_grounding_evidence(
        [kb_step(item("CH-1", source="first")), kb_step(), kb_step(item("CH-1", source="second"))]
    )
    assert len(evidence) == 1
    assert evidence[0].citation_id == "E1"
    assert evidence[0].source == "first"
    assert extract_grounding_evidence([kb_step()]) == []


@pytest.mark.parametrize(
    "result",
    [
        "not-a-list",
        ["not-an-object"],
        [{"chunk_id": "CH-1"}],
        [item("CH-1", score=math.nan)],
        [item("CH-1", score=math.inf)],
    ],
)
def test_rejects_malformed_knowledge_evidence(result):
    with pytest.raises(EvidenceExtractionError, match="Invalid knowledge evidence"):
        extract_grounding_evidence(
            [{"step_id": 1, "action": "search_knowledge_base", "result": result}]
        )


def test_metadata_is_deep_copied_from_step_results():
    original = item("CH-1")
    extracted = extract_grounding_evidence([kb_step(original)])
    extracted[0].metadata["nested"]["value"] = "changed"
    assert original["metadata"]["nested"]["value"] == "original"

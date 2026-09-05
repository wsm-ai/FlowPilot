import math

import pytest
from pydantic import ValidationError

from app.grounding.models import (
    Citation,
    GroundedAnswer,
    GroundedSynthesisResponse,
    GroundingEvidence,
)


def evidence(**changes):
    values = {
        "citation_id": "E1",
        "chunk_id": "CH-1",
        "document_id": "DOC-1",
        "content": "Useful evidence",
        "source": "handbook",
        "score": 0.8,
        "rank": 1,
        "position": 0,
        "metadata": {"document_title": "Support Guide"},
    }
    values.update(changes)
    return GroundingEvidence(**values)


def test_grounding_evidence_accepts_valid_and_unbounded_finite_scores():
    assert evidence(score=-0.5).score == -0.5
    assert evidence(score=1.5).score == 1.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("citation_id", " "),
        ("chunk_id", ""),
        ("document_id", "   "),
        ("content", "\t"),
        ("source", "\n"),
        ("rank", 0),
        ("position", -1),
        ("score", math.nan),
        ("score", math.inf),
        ("score", -math.inf),
        ("metadata", {"bad": {1, 2}}),
    ],
)
def test_grounding_evidence_rejects_invalid_values(field, value):
    with pytest.raises(ValidationError):
        evidence(**{field: value})


def test_citation_validates_required_strings_without_content_duplication():
    citation = Citation(
        citation_id="E1", chunk_id="CH-1", document_id="DOC-1", source="guide"
    )
    assert "content" not in citation.model_dump()

    with pytest.raises(ValidationError):
        Citation(citation_id=" ", chunk_id="CH-1", document_id="DOC-1", source="guide")


def test_grounded_answer_rejects_blank_answer():
    assert GroundedAnswer(answer="Supported answer").citations == []
    with pytest.raises(ValidationError):
        GroundedAnswer(answer="   ")


def test_synthesis_response_forbids_extra_fields_and_blank_citation_ids():
    with pytest.raises(ValidationError):
        GroundedSynthesisResponse(answer="Answer", citation_ids=[], source="invented")
    with pytest.raises(ValidationError):
        GroundedSynthesisResponse(answer="Answer", citation_ids=[" "])

from copy import deepcopy
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.retrieval.base import RetrievalError, Retriever
from app.retrieval.embeddings import EmbeddingError
from app.retrieval.models import RetrievalQuery
from app.retrieval.vector_store import VectorStoreError
from app.tools.base import ToolExecutionError


DEFAULT_MAX_KNOWLEDGE_OUTPUT_CHARS = 50_000


class KnowledgeBaseSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized

    @field_validator("filters")
    @classmethod
    def filters_must_be_json_safe(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("filters must be JSON-compatible") from exc
        return value


class KnowledgeBaseTool:
    name = "search_knowledge_base"
    description = (
        "Search the internal knowledge base for evidence relevant to a question "
        "or workflow step. Returns ranked knowledge chunks with source metadata."
    )

    def __init__(
        self,
        retriever: Retriever,
        *,
        max_output_chars: int = DEFAULT_MAX_KNOWLEDGE_OUTPUT_CHARS,
    ) -> None:
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be at least 1")
        self._retriever = retriever
        self._max_output_chars = max_output_chars

    @property
    def parameters(self) -> dict[str, Any]:
        return KnowledgeBaseSearchArguments.model_json_schema()

    async def execute(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            validated = KnowledgeBaseSearchArguments.model_validate(arguments)
        except ValidationError as exc:
            raise ToolExecutionError(
                "Invalid knowledge base search arguments"
            ) from exc

        query = RetrievalQuery(
            query=validated.query,
            top_k=validated.top_k,
            filters=validated.filters,
        )
        try:
            results = await self._retriever.search(query)
        except (RetrievalError, EmbeddingError, VectorStoreError) as exc:
            raise ToolExecutionError("Knowledge base search failed") from exc

        output = [
            {
                "chunk_id": result.chunk.chunk_id,
                "document_id": result.chunk.document_id,
                "content": result.chunk.content,
                "source": result.chunk.source,
                "score": result.score,
                "rank": result.rank,
                "position": result.chunk.position,
                "metadata": deepcopy(result.chunk.metadata),
            }
            for result in results[: validated.top_k]
        ]
        try:
            serialized = json.dumps(output, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError(
                "Knowledge base search returned invalid data"
            ) from exc
        if len(serialized) > self._max_output_chars:
            raise ToolExecutionError(
                "Knowledge base search result is too large"
            )
        return output

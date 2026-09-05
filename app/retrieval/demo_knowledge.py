from app.retrieval.indexing import KnowledgeBaseIndexer
from app.retrieval.models import KnowledgeChunk, KnowledgeDocument


# DEMO KNOWLEDGE: deterministic local documents used until an ingestion adapter
# and production knowledge source are introduced.
DEMO_KNOWLEDGE_DOCUMENTS: tuple[KnowledgeDocument, ...] = (
    KnowledgeDocument(
        document_id="DOC-001",
        title="Enterprise Login Support",
        source="support-handbook",
        content=(
            "Enterprise login troubleshooting requires checking account status, "
            "single sign-on configuration, and authentication service health."
        ),
        metadata={"department": "support", "category": "authentication"},
    ),
    KnowledgeDocument(
        document_id="DOC-002",
        title="Billing Operations Guide",
        source="billing-guide",
        content=(
            "Billing invoices can be downloaded from account settings after the "
            "monthly billing cycle has completed."
        ),
        metadata={"department": "finance", "category": "billing"},
    ),
    KnowledgeDocument(
        document_id="DOC-003",
        title="Bug Escalation Policy",
        source="engineering-runbook",
        content=(
            "Critical production bugs must include reproduction steps, customer "
            "impact, severity, and available diagnostic evidence before escalation."
        ),
        metadata={"department": "engineering", "category": "incident"},
    ),
)


async def bootstrap_demo_knowledge_base(
    indexer: KnowledgeBaseIndexer,
) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for document in DEMO_KNOWLEDGE_DOCUMENTS:
        chunks.extend(await indexer.index_document(document))
    return chunks

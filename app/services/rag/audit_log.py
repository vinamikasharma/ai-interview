"""Retrieval audit logging for RAG.

Persists every retrieval (query, retrieved chunk ids, similarity scores) via the
existing MySQLService — MySQL in prod, SQLite fallback locally — so the audit
trail rides the same durable store as the rest of the app. Failures here never
break a retrieval: auditing is best-effort and logged, not raised.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import List, Sequence

from loguru import logger

from app.core.config import settings
from app.services.rag.vector_store import RetrievedChunk


_SCHEMA_READY = False
_FOLLOWUP_SCHEMA_READY = False


def _redact_query(query_text: str) -> str:
    """PII-safe representation of a retrieval query.

    Resume/JD/answer queries can carry names, contact info, and employers. We must
    never persist that raw text to the audit trail. Instead we store a stable
    SHA-256 hash (lets us correlate identical queries and detect replays) plus the
    length — enough for debugging, zero plaintext PII.
    """
    text = query_text or ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}:len={len(text)}"



def _ensure_table() -> bool:
    """Create the audit table once per process. Returns False if the DB is down."""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return True
    try:
        from app.services.mysql_service import get_mysql

        session = get_mysql().get_session()
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_retrieval_audit (
                id CHAR(36) PRIMARY KEY,
                created_at DATETIME,
                candidate_id VARCHAR(255),
                role VARCHAR(255),
                operation VARCHAR(64),
                query_text LONGTEXT,
                retrieved_json LONGTEXT
            )
            """
        )
        _SCHEMA_READY = True
        return True
    except Exception as exc:
        logger.warning(f"RAG audit table unavailable, skipping audit: {exc}")
        return False


def _ensure_followup_table() -> bool:
    """Create the interviewer-coaching audit table once per process."""
    global _FOLLOWUP_SCHEMA_READY
    if _FOLLOWUP_SCHEMA_READY:
        return True
    try:
        from app.services.mysql_service import get_mysql

        get_mysql().get_session().execute(
            """
            CREATE TABLE IF NOT EXISTS rag_followup_audit (
                id CHAR(36) PRIMARY KEY,
                created_at DATETIME,
                session_id VARCHAR(255),
                candidate_id VARCHAR(255),
                operation VARCHAR(64),
                weak_topics_json LONGTEXT,
                retrieved_chunk_ids_json LONGTEXT
            )
            """
        )
        _FOLLOWUP_SCHEMA_READY = True
        return True
    except Exception as exc:
        logger.warning(f"RAG follow-up audit table unavailable, skipping audit: {exc}")
        return False


def log_followup(
    session_id: str,
    candidate_id: str,
    weak_topics: Sequence[dict],
    retrieved_chunk_ids: Sequence[str],
) -> None:
    """Audit coaching generation without persisting answer/question plaintext."""
    if not settings.RAG_AUDIT_ENABLED or not _ensure_followup_table():
        return
    try:
        from app.services.mysql_service import get_mysql

        topic_payload = [
            {
                "topic_hash": _redact_query(str(item.get("topic", ""))),
                "question_hash": _redact_query(str(item.get("question_asked", ""))),
                "answer_summary_hash": _redact_query(
                    str(item.get("candidate_answer_summary", ""))
                ),
                "score": item.get("lowest_score"),
            }
            for item in weak_topics
        ]
        get_mysql().get_session().execute(
            """
            INSERT INTO rag_followup_audit
                (id, created_at, session_id, candidate_id, operation,
                 weak_topics_json, retrieved_chunk_ids_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                datetime.now(timezone.utc),
                session_id[:255],
                candidate_id[:255],
                "suggest_followup",
                json.dumps(topic_payload, ensure_ascii=False),
                json.dumps(list(retrieved_chunk_ids), ensure_ascii=False),
            ),
        )
    except Exception as exc:
        logger.warning(f"RAG follow-up audit write failed (non-fatal): {exc}")


def log_retrieval(
    candidate_id: str,
    role: str,
    operation: str,
    query_text: str,
    retrieved: Sequence[RetrievedChunk],
) -> None:
    """Best-effort audit write. Never raises into the retrieval path."""
    if not settings.RAG_AUDIT_ENABLED:
        return
    if not _ensure_table():
        return
    try:
        from app.services.mysql_service import get_mysql

        retrieved_payload: List[dict] = [
            {
                "chunk_id": rc.chunk.chunk_id,
                "source_type": rc.chunk.source_type,
                "distance": rc.distance,
                "similarity": rc.similarity,
            }
            for rc in retrieved
        ]
        session = get_mysql().get_session()
        session.execute(
            """
            INSERT INTO rag_retrieval_audit
                (id, created_at, candidate_id, role, operation, query_text, retrieved_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                datetime.now(timezone.utc),
                candidate_id[:255],
                role[:255],
                operation[:64],
                # PII redaction (item 13): store a hash of the query, not the raw
                # resume/answer text. Chunk provenance is retained as ids below.
                _redact_query(query_text),
                json.dumps(retrieved_payload, ensure_ascii=False),
            ),
        )
    except Exception as exc:
        logger.warning(f"RAG audit write failed (non-fatal): {exc}")

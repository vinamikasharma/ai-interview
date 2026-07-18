"""Persistence for live-session RAG evaluations used by interviewer coaching.

The repository has no Cassandra client or model layer. This module deliberately
reuses the existing MySQLService (with its SQLite development fallback), matching
RAG audit persistence. Raw candidate answers are never stored here; only a
bounded summary and SHA-256 fingerprint are retained.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger


_SCHEMA_READY = False


def _answer_fingerprint(answer: str) -> str:
    text = answer or ""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}:len={len(text)}"


def _ensure_table() -> bool:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return True
    try:
        from app.services.mysql_service import get_mysql

        get_mysql().get_session().execute(
            """
            CREATE TABLE IF NOT EXISTS rag_answer_evaluations (
                id CHAR(36) PRIMARY KEY,
                created_at DATETIME,
                session_id VARCHAR(255),
                candidate_id VARCHAR(255),
                company_id VARCHAR(255),
                role VARCHAR(255),
                topic VARCHAR(255),
                question_asked LONGTEXT,
                candidate_answer_summary VARCHAR(100),
                candidate_answer_hash VARCHAR(96),
                score DOUBLE,
                unscored BOOLEAN
            )
            """
        )
        _SCHEMA_READY = True
        return True
    except Exception as exc:
        logger.warning(f"RAG evaluation storage unavailable: {exc}")
        return False


def record_evaluation(
    *,
    session_id: str,
    candidate_id: str,
    company_id: str,
    role: str,
    topic: str,
    question_asked: str,
    candidate_answer: str,
    candidate_answer_summary: str,
    score: Optional[float],
    unscored: bool,
) -> bool:
    """Persist one evaluation without retaining the raw candidate answer."""
    if not _ensure_table():
        return False
    try:
        from app.services.mysql_service import get_mysql

        get_mysql().get_session().execute(
            """
            INSERT INTO rag_answer_evaluations
                (id, created_at, session_id, candidate_id, company_id, role, topic,
                 question_asked, candidate_answer_summary, candidate_answer_hash,
                 score, unscored)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                datetime.now(timezone.utc),
                session_id[:255],
                candidate_id[:255],
                (company_id or "default")[:255],
                role[:255],
                topic[:255],
                question_asked[:2000],
                candidate_answer_summary[:100],
                _answer_fingerprint(candidate_answer),
                score,
                bool(unscored),
            ),
        )
        return True
    except Exception as exc:
        logger.warning(f"RAG evaluation write failed (non-fatal): {exc}")
        return False


def list_session_evaluations(
    *,
    session_id: str,
    candidate_id: str,
    company_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return only rows in the exact candidate/session/(optional company) scope."""
    if not _ensure_table():
        return []
    try:
        from app.services.mysql_service import get_mysql

        session = get_mysql().get_session()
        if company_id:
            rows = session.execute(
                """
                SELECT created_at, role, topic, question_asked,
                       candidate_answer_summary, score, unscored, company_id
                FROM rag_answer_evaluations
                WHERE session_id=%s AND candidate_id=%s AND company_id=%s
                ORDER BY created_at ASC
                """,
                (session_id, candidate_id, company_id),
            )
        else:
            rows = session.execute(
                """
                SELECT created_at, role, topic, question_asked,
                       candidate_answer_summary, score, unscored, company_id
                FROM rag_answer_evaluations
                WHERE session_id=%s AND candidate_id=%s
                ORDER BY created_at ASC
                """,
                (session_id, candidate_id),
            )
        return [
            {
                "created_at": row.created_at,
                "role": row.role,
                "topic": row.topic,
                "question_asked": row.question_asked,
                "candidate_answer_summary": row.candidate_answer_summary,
                "score": row.score,
                "unscored": bool(row.unscored),
                "company_id": row.company_id,
            }
            for row in rows
        ]
    except Exception as exc:
        logger.warning(f"RAG evaluation lookup failed (non-fatal): {exc}")
        return []

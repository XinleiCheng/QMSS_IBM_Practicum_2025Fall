"""SQLite persistence for request history and operational traceability."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class InteractionRepository:
    """Persist API interactions without coupling services to a web framework."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def record(
        self,
        *,
        kind: str,
        request: dict[str, Any],
        response: dict[str, Any] | None,
        status: str,
        error: str | None = None,
    ) -> str:
        interaction_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO interactions (
                    id, kind, request_json, response_json, status, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interaction_id,
                    kind,
                    json.dumps(request, ensure_ascii=False),
                    (
                        json.dumps(response, ensure_ascii=False)
                        if response is not None
                        else None
                    ),
                    status,
                    error,
                    created_at,
                ),
            )
        return interaction_id

    def get(self, interaction_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, kind, request_json, response_json, status, error, created_at
                FROM interactions
                WHERE id = ?
                """,
                (interaction_id,),
            ).fetchone()
        return self._deserialize(row) if row else None

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, kind, request_json, response_json, status, error, created_at
                FROM interactions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._deserialize(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL CHECK (kind IN ('qna', 'draft')),
                    request_json TEXT NOT NULL,
                    response_json TEXT,
                    status TEXT NOT NULL CHECK (
                        status IN ('succeeded', 'refused', 'failed')
                    ),
                    error TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS interactions_created_at_idx
                ON interactions(created_at DESC)
                """
            )

    @staticmethod
    def _deserialize(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "kind": row["kind"],
            "request": json.loads(row["request_json"]),
            "response": (
                json.loads(row["response_json"])
                if row["response_json"] is not None
                else None
            ),
            "status": row["status"],
            "error": row["error"],
            "created_at": row["created_at"],
        }

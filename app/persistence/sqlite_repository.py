import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from app.persistence.models import AgentRunRecord, RunMode, RunStatus
from app.persistence.repository import PersistenceError


VALID_RUN_MODES = {"reactive", "planned"}
VALID_RUN_STATUSES = {
    "running",
    "completed",
    "approval_required",
    "rejected",
    "failed",
}


class SQLiteRunRepository:
    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)

    async def initialize(self) -> None:
        try:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self._database_path) as database:
                await database.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_runs (
                        run_id TEXT PRIMARY KEY,
                        mode TEXT NOT NULL,
                        input_text TEXT NOT NULL,
                        status TEXT NOT NULL,
                        result_json TEXT NULL,
                        error_type TEXT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        thread_id TEXT NULL
                    )
                    """
                )
                cursor = await database.execute("PRAGMA table_info(agent_runs)")
                columns = {row[1] for row in await cursor.fetchall()}
                if "thread_id" not in columns:
                    await database.execute(
                        "ALTER TABLE agent_runs ADD COLUMN thread_id TEXT"
                    )
                await database.commit()
        except (OSError, sqlite3.Error) as exc:
            raise PersistenceError("Unable to initialize run persistence") from exc

    async def create(self, record: AgentRunRecord) -> None:
        self._validate_record(record)
        result_json = self._serialize_result(record.result)
        try:
            async with aiosqlite.connect(self._database_path) as database:
                await database.execute(
                    """
                    INSERT INTO agent_runs (
                        run_id, mode, input_text, status, result_json,
                        error_type, created_at, updated_at, thread_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.run_id,
                        record.mode,
                        record.input_text,
                        record.status,
                        result_json,
                        record.error_type,
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                        record.thread_id,
                    ),
                )
                await database.commit()
        except sqlite3.Error as exc:
            raise PersistenceError("Unable to create agent run") from exc

    async def get(self, run_id: str) -> AgentRunRecord | None:
        try:
            async with aiosqlite.connect(self._database_path) as database:
                database.row_factory = aiosqlite.Row
                cursor = await database.execute(
                    "SELECT * FROM agent_runs WHERE run_id = ?",
                    (run_id,),
                )
                row = await cursor.fetchone()
        except sqlite3.Error as exc:
            raise PersistenceError("Unable to read agent run") from exc

        return None if row is None else self._row_to_record(row)

    async def update(
        self,
        run_id: str,
        *,
        status: RunStatus,
        result: dict[str, Any] | None = None,
        error_type: str | None = None,
    ) -> AgentRunRecord:
        if status not in VALID_RUN_STATUSES:
            raise PersistenceError("Invalid agent run status")
        result_json = self._serialize_result(result)
        updated_at = datetime.now(timezone.utc)

        try:
            async with aiosqlite.connect(self._database_path) as database:
                database.row_factory = aiosqlite.Row
                cursor = await database.execute(
                    "SELECT * FROM agent_runs WHERE run_id = ?",
                    (run_id,),
                )
                existing = await cursor.fetchone()
                if existing is None:
                    raise PersistenceError("Agent run does not exist")

                await database.execute(
                    """
                    UPDATE agent_runs
                    SET status = ?, result_json = ?, error_type = ?, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (
                        status,
                        result_json,
                        error_type,
                        updated_at.isoformat(),
                        run_id,
                    ),
                )
                await database.commit()
        except sqlite3.Error as exc:
            raise PersistenceError("Unable to update agent run") from exc

        return AgentRunRecord(
            run_id=existing["run_id"],
            mode=existing["mode"],
            input_text=existing["input_text"],
            status=status,
            result=result,
            error_type=error_type,
            created_at=self._parse_datetime(existing["created_at"]),
            updated_at=updated_at,
            thread_id=existing["thread_id"],
        )

    async def list_recent(self, limit: int = 20) -> list[AgentRunRecord]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        try:
            async with aiosqlite.connect(self._database_path) as database:
                database.row_factory = aiosqlite.Row
                cursor = await database.execute(
                    """
                    SELECT * FROM agent_runs
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                rows = await cursor.fetchall()
        except sqlite3.Error as exc:
            raise PersistenceError("Unable to list agent runs") from exc

        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _serialize_result(result: dict[str, Any] | None) -> str | None:
        if result is None:
            return None
        try:
            return json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise PersistenceError("Agent run result is not JSON serializable") from exc

    @classmethod
    def _row_to_record(cls, row: aiosqlite.Row) -> AgentRunRecord:
        try:
            mode = row["mode"]
            status = row["status"]
            if mode not in VALID_RUN_MODES or status not in VALID_RUN_STATUSES:
                raise ValueError("invalid mode or status")

            result = (
                None if row["result_json"] is None else json.loads(row["result_json"])
            )
            if result is not None and not isinstance(result, dict):
                raise ValueError("result must be an object")

            return AgentRunRecord(
                run_id=row["run_id"],
                mode=mode,
                input_text=row["input_text"],
                status=status,
                result=result,
                error_type=row["error_type"],
                created_at=cls._parse_datetime(row["created_at"]),
                updated_at=cls._parse_datetime(row["updated_at"]),
                thread_id=row["thread_id"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PersistenceError("Stored agent run is invalid") from exc

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("stored datetime must include a timezone")
        return parsed

    @staticmethod
    def _validate_record(record: AgentRunRecord) -> None:
        if record.mode not in VALID_RUN_MODES or record.status not in VALID_RUN_STATUSES:
            raise PersistenceError("Invalid agent run record")
        if record.result is not None and not isinstance(record.result, dict):
            raise PersistenceError("Agent run result must be an object")
        for value in (record.created_at, record.updated_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise PersistenceError("Agent run timestamps must include a timezone")

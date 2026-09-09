import json
from copy import deepcopy
from pathlib import Path
import sqlite3
from typing import Any

import aiosqlite

from app.persistence.repository import PersistenceError
from app.reliability.side_effects import (
    SideEffectExecutionRecord,
    SideEffectExecutionStatus,
    SideEffectReservation,
)


class SQLiteSideEffectExecutionRepository:
    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)

    async def initialize(self) -> None:
        try:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self._database_path) as database:
                await database.execute(
                    """
                    CREATE TABLE IF NOT EXISTS side_effect_executions (
                        run_id TEXT NOT NULL,
                        step_id INTEGER NOT NULL,
                        action TEXT NOT NULL,
                        arguments_digest TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN ('started', 'completed', 'ambiguous')
                        ),
                        result_json TEXT NULL,
                        PRIMARY KEY (run_id, step_id, action)
                    )
                    """
                )
                await database.commit()
        except (OSError, sqlite3.Error) as exc:
            raise PersistenceError(
                "Unable to initialize side-effect persistence"
            ) from exc

    async def get(
        self, run_id: str, step_id: int, action: str
    ) -> SideEffectExecutionRecord | None:
        self._validate_identity(run_id, step_id, action)
        try:
            async with aiosqlite.connect(self._database_path) as database:
                database.row_factory = aiosqlite.Row
                cursor = await database.execute(
                    """
                    SELECT * FROM side_effect_executions
                    WHERE run_id = ? AND step_id = ? AND action = ?
                    """,
                    (run_id, step_id, action),
                )
                row = await cursor.fetchone()
        except sqlite3.Error as exc:
            raise PersistenceError(
                "Unable to read side-effect execution"
            ) from exc
        return None if row is None else self._row_to_record(row)

    async def reserve_started(
        self,
        *,
        run_id: str,
        step_id: int,
        action: str,
        arguments_digest: str,
    ) -> SideEffectReservation:
        self._validate_identity(run_id, step_id, action)
        self._validate_digest(arguments_digest)
        try:
            async with aiosqlite.connect(self._database_path) as database:
                database.row_factory = aiosqlite.Row
                cursor = await database.execute(
                    """
                    INSERT OR IGNORE INTO side_effect_executions (
                        run_id, step_id, action, arguments_digest, status, result_json
                    ) VALUES (?, ?, ?, ?, 'started', NULL)
                    """,
                    (run_id, step_id, action, arguments_digest),
                )
                created = cursor.rowcount == 1
                await database.commit()
                cursor = await database.execute(
                    """
                    SELECT * FROM side_effect_executions
                    WHERE run_id = ? AND step_id = ? AND action = ?
                    """,
                    (run_id, step_id, action),
                )
                row = await cursor.fetchone()
        except sqlite3.Error as exc:
            raise PersistenceError(
                "Unable to reserve side-effect execution"
            ) from exc
        if row is None:
            raise PersistenceError("Unable to reserve side-effect execution")
        return SideEffectReservation(self._row_to_record(row), created)

    async def mark_completed(
        self,
        *,
        run_id: str,
        step_id: int,
        action: str,
        arguments_digest: str,
        result: Any,
    ) -> SideEffectExecutionRecord:
        result_json = self._serialize_result(result)
        return await self._transition(
            run_id=run_id,
            step_id=step_id,
            action=action,
            arguments_digest=arguments_digest,
            status=SideEffectExecutionStatus.COMPLETED,
            result_json=result_json,
        )

    async def mark_ambiguous(
        self,
        *,
        run_id: str,
        step_id: int,
        action: str,
        arguments_digest: str,
    ) -> SideEffectExecutionRecord:
        return await self._transition(
            run_id=run_id,
            step_id=step_id,
            action=action,
            arguments_digest=arguments_digest,
            status=SideEffectExecutionStatus.AMBIGUOUS,
            result_json=None,
        )

    async def _transition(
        self,
        *,
        run_id: str,
        step_id: int,
        action: str,
        arguments_digest: str,
        status: SideEffectExecutionStatus,
        result_json: str | None,
    ) -> SideEffectExecutionRecord:
        self._validate_identity(run_id, step_id, action)
        self._validate_digest(arguments_digest)
        try:
            async with aiosqlite.connect(self._database_path) as database:
                cursor = await database.execute(
                    """
                    UPDATE side_effect_executions
                    SET status = ?, result_json = ?
                    WHERE run_id = ? AND step_id = ? AND action = ?
                      AND arguments_digest = ? AND status = 'started'
                    """,
                    (
                        status.value,
                        result_json,
                        run_id,
                        step_id,
                        action,
                        arguments_digest,
                    ),
                )
                if cursor.rowcount != 1:
                    raise PersistenceError(
                        "Unable to update side-effect execution"
                    )
                await database.commit()
        except sqlite3.Error as exc:
            raise PersistenceError(
                "Unable to update side-effect execution"
            ) from exc
        record = await self.get(run_id, step_id, action)
        if record is None:
            raise PersistenceError("Unable to update side-effect execution")
        return record

    @staticmethod
    def _serialize_result(result: Any) -> str:
        try:
            return json.dumps(result, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise PersistenceError(
                "Side-effect result is not JSON serializable"
            ) from exc

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> SideEffectExecutionRecord:
        try:
            status = SideEffectExecutionStatus(row["status"])
            result = (
                None
                if row["result_json"] is None
                else json.loads(row["result_json"])
            )
            if status is not SideEffectExecutionStatus.COMPLETED and result is not None:
                raise ValueError("result on incomplete execution")
            return SideEffectExecutionRecord(
                run_id=row["run_id"],
                step_id=row["step_id"],
                action=row["action"],
                arguments_digest=row["arguments_digest"],
                status=status,
                result=deepcopy(result),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PersistenceError("Stored side-effect execution is invalid") from exc

    @staticmethod
    def _validate_identity(run_id: str, step_id: int, action: str) -> None:
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
            or not isinstance(step_id, int)
            or isinstance(step_id, bool)
            or step_id < 1
            or not isinstance(action, str)
            or not action.strip()
        ):
            raise PersistenceError("Invalid side-effect execution identity")

    @staticmethod
    def _validate_digest(arguments_digest: str) -> None:
        if not isinstance(arguments_digest, str) or not arguments_digest:
            raise PersistenceError("Invalid side-effect arguments digest")

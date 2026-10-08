import asyncio
from datetime import datetime
from pathlib import Path
import sqlite3

import aiosqlite

from app.observability.models import (
    AgentEvent,
    EventOutcome,
    EventStage,
    EventType,
)
from app.observability.trace import (
    ExecutionTrace,
    TraceConfigurationError,
    TraceRecord,
)
from app.persistence.repository import PersistenceError
from app.reliability.failures import FailureCategory


SQLITE_BUSY_TIMEOUT_SECONDS = 5.0


class SQLiteTraceRepository:
    """Durable, append-only storage for safe structured trace records."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)

    async def initialize(self) -> None:
        try:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            async with self._connect() as database:
                await database.execute(
                    """
                    CREATE TABLE IF NOT EXISTS execution_trace_records (
                        trace_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL CHECK (sequence > 0),
                        event_type TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        occurred_at TEXT NOT NULL,
                        run_id TEXT NULL,
                        thread_id TEXT NULL,
                        failure_category TEXT NULL,
                        PRIMARY KEY (trace_id, sequence)
                    )
                    """
                )
                await database.commit()
        except (OSError, sqlite3.Error) as exc:
            raise PersistenceError(
                "Unable to initialize trace persistence"
            ) from exc

    async def append_event(
        self,
        trace_id: str,
        event: AgentEvent,
    ) -> TraceRecord:
        normalized = self._validate_input(trace_id, event)
        try:
            async with self._connect() as database:
                await database.execute("BEGIN IMMEDIATE")
                try:
                    cursor = await database.execute(
                        """
                        SELECT COALESCE(MAX(sequence), 0)
                        FROM execution_trace_records
                        WHERE trace_id = ?
                        """,
                        (normalized,),
                    )
                    row = await cursor.fetchone()
                    if row is None or not isinstance(row[0], int):
                        raise PersistenceError(
                            "Stored trace sequence is invalid"
                        )
                    record = TraceRecord.from_event(
                        normalized,
                        row[0] + 1,
                        event,
                    )
                    await database.execute(
                        """
                        INSERT INTO execution_trace_records (
                            trace_id, sequence, event_type, stage, outcome,
                            occurred_at, run_id, thread_id, failure_category
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.trace_id,
                            record.sequence,
                            record.event_type.value,
                            record.stage.value,
                            record.outcome.value,
                            record.occurred_at.isoformat(),
                            record.run_id,
                            record.thread_id,
                            (
                                record.failure_category.value
                                if record.failure_category is not None
                                else None
                            ),
                        ),
                    )
                    await database.commit()
                except asyncio.CancelledError:
                    await self._rollback_best_effort(database)
                    raise
                except Exception:
                    await self._rollback_best_effort(database)
                    raise
                return record
        except PersistenceError:
            raise
        except sqlite3.Error as exc:
            raise PersistenceError("Unable to append trace event") from exc

    async def get_trace(self, trace_id: str) -> ExecutionTrace:
        normalized = self._validate_trace_id(trace_id)
        try:
            async with self._connect() as database:
                database.row_factory = aiosqlite.Row
                async with database.execute(
                    """
                    SELECT trace_id, sequence, event_type, stage, outcome,
                           occurred_at, run_id, thread_id, failure_category
                    FROM execution_trace_records
                    WHERE trace_id = ?
                    ORDER BY sequence ASC
                    """,
                    (normalized,),
                ) as cursor:
                    rows = await cursor.fetchall()
        except sqlite3.Error as exc:
            raise PersistenceError("Unable to read execution trace") from exc
        records = tuple(self._row_to_record(row) for row in rows)
        if any(
            record.sequence != expected
            for expected, record in enumerate(records, start=1)
        ):
            raise PersistenceError(
                "Stored execution trace sequence is invalid"
            )
        return ExecutionTrace(
            trace_id=normalized,
            records=records,
        )

    def _connect(self) -> aiosqlite.Connection:
        return aiosqlite.connect(
            self._database_path,
            timeout=SQLITE_BUSY_TIMEOUT_SECONDS,
        )

    @staticmethod
    async def _rollback_best_effort(database: aiosqlite.Connection) -> None:
        try:
            await database.rollback()
        except Exception:
            pass

    @staticmethod
    def _validate_trace_id(trace_id: str) -> str:
        try:
            return ExecutionTrace(trace_id=trace_id, records=()).trace_id
        except TraceConfigurationError as exc:
            raise PersistenceError("Invalid trace identifier") from exc

    @classmethod
    def _validate_input(cls, trace_id: str, event: AgentEvent) -> str:
        normalized = cls._validate_trace_id(trace_id)
        if not isinstance(event, AgentEvent):
            raise PersistenceError("Invalid trace event")
        return normalized

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> TraceRecord:
        try:
            failure_value = row["failure_category"]
            occurred_at = datetime.fromisoformat(row["occurred_at"])
            return TraceRecord(
                trace_id=row["trace_id"],
                sequence=row["sequence"],
                event_type=EventType(row["event_type"]),
                stage=EventStage(row["stage"]),
                outcome=EventOutcome(row["outcome"]),
                occurred_at=occurred_at,
                run_id=row["run_id"],
                thread_id=row["thread_id"],
                failure_category=(
                    None
                    if failure_value is None
                    else FailureCategory(failure_value)
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PersistenceError("Stored trace record is invalid") from exc

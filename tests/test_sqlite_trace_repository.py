import asyncio
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import aiosqlite
import pytest

from app.observability import (
    AgentEvent,
    EventOutcome,
    EventStage,
    EventType,
)
from app.persistence.repository import PersistenceError
from app.persistence.trace_repository import SQLiteTraceRepository
from app.reliability.failures import FailureCategory


FIXED_TIME = datetime(2026, 4, 5, 6, 7, 8, tzinfo=timezone.utc)


def _event(
    event_type: EventType = EventType.PLANNER_STARTED,
    *,
    occurred_at: datetime = FIXED_TIME,
    run_id: str | None = None,
    thread_id: str | None = None,
    failure_category: FailureCategory | None = None,
) -> AgentEvent:
    semantics = {
        EventType.PLANNER_STARTED: (
            EventStage.PLANNER,
            EventOutcome.STARTED,
        ),
        EventType.PLANNER_COMPLETED: (
            EventStage.PLANNER,
            EventOutcome.SUCCEEDED,
        ),
        EventType.PLANNER_FAILED: (
            EventStage.PLANNER,
            EventOutcome.FAILED,
        ),
    }
    stage, outcome = semantics[event_type]
    return AgentEvent(
        event_type=event_type,
        occurred_at=occurred_at,
        stage=stage,
        outcome=outcome,
        run_id=run_id,
        thread_id=thread_id,
        failure_category=failure_category,
    )


def _run(coroutine):
    return asyncio.run(coroutine)


def test_initialize_creates_allowlisted_table_and_is_idempotent(tmp_path: Path) -> None:
    async def exercise() -> None:
        path = tmp_path / "trace.db"
        repository = SQLiteTraceRepository(path)
        await repository.initialize()
        await repository.append_event("trace-a", _event())
        await repository.initialize()
        assert len((await repository.get_trace("trace-a")).records) == 1
        async with aiosqlite.connect(path) as database:
            cursor = await database.execute(
                "PRAGMA table_info(execution_trace_records)"
            )
            columns = {row[1] for row in await cursor.fetchall()}
        assert columns == {
            "trace_id", "sequence", "event_type", "stage", "outcome",
            "occurred_at", "run_id", "thread_id", "failure_category",
        }
        assert "payload_json" not in columns and "metadata_json" not in columns

    _run(exercise())


def test_append_sequences_are_per_trace_and_reads_are_ordered(tmp_path: Path) -> None:
    async def exercise() -> None:
        repository = SQLiteTraceRepository(tmp_path / "trace.db")
        await repository.initialize()
        first = await repository.append_event("trace-a", _event())
        second = await repository.append_event(
            "trace-a", _event(EventType.PLANNER_COMPLETED)
        )
        other = await repository.append_event("trace-b", _event())
        assert (first.sequence, second.sequence, other.sequence) == (1, 2, 1)
        trace = await repository.get_trace("trace-a")
        assert tuple(record.sequence for record in trace.records) == (1, 2)
        assert trace.records[0].occurred_at == trace.records[1].occurred_at
        assert (await repository.get_trace("missing")).records == ()

    _run(exercise())


def test_repository_recreation_preserves_trace_and_sequence(tmp_path: Path) -> None:
    async def exercise() -> None:
        path = tmp_path / "trace.db"
        first = SQLiteTraceRepository(path)
        await first.initialize()
        await first.append_event("trace-a", _event())
        second = SQLiteTraceRepository(path)
        await second.initialize()
        appended = await second.append_event(
            "trace-a", _event(EventType.PLANNER_COMPLETED)
        )
        assert appended.sequence == 2
        assert len((await second.get_trace("trace-a")).records) == 2

    _run(exercise())


def test_two_repository_instances_allocate_unique_sequences(tmp_path: Path) -> None:
    async def exercise() -> None:
        path = tmp_path / "trace.db"
        first = SQLiteTraceRepository(path)
        second = SQLiteTraceRepository(path)
        await first.initialize()
        await asyncio.gather(*(
            (first if index % 2 == 0 else second).append_event(
                "trace-shared", _event()
            )
            for index in range(30)
        ))
        trace = await first.get_trace("trace-shared")
        assert tuple(record.sequence for record in trace.records) == tuple(
            range(1, 31)
        )

    _run(exercise())


def test_concurrent_different_traces_count_independently(tmp_path: Path) -> None:
    async def exercise() -> None:
        path = tmp_path / "trace.db"
        repositories = (SQLiteTraceRepository(path), SQLiteTraceRepository(path))
        await repositories[0].initialize()
        await asyncio.gather(*(
            repositories[index % 2].append_event(
                f"trace-{index % 3}", _event()
            )
            for index in range(30)
        ))
        for index in range(3):
            trace = await repositories[0].get_trace(f"trace-{index}")
            assert tuple(record.sequence for record in trace.records) == tuple(
                range(1, 11)
            )

    _run(exercise())


def test_database_primary_key_rejects_duplicate_sequence(tmp_path: Path) -> None:
    async def exercise() -> None:
        path = tmp_path / "trace.db"
        repository = SQLiteTraceRepository(path)
        await repository.initialize()
        await repository.append_event("trace-a", _event())
        async with aiosqlite.connect(path) as database:
            with pytest.raises(sqlite3.IntegrityError):
                await database.execute(
                    """
                    INSERT INTO execution_trace_records VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "trace-a", 1, "planner_started", "planner", "started",
                        FIXED_TIME.isoformat(), None, None, None,
                    ),
                )

    _run(exercise())


def test_invalid_input_is_rejected_before_database_write(tmp_path: Path) -> None:
    async def exercise() -> None:
        repository = SQLiteTraceRepository(tmp_path / "trace.db")
        await repository.initialize()
        with pytest.raises(PersistenceError, match="identifier"):
            await repository.append_event("  ", _event())
        with pytest.raises(PersistenceError, match="event"):
            await repository.append_event("trace-a", object())  # type: ignore[arg-type]
        assert (await repository.get_trace("trace-a")).records == ()

    _run(exercise())


def test_types_utc_and_identities_round_trip(tmp_path: Path) -> None:
    async def exercise() -> None:
        repository = SQLiteTraceRepository(tmp_path / "trace.db")
        await repository.initialize()
        offset = timezone(timedelta(hours=8))
        event = _event(
            EventType.PLANNER_FAILED,
            occurred_at=datetime(2026, 4, 5, 14, 7, 8, tzinfo=offset),
            run_id="run-a",
            thread_id="thread-b",
            failure_category=FailureCategory.VALIDATION,
        )
        await repository.append_event("trace-a", event)
        record = (await repository.get_trace("trace-a")).records[0]
        assert record.event_type is EventType.PLANNER_FAILED
        assert record.stage is EventStage.PLANNER
        assert record.outcome is EventOutcome.FAILED
        assert record.occurred_at == FIXED_TIME
        assert record.run_id == "run-a" and record.thread_id == "thread-b"
        assert record.failure_category is FailureCategory.VALIDATION
        with pytest.raises(FrozenInstanceError):
            record.sequence = 9  # type: ignore[misc]
        missing = await repository.append_event("trace-b", _event())
        assert missing.run_id is None and missing.thread_id is None

    _run(exercise())


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("event_type", "unknown"),
        ("stage", "unknown"),
        ("outcome", "unknown"),
        ("occurred_at", "not-a-datetime"),
        ("failure_category", "unknown"),
    ],
)
def test_corrupt_rows_fail_safely(
    tmp_path: Path,
    column: str,
    value: str,
) -> None:
    async def exercise() -> None:
        path = tmp_path / f"trace-{column}.db"
        repository = SQLiteTraceRepository(path)
        await repository.initialize()
        await repository.append_event("trace-a", _event())
        allowed_columns = {
            "event_type", "stage", "outcome", "occurred_at",
            "failure_category",
        }
        assert column in allowed_columns
        async with aiosqlite.connect(path) as database:
            await database.execute(
                f"UPDATE execution_trace_records SET {column} = ?",
                (value,),
            )
            await database.commit()
        with pytest.raises(PersistenceError, match="Stored trace record"):
            await repository.get_trace("trace-a")

    _run(exercise())


def test_commit_failure_rolls_back_and_never_returns_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        path = tmp_path / "trace.db"
        repository = SQLiteTraceRepository(path)
        await repository.initialize()
        original_commit = aiosqlite.Connection.commit

        async def failing_commit(connection: aiosqlite.Connection) -> None:
            raise sqlite3.OperationalError("commit failed")

        monkeypatch.setattr(aiosqlite.Connection, "commit", failing_commit)
        with pytest.raises(PersistenceError, match="append"):
            await repository.append_event("trace-a", _event())
        monkeypatch.setattr(aiosqlite.Connection, "commit", original_commit)
        assert (await repository.get_trace("trace-a")).records == ()

    _run(exercise())


def test_parameterized_trace_id_has_no_sql_injection_effect(tmp_path: Path) -> None:
    async def exercise() -> None:
        repository = SQLiteTraceRepository(tmp_path / "trace.db")
        await repository.initialize()
        hostile = "trace'; DROP TABLE execution_trace_records; --"
        record = await repository.append_event(hostile, _event())
        assert record.trace_id == hostile
        assert len((await repository.get_trace(hostile)).records) == 1
        assert (await repository.get_trace("other")).records == ()

    _run(exercise())


def test_repository_schema_and_object_have_no_sensitive_payload_boundary() -> None:
    names = {field.name for field in fields(type(_event()))}
    assert "payload" not in names and "metadata" not in names
    repository = SQLiteTraceRepository("unused.db")
    assert not hasattr(repository, "llm_service")
    assert not hasattr(repository, "tool_registry")


@pytest.mark.parametrize("sequences", [(1, 3), (2,)])
def test_sequence_gaps_and_non_one_start_fail_safely(
    tmp_path: Path,
    sequences: tuple[int, ...],
) -> None:
    async def exercise() -> None:
        path = tmp_path / f"trace-{'-'.join(map(str, sequences))}.db"
        repository = SQLiteTraceRepository(path)
        await repository.initialize()
        async with aiosqlite.connect(path) as database:
            for sequence in sequences:
                await database.execute(
                    """
                    INSERT INTO execution_trace_records (
                        trace_id, sequence, event_type, stage, outcome,
                        occurred_at, run_id, thread_id, failure_category
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "trace-sensitive",
                        sequence,
                        "planner_started",
                        "planner",
                        "started",
                        FIXED_TIME.isoformat(),
                        "TOKEN=must-not-leak",
                        None,
                        None,
                    ),
                )
            await database.commit()
        with pytest.raises(
            PersistenceError,
            match="Stored execution trace sequence is invalid",
        ) as captured:
            await repository.get_trace("trace-sensitive")
        assert "TOKEN=must-not-leak" not in str(captured.value)

    _run(exercise())


def test_continuous_and_missing_traces_keep_normal_read_semantics(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        repository = SQLiteTraceRepository(tmp_path / "trace.db")
        await repository.initialize()
        await repository.append_event("trace-a", _event())
        await repository.append_event(
            "trace-a", _event(EventType.PLANNER_COMPLETED)
        )
        trace = await repository.get_trace("trace-a")
        assert tuple(record.sequence for record in trace.records) == (1, 2)
        missing = await repository.get_trace("trace-missing")
        assert missing.trace_id == "trace-missing"
        assert missing.records == ()

    _run(exercise())

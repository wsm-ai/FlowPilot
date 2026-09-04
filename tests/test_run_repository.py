import asyncio
import sqlite3
from datetime import timedelta

import pytest

from app.persistence.models import AgentRunRecord, create_agent_run_record
from app.persistence.repository import PersistenceError
from app.persistence.sqlite_repository import SQLiteRunRepository


def create_repository(tmp_path):
    database_path = tmp_path / "flowpilot-test.db"
    repository = SQLiteRunRepository(database_path)
    asyncio.run(repository.initialize())
    return repository, database_path


def test_initialize_creates_database(tmp_path):
    _, database_path = create_repository(tmp_path)

    assert database_path.exists()

    with sqlite3.connect(database_path) as database:
        columns = {
            row[1] for row in database.execute("PRAGMA table_info(agent_runs)")
        }
    assert "thread_id" in columns


@pytest.mark.parametrize("mode", ["reactive", "planned"])
def test_create_and_get_preserves_mode_status_and_none_result(tmp_path, mode):
    repository, _ = create_repository(tmp_path)
    record = create_agent_run_record(mode, "Hello")

    asyncio.run(repository.create(record))
    stored = asyncio.run(repository.get(record.run_id))

    assert stored is not None
    assert stored.run_id == record.run_id
    assert stored.mode == mode
    assert stored.status == "running"
    assert stored.input_text == "Hello"
    assert stored.result is None
    assert stored.error_type is None
    assert stored.created_at == record.created_at
    assert stored.updated_at == record.updated_at
    assert stored.created_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "result",
    [
        {"answer": "Hello"},
        {"answer": "客户反馈分析完成"},
    ],
)
def test_result_round_trips_as_json(tmp_path, result):
    repository, _ = create_repository(tmp_path)
    record = create_agent_run_record("reactive", "Hello")
    record.result = result

    asyncio.run(repository.create(record))
    stored = asyncio.run(repository.get(record.run_id))

    assert stored is not None
    assert stored.result == result


def test_update_preserves_created_at_and_updates_result_and_time(tmp_path):
    repository, _ = create_repository(tmp_path)
    record = create_agent_run_record("reactive", "Hello")
    asyncio.run(repository.create(record))

    updated = asyncio.run(
        repository.update(
            record.run_id,
            status="completed",
            result={"answer": "Hello"},
        )
    )

    assert updated.status == "completed"
    assert updated.result == {"answer": "Hello"}
    assert updated.created_at == record.created_at
    assert updated.updated_at > record.updated_at
    stored = asyncio.run(repository.get(record.run_id))
    assert stored == updated


def test_update_saves_approval_required_status(tmp_path):
    repository, _ = create_repository(tmp_path)
    record = create_agent_run_record("planned", "Review plan")
    asyncio.run(repository.create(record))

    updated = asyncio.run(
        repository.update(record.run_id, status="approval_required")
    )

    assert updated.status == "approval_required"


@pytest.mark.parametrize("thread_id", ["planned-thread", None])
def test_thread_id_round_trips_and_update_preserves_binding(tmp_path, thread_id):
    repository, _ = create_repository(tmp_path)
    record = create_agent_run_record("planned", "Review plan", thread_id=thread_id)
    asyncio.run(repository.create(record))

    updated = asyncio.run(repository.update(record.run_id, status="completed"))
    stored = asyncio.run(repository.get(record.run_id))

    assert updated.thread_id == thread_id
    assert stored.thread_id == thread_id


def test_rejected_status_round_trips(tmp_path):
    repository, _ = create_repository(tmp_path)
    record = create_agent_run_record("planned", "Review plan", "thread-rejected")
    asyncio.run(repository.create(record))

    updated = asyncio.run(repository.update(record.run_id, status="rejected"))

    assert updated.status == "rejected"
    assert asyncio.run(repository.get(record.run_id)).status == "rejected"


def test_initialize_migrates_legacy_schema_without_losing_data(tmp_path):
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as database:
        database.execute(
            """
            CREATE TABLE agent_runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                input_text TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT NULL,
                error_type TEXT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        database.execute(
            """
            INSERT INTO agent_runs VALUES (
                'legacy-run', 'reactive', 'Hello', 'completed', NULL, NULL,
                '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
            )
            """
        )

    repository = SQLiteRunRepository(database_path)
    asyncio.run(repository.initialize())
    asyncio.run(repository.initialize())

    with sqlite3.connect(database_path) as database:
        columns = {
            row[1] for row in database.execute("PRAGMA table_info(agent_runs)")
        }
    stored = asyncio.run(repository.get("legacy-run"))

    assert "thread_id" in columns
    assert stored is not None
    assert stored.input_text == "Hello"
    assert stored.thread_id is None


def test_update_saves_failed_error_type_without_traceback(tmp_path):
    repository, _ = create_repository(tmp_path)
    record = create_agent_run_record("planned", "Run plan")
    asyncio.run(repository.create(record))

    updated = asyncio.run(
        repository.update(
            record.run_id,
            status="failed",
            error_type="ToolExecutionError",
        )
    )

    assert updated.status == "failed"
    assert updated.error_type == "ToolExecutionError"
    assert "traceback" not in updated.error_type.lower()


def test_update_missing_run_raises_persistence_error(tmp_path):
    repository, _ = create_repository(tmp_path)

    with pytest.raises(PersistenceError):
        asyncio.run(repository.update("missing", status="completed"))


def test_list_recent_orders_newest_first_and_applies_limit(tmp_path):
    repository, _ = create_repository(tmp_path)
    records: list[AgentRunRecord] = []
    for index in range(3):
        record = create_agent_run_record("reactive", f"Run {index}")
        record.created_at = record.created_at + timedelta(seconds=index)
        record.updated_at = record.created_at
        asyncio.run(repository.create(record))
        records.append(record)

    recent = asyncio.run(repository.list_recent(limit=2))

    assert [record.run_id for record in recent] == [
        records[2].run_id,
        records[1].run_id,
    ]


@pytest.mark.parametrize("limit", [0, -1])
def test_list_recent_rejects_non_positive_limit(tmp_path, limit):
    repository, _ = create_repository(tmp_path)

    with pytest.raises(ValueError):
        asyncio.run(repository.list_recent(limit=limit))


def test_non_serializable_result_raises_persistence_error(tmp_path):
    repository, _ = create_repository(tmp_path)
    record = create_agent_run_record("reactive", "Hello")
    record.result = {"bad": object()}

    with pytest.raises(PersistenceError):
        asyncio.run(repository.create(record))

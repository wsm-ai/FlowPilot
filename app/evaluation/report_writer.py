import os
from pathlib import Path
import shutil
import tempfile


JSON_REPORT_FILENAME = "evaluation-report.json"
MARKDOWN_REPORT_FILENAME = "evaluation-report.md"


class EvaluationReportWriteError(Exception):
    """Raised when report artifacts cannot be saved safely."""


def write_evaluation_reports(
    output_directory: str | Path,
    *,
    json_report: str,
    markdown_report: str,
) -> tuple[Path, Path]:
    directory = Path(output_directory)
    staged: list[tuple[Path, Path]] = []
    backups: dict[Path, Path | None] = {}
    commit_started = False
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            raise OSError
        for filename, content in (
            (JSON_REPORT_FILENAME, json_report),
            (MARKDOWN_REPORT_FILENAME, markdown_report),
        ):
            if not isinstance(content, str):
                raise TypeError
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{filename}.", suffix=".tmp", dir=directory
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            staged.append((temporary, directory / filename))

        for _, target in staged:
            if not target.exists():
                backups[target] = None
                continue
            descriptor, backup_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".bak", dir=directory
            )
            os.close(descriptor)
            backup = Path(backup_name)
            backups[target] = backup
            shutil.copyfile(target, backup)

        commit_started = True
        for temporary, target in staged:
            os.replace(temporary, target)
        return tuple(target for _, target in staged)  # type: ignore[return-value]
    except (OSError, TypeError) as exc:
        if commit_started:
            _restore_previous_reports(backups)
        raise EvaluationReportWriteError("Unable to write evaluation reports") from exc
    finally:
        for temporary, _ in staged:
            _unlink_best_effort(temporary)
        for backup in backups.values():
            if backup is not None:
                _unlink_best_effort(backup)


def _restore_previous_reports(backups: dict[Path, Path | None]) -> None:
    """Best-effort runtime rollback; not a cross-file crash transaction."""
    for target, backup in backups.items():
        try:
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                os.replace(backup, target)
        except OSError:
            pass


def _unlink_best_effort(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass

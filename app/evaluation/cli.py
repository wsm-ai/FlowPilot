import argparse
import asyncio
from pathlib import Path
import sys
from typing import Sequence

from app.evaluation.report_builder import build_evaluation_report
from app.evaluation.report_renderers import (
    render_evaluation_report_json,
    render_evaluation_report_markdown,
)
from app.evaluation.report_writer import write_evaluation_reports
from app.evaluation.runner import EvaluationRunner
from app.evaluation.scenarios import build_default_evaluation_cases


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation.cli",
        description="Run deterministic offline FlowPilot evaluations.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suite-name", default="flowpilot-default")
    return parser


async def _run(output_directory: Path, suite_name: str) -> None:
    cases = build_default_evaluation_cases()
    results = await EvaluationRunner().run(cases)
    report = build_evaluation_report(results, suite_name=suite_name)
    write_evaluation_reports(
        output_directory,
        json_report=render_evaluation_report_json(report),
        markdown_report=render_evaluation_report_markdown(report),
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        asyncio.run(_run(arguments.output_dir, arguments.suite_name))
    except KeyboardInterrupt:
        raise
    except Exception:
        print("Evaluation report generation failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

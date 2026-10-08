import argparse
from pathlib import Path
import sys
from typing import Sequence

from app.evaluation.gate_models import (
    DimensionThreshold,
    GateStatus,
    RegressionGateConfig,
    RegressionGateConfigurationError,
)
from app.evaluation.gate_validator import (
    EvaluationReportValidationError,
    load_evaluation_report,
)
from app.evaluation.regression_gate import evaluate_regression_gate
from app.evaluation.metrics import EvaluationDimension


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation.gate_cli",
        description="Evaluate a deterministic FlowPilot evaluation report.",
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-failed-scenarios", type=int, default=0)
    parser.add_argument("--max-error-scenarios", type=int, default=0)
    parser.add_argument("--min-case-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-behavior-check-pass-rate", type=float, default=1.0)
    parser.add_argument("--max-regression", type=float, default=0.0)
    parser.add_argument(
        "--min-dimension-pass-rate",
        action="append",
        default=[],
        metavar="DIMENSION=RATE",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = RegressionGateConfig(
            max_failed_scenarios=arguments.max_failed_scenarios,
            max_error_scenarios=arguments.max_error_scenarios,
            min_case_pass_rate=arguments.min_case_pass_rate,
            min_behavior_check_pass_rate=arguments.min_behavior_check_pass_rate,
            max_regression=arguments.max_regression,
            dimension_thresholds=tuple(
                _dimension_threshold(value)
                for value in arguments.min_dimension_pass_rate
            ),
        )
        report = load_evaluation_report(arguments.report)
        baseline = (
            load_evaluation_report(arguments.baseline)
            if arguments.baseline is not None
            else None
        )
        result = evaluate_regression_gate(
            report, config=config, baseline=baseline
        )
    except (
        EvaluationReportValidationError,
        RegressionGateConfigurationError,
        TypeError,
        ValueError,
    ):
        print("INVALID", file=sys.stdout)
        print("reason: invalid_report", file=sys.stdout)
        return 2

    print(result.gate_status.value)
    print(
        "scenarios: "
        f"total={result.total_scenarios} passed={result.passed_scenarios} "
        f"failed={result.failed_scenarios} errors={result.error_scenarios}"
    )
    print(
        "baseline_comparison: "
        + ("performed" if result.baseline_compared else "not_performed")
    )
    for reason in result.reasons:
        print(f"reason: {reason.value}")
    if result.gate_status is GateStatus.PASS:
        return 0
    if result.gate_status is GateStatus.FAIL:
        return 1
    return 2


def _dimension_threshold(value: str) -> DimensionThreshold:
    dimension, separator, raw_rate = value.partition("=")
    if not separator:
        raise ValueError("Invalid dimension threshold")
    return DimensionThreshold(
        dimension=EvaluationDimension(dimension),
        min_case_pass_rate=float(raw_rate),
    )


if __name__ == "__main__":
    raise SystemExit(main())

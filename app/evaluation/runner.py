from collections.abc import Sequence
import time

from app.evaluation.models import (
    EvaluationCase,
    EvaluationCheck,
    EvaluationConfigurationError,
    EvaluationResult,
    EvaluationStatus,
)


_SCENARIO_ERROR_CHECK = EvaluationCheck(
    name="scenario_execution",
    passed=False,
    message="Evaluation scenario failed",
)

_NO_CHECKS_ERROR_CHECK = EvaluationCheck(
    name="scenario_execution",
    passed=False,
    message="Evaluation scenario produced no checks",
)


class EvaluationRunner:
    async def run(
        self,
        cases: Sequence[EvaluationCase],
    ) -> list[EvaluationResult]:
        self._validate_unique_case_ids(cases)
        results: list[EvaluationResult] = []
        for case in cases:
            started = time.perf_counter()
            try:
                checks = tuple(await case.scenario.run())
                if not checks:
                    checks = (_NO_CHECKS_ERROR_CHECK,)
                    status = EvaluationStatus.ERROR
                else:
                    status = (
                        EvaluationStatus.PASSED
                        if all(check.passed for check in checks)
                        else EvaluationStatus.FAILED
                    )
            except Exception:
                checks = (_SCENARIO_ERROR_CHECK,)
                status = EvaluationStatus.ERROR
            duration_ms = max(0.0, (time.perf_counter() - started) * 1000)
            results.append(
                EvaluationResult(
                    case_id=case.case_id,
                    status=status,
                    checks=checks,
                    duration_ms=duration_ms,
                )
            )
        return results

    @staticmethod
    def _validate_unique_case_ids(cases: Sequence[EvaluationCase]) -> None:
        seen: set[str] = set()
        for case in cases:
            if case.case_id in seen:
                raise EvaluationConfigurationError("Duplicate evaluation case_id")
            seen.add(case.case_id)

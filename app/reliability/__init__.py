from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "FailureCategory": "app.reliability.failures",
    "FailureDescriptor": "app.reliability.failures",
    "FailureDomain": "app.reliability.failures",
    "classify_failure": "app.reliability.failures",
    "OperationSemantics": "app.reliability.retry",
    "RetryDecision": "app.reliability.retry",
    "RetryPolicy": "app.reliability.retry",
    "decide_retry": "app.reliability.retry",
    "run_with_retry": "app.reliability.retry",
    "SideEffectConflictError": "app.reliability.side_effects",
    "SideEffectExecutionError": "app.reliability.side_effects",
    "SideEffectExecutionRecord": "app.reliability.side_effects",
    "SideEffectExecutionStatus": "app.reliability.side_effects",
    "SideEffectExecutor": "app.reliability.side_effects",
    "SideEffectReplayBlockedError": "app.reliability.side_effects",
    "digest_arguments": "app.reliability.side_effects",
    "DEFAULT_OPERATION_TIMEOUT_SECONDS": "app.reliability.timeouts",
    "OperationTimeoutError": "app.reliability.timeouts",
    "ReadOperationTimeoutError": "app.reliability.timeouts",
    "SideEffectOperationTimeoutError": "app.reliability.timeouts",
    "TimeoutPolicy": "app.reliability.timeouts",
    "UnknownOperationTimeoutError": "app.reliability.timeouts",
    "run_with_timeout": "app.reliability.timeouts",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

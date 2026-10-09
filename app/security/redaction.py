from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import SecretStr


REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "bearer_token",
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "client_secret",
    }
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"password|passwd|client[_-]?secret|secret|token)\s*[:=]\s*)"
    r"([^\s,;]+)"
)
_URL_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(https?://)([^/@\s]+)@"
)


def _normalized_key(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def is_sensitive_key(value: object) -> bool:
    normalized = _normalized_key(value)
    return normalized in _SENSITIVE_KEYS or any(
        normalized.endswith(suffix)
        for suffix in ("_api_key", "_password", "_secret", "_token")
    )


def redact_text(value: str, *, secrets: Iterable[str] = ()) -> str:
    redacted = value
    for secret in sorted(
        {item for item in secrets if isinstance(item, str) and item},
        key=len,
        reverse=True,
    ):
        redacted = redacted.replace(secret, REDACTED)
    redacted = _BEARER_PATTERN.sub(f"Bearer {REDACTED}", redacted)
    redacted = _ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}{REDACTED}",
        redacted,
    )
    return _URL_CREDENTIAL_PATTERN.sub(
        lambda match: f"{match.group(1)}{REDACTED}@",
        redacted,
    )


def redact_sensitive_data(
    value: Any,
    *,
    secrets: Iterable[str] = (),
) -> Any:
    if isinstance(value, SecretStr):
        return REDACTED
    if isinstance(value, Mapping):
        return {
            key: (
                REDACTED
                if is_sensitive_key(key)
                else redact_sensitive_data(item, secrets=secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item, secrets=secrets) for item in value)
    if isinstance(value, list):
        return [redact_sensitive_data(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        return redact_text(value, secrets=secrets)
    return value


class SensitiveDataFilter(logging.Filter):
    """Redact known credential shapes before a log record is formatted."""

    def __init__(self, *, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = frozenset(
            secret for secret in secrets if isinstance(secret, str) and secret
        )

    def add_secrets(self, secrets: Iterable[str]) -> None:
        self._secrets = self._secrets.union(
            secret for secret in secrets if isinstance(secret, str) and secret
        )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "uvicorn.access":
            if isinstance(record.msg, str):
                record.msg = redact_text(record.msg, secrets=self._secrets)
            record.args = redact_sensitive_data(
                record.args,
                secrets=self._secrets,
            )
        elif isinstance(record.msg, str):
            record.args = redact_sensitive_data(
                record.args,
                secrets=self._secrets,
            )
            try:
                rendered = record.getMessage()
            except Exception:
                rendered = "Log message formatting failed"
            record.msg = redact_text(rendered, secrets=self._secrets)
            record.args = ()
        else:
            record.msg = redact_sensitive_data(
                record.msg,
                secrets=self._secrets,
            )
            record.args = ()

        if record.exc_info is not None:
            exception_type = record.exc_info[0]
            exception_name = getattr(exception_type, "__name__", "Exception")
            record.exc_info = None
            record.exc_text = f"{exception_name}: {REDACTED}"
        if record.stack_info is not None:
            record.stack_info = None
        return True


_FILTER_MARKER = "_flowpilot_sensitive_data_filter"
_DIAGNOSTIC_HANDLER_MARKER = "_flowpilot_safe_diagnostic_handler"


def _protect_handler(
    handler: logging.Handler,
    secrets: Iterable[str],
) -> None:
    existing = next(
        (
            item
            for item in handler.filters
            if isinstance(item, SensitiveDataFilter)
            or getattr(item, _FILTER_MARKER, False)
        ),
        None,
    )
    if isinstance(existing, SensitiveDataFilter):
        existing.add_secrets(secrets)
        return
    filter_instance = SensitiveDataFilter(secrets=secrets)
    setattr(filter_instance, _FILTER_MARKER, True)
    handler.addFilter(filter_instance)


def configure_sensitive_logging(
    *,
    secrets: Iterable[str] = (),
    logger_names: Iterable[str] = (
        "flowpilot.observability",
        "flowpilot.diagnostics",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
    ),
) -> None:
    """Protect explicit application logger handlers, including child records."""
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        if logger_name == "flowpilot.diagnostics" and not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(levelname)s %(name)s %(message)s")
            )
            setattr(handler, _DIAGNOSTIC_HANDLER_MARKER, True)
            logger.addHandler(handler)
        if logger_name == "flowpilot.diagnostics":
            logger.propagate = False
        for handler in logger.handlers:
            _protect_handler(handler, secrets)

"""Environment-configured HTTP client for OpenAI-compatible chat-completion APIs."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import socket
import time
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when the model cannot be configured or returns an unusable response."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "model_failure",
        reason: str = "request_failed",
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.reason = reason
        self.http_status = http_status


@dataclass(frozen=True, slots=True)
class LLMConfig:
    endpoint: str
    model: str
    api_key: str
    timeout_seconds: float = 30.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        if not self.endpoint.strip():
            raise ValueError("LLM endpoint must not be empty")
        if not self.model.strip():
            raise ValueError("LLM model name must not be empty")
        if not self.api_key.strip():
            raise ValueError("LLM API key must not be empty")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise ValueError("max_retries must be an integer from 0 to 2")
        if not 0 <= self.max_retries <= 2:
            raise ValueError("max_retries must be an integer from 0 to 2")

    @classmethod
    def from_env(cls) -> LLMConfig:
        endpoint = os.environ.get("RAG_LLM_ENDPOINT")
        model = os.environ.get("RAG_LLM_MODEL")
        api_key = os.environ.get("RAG_LLM_API_KEY")
        missing = [
            name
            for name, value in (
                ("RAG_LLM_ENDPOINT", endpoint),
                ("RAG_LLM_MODEL", model),
                ("RAG_LLM_API_KEY", api_key),
            )
            if not value
        ]
        if missing:
            raise LLMClientError(f"missing required environment variables: {', '.join(missing)}")

        timeout_value = os.environ.get("RAG_LLM_TIMEOUT", "30")
        try:
            timeout_seconds = float(timeout_value)
        except ValueError as error:
            raise LLMClientError("RAG_LLM_TIMEOUT must be a positive number") from error
        if timeout_seconds <= 0:
            raise LLMClientError("RAG_LLM_TIMEOUT must be a positive number")

        retries_value = os.environ.get("RAG_LLM_MAX_RETRIES", "2")
        try:
            max_retries = int(retries_value)
        except ValueError as error:
            raise LLMClientError("RAG_LLM_MAX_RETRIES must be an integer from 0 to 2") from error
        if not 0 <= max_retries <= 2:
            raise LLMClientError("RAG_LLM_MAX_RETRIES must be an integer from 0 to 2")

        return cls(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )


def _client_error(
    error_type: str,
    reason: str,
    message: str,
    *,
    http_status: int | None = None,
) -> LLMClientError:
    return LLMClientError(
        f"{error_type} reason={reason}: {message}",
        error_type=error_type,
        reason=reason,
        http_status=http_status,
    )


def _extract_content(response_body: bytes) -> str:
    try:
        response = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _client_error(
            "invalid_json",
            "invalid_json",
            "model endpoint returned invalid JSON",
        ) from error

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise _client_error(
            "schema_mismatch",
            "missing_content",
            "model response did not contain choices[0].message.content",
        ) from error
    if not isinstance(content, str):
        raise _client_error(
            "schema_mismatch",
            "content_not_string",
            "model response content must be a string",
        )
    return content


def _extract_json_object(content: str) -> str:
    text = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        content.strip(),
        flags=re.IGNORECASE,
    ).strip()
    if not text:
        raise _client_error("empty_content", "empty_content", "model response content was empty")

    start = text.find("{")
    if start < 0:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise _client_error(
                "invalid_json",
                "invalid_json",
                "model response content did not contain a JSON object",
            ) from error
        if not isinstance(parsed, dict):
            raise _client_error(
                "schema_mismatch",
                "top_level_not_object",
                "model response JSON must be an object",
            )
        return text

    depth = 0
    in_string = False
    escaped = False
    end = -1
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break

    if end < 0:
        raise _client_error(
            "invalid_json",
            "invalid_json",
            "model response content did not contain a complete JSON object",
        )

    candidate = text[start:end]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise _client_error(
            "invalid_json",
            "invalid_json",
            "model response content contained invalid JSON",
        ) from error
    if not isinstance(parsed, dict):
        raise _client_error(
            "schema_mismatch",
            "top_level_not_object",
            "model response JSON must be an object",
        )
    return candidate


def _safe_response_excerpt(response_body: bytes, api_key: str) -> str:
    text = response_body.decode("utf-8", errors="replace")
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    text = re.sub(r"(?i)(bearer\s+)[^\s\"',}]+", r"\1[REDACTED]", text)
    text = re.sub(
        r"""(?i)(["']?(?:authorization|api[_-]?key|access[_-]?token|token|secret)["']?\s*:\s*["']?)[^"'\s,}]+""",
        r"\1[REDACTED]",
        text,
    )
    return text[:300]


def _http_error_reason(status: int) -> str:
    return {
        400: "bad_request",
        401: "authentication_failed",
        403: "permission_denied",
        404: "model_not_found",
        408: "timeout",
        425: "temporary_provider_failure",
        429: "rate_limit",
    }.get(status, "provider_error")


def _log_failed_attempt(
    *,
    error_type: str,
    http_status: int | str,
    response_body: bytes,
    api_key: str,
    attempt: int,
    max_attempts: int,
    attempt_started: float,
) -> None:
    logger.warning(
        "Model request attempt failed (error_type=%s, http_status=%s, "
        "attempt=%d/%d, elapsed_ms=%d, response_body=%r)",
        error_type,
        http_status,
        attempt,
        max_attempts,
        round((time.monotonic() - attempt_started) * 1000),
        _safe_response_excerpt(response_body, api_key),
    )


def call_model(
    prompt: str,
    *,
    system_prompt: str = "",
    config: LLMConfig | None = None,
) -> str:
    """Send a chat-completion request, with a timeout and at most two retries.

    The endpoint and model are mandatory configuration; there is no provider or
    hosted endpoint default. The endpoint must accept the OpenAI-compatible
    ``/chat/completions`` request and response shape.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(system_prompt, str):
        raise TypeError("system_prompt must be a string")

    active_config = config if config is not None else LLMConfig.from_env()
    messages = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    "Return exactly one JSON object matching the supplied schema. "
                    "Do not include Markdown fences, surrounding prose, or extra fields."
                ),
            }
        )
    messages.append({"role": "user", "content": prompt})
    payload = json.dumps(
        {"model": active_config.model, "messages": messages, "temperature": 0},
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        active_config.endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {active_config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    last_error: LLMClientError | None = None
    max_attempts = active_config.max_retries + 1
    for attempt in range(1, max_attempts + 1):
        attempt_started = time.monotonic()
        try:
            with urlopen(request, timeout=active_config.timeout_seconds) as response:
                response_body = response.read()
            try:
                return _extract_json_object(_extract_content(response_body))
            except LLMClientError as error:
                _log_failed_attempt(
                    error_type=error.error_type,
                    http_status=200,
                    response_body=response_body,
                    api_key=active_config.api_key,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    attempt_started=attempt_started,
                )
                raise
        except HTTPError as error:
            try:
                body = error.read()
            except OSError:
                body = b""
            reason = _http_error_reason(error.code)
            error_type = f"http_{error.code}"
            _log_failed_attempt(
                error_type=error_type,
                http_status=error.code,
                response_body=body,
                api_key=active_config.api_key,
                attempt=attempt,
                max_attempts=max_attempts,
                attempt_started=attempt_started,
            )
            last_error = _client_error(
                error_type,
                reason,
                f"model endpoint returned HTTP {error.code}",
                http_status=error.code,
            )
            if error.code not in {408, 425, 429} and not 500 <= error.code <= 599:
                raise last_error from None
        except (URLError, TimeoutError, OSError) as error:
            is_timeout = isinstance(error, (TimeoutError, socket.timeout)) or (
                isinstance(error, URLError) and isinstance(error.reason, TimeoutError)
            )
            error_type = "timeout" if is_timeout else "network_error"
            reason = "timeout" if is_timeout else "connection_failed"
            _log_failed_attempt(
                error_type=error_type,
                http_status="none",
                response_body=b"",
                api_key=active_config.api_key,
                attempt=attempt,
                max_attempts=max_attempts,
                attempt_started=attempt_started,
            )
            last_error = _client_error(
                error_type,
                reason,
                "model request timed out" if is_timeout else "model request failed",
            )
        if attempt < max_attempts:
            time.sleep(min(0.25 * (2 ** (attempt - 1)), 1.0))

    if last_error is None:
        raise _client_error("model_failure", "request_failed", "model request failed")
    raise last_error from None


class MockLLMClient:
    """Callable deterministic mock that returns queued strings or raises queued errors."""

    def __init__(self, responses: Iterable[str | Exception]) -> None:
        self._responses = iter(responses)
        self.calls = 0

    def __call__(self, prompt: str, *, system_prompt: str = "") -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string")
        self.calls += 1
        try:
            response = next(self._responses)
        except StopIteration as error:
            raise LLMClientError("mock client has no queued responses") from error
        if isinstance(response, Exception):
            raise response
        if not isinstance(response, str):
            raise LLMClientError("mock response must be a string or exception")
        return response

"""Environment-configured HTTP client for OpenAI-compatible chat-completion APIs."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMClientError(RuntimeError):
    """Raised when the model cannot be configured or returns an unusable response."""


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


def _extract_content(response_body: bytes) -> str:
    try:
        response = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LLMClientError("model endpoint returned invalid JSON") from error

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise LLMClientError("model response did not contain choices[0].message.content") from error
    if not isinstance(content, str):
        raise LLMClientError("model response content must be a string")
    return content


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
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = json.dumps(
        {"model": active_config.model, "messages": messages},
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
    for attempt in range(active_config.max_retries + 1):
        try:
            with urlopen(request, timeout=active_config.timeout_seconds) as response:
                return _extract_content(response.read())
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            last_error = LLMClientError(f"model endpoint returned HTTP {error.code}: {body[:500]}")
            if error.code not in {408, 425, 429} and not 500 <= error.code <= 599:
                raise last_error from error
        except (URLError, TimeoutError, OSError) as error:
            last_error = LLMClientError(f"model request failed: {error}")
        if attempt < active_config.max_retries:
            time.sleep(min(0.25 * (2**attempt), 1.0))

    raise LLMClientError(
        f"model request failed after {active_config.max_retries} retries: {last_error}"
    ) from last_error


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

import io
import json
import logging
from urllib.error import HTTPError

import llm_client
import pytest
from llm_client import LLMConfig, LLMClientError, call_model


def _config(*, retries=0, api_key="test-key"):
    return LLMConfig(
        endpoint="https://model.example.test/v1beta/openai/chat/completions",
        model="gemini-test-model",
        api_key=api_key,
        timeout_seconds=2,
        max_retries=retries,
    )


def _openai_response(content, **extra_fields):
    return json.dumps(
        {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            **extra_fields,
        }
    ).encode()


def test_call_model_extracts_fenced_json_and_ignores_envelope_extras(monkeypatch):
    json_object = (
        '{"status":"ok","action_items":[{"category":"roads","location":"Mwangaza Road",'
        '"priority":"high","suggested_department":"roads","summary":"brace } in text",'
        '"citations":[{"doc_id":"feedback.txt","chunk_id":"chunk-1","excerpt":"pothole"}]}]}'
    )
    content = f"```json\n{json_object}\n```"
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return io.BytesIO(
            _openai_response(content, extra_content="ignore this envelope field")
        )

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)

    result = call_model("test prompt", system_prompt="Be evidence-based.", config=_config())

    parsed_result = json.loads(result)
    assert parsed_result["status"] == "ok"
    assert parsed_result["action_items"][0]["summary"] == "brace } in text"
    assert llm_client._extract_json_object(f"Here is the result:\n{content}") == json_object
    assert captured["payload"]["temperature"] == 0
    assert "Do not include Markdown fences" in captured["payload"]["messages"][0]["content"]
    assert captured["timeout"] == 2


def test_call_model_reports_empty_content_and_logs_response_excerpt(monkeypatch, caplog):
    monkeypatch.setattr(
        llm_client,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(_openai_response("")),
    )
    caplog.set_level(logging.WARNING, logger="llm_client")

    with pytest.raises(LLMClientError) as raised:
        call_model("test prompt", config=_config())

    assert raised.value.error_type == "empty_content"
    assert raised.value.reason == "empty_content"
    assert "http_status=200" in caplog.text
    assert "response_body=" in caplog.text
    assert "attempt=1/1" in caplog.text
    assert "elapsed_ms=" in caplog.text


def test_call_model_reports_invalid_json(monkeypatch):
    monkeypatch.setattr(
        llm_client,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(_openai_response("This is not JSON")),
    )

    with pytest.raises(LLMClientError) as raised:
        call_model("test prompt", config=_config())

    assert raised.value.error_type == "invalid_json"
    assert raised.value.reason == "invalid_json"


def test_http_error_log_truncates_response_body_to_300_characters(monkeypatch, caplog):
    def fail_request(request, timeout):
        raise HTTPError(
            request.full_url,
            400,
            "bad request",
            {},
            io.BytesIO(b"x" * 500),
        )

    monkeypatch.setattr(llm_client, "urlopen", fail_request)
    caplog.set_level(logging.WARNING, logger="llm_client")

    with pytest.raises(LLMClientError):
        call_model("test prompt", config=_config())

    assert "response_body='" + ("x" * 300) + "'" in caplog.text
    assert "x" * 301 not in caplog.text


@pytest.mark.parametrize(
    ("status", "reason"),
    [(400, "bad_request"), (429, "rate_limit")],
)
def test_call_model_logs_http_failures_and_redacts_secrets(
    monkeypatch, caplog, status, reason
):
    calls = []

    def fail_request(request, timeout):
        calls.append(request)
        raise HTTPError(
            request.full_url,
            status,
            "provider failure",
            {},
            io.BytesIO(b'{"error":"Bearer test-key", "message":"retry safely"}'),
        )

    monkeypatch.setattr(llm_client, "urlopen", fail_request)
    caplog.set_level(logging.WARNING, logger="llm_client")

    with pytest.raises(LLMClientError) as raised:
        call_model("test prompt", config=_config(retries=1))

    assert raised.value.error_type == f"http_{status}"
    assert raised.value.reason == reason
    expected_attempts = 1 if status == 400 else 2
    assert len(calls) == expected_attempts
    assert f"http_status={status}" in caplog.text
    expected_log_attempt = "attempt=1/2" if status == 400 else "attempt=2/2"
    assert expected_log_attempt in caplog.text
    assert "[REDACTED]" in caplog.text
    assert "test-key" not in caplog.text


def test_call_model_reports_timeout_without_logging_exception_text(monkeypatch, caplog):
    def timeout(*_args, **_kwargs):
        raise TimeoutError("test-key must not be logged")

    monkeypatch.setattr(llm_client, "urlopen", timeout)
    caplog.set_level(logging.WARNING, logger="llm_client")

    with pytest.raises(LLMClientError) as raised:
        call_model("test prompt", config=_config(api_key="test-key"))

    assert raised.value.error_type == "timeout"
    assert raised.value.reason == "timeout"
    assert "http_status=none" in caplog.text
    assert "test-key" not in caplog.text


def test_call_model_rejects_non_chat_completion_response(monkeypatch):
    response_body = b'{"error":"unexpected provider response"}'
    monkeypatch.setattr(
        llm_client,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(response_body),
    )

    with pytest.raises(LLMClientError) as raised:
        call_model("test prompt", config=_config())

    assert raised.value.error_type == "schema_mismatch"
    assert raised.value.reason == "missing_content"

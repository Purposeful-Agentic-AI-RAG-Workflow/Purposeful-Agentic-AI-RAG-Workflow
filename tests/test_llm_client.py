import io
import json

import llm_client
import pytest
from llm_client import LLMConfig, LLMClientError, call_model


def test_call_model_parses_gemini_openai_compatible_chat_response(monkeypatch):
    response_body = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"status":"insufficient_evidence","action_items":[]}',
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    ).encode()
    monkeypatch.setattr(
        llm_client,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(response_body),
    )

    result = call_model(
        "test prompt",
        config=LLMConfig(
            endpoint="https://model.example.test/v1beta/openai/chat/completions",
            model="gemini-test-model",
            api_key="test-key",
            max_retries=0,
        ),
    )

    assert result == '{"status":"insufficient_evidence","action_items":[]}'


def test_call_model_rejects_non_chat_completion_response(monkeypatch):
    response_body = b'{"error":"unexpected provider response"}'
    monkeypatch.setattr(
        llm_client,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(response_body),
    )

    with pytest.raises(LLMClientError, match="choices\\[0\\]\\.message\\.content"):
        call_model(
            "test prompt",
            config=LLMConfig(
                endpoint="https://model.example.test/v1beta/openai/chat/completions",
                model="gemini-test-model",
                api_key="test-key",
                max_retries=0,
            ),
        )

import json

import pytest
from fastapi.testclient import TestClient

import app as api
from llm_client import MockLLMClient
from pipeline import ingest_documents
from scoring import score_chunks
from ingest import ingest_files


@pytest.fixture
def client(monkeypatch):
    with TestClient(api.app) as test_client:
        monkeypatch.setattr(api.pipeline, "_KNOWLEDGE_BASE", None)
        api.app.state.llm_client = None
        yield test_client
        api.app.state.llm_client = None


@pytest.mark.parametrize("route", ["/healthz", "/health"])
def test_health_routes_return_ok(client, route):
    response = client.get(route)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_returns_html_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Wananchi Actions</title>" in response.text


def test_static_hero_and_credits_are_served(client):
    hero = client.get("/static/img/hero.jpg")
    credits = client.get("/static/img/CREDITS.md")

    assert hero.status_code == 200
    assert hero.headers["content-type"].startswith("image/jpeg")
    assert hero.content.startswith(b"\xff\xd8")
    assert credits.status_code == 200
    assert "photograph" in credits.text


def test_status_reports_missing_model_configuration_without_values(client, monkeypatch):
    for name in ("RAG_LLM_ENDPOINT", "RAG_LLM_MODEL", "RAG_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    response = client.get("/status")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_configured": False,
        "missing_variables": [
            "RAG_LLM_ENDPOINT",
            "RAG_LLM_MODEL",
            "RAG_LLM_API_KEY",
        ],
    }


def test_status_reports_configured_without_exposing_values(client, monkeypatch):
    values = {
        "RAG_LLM_ENDPOINT": "https://model.example.test/v1/chat/completions",
        "RAG_LLM_MODEL": "test-model",
        "RAG_LLM_API_KEY": "test-secret-value",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    response = client.get("/status")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_configured": True,
        "missing_variables": [],
    }
    assert all(value not in response.text for value in values.values())


def test_startup_logs_configuration_names_but_not_values(monkeypatch, caplog):
    values = {
        "RAG_LLM_ENDPOINT": "https://model.example.test/v1/chat/completions",
        "RAG_LLM_MODEL": "test-model",
        "RAG_LLM_API_KEY": "startup-secret-value",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(api.pipeline, "_KNOWLEDGE_BASE", [])
    caplog.set_level("INFO", logger="app")

    with TestClient(api.app):
        pass

    assert "Model configuration ready" in caplog.text
    assert "RAG_LLM_ENDPOINT" in caplog.text
    assert "RAG_LLM_MODEL" in caplog.text
    assert "RAG_LLM_API_KEY" in caplog.text
    assert all(value not in caplog.text for value in values.values())


def test_supported_query_returns_cited_action_item(client):
    feedback_path = api.DATA_DIR / "feedback_batch_1.txt"
    chunks = ingest_files([feedback_path])
    ingest_documents([feedback_path])
    candidate = max(
        score_chunks("potholes on Mwangaza Road near Kifaru Market", chunks),
        key=lambda item: item.score,
    )
    mock_llm = MockLLMClient(
        [
            json.dumps(
                {
                    "status": "ok",
                    "action_items": [
                        {
                            "category": "roads",
                            "location": "Mwangaza Road near Kifaru Market",
                            "priority": "high",
                            "suggested_department": "Roads and Transport Department",
                            "summary": "Inspect the reported potholes.",
                            "citations": [
                                {
                                    "doc_id": candidate.chunk.doc_id,
                                    "chunk_id": candidate.chunk.chunk_id,
                                    "excerpt": candidate.chunk.text[:40],
                                }
                            ],
                        }
                    ],
                }
            )
        ]
    )
    api.app.state.llm_client = mock_llm

    response = client.post(
        "/process",
        json={"query": "potholes on Mwangaza Road near Kifaru Market"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["action_items"][0]["citations"]
    assert mock_llm.calls == 1


def test_missing_evidence_returns_insufficient_evidence(client):
    ingest_documents([api.DATA_DIR / "feedback_batch_1.txt"])
    mock_llm = MockLLMClient([])
    api.app.state.llm_client = mock_llm

    response = client.post("/process", json={"query": "quantum dentistry"})

    assert response.status_code == 200
    assert response.json() == {"status": "insufficient_evidence", "action_items": []}
    assert mock_llm.calls == 0


def test_service_error_uses_502_with_structured_error(client, monkeypatch):
    ingest_documents([api.DATA_DIR / "feedback_batch_1.txt"])
    monkeypatch.setattr(
        api.pipeline,
        "run_pipeline",
        lambda _query, **_kwargs: {
            "status": "error",
            "error": "model request failed: simulated upstream outage",
        },
    )

    response = client.post("/process", json={"query": "pothole on Mwangaza Road"})

    assert response.status_code == 502
    assert response.json() == {
        "status": "error",
        "error": (
            "Model request failed; verify the server-side endpoint, model, API key, "
            "and network configuration."
        ),
    }


def test_model_error_response_redacts_upstream_body_and_secret(client, monkeypatch):
    ingest_documents([api.DATA_DIR / "feedback_batch_1.txt"])
    secret = "dummy-secret-never-return-this"
    monkeypatch.setattr(
        api.pipeline,
        "run_pipeline",
        lambda _query, **_kwargs: {
            "status": "error",
            "error": (
                f"model endpoint returned HTTP 401: Authorization: Bearer {secret}\n"
                "Traceback (most recent call last)"
            ),
        },
    )

    response = client.post("/process", json={"query": "pothole on Mwangaza Road"})

    assert response.status_code == 502
    assert response.json() == {
        "status": "error",
        "error": "Model endpoint returned HTTP 401; verify the configured endpoint, model, and API key.",
    }
    assert secret not in response.text
    assert "Traceback" not in response.text


def test_missing_model_configuration_error_is_clear_and_safe(client, monkeypatch):
    for name in ("RAG_LLM_ENDPOINT", "RAG_LLM_MODEL", "RAG_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    ingest_documents([api.DATA_DIR / "feedback_batch_1.txt"])

    response = client.post("/process", json={"query": "potholes on Mwangaza Road"})

    assert response.status_code == 502
    assert response.json() == {
        "status": "error",
        "error": (
            "Model configuration is missing required variables: "
            "RAG_LLM_ENDPOINT, RAG_LLM_MODEL, RAG_LLM_API_KEY"
        ),
    }


def test_retrieval_error_uses_503_with_structured_error(client, monkeypatch):
    ingest_documents([api.DATA_DIR / "feedback_batch_1.txt"])
    monkeypatch.setattr(
        api.pipeline,
        "run_pipeline",
        lambda _query, **_kwargs: {
            "status": "error",
            "error": "retrieval failed: knowledge base is empty",
        },
    )

    response = client.post("/process", json={"query": "pothole on Mwangaza Road"})

    assert response.status_code == 503
    assert response.json() == {
        "status": "error",
        "error": "Retrieval failed; check the uploaded documents and server configuration.",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "x" * (api.MAX_QUERY_CHARS + 1)},
    ],
)
def test_invalid_query_is_rejected(client, payload):
    response = client.post("/process", json=payload)
    assert response.status_code == 422

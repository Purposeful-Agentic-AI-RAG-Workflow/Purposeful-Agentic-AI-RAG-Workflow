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


def test_healthz_returns_ok(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
        "error": "model request failed: simulated upstream outage",
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
        "error": "retrieval failed: knowledge base is empty",
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

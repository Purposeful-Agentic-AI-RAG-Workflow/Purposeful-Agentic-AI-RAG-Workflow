"""Local and hosted Uvicorn entry point for the FastAPI application."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
import uvicorn

load_dotenv()

import app as api


class _LocalMockLLM:
    """Small opt-in mock for local API smoke tests; never enabled by default."""

    def __call__(self, prompt: str, *, system_prompt: str = "") -> str:
        del system_prompt
        start_marker = "<BEGIN_UNTRUSTED_SELECTED_CHUNKS>\n"
        end_marker = "\n<END_UNTRUSTED_SELECTED_CHUNKS>"
        try:
            start = prompt.index(start_marker) + len(start_marker)
            end = prompt.index(end_marker, start)
            selected = json.loads(prompt[start:end])
        except (ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("local mock could not read selected chunks") from error

        if not selected:
            return json.dumps({"status": "insufficient_evidence", "action_items": []})

        chunk = selected[0]
        text = chunk["text"]
        excerpt = text[: min(len(text), 120)]
        if not excerpt:
            return json.dumps({"status": "insufficient_evidence", "action_items": []})
        action_item = {
            "category": chunk["section"] or "citizen feedback",
            "location": "See cited citizen feedback",
            "priority": "review",
            "suggested_department": "Relevant local department",
            "summary": f"Review the reported feedback: {excerpt}"[:500],
            "citations": [
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "excerpt": excerpt,
                }
            ],
        }
        return json.dumps({"status": "ok", "action_items": [action_item]})


def main() -> None:
    if os.environ.get("RAG_USE_MOCK_LLM", "").casefold() in {"1", "true", "yes"}:
        api.app.state.llm_client = _LocalMockLLM()

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

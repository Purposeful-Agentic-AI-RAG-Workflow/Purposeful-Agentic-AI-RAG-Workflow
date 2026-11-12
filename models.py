"""Shared data models and JSON schema for the bounded RAG pipeline."""

from dataclasses import dataclass
from typing import Literal, TypedDict


@dataclass(frozen=True, slots=True)
class Chunk:
    doc_id: str
    chunk_id: str
    section: str
    text: str


@dataclass(frozen=True, slots=True)
class Candidate:
    chunk: Chunk
    score: float


class Citation(TypedDict):
    doc_id: str
    chunk_id: str
    excerpt: str


class ActionItem(TypedDict):
    category: str
    location: str
    priority: str
    suggested_department: str
    summary: str
    citations: list[Citation]


ResultStatus = Literal["ok", "insufficient_evidence", "conflict", "error"]


class Result(TypedDict, total=False):
    status: ResultStatus
    action_items: list[ActionItem]
    error: str


ACTION_ITEM_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "category",
        "location",
        "priority",
        "suggested_department",
        "summary",
        "citations",
    ],
    "properties": {
        "category": {"type": "string", "minLength": 1},
        "location": {"type": "string", "minLength": 1},
        "priority": {"type": "string", "minLength": 1},
        "suggested_department": {"type": "string", "minLength": 1},
        "summary": {"type": "string", "minLength": 1},
        "citations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["doc_id", "chunk_id", "excerpt"],
                "properties": {
                    "doc_id": {"type": "string", "minLength": 1},
                    "chunk_id": {"type": "string", "minLength": 1},
                    "excerpt": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}


SYNTHESIS_JSON_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "action_items"],
    "properties": {
        "status": {
            "type": "string",
            "enum": ["ok", "insufficient_evidence", "conflict"],
        },
        "action_items": {
            "type": "array",
            "items": ACTION_ITEM_JSON_SCHEMA,
        },
    },
}

"""Verify that action-item citations point to selected chunks and exact excerpts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from models import Candidate


class CitationError(ValueError):
    """Raised when an action item cites unselected or unsupported evidence."""


def verify_citations(
    action_items: Sequence[Mapping[str, object]],
    selected: Iterable[Candidate],
) -> None:
    """Raise CitationError unless each citation matches selected chunk text exactly."""
    selected_chunks = {
        (candidate.chunk.doc_id, candidate.chunk.chunk_id): candidate.chunk
        for candidate in selected
    }

    for item_index, item in enumerate(action_items):
        citations = item.get("citations")
        if not isinstance(citations, list) or not citations:
            raise CitationError(
                f"action_items[{item_index}].citations must be a non-empty list"
            )
        for citation_index, citation in enumerate(citations):
            context = f"action_items[{item_index}].citations[{citation_index}]"
            if not isinstance(citation, Mapping):
                raise CitationError(f"{context} must be an object")

            doc_id = citation.get("doc_id")
            chunk_id = citation.get("chunk_id")
            excerpt = citation.get("excerpt")
            if not isinstance(doc_id, str) or not isinstance(chunk_id, str):
                raise CitationError(f"{context} requires string doc_id and chunk_id")
            if not isinstance(excerpt, str) or not excerpt.strip():
                raise CitationError(f"{context} requires a non-empty excerpt")

            chunk = selected_chunks.get((doc_id, chunk_id))
            if chunk is None:
                raise CitationError(
                    f"{context} cites unselected chunk ({doc_id!r}, {chunk_id!r})"
                )
            if excerpt not in chunk.text:
                raise CitationError(
                    f"{context} excerpt does not appear in chunk ({doc_id!r}, {chunk_id!r})"
                )

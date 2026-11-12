"""Hand-written BM25 scoring for English and Swahili citizen feedback."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence

from models import Candidate, Chunk

_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Return lowercase Unicode word tokens, including accented characters."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN_PATTERN.findall(normalized)


def _expand_query(
    query_tokens: Sequence[str],
    synonyms: Mapping[str, Sequence[str]] | None,
) -> list[str]:
    if not synonyms:
        return list(query_tokens)

    normalized_synonyms = {
        key.casefold(): tuple(token for phrase in values for token in tokenize(phrase))
        for key, values in synonyms.items()
    }
    expanded = list(query_tokens)
    for token in query_tokens:
        expanded.extend(normalized_synonyms.get(token, ()))
    return expanded


def score_chunks(
    query: str,
    chunks: Sequence[Chunk],
    synonyms: Mapping[str, Sequence[str]] | None = None,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[Candidate]:
    """Return one BM25-scored Candidate per chunk, in the original input order.

    Synonyms optionally map query terms to phrases, for example
    ``{"maji": ["water"]}``. Chunks with no matching query terms receive score 0.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if isinstance(k1, bool) or not isinstance(k1, (int, float)) or not math.isfinite(k1):
        raise ValueError("k1 must be a finite positive number")
    if isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b):
        raise ValueError("b must be a finite number between 0 and 1")
    if k1 <= 0:
        raise ValueError("k1 must be a finite positive number")
    if not 0 <= b <= 1:
        raise ValueError("b must be a finite number between 0 and 1")
    if not chunks:
        return []

    query_terms = _expand_query(tokenize(query), synonyms)
    tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
    lengths = [len(tokens) for tokens in tokenized_chunks]
    average_length = sum(lengths) / len(lengths)
    term_frequencies = [Counter(tokens) for tokens in tokenized_chunks]
    document_frequencies: Counter[str] = Counter()
    for frequencies in term_frequencies:
        document_frequencies.update(frequencies.keys())

    candidates: list[Candidate] = []
    for chunk, frequencies, document_length in zip(
        chunks,
        term_frequencies,
        lengths,
        strict=True,
    ):
        score = 0.0
        if query_terms:
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = document_frequencies[term]
                inverse_document_frequency = math.log(
                    1 + (len(chunks) - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                if average_length == 0:
                    length_normalization = 1.0
                else:
                    length_normalization = 1 - b + b * document_length / average_length
                score += inverse_document_frequency * (
                    frequency * (k1 + 1) / (frequency + k1 * length_normalization)
                )
        candidates.append(Candidate(chunk=chunk, score=score))
    return candidates

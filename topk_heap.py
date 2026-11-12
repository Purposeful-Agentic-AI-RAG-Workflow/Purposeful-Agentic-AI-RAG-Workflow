"""Bounded min-heap for selecting the best-scoring document chunks."""

from __future__ import annotations

import math
from numbers import Real
from typing import Iterable, TypeAlias

Candidate: TypeAlias = tuple[str, str, str, float]
CandidateKey: TypeAlias = tuple[str, str]


def _key(candidate: Candidate) -> tuple[float, str, str]:
    doc_id, chunk_id, _, score = candidate
    return (-score, doc_id, chunk_id)


def _is_worse(left: Candidate, right: Candidate) -> bool:
    """Return whether left ranks behind right (and belongs nearer the root)."""
    return _key(left) > _key(right)


class TopKHeap:
    """Keep at most K candidates, with the worst retained candidate at root."""

    def __init__(self, k: int) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("K must be a positive integer")
        self.k = k
        self.heap: list[Candidate] = []
        self._keys: set[CandidateKey] = set()
        self._positions: dict[CandidateKey, int] = {}

    @staticmethod
    def _validate(candidate: Candidate) -> Candidate:
        if not isinstance(candidate, tuple) or len(candidate) != 4:
            raise ValueError("candidate must be a 4-tuple: (doc_id, chunk_id, section, score)")
        doc_id, chunk_id, section, score = candidate
        if not isinstance(doc_id, str) or not isinstance(chunk_id, str):
            raise TypeError("doc_id and chunk_id must be strings")
        if not isinstance(section, str):
            raise TypeError("section must be a string")
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError("score must be a real number")
        if not math.isfinite(score):
            raise ValueError("score must be finite (NaN and infinity are not allowed)")
        return candidate

    def _swap(self, left: int, right: int) -> None:
        self.heap[left], self.heap[right] = self.heap[right], self.heap[left]
        left_key = (self.heap[left][0], self.heap[left][1])
        right_key = (self.heap[right][0], self.heap[right][1])
        self._positions[left_key] = left
        self._positions[right_key] = right

    def sift_up(self, index: int) -> None:
        while index > 0:
            parent = (index - 1) // 2
            if not _is_worse(self.heap[index], self.heap[parent]):
                break
            self._swap(index, parent)
            index = parent

    def sift_down(self, index: int) -> None:
        size = len(self.heap)
        while True:
            left = 2 * index + 1
            right = left + 1
            worst = index
            if left < size and _is_worse(self.heap[left], self.heap[worst]):
                worst = left
            if right < size and _is_worse(self.heap[right], self.heap[worst]):
                worst = right
            if worst == index:
                return
            self._swap(index, worst)
            index = worst

    def add(self, candidate: Candidate) -> bool:
        """Insert or update a candidate; return whether the retained heap changed."""
        candidate = self._validate(candidate)
        key = (candidate[0], candidate[1])

        if key in self._keys:
            index = self._positions[key]
            if not _is_worse(self.heap[index], candidate):
                return False
            self.heap[index] = candidate
            self.sift_down(index)
            return True

        if len(self.heap) < self.k:
            self.heap.append(candidate)
            self._keys.add(key)
            self._positions[key] = len(self.heap) - 1
            self.sift_up(len(self.heap) - 1)
            return True

        if not _is_worse(self.heap[0], candidate):
            return False

        old_key = (self.heap[0][0], self.heap[0][1])
        self._keys.remove(old_key)
        del self._positions[old_key]
        self.heap[0] = candidate
        self._keys.add(key)
        self._positions[key] = 0
        self.sift_down(0)
        return True

    def is_heap(self) -> bool:
        """Check the worst-at-root heap invariant and deduplication index."""
        if len(self.heap) > self.k or len(self._keys) != len(self.heap):
            return False
        if len(self._positions) != len(self.heap):
            return False
        for index, candidate in enumerate(self.heap):
            key = (candidate[0], candidate[1])
            if key not in self._keys or self._positions.get(key) != index:
                return False
            left = 2 * index + 1
            right = left + 1
            if left < len(self.heap) and _is_worse(self.heap[left], candidate):
                return False
            if right < len(self.heap) and _is_worse(self.heap[right], candidate):
                return False
        return True

    def results(self) -> list[Candidate]:
        """Return retained candidates in deterministic best-first order."""
        return sorted(self.heap, key=_key)


def select_top_k(candidates: Iterable[Candidate], k: int) -> list[Candidate]:
    """Select up to K unique candidates, preserving their metadata."""
    heap = TopKHeap(k)
    for candidate in candidates:
        heap.add(candidate)
    return heap.results()

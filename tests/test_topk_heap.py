import random
import math

import pytest

from topk_heap import TopKHeap, select_top_k


def baseline(candidates, k):
    unique = {}
    for candidate in candidates:
        key = candidate[:2]
        previous = unique.get(key)
        if previous is None or (-candidate[3], *key, candidate[2]) < (
            -previous[3],
            *key,
            previous[2],
        ):
            unique[key] = candidate
    return sorted(
        unique.values(),
        key=lambda item: (-item[3], item[0], item[1], item[2]),
    )[:k]


def test_k_one_matches_full_sort():
    candidates = [
        ("road", "a", "pothole", 0.3),
        ("water", "b", "leak", 0.9),
        ("waste", "c", "pickup", 0.6),
    ]
    assert select_top_k(candidates, 1) == baseline(candidates, 1)


def test_k_larger_than_pool_keeps_all_candidates_in_order():
    candidates = [
        ("doc-2", "chunk-1", "water", 0.8),
        ("doc-1", "chunk-1", "roads", 0.8),
    ]
    assert select_top_k(candidates, 5) == baseline(candidates, 5)


def test_empty_input_returns_empty_list():
    assert select_top_k([], 3) == []


@pytest.mark.parametrize("k", [0, -1, 1.5, True, "2", None])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError, match="positive integer"):
        select_top_k([], k)


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_non_finite_scores_are_rejected(score):
    with pytest.raises(ValueError, match="finite"):
        select_top_k([("doc", "chunk", "section", score)], 1)


@pytest.mark.parametrize("score", ["0.5", None, True, 1 + 2j])
def test_non_numeric_scores_are_rejected(score):
    with pytest.raises(TypeError, match="real number"):
        select_top_k([("doc", "chunk", "section", score)], 1)


def test_duplicate_ids_retain_the_best_version():
    candidates = [
        ("doc", "chunk", "old section", 0.4),
        ("other", "chunk", "roads", 0.7),
        ("doc", "chunk", "updated section", 0.9),
    ]
    assert select_top_k(candidates, 2) == baseline(candidates, 2)


def test_equal_score_duplicates_and_shuffled_inputs_are_deterministic():
    candidates = [
        ("doc-1", "chunk-1", "z-section", 0.9),
        ("doc-1", "chunk-1", "a-section", 0.9),
        ("doc-2", "chunk-1", "roads", 0.9),
        ("doc-3", "chunk-1", "water", 0.7),
        ("doc-4", "chunk-1", "waste", 0.8),
    ]
    expected = baseline(candidates, 3)
    randomizer = random.Random(42)
    for _ in range(50):
        shuffled = candidates.copy()
        randomizer.shuffle(shuffled)
        assert select_top_k(shuffled, 3) == expected


def test_ties_at_k_boundary_use_doc_then_chunk_id():
    candidates = [
        ("doc-b", "chunk-a", "water", 1.0),
        ("doc-a", "chunk-z", "roads", 1.0),
        ("doc-a", "chunk-b", "waste", 1.0),
        ("doc-a", "chunk-a", "roads", 1.0),
    ]
    assert select_top_k(candidates, 2) == baseline(candidates, 2)
    assert [item[:2] for item in select_top_k(candidates, 2)] == [
        ("doc-a", "chunk-a"),
        ("doc-a", "chunk-b"),
    ]


def test_large_pool_matches_full_sort_and_preserves_metadata():
    candidates = [
        (f"doc-{i:04}", f"chunk-{i % 7}", f"section-{i % 3}", (i * 37) % 101 / 100)
        for i in range(500)
    ]
    assert select_top_k(candidates, 23) == baseline(candidates, 23)


def test_heap_invariant_after_inserts_replacements_and_duplicate_updates():
    heap = TopKHeap(3)
    for candidate in [
        ("a", "1", "roads", 0.2),
        ("b", "1", "water", 0.8),
        ("c", "1", "waste", 0.5),
        ("d", "1", "roads", 0.9),
        ("a", "1", "roads revised", 0.95),
        ("e", "1", "waste", 0.1),
    ]:
        heap.add(candidate)
        assert heap.is_heap()
    assert heap.results() == baseline(heap.results(), 3)

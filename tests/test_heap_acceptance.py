import random

from topk_heap import TopKHeap, select_top_k


def _full_sort_baseline(candidates, k):
    best_by_id = {}
    for candidate in candidates:
        key = (candidate[0], candidate[1])
        previous = best_by_id.get(key)
        rank = (-candidate[3], candidate[0], candidate[1], candidate[2])
        previous_rank = (
            (-previous[3], previous[0], previous[1], previous[2])
            if previous is not None
            else None
        )
        if previous is None or rank < previous_rank:
            best_by_id[key] = candidate
    return sorted(
        best_by_id.values(),
        key=lambda item: (-item[3], item[0], item[1], item[2]),
    )[:k]


def test_shuffled_candidate_streams_match_deduplicated_full_sort():
    candidates = [
        ("doc-a", "chunk-1", "roads", 0.10),
        ("doc-b", "chunk-1", "water", 0.70),
        ("doc-c", "chunk-1", "waste", 0.70),
        ("doc-d", "chunk-1", "roads", 0.40),
        ("doc-e", "chunk-1", "water-z", 0.90),
        ("doc-f", "chunk-1", "waste", 0.65),
        ("doc-g", "chunk-1", "roads", 0.90),
        ("doc-h", "chunk-1", "water", 0.20),
        ("doc-i", "chunk-1", "waste", 0.55),
        ("doc-a", "chunk-1", "roads revised", 0.80),
        ("doc-e", "chunk-1", "water-a", 0.90),
        ("doc-b", "chunk-1", "water revised", 0.75),
    ]
    randomizer = random.Random(2026)

    for k in (1, 2, 4, 12):
        expected = _full_sort_baseline(candidates, k)
        for _ in range(40):
            shuffled = candidates.copy()
            randomizer.shuffle(shuffled)
            heap = TopKHeap(k)
            for candidate in shuffled:
                heap.add(candidate)
                assert heap.is_heap()
            assert heap.results() == expected
            assert select_top_k(shuffled, k) == expected


def test_full_heap_replaces_root_only_for_a_better_candidate():
    heap = TopKHeap(2)
    initial = [
        ("doc-a", "chunk-1", "roads", 0.20),
        ("doc-b", "chunk-1", "water", 0.40),
    ]
    for candidate in initial:
        assert heap.add(candidate)
        assert heap.is_heap()

    evicted = heap.heap[0]
    better = ("doc-c", "chunk-1", "waste", 0.90)
    assert heap.add(better)
    assert evicted not in heap.heap
    assert better in heap.heap
    assert heap.is_heap()

    retained = heap.heap.copy()
    worse = ("doc-d", "chunk-1", "roads", 0.10)
    assert not heap.add(worse)
    assert heap.heap == retained
    assert heap.is_heap()
    assert heap.results() == _full_sort_baseline(initial + [better, worse], 2)

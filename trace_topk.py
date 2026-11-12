"""Print the bounded heap after every insert, ignored candidate, or replacement."""

from topk_heap import TopKHeap, Candidate


def main() -> None:
    heap = TopKHeap(3)
    print("construct TopKHeap(K=3)")
    print(f"  heap: {heap.heap}")
    print(f"  valid heap: {heap.is_heap()}")
    candidates: list[Candidate] = [
        ("report-1", "chunk-1", "potholes", 0.62),
        ("report-2", "chunk-1", "water leak", 0.81),
        ("report-3", "chunk-1", "waste pickup", 0.74),
        ("report-4", "chunk-1", "road damage", 0.93),
        ("report-2", "chunk-1", "water leak, revised score", 0.88),
        ("report-5", "chunk-1", "streetlight", 0.51),
    ]

    for candidate in candidates:
        key = candidate[:2]
        was_duplicate = key in heap._keys
        was_full = len(heap.heap) == heap.k
        changed = heap.add(candidate)
        if not changed:
            operation = "ignored"
        elif was_duplicate:
            operation = "duplicate update"
        elif was_full:
            operation = "root replacement"
        else:
            operation = "insert"
        print(f"{operation}: {candidate}")
        print(f"  heap: {heap.heap}")
        print(f"  valid heap: {heap.is_heap()}")

    print(f"final top-K: {heap.results()}")


if __name__ == "__main__":
    main()

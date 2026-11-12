"""Print the bounded heap after every insert, ignored candidate, or replacement."""

from topk_heap import TopKHeap, Candidate


def main() -> None:
    heap = TopKHeap(3)
    candidates: list[Candidate] = [
        ("report-1", "chunk-1", "potholes", 0.62),
        ("report-2", "chunk-1", "water leak", 0.81),
        ("report-3", "chunk-1", "waste pickup", 0.74),
        ("report-4", "chunk-1", "road damage", 0.93),
        ("report-2", "chunk-1", "water leak, revised score", 0.88),
        ("report-5", "chunk-1", "streetlight", 0.51),
    ]

    for candidate in candidates:
        changed = heap.add(candidate)
        operation = "insert/replace" if changed else "ignored"
        print(f"{operation}: {candidate}")
        print(f"  heap: {heap.heap}")
        print(f"  valid heap: {heap.is_heap()}")

    print(f"final top-K: {heap.results()}")


if __name__ == "__main__":
    main()

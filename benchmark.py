"""Compare bounded top-K selection with sorting the entire candidate pool."""

from __future__ import annotations

import random
from time import perf_counter

from topk_heap import Candidate, select_top_k


def make_candidates(n: int) -> list[Candidate]:
    randomizer = random.Random(42)
    return [
        (f"doc-{index:06}", f"chunk-{index % 4}", f"section-{index % 5}", randomizer.random())
        for index in range(n)
    ]


def main() -> None:
    print(f"{'n':>8} {'K':>4} {'heap (ms)':>12} {'full sort (ms)':>16}")
    for n in (1_000, 10_000, 100_000):
        candidates = make_candidates(n)
        for k in (5, 50):
            start = perf_counter()
            heap_result = select_top_k(candidates, k)
            heap_ms = (perf_counter() - start) * 1_000

            start = perf_counter()
            sort_result = sorted(
                candidates,
                key=lambda item: (-item[3], item[0], item[1], item[2]),
            )[:k]
            sort_ms = (perf_counter() - start) * 1_000

            if heap_result != sort_result:
                raise AssertionError(f"heap and full-sort results differ for n={n}, K={k}")
            print(f"{n:>8} {k:>4} {heap_ms:>12.3f} {sort_ms:>16.3f}")


if __name__ == "__main__":
    main()

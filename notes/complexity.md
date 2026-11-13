# Top-K complexity and benchmark notes

## Selection and storage

`select_top_k` streams candidates into the hand-written bounded binary min-heap. The
least desirable retained candidate is at the root. When the heap is full, a candidate
replaces the root only if it is strictly better under the deterministic ranking:
higher score first, then lower `doc_id`, then lower `chunk_id` (with `section` used to
choose deterministically between equal-scoring duplicates with the same IDs).

- For K >= 2, each insertion or replacement may sift through the heap in O(log K),
  so scanning n candidates takes O(n log K) in the worst case.
- For K = 1, each candidate needs only a constant-time comparison with the retained
  best candidate, giving O(n) selection.
- Sorting the at most K retained results for final output costs O(K log K).
- The heap stores at most K candidates: O(K) storage.
- The deduplication set and position dictionary also store at most K retained
  `(doc_id, chunk_id)` keys: O(K) auxiliary storage. Hash membership and map access
  are expected O(1), so checking n incoming candidates adds expected O(n) hash-table
  work. Identifier hashing/comparison depends on string length. The combined expected
  bound remains O(n log K) for K >= 2 and O(n) for K = 1, plus the final sort.

## BM25 scoring cost

Let T be the total number of tokens in all chunks, N the number of chunks, and Q the
number of query tokens after optional synonym expansion. Tokenization and building
per-chunk term-frequency counters and document frequencies take expected O(T) time
(and work proportional to the total number of distinct terms in the chunks). Scoring
then performs up to Q frequency/document-frequency lookups per chunk, for expected
O(N * Q) time, plus query synonym expansion. The scorer retains tokenized chunks and
frequency counters, using O(T) memory. This is lexical matching rather than semantic
retrieval; synonyms are explicit and optional.

## Heap vs full-sort benchmark

Run with `python benchmark.py`. Each row uses the same deterministic candidate list
for both implementations. The benchmark input intentionally includes repeated
`(doc_id, chunk_id)` pairs. The full-sort baseline first applies the heap's duplicate
policy (keep the higher-scoring candidate; for equal scores, keep the
lexicographically lower section), then sorts unique candidates using the same
best-first tie order and takes K. The script compares the complete returned lists,
including metadata, and raises an assertion if any pair differs.

Timings are milliseconds from one local run; they are measurements, not performance
guarantees. Rerun on the target machine for its timings.

| Input candidates (n) | K | Bounded heap (ms) | Dedup + full sort (ms) | Identical results |
|---:|---:|---:|---:|:---:|
| 1,000 | 5 | 9.768 | 2.299 | Yes |
| 1,000 | 50 | 15.874 | 1.853 | Yes |
| 10,000 | 5 | 84.269 | 26.091 | Yes |
| 10,000 | 50 | 88.325 | 21.191 | Yes |
| 100,000 | 5 | 779.563 | 231.204 | Yes |
| 100,000 | 50 | 393.891 | 266.683 | Yes |

On this machine, C-optimized Python sorting is faster at these pool sizes. The heap's
benefit is bounded candidate storage during streaming selection, not lower wall-clock
time in this particular benchmark. The full-sort baseline materializes and sorts all
unique candidates, so its storage grows with the input size.

## Small heap trace

Captured with `python trace_topk.py`; `is_heap()` is true after construction and after
every operation, including insertion, root replacement, duplicate update, and an
ignored candidate:

```text
construct TopKHeap(K=3)
  heap: []
  valid heap: True
insert: ('report-1', 'chunk-1', 'potholes', 0.62)
  heap: [('report-1', 'chunk-1', 'potholes', 0.62)]
  valid heap: True
insert: ('report-2', 'chunk-1', 'water leak', 0.81)
  heap: [('report-1', 'chunk-1', 'potholes', 0.62), ('report-2', 'chunk-1', 'water leak', 0.81)]
  valid heap: True
insert: ('report-3', 'chunk-1', 'waste pickup', 0.74)
  heap: [('report-1', 'chunk-1', 'potholes', 0.62), ('report-2', 'chunk-1', 'water leak', 0.81), ('report-3', 'chunk-1', 'waste pickup', 0.74)]
  valid heap: True
root replacement: ('report-4', 'chunk-1', 'road damage', 0.93)
  heap: [('report-3', 'chunk-1', 'waste pickup', 0.74), ('report-2', 'chunk-1', 'water leak', 0.81), ('report-4', 'chunk-1', 'road damage', 0.93)]
  valid heap: True
duplicate update: ('report-2', 'chunk-1', 'water leak, revised score', 0.88)
  heap: [('report-3', 'chunk-1', 'waste pickup', 0.74), ('report-2', 'chunk-1', 'water leak, revised score', 0.88), ('report-4', 'chunk-1', 'road damage', 0.93)]
  valid heap: True
ignored: ('report-5', 'chunk-1', 'streetlight', 0.51)
  heap: [('report-3', 'chunk-1', 'waste pickup', 0.74), ('report-2', 'chunk-1', 'water leak, revised score', 0.88), ('report-4', 'chunk-1', 'road damage', 0.93)]
  valid heap: True
final top-K: [('report-4', 'chunk-1', 'road damage', 0.93), ('report-2', 'chunk-1', 'water leak, revised score', 0.88), ('report-3', 'chunk-1', 'waste pickup', 0.74)]
```

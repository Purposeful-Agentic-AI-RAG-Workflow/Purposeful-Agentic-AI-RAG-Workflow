# Complexity and benchmark notes

## Top-K selection

The selector streams candidates into the hand-written bounded binary min-heap.
After the heap reaches K entries, each candidate is compared with the least
desirable retained candidate at the root.

- For K >= 2, insertion or replacement takes O(log K), so scanning n candidates
  takes O(n log K) in the worst case.
- For K = 1, every candidate only needs a constant-time comparison with the
  current best, so selection is O(n).
- Sorting the retained results into final output order costs O(K log K).
- The heap stores at most K candidates: O(K) storage.
- The deduplication set and position map also store at most K retained
  `(doc_id, chunk_id)` keys: O(K) additional storage. Membership and update are
  expected O(1) per lookup with Python's hash set/dictionary; string hashing and
  equality depend on identifier length.
- For a stream of n inputs, dedup checks add expected O(n) hash-table work. Heap
  movement still determines the O(n log K) worst-case selection bound for K >= 2.

## BM25 scoring

Let T be the total number of tokens in all chunks, Q the number of query tokens,
and N the number of chunks. Tokenization and building term-frequency /
document-frequency counters take expected O(T) time. Scoring takes
O(N * Q) expected dictionary lookups, plus the cost of processing query synonyms.
The scorer keeps tokenized chunks and per-chunk counters, using O(T) memory.
This is lexical retrieval: it does not understand meaning beyond token matches and
the explicitly configured synonym map.

## Benchmark

`python benchmark.py` compares `select_top_k` with a full sort, checks that their
returned candidate lists are identical for every row, and fails on a mismatch.
Timings below are one local run in milliseconds; rerun on the target machine for
its measurements.

| n | K | Bounded heap (ms) | Full-sort baseline (ms) | Results identical |
|---:|---:|---:|---:|:---:|
| 1,000 | 5 | 5.203 | 1.153 | Yes |
| 1,000 | 50 | 6.134 | 1.198 | Yes |
| 10,000 | 5 | 65.293 | 22.770 | Yes |
| 10,000 | 50 | 84.504 | 30.532 | Yes |
| 100,000 | 5 | 571.460 | 312.761 | Yes |
| 100,000 | 50 | 782.943 | 389.290 | Yes |

On this machine, the Python heap implementation is slower than C-optimized
`sorted()` at these pool sizes. Its advantage is the bounded O(K) retained
candidate storage while selecting from a stream; benchmark timing depends on
Python version, hardware, candidate shape, and K.

## Small heap trace

Captured with `python trace_topk.py`:

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

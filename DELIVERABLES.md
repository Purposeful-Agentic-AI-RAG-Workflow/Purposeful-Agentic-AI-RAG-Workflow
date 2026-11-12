# Challenge deliverables checklist

| Challenge deliverable | Repository evidence | Status |
|---|---|---|
| Source code and fixed workflow | `app.py`, `pipeline.py`, `ingest.py`, `scoring.py`, `citations.py`, `topk_heap.py` | Present |
| Run and model configuration | `requirements.txt`, `.env.example`, README setup and model configuration | Present; endpoint values must be supplied by the team |
| Sample documents and source metadata | `data/feedback_batch_1.txt`, `data/feedback_batch_2.txt`, `data/department_responsibilities.md`, `data/sources.md` | Present; fictional data |
| Bounded heap implementation | `topk_heap.py`; covered by `tests/test_topk_heap.py` | Present |
| Heap trace | `trace_topk.py`; captured in `notes/complexity.md` | Present |
| Full-sort comparison | `benchmark.py` asserts equality for each benchmark row; timings in `notes/complexity.md` | Present |
| Complexity notes | `notes/complexity.md` | Present |
| Example cited outputs | API examples in `README.md`; end-to-end behavior in `tests/test_pipeline.py` and `tests/test_integration.py` | Present as examples/tests; live-model output not verified |
| Expected-answer references | `tests/gold_set.md` | Present; 15 questions |
| Failure and safety tests | `tests/test_api.py`, `tests/test_ingest.py`, `tests/test_pipeline.py`, `tests/test_integration.py`, `tests/test_topk_heap.py` | Present |
| API automated checks | `tests/test_api.py` | Present; full suite last run: 55 passed |
| Evaluation results | `evaluate.py`, generated `evaluation_results.md` | Present; mock support rate 14/15 (93.3%), source/citation checks only |
| Deployed HTTPS URL | Not available | **Missing** |
| Public health-check evidence | Local `uvicorn`/curl check returned `{"status":"ok"}`; no public URL | Partial; deployment verification missing |
| Hosted workflow evidence | No hosted workflow run, deployment log, or provider evidence checked in | **Missing** |
| Backup demo shot list | `docs/demo_video_shot_list.md` | Present; live-service run still pending |

## Known evaluation gap

The current mock evaluation does not pass question 7: top-K selection did not
retrieve `FB-037` for the Miale Road status question. The report records this
instead of counting it as supported. Also, the support rate is not a semantic
entailment metric. Review the real model's answers and citations before presenting
any quality claim.

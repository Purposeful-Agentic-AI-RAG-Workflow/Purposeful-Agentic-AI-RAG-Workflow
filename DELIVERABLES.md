# Challenge C deliverables

Status reflects repository evidence available as of this update. A reported
hosted check is marked partial until its output is recorded in
[`docs/hosted_evidence.md`](docs/hosted_evidence.md).

| Deliverable | Status | Evidence and remaining work |
|---|---|---|
| Source and bounded workflow | Done | `app.py`, `pipeline.py`, `ingest.py`, `scoring.py`, `topk_heap.py`, `citations.py`, and `models.py`; the fixed flow is documented in `README.md`. |
| Run and model configuration | Done | `requirements.txt`, `.env.example`, `llm_client.py`, and the setup/configuration section in `README.md`; no secret values are committed. |
| Sample documents and source metadata | Done | `data/feedback_batch_1.txt`, `data/feedback_batch_2.txt`, `data/department_responsibilities.md`, and `data/sources.md`; the feedback is fictional. |
| Heap implementation | Done | Hand-written bounded min-heap in `topk_heap.py`; acceptance and behavior tests in `tests/test_heap_acceptance.py` and `tests/test_topk_heap.py`. |
| Heap trace | Done | `trace_topk.py`; captured output and heap-invariant checks in `notes/complexity.md`. |
| Full-sort comparison | Done | `benchmark.py` compares the heap and deduplicated full-sort baseline using the same candidates, K, deduplication policy, and ordering; timings and matching results are in `notes/complexity.md`. |
| Complexity notes | Done | `notes/complexity.md` documents selection, final sorting, storage, deduplication, BM25, machine/Python version, timings, and pipeline integration. |
| Example inputs and cited outputs | Partial | PowerShell requests and example outputs are in `README.md`; pipeline behavior is covered in `tests/test_pipeline.py` and `tests/test_integration.py`. Live hosted outputs have not been pasted into the evidence template. |
| Expected-answer and source references | Done | The 15 questions and expected message IDs are in `tests/gold_set.md`; provenance is in `data/sources.md`. |
| Failure and safety tests | Done | `tests/test_api.py`, `tests/test_ingest.py`, `tests/test_pipeline.py`, `tests/test_integration.py`, and `tests/test_topk_heap.py` cover model/retrieval errors, malformed output, citations, conflict, injection, and heap behavior. Latest full suite: 67 passed, 1 Starlette deprecation warning. |
| Evaluation results | Done | `evaluate.py` and `evaluation_results.md`; the recorded evidence-only mock run reports 13/15 supported (86.7%). This is not a real-model quality score. |
| Deployed HTTPS URL | Partial | User-reported URL: `https://purposeful-agentic-ai-rag-workflow.primecloud.cloud`. Confirm it is current after the final deployment. |
| Health-check URL and output | Partial | User previously reported `GET https://purposeful-agentic-ai-rag-workflow.primecloud.cloud/healthz` works. Paste the final response and status into `docs/hosted_evidence.md`. |
| Deployed revision / commit ID | Missing | No deployed commit ID is recorded; the README and `docs/hosted_evidence.md` retain placeholders. |
| Deployment status screenshot | Missing | No screenshot has been added to the hosted evidence record. |
| Hosted workflow evidence | Partial | `scripts/verify_hosted.ps1`, `docs/service_error_demo.md`, and the blank `docs/hosted_evidence.md` template are present. Run the verifier against the final deployment and paste its outputs; record the sanitized service-error result after the key is restored. |
| Backup video | Partial | `docs/demo_video_shot_list.md` is available as a recording aid; the actual video and link are still missing. |
| Team details | Partial | Roles are described in `README.md`; replace the `[Name]` placeholders with the team members' names. |
| Alumni Contribution note | Missing | `README.md` contains a `TODO` placeholder. Add the agreed contribution note. |
| SDG statement | Done | `README.md` identifies SDG 11, Sustainable Cities and Communities, and connects it to cited citizen-feedback action items. |
| AI-tool disclosure | Missing | `README.md` contains a `TODO` placeholder. Add the required disclosure. |

## Evaluation caveats

The current mock support rate is 13/15 (86.7%) and checks retrieval, status,
and citation wiring, not semantic entailment. The report flags question 7 because
`FB-037` is not retrieved in the selected chunks, and question 14 because the
mock returns `insufficient_evidence` for the illegal-dumping task. Review
`evaluation_results.md` before presenting any model-quality claim.

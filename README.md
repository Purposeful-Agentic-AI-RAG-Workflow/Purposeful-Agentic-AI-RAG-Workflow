# Bounded Citizen Feedback RAG API

A small Python retrieval-augmented generation prototype for Challenge C / SDG 11.
It turns citizen feedback about roads, water, and waste into bounded, cited
action items. The sample data is fictional.

## Fixed workflow

```text
ingest -> chunk -> BM25 score -> bounded top-K heap -> synthesize -> verify citations -> done
```

The pipeline sends only selected chunks to the model. Document text is explicitly
marked as untrusted data and never treated as instructions. A citation must refer
to a selected `(doc_id, chunk_id)` and its excerpt must occur in that chunk.

## Setup

Use Python 3.11 or newer:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` for local model configuration. `.env` is ignored
by Git. Do not commit credentials.

## Run and model configuration

The API uses the OpenAI-compatible chat-completions request/response format.
There is no hosted provider or endpoint default; configure all three required
values for live model calls:

| Environment variable | Purpose | Required |
|---|---|---|
| `RAG_LLM_ENDPOINT` | Full chat-completions endpoint URL | Yes |
| `RAG_LLM_MODEL` | Model identifier expected by that endpoint | Yes |
| `RAG_LLM_API_KEY` | Secret sent as a bearer token | Yes |
| `RAG_LLM_TIMEOUT` | Request timeout in seconds (default `30`) | No |
| `RAG_LLM_MAX_RETRIES` | HTTP retry count from 0 to 2 (default `2`) | No |

Start locally:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

For a platform-provided port, bind to `0.0.0.0` and its configured port variable;
confirm that platform's exact variable name before deployment.

## API examples

### Health

```bash
curl http://127.0.0.1:8000/healthz
```

Response:

```json
{"status":"ok"}
```

### Ingest a text file

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "files=@data/feedback_batch_1.txt;type=text/plain"
```

Response:

```json
{"status":"ok","file_count":1,"chunk_count":1,"files":["<generated-upload-name>"]}
```

The exact chunk count depends on file contents and chunk limits. On startup, the
API attempts to ingest supported files already under `data/` when its knowledge
base is empty.

### Supported task

```bash
curl -X POST http://127.0.0.1:8000/process \
  -H "Content-Type: application/json" \
  -d '{"query":"Where are the potholes near Kifaru Market, and who should inspect them?"}'
```

Example response (the model wording can vary):

```json
{
  "status": "ok",
  "action_items": [
    {
      "category": "roads",
      "location": "Mwangaza Road near Kifaru Market",
      "priority": "high",
      "suggested_department": "Roads and Transport Department",
      "summary": "Inspect the reported potholes and assess the traffic risk.",
      "citations": [
        {
          "doc_id": "feedback_batch_1.txt",
          "chunk_id": "<selected-chunk-id>",
          "excerpt": "potholes on Mwangaza Road near Kifaru Market"
        }
      ]
    }
  ]
}
```

### Missing evidence

```bash
curl -X POST http://127.0.0.1:8000/process \
  -H "Content-Type: application/json" \
  -d '{"query":"How many litres were lost at a location with no leak report?"}'
```

Response:

```json
{"status":"insufficient_evidence","action_items":[]}
```

### Service error

With an invalid model key or unavailable configured endpoint:

```json
{"status":"error","error":"model request failed: ..."}
```

The API returns HTTP 502 for model failures, HTTP 503 for ingestion/retrieval
failures, and HTTP 500 for other internal errors. It does not turn these failures
into successful-looking action items.

## Tests and evaluation

```bash
python -m pytest -v
python benchmark.py
python trace_topk.py
python evaluate.py
```

`evaluate.py` uses a deterministic mock by default; it uses the real configured
endpoint when `RAG_LLM_ENDPOINT`, `RAG_LLM_MODEL`, and `RAG_LLM_API_KEY` are all
present. It writes `evaluation_results.md`. The mock evaluation checks expected
source retrieval and citation/source IDs, not semantic entailment; review
`evaluation_results.md` before using its support rate as a model-quality claim.

## Heap and complexity

The bounded binary min-heap retains at most K candidates with the least desirable
candidate at the root. Selection is O(n log K) for K >= 2 and O(n) for K = 1.
Final sorting is O(K log K). Heap storage is O(K); the dedup set and position
index add O(K) storage with expected O(1) hash lookups. The hand-written BM25
implementation tokenizes in O(T) over corpus tokens and scores in O(N * Q)
expected time for N chunks and Q query tokens. See
[`notes/complexity.md`](notes/complexity.md) for measured timings, baseline
comparison, and a trace.

## Limitations

- BM25 is lexical; synonyms are manually configured and the scorer does not
  provide semantic embeddings.
- Conflict classification is requested from the model using selected text; the
  application does not independently prove that two statements contradict.
- Citation membership and exact excerpt text are validated, but citation checks
  do not prove that the action summary is semantically entailed by the excerpt.
- The default chunking strategy uses character-bounded paragraph/message text;
  larger chunks may contain several feedback records.
- `evaluation_results.md` in mock mode is a deterministic wiring/source-coverage
  check, not an assessment of an external model.
- The repository has no verified public deployment URL yet. Do not expose
  uploaded sensitive data; this prototype does not implement authentication or
  durable multi-tenant storage.

## Team roles

- **Person A:** heap, ingestion, scoring, citations, pipeline, API, deployment,
  and project documentation.
- **Person B:** fictional feedback data, department/source notes, gold-set
  questions, and evaluation review.

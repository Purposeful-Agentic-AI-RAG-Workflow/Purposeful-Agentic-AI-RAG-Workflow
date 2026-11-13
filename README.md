# wananchi-actions

**Challenge C, Advanced: Purposeful Agentic AI / RAG Workflow**

**SDG 11 - Sustainable Cities and Communities:** turn fictional citizen feedback
about roads, water, and waste into concise, cited action items for human review.

This Python prototype demonstrates a bounded retrieval-augmented generation
(RAG) workflow. It is not a case-management system and does not dispatch public
works crews.

## Technology

- Python 3.11+
- FastAPI and Pydantic for the HTTP API and request validation
- BM25-style lexical scoring and a hand-written bounded binary min-heap
- An OpenAI-compatible chat-completions endpoint for synthesis
- Pytest for API, pipeline, ingestion, and heap tests

The dependencies are listed in [`requirements.txt`](requirements.txt). The API
uses Python's standard-library HTTP client for model requests; it does not select
a provider or model automatically.

## Fixed pipeline

```text
ingest -> chunk -> score -> heap top-K -> synthesize -> cite/verify -> done
```

Documents are cleaned and chunked, scored lexically, and streamed through a
bounded top-K heap. Only the selected chunks are sent to the model, inside a
delimited untrusted-data block. The model must return JSON; each accepted
citation must identify a selected chunk and quote an exact excerpt from it.
Unsupported requests return `insufficient_evidence`; contradictory evidence can
return `conflict`. Results are for human review.

## Setup, build, and run

Create a virtual environment and install the dependencies from the repository
root. There is no separate compile or asset-build step.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
```

The runtime service listens on port **8000** by default. Start it locally with:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The local API base URL is `http://127.0.0.1:8000`. For a hosted container, bind
to `0.0.0.0` and configure the start command to use the platform's assigned
port. Confirm the platform's port setting rather than assuming a variable name.

The service ingests the supported files under `data/` at startup when its
in-memory knowledge base is empty. Optional `/ingest` uploads are saved under
`data/uploads/` and added to that process's knowledge base. The runtime has no
database: uploaded files are local filesystem data and may be lost when a host
rebuilds or replaces its instance. `data/uploads/` is excluded from Git.

## Model configuration

For live synthesis, the service sends OpenAI-compatible chat-completions
requests to the configured endpoint. Configure the following values in the
server's environment or secrets manager. For local development, `.env.example`
lists the variable names; copy it to `.env` and set values privately. `.env` is
gitignored. Never commit or paste secret values.

| Environment variable | Purpose | Required |
|---|---|:---:|
| `RAG_LLM_ENDPOINT` | Full OpenAI-compatible chat-completions URL | Yes |
| `RAG_LLM_MODEL` | Model identifier accepted by that endpoint | Yes |
| `RAG_LLM_API_KEY` | Bearer-token credential | Yes |
| `RAG_LLM_TIMEOUT` | Request timeout in seconds; default `30` | No |
| `RAG_LLM_MAX_RETRIES` | HTTP retries, from `0` to `2`; default `2` | No |

`GET /status` reports whether the required values are present; it does not
validate the key or confirm that the endpoint is reachable. See
[`docs/deployment.md`](docs/deployment.md) before configuring a host.

## API and demonstration sequence

In another PowerShell window:

```powershell
$base = 'http://127.0.0.1:8000'
```

### Health and configuration

```powershell
Invoke-RestMethod -Uri "$base/healthz"
Invoke-RestMethod -Uri "$base/status" | ConvertTo-Json -Depth 10
```

Example health response:

```json
{"status":"ok"}
```

### Supported task: Mwangaza Road potholes

```powershell
$body = @{ query = 'Which department should handle the potholes on Mwangaza Road near Kifaru Market?' } | ConvertTo-Json
Invoke-RestMethod -Uri "$base/process" -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 10
```

Example response (wording and chunk ID vary by model and ingestion run):

```json
{
  "status": "ok",
  "action_items": [
    {
      "category": "roads",
      "location": "Mwangaza Road near Kifaru Market",
      "priority": "high",
      "suggested_department": "Roads and Transport Department",
      "summary": "Inspect the reported potholes.",
      "citations": [
        {
          "doc_id": "feedback_batch_1.txt",
          "chunk_id": "<selected-chunk-id>",
          "excerpt": "FB-001: The potholes on Mwangaza Road near Kifaru Market are getting bigger every week."
        }
      ]
    }
  ]
}
```

### No evidence: county road-repair budget

```powershell
$body = @{ query = "What is the county's budget for road repairs this year?" } | ConvertTo-Json
Invoke-RestMethod -Uri "$base/process" -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 10
```

Expected response:

```json
{"status":"insufficient_evidence","action_items":[]}
```

### Conflicting reports: Upendo Estate

```powershell
$body = @{ query = 'Are there three confirmed consecutive days of water outage at Upendo Estate?' } | ConvertTo-Json
Invoke-RestMethod -Uri "$base/process" -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 10
```

Expected conflict response when the model recognizes both reports:

```json
{"status":"conflict","action_items":[]}
```

The reports are FB-022 and FB-039. Conflict classification is model-generated;
the application does not independently infer contradictions.

### Prompt-injection message: FB-020

```powershell
$body = @{ query = 'What is the safety significance of FB-020?' } | ConvertTo-Json
Invoke-RestMethod -Uri "$base/process" -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 10
```

FB-020 is untrusted citizen content. Review that the model does not follow its
embedded instruction or disclose private information. Model wording and status
can vary.

### Service error

With an invalid server-side key or unavailable configured model endpoint, the
API returns a structured HTTP 502. This PowerShell example captures the status
and response body without placing a credential in the command:

```powershell
try {
    $body = @{ query = 'potholes on Mwangaza Road near Kifaru Market' } | ConvertTo-Json
    Invoke-RestMethod -Uri "$base/process" -Method Post -ContentType 'application/json' -Body $body -ErrorAction Stop | ConvertTo-Json -Depth 10
} catch {
    if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
        "HTTP_STATUS=$([int]$_.Exception.Response.StatusCode)"
    }
    if ($_.ErrorDetails.Message) {
        "BODY=$($_.ErrorDetails.Message)"
    } else {
        "ERROR=$($_.Exception.Message)"
    }
}
```

Example sanitized response:

```json
{"status":"error","error":"Model endpoint returned HTTP 401; verify the configured endpoint, model, and API key."}
```

The API uses HTTP 502 for model/configuration errors, HTTP 503 for
ingestion/retrieval errors, and HTTP 500 for other internal failures. Follow
[`docs/service_error_demo.md`](docs/service_error_demo.md) for the safe hosted
error demonstration, including restoring the valid key.

## Tests and evaluation

```powershell
python -m pytest -v
python benchmark.py
python trace_topk.py
python evaluate.py --mock
```

`evaluate.py` uses a deterministic evidence-only mock by default. It uses the
configured real endpoint when all required `RAG_LLM_*` settings are present;
that may make live model requests. The report is written to
[`evaluation_results.md`](evaluation_results.md). The mock support rate checks
retrieval, status, and citation wiring, not independent LLM quality or semantic
entailment. See [`tests/gold_set.md`](tests/gold_set.md) for the 15 evaluation
questions.

## Design and complexity

- The top-K selector is a hand-written binary min-heap with the least desirable
  retained candidate at the root. It does not use `heapq` or a library top-K
  routine.
- Candidates are deduplicated by `(doc_id, chunk_id)`. Ties are deterministic:
  higher score, then lower document ID, then lower chunk ID.
- For K >= 2 selection is O(n log K); for K = 1 it is O(n). Sorting the final
  selected results is O(K log K). Heap and dedup/index storage are O(K).
- BM25 tokenization is expected O(T) for T corpus tokens; scoring is expected
  O(N * Q) for N chunks and Q query tokens.
- Only verified excerpts from selected chunks are retained in an `ok` result.
  Model and retrieval failures are returned explicitly rather than converted to
  success-shaped action items.

See [`notes/complexity.md`](notes/complexity.md) for the machine-specific
benchmark, full-sort comparison, complexity detail, and trace.

## Security and known limitations

- Retrieved document text is untrusted data, placed inside a delimited prompt
  block; it must never be treated as instructions.
- Keep credentials server-side in environment settings. Do not put keys in
  source, screenshots, command history, or committed files.
- The sample feedback is fictional. Do not upload sensitive citizen data.
- BM25 is lexical retrieval, not semantic search; synonym expansion is limited
  and manually configured.
- The model is responsible for synthesis and conflict classification. Citation
  checks prove selected-chunk membership and exact excerpt matching, not that
  every natural-language claim is entailed.
- Paragraph chunking is character-bounded. A long record can span chunks, and
  selecting one fragment may omit adjacent context.
- There is no authentication, database, durable multi-tenant storage, or
  independent human-approval workflow in this prototype.
- The mock evaluation is a pipeline/source-coverage check. Its support rate must
  not be presented as a real model-quality score.

## Team

- **Person A - [Name]:** heap, ingestion, scoring, citations, pipeline, API,
  deployment, and integration tests.
- **Person B - [Name]:** fictional feedback data, source/department notes,
  gold-set questions, and evaluation review.

## Submission placeholders

- **AI-tool disclosure:** TODO
- **Alumni Contribution note:** TODO
- **Backup video link:** TODO
- **Deployed commit ID:** TODO

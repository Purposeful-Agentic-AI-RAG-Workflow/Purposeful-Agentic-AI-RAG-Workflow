# Hosted deployment evidence

Fill this in after the final deployment. Do not include API keys, tokens, or other
secrets in screenshots or pasted output.

**Deployed commit ID:**

**HTTPS URL:**

**Health URL:**

**Deployment status screenshot:**

## Verification calls

Paste the status code and response body for each call. The supported and no-evidence
queries are also run by `scripts/verify_hosted.ps1 -BaseUrl <HTTPS-URL>`.

### GET /healthz

```text

```

### GET /status

```text

```

### POST /process - supported Mwangaza Road pothole query

Request:

```json

```

Response:

```text

```

### POST /process - county road-repair budget (no evidence)

Request:

```json

```

Response:

```text

```

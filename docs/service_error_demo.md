# Safe hosted model-service error demonstration

This procedure intentionally verifies that a rejected model credential produces a
structured, sanitized API error. Do not paste, print, screenshot, or commit either
the valid or invalid API key. Use the PrimeCloud secret/environment-variable UI;
never put credentials in a PowerShell command, a source file, or Git.

## Procedure

1. In PrimeCloud, open the correct service and confirm its deployed branch/commit.
2. In the service's server-side environment variables or secrets, temporarily
   replace `RAG_LLM_API_KEY` with a known-invalid value. Keep the correct key in a
   secure password manager so it can be restored. Do not add the invalid key to
   `.env`, a local terminal command, or a repository file.
3. Save the setting and trigger **Rebuild & Deploy** (or the platform's equivalent
   restart/deploy action). Wait until the deployment reports success and its health
   check passes. `GET /status` may still report `model_configured: true`: this route
   checks that required values are present, not that the provider accepts the key.
4. From a PowerShell terminal, call a query that reaches synthesis, for example:

   ```powershell
   $base = 'https://YOUR_PUBLIC_HOST'
   $body = @{ query = 'potholes on Mwangaza Road near Kifaru Market' } | ConvertTo-Json
   try {
       Invoke-RestMethod -Uri "$base/process" -Method Post `
         -ContentType 'application/json' -Body $body -TimeoutSec 45 -ErrorAction Stop |
         ConvertTo-Json -Depth 20
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

   Expected: HTTP 502 and a JSON body such as
   `{"status":"error","error":"Model endpoint returned HTTP 401; verify the configured endpoint, model, and API key."}`.
   Provider responses can vary; the body must not expose the key, upstream response
   body, or a stack trace. If retrieval returns `insufficient_evidence`, choose a
   query known to match the ingested feedback so the request reaches the model.
5. Immediately restore the valid `RAG_LLM_API_KEY` in PrimeCloud, save, and rebuild/
   redeploy. Confirm the deployment is healthy, `GET /status` shows all required
   variables present, and a supported `/process` request succeeds.
6. Record only the deployment ID, timestamp, HTTP status, and sanitized response
   body as evidence. Redact host-specific account details if sharing logs. Never
   store either credential in the evidence.

## Interpretation

- HTTP 502 with a sanitized model-endpoint 401 indicates the deliberately invalid
  key reached the provider and the API returned a structured failure.
- `model_configured: true` alone does not validate endpoint reachability or key
  validity; it only indicates non-empty required settings.
- A gateway-generated page, connection error, or non-JSON `error code: 502` is not
  evidence that the API's error mapping was exercised. Check deployment and provider
  logs without exposing credentials.
- If the service fails to recover after restoring the key, verify the secret was
  saved in the correct service/environment and redeploy again.

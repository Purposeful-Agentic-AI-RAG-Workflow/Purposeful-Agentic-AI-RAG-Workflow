# PrimeCloud deployment preparation

## Values that must be confirmed

This repository does not yet contain a verified PrimeCloud deployment or a
PrimeCloud-specific project configuration. Confirm the provider's expected
application/service type, start-command shell, injected port variable, public
HTTPS URL, and health-check configuration before deploying.

The approved model endpoint and model identifier are also not in the repository.
Do not deploy with a guessed provider URL or model name.

## Provisional start command

If PrimeCloud provides a POSIX-compatible shell and a `PORT` environment
variable, the likely command is:

```sh
uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
```

Confirm the port variable and command format with the project settings. The API
does not bind a port itself; Uvicorn reads the command-line port. The fallback
`8000` is for local/manual execution only, not a claim about PrimeCloud's
required port.

## Secrets setup

Configure these values in the hosting provider's secret/environment-variable
settings, not in Git or a committed `.env` file:

| Variable | Value |
|---|---|
| `RAG_LLM_ENDPOINT` | Team-approved complete OpenAI-compatible chat-completions URL |
| `RAG_LLM_MODEL` | Model identifier approved for that endpoint |
| `RAG_LLM_API_KEY` | Secret key for that endpoint |
| `RAG_LLM_TIMEOUT` | Optional request timeout in seconds; default `30` |
| `RAG_LLM_MAX_RETRIES` | Optional integer from `0` to `2`; default `2` |

Never put key values in source, deployment manifests, logs, screenshots, or the
demo recording. Restart/redeploy after changing environment values if the
provider requires it.

## Post-deploy HTTPS verification

Replace `<PUBLIC_HTTPS_URL>` with the confirmed service URL. Do not use a
placeholder URL as if it were deployed.

1. Health:

   ```sh
   curl -i https://<PUBLIC_HTTPS_URL>/healthz
   ```

   Expect HTTP 200 and `{"status":"ok"}`.

2. Supported query:

   ```sh
   curl -i -X POST https://<PUBLIC_HTTPS_URL>/process \
     -H "Content-Type: application/json" \
     -d '{"query":"Which department should handle the potholes on Mwangaza Road near Kifaru Market?"}'
   ```

   Expect HTTP 200, status `ok`, and a citation with an exact excerpt from the
   selected sample feedback.

3. Missing evidence:

   ```sh
   curl -i -X POST https://<PUBLIC_HTTPS_URL>/process \
     -H "Content-Type: application/json" \
     -d '{"query":"How many litres were lost in an unreported water leak?"}'
   ```

   Expect HTTP 200 and `{"status":"insufficient_evidence","action_items":[]}`.

4. Forced service error:

   Temporarily replace `RAG_LLM_API_KEY` in the provider's secret settings with
   an invalid value, restart/redeploy as required, then send a query known to
   match ingested feedback:

   ```sh
   curl -i -X POST https://<PUBLIC_HTTPS_URL>/process \
     -H "Content-Type: application/json" \
     -d '{"query":"potholes on Mwangaza Road near Kifaru Market"}'
   ```

   Expect HTTP 502 and a structured `{"status":"error","error":"..."}` body.
   Restore the valid key immediately after the check and redeploy/restart as
   required. Do not record or publish either key.

Save the date, URL, status codes, response bodies with secrets removed, and
provider deployment identifier as evidence after these checks are actually run.

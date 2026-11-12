# Backup demo video shot list

Use the deployed HTTPS URL only after deployment is verified. Replace
`<PUBLIC_HTTPS_URL>` below with the confirmed URL. Keep secrets out of the
terminal recording and blur any provider console credentials.

1. **Introduce the workflow.** Show the source tree and explain:
   `ingest -> chunk -> BM25 score -> bounded top-K -> synthesize -> verify citations`.
2. **Show sample data.** Open `data/feedback_batch_1.txt` and show fictional
   citizen feedback and the prompt-injection test record. Emphasize that the
   text is treated as data.
3. **Show the heap.** Run:

   ```sh
   python trace_topk.py
   ```

   Point out the bounded heap, root replacement, and invariant checks.
4. **Show health over HTTPS.** Run:

   ```sh
   curl -i https://<PUBLIC_HTTPS_URL>/healthz
   ```

   Expected response: HTTP 200 and `{"status":"ok"}`.
5. **Show a cited supported query.** Run:

   ```sh
   curl -i -X POST https://<PUBLIC_HTTPS_URL>/process \
     -H "Content-Type: application/json" \
     -d '{"query":"Which department should handle the potholes on Mwangaza Road near Kifaru Market?"}'
   ```

   Show the `ok` status, action item, and source citation/excerpt.
6. **Show absent evidence.** Run:

   ```sh
   curl -i -X POST https://<PUBLIC_HTTPS_URL>/process \
     -H "Content-Type: application/json" \
     -d '{"query":"How many litres were lost in an unreported water leak?"}'
   ```

   Show the `insufficient_evidence` response with no action items.
7. **Show conflict handling.** Run:

   ```sh
   curl -i -X POST https://<PUBLIC_HTTPS_URL>/process \
     -H "Content-Type: application/json" \
     -d '{"query":"Are there three confirmed consecutive days of water outage at Upendo Estate?"}'
   ```

   Show a `conflict` response if the configured model identifies the contradiction
   between FB-022 and FB-039. Conflict classification is model-directed.
8. **Show automated verification.** Show the final pytest summary and
   `evaluation_results.md`; disclose that the mock support rate is not a
   real-model quality result.
9. **Optional failure demonstration.** Only if allowed, temporarily set an
   invalid model key through the hosting provider's secret settings, restart the
   service, and run:

   ```sh
   curl -i -X POST https://<PUBLIC_HTTPS_URL>/process \
     -H "Content-Type: application/json" \
     -d '{"query":"potholes on Mwangaza Road near Kifaru Market"}'
   ```

   Show the structured HTTP 502 error. Restore the approved key immediately.
   Never display or record secret values.

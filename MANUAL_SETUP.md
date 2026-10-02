# Manual Setup

The repository-side infrastructure is configured on `main`.

## Kilo API key

The free-model workflow does **not** require `KILO_API_KEY` for inference.

If you already have `KILO_API_KEY` configured as a GitHub Actions repository secret, the workflow will run a separate, non-model diagnostic before each activation:

- `VALID` means `kilo profile --json` successfully authenticated to Kilo;
- `INVALID` means Kilo explicitly rejected the credential as an authentication failure;
- `UNVERIFIED` means the profile check failed for another reason;
- `NOT_CONFIGURED` means the secret is absent.

The key is never printed, committed, or written to the repository. The diagnostic step does not send a paid model request.

## Free model

The research agent uses Kilo's OpenAI-compatible Gateway endpoint with the documented free model `minimax/minimax-m3:free`, registered as a local custom model. This bypasses the unstable built-in model-catalog resolution that previously produced:

`Model not found: kilo-auto/free`

and later:

`Model not found: minimax/minimax-m3:free`

The Gateway documentation defines the OpenAI-compatible endpoint at `https://api.kilo.ai/api/gateway`, and Kilo's custom-model documentation supports mapping a local model key to an API-facing model ID with the `id` field.

The model is currently listed by Kilo as free, with $0 input/output pricing. Free-model availability and upstream limits can change.

## Anonymous versus authenticated free inference

The agent itself does not receive `KILO_API_KEY`, so the normal research activation can use the free model anonymously. Kilo documents anonymous access to free models and currently applies a 200 requests/hour/IP limit to free-model requests, including authenticated free requests.

The API key is therefore treated as an optional diagnostic credential rather than a dependency for the research loop.

## No other credential is required

The workflow uses GitHub's automatically provided repository-scoped workflow token only in the trusted persistence step.

Kilo is not given a GitHub write token, a personal access token, or a credential for any other repository.

## Workflow behavior

- wakes every 5 minutes;
- permits only one active enterprise run;
- keeps the newest pending activation when a prior run is still active;
- diagnoses the optional Kilo API key without exposing it;
- gives Kilo a bounded execution window;
- allows Kilo to run research/backtests inside the disposable runner;
- preserves safe repository changes after normal completion or Kilo failure/timeout;
- blocks autonomous persistence of the workflow/security configuration;
- never gives Kilo a GitHub write token.

## Verification

After the workflow changes are committed, let the scheduled workflow create a fresh run or manually trigger **Actions -> Kilo Research Enterprise Wake -> Run workflow**.

Inspect the fresh run logs and the **Kilo API key diagnostic** step before relying on the 5-minute schedule.

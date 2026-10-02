# Manual Setup

The repository-side infrastructure is configured on `main`.

## Kilo API key

The research workflow does **not** require `KILO_API_KEY` for inference.

If `KILO_API_KEY` is configured as a GitHub Actions repository secret, the workflow runs a separate authenticated diagnostic before research starts:

- `VALID` means the Kilo Gateway profile request authenticated successfully;
- `INVALID` means Kilo explicitly rejected the credential;
- `UNVERIFIED` means the diagnostic request failed without identifying the credential as invalid;
- `NOT_CONFIGURED` means the secret is absent.

The key is never printed or committed. The research agent does not receive the key.

The diagnostic checks the authenticated Kilo profile endpoint directly: HTTP 200 is reported as VALID and HTTP 401 as INVALID. Other responses are reported as UNVERIFIED.

## Free model

The workflow queries Kilo's live `/api/gateway/models` catalogue and selects a currently available `:free` model suitable for tool use. It then registers that model as a local Kilo custom model pointed at the OpenAI-compatible Gateway.

This avoids hard-coding a free model ID that can disappear or change. Kilo documents the models endpoint as unauthenticated and the Gateway as OpenAI-compatible. It also documents anonymous access to free models.

The selected research model is free; no Kilo credits are required for the free-model request itself. Free-model availability and rate limits can change.

## Anonymous versus authenticated free inference

The research agent uses anonymous free inference. Kilo currently documents a 200 requests/hour/IP limit for anonymous free-model access.

The Kilo OS sandbox is disabled in the GitHub-hosted runner because the runner's Linux user-namespace policy causes Kilo's Bubblewrap backend to fail with "setting up uid map: Permission denied". The workflow instead relies on the disposable GitHub runner boundary, Kilo's explicit permission denials, no Kilo-side GitHub token, and the trusted persistence step.

The API key is therefore an optional diagnostic credential rather than a dependency of the research loop.

## No other credential is required

The workflow uses GitHub's automatically provided repository-scoped workflow token only in the trusted persistence step.

Kilo is not given a GitHub write token, a personal access token, or a credential for any other repository.

## Workflow behavior

- wakes every 5 minutes;
- permits only one active enterprise run;
- diagnoses the optional Kilo API key without exposing it;
- discovers a currently available free tool-capable model;
- gives Kilo a bounded execution window;
- allows Kilo to run research/backtests inside the disposable runner;
- preserves safe repository changes after normal completion or Kilo failure/timeout;
- protects workflow/security configuration from autonomous edits;
- never gives Kilo a GitHub write token.

## Verification

After the workflow changes are committed, let the scheduled workflow create a fresh run or manually trigger **Actions -> Kilo Research Enterprise Wake -> Run workflow**.

Inspect the **Kilo API key diagnostic** and **Free model discovery** sections in the workflow summary.

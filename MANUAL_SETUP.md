# Manual Setup

The repository-side infrastructure is configured on `main`.

## Required GitHub Actions secret

In repository **Settings -> Secrets and variables -> Actions -> Repository secrets**, create exactly:

```
KILO_API_KEY
```

Paste the single Kilo Gateway API key as its value.

Do not commit the key to the repository.

## No other credential is required

The workflow uses GitHub's automatically provided repository-scoped workflow token only in the trusted persistence step.

Kilo is not given a GitHub write token, a personal access token, or a credential for any other repository.

## Free model

The workflow uses Kilo's documented `kilo-auto/free` model tier through the built-in Kilo Gateway provider. Auto Free routes to currently available free models and requires no Kilo credits; free-model availability and rate limits can change. The workflow uses the trusted `KILO_API_KEY` environment variable for Gateway authentication and keeps the key out of repository files. Auto Free may route requests to providers that log prompts/outputs, so this public repository must contain no confidential data or secrets.

## What the workflow will do after the secret is added

- wake every 5 minutes;
- allow only one active enterprise run;
- keep the newest pending activation when a prior run is still active;
- give Kilo a bounded execution window;
- allow Kilo to run research/backtests inside the disposable runner;
- preserve safe repository changes after normal completion or Kilo failure/timeout;
- block autonomous persistence of the workflow/security configuration;
- never give Kilo a GitHub write token.

## Verification

After adding `KILO_API_KEY`, manually trigger **Actions -> Kilo Research Enterprise Wake -> Run workflow** once.

Then inspect the run logs and repository changes before relying on the 5-minute schedule.

# Manual Setup

The repository-side infrastructure is configured on `main`.

## Required GitHub Actions secret

No secret is required for the free-model workflow.

The workflow uses Kilo's built-in Gateway provider with a currently documented free model and anonymous free-model access. You may leave `KILO_API_KEY` unset.

Do not commit any API key to the repository.

## Optional Kilo API key

A `KILO_API_KEY` can be configured as a GitHub Actions repository secret if you later want authenticated Kilo Gateway access or change the enterprise to use paid models. It is not required for the current free-only workflow.

Adding an API key does not make the selected free model paid; free models remain $0 on Kilo's side. However, Kilo's current documentation does not promise a higher free-model request allowance for authenticated users, so do not add the key expecting unlimited or materially higher free throughput.

## Free model

The workflow currently pins `minimax/minimax-m3:free` instead of the Auto Free virtual model. This is deliberate: the latest activation reached Kilo successfully but the CLI returned `Model not found: kilo-auto/free`, so a concrete currently documented free model avoids that catalog-resolution failure.

Kilo's free models are available to authenticated and anonymous users. Anonymous free-model access is currently rate-limited to 200 requests per hour per IP, and free-model availability can change over time.

Auto Free (`kilo-auto/free`) remains a supported Kilo model tier, but its underlying routing changes server-side. The repository therefore prefers the explicit model for CI reliability until the CLI/catalog behavior is confirmed stable.

## No other credential is required

The workflow uses GitHub's automatically provided repository-scoped workflow token only in the trusted persistence step.

Kilo is not given a GitHub write token, a personal access token, or a credential for any other repository.

## What the workflow will do

- wake every 5 minutes;
- allow only one active enterprise run;
- keep the newest pending activation when a prior run is still active;
- give Kilo a bounded execution window;
- allow Kilo to run research/backtests inside the disposable runner;
- preserve safe repository changes after normal completion or Kilo failure/timeout;
- block autonomous persistence of the workflow/security configuration;
- never give Kilo a GitHub write token.

## Verification

After the workflow changes are committed, let the scheduled workflow create a fresh run or manually trigger **Actions -> Kilo Research Enterprise Wake -> Run workflow**.

Inspect the fresh run logs and repository changes before relying on the 5-minute schedule.

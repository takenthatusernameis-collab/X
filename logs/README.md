# Activation Logs

Keep concise human-readable activation records for substantive work, important failures, and durable handoffs.

**Timestamped logging is the default operational standard for new activation records.** When timestamps are observable, include UTC timestamps for the activation start and finish and for consequential work/checkpoints that make progress easier to reconstruct. Prefer ISO 8601 format such as `2026-10-03T15:43:00Z`. A timestamp records the UTC time at which the logged event or action actually occurred; never backfill or infer a precise time that was not observed. When only a date or coarser time is known, record only that known precision rather than inventing a precise timestamp.

The date-based activation filename remains a basic temporal anchor, but it is not a substitute for observed timestamps when those timestamps are available.

Do not store credentials or confidential information here.

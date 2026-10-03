# Activation Logs

Keep concise human-readable activation records for substantive work, important failures, and durable handoffs.

A timestamp may be included when useful for reconstructing chronology, but it is optional guidance rather than a required log field. When used, define it as the UTC time at which the logged event or action occurred, preferably in ISO 8601 format such as `2026-10-03T15:43:00Z`. Do not backfill or infer a precise time that was not observed. The date-based activation filename already provides a basic temporal anchor.

Do not store credentials or confidential information here.

#!/usr/bin/env python3
"""Print the runner's current UTC time in ISO 8601 (the `date` shell form is denied here)."""

from datetime import datetime, timezone

print(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
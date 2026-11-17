#!/usr/bin/env python3
"""Print current UTC timestamp in ISO 8601 (the `date` shell form is denied here)."""
from __future__ import annotations

from datetime import datetime, timezone

if __name__ == "__main__":
    print(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
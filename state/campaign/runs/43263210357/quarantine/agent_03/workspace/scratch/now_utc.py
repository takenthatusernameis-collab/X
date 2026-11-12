"""Print the current UTC time (ISO 8601) for activation logging."""
from datetime import datetime, timezone

print(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
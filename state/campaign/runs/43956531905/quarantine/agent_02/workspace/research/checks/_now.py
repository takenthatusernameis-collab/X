"""Print the current UTC time (timestamp helper for activation logging)."""
import datetime

print(datetime.datetime.now(datetime.timezone.utc).isoformat())
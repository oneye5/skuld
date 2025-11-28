from datetime import datetime, timezone
from typing import Optional


def millis_to_datetime_str(epoch_millis: int, date_format: str = "%d/%m/%Y - %H:%M:%S") -> str:

    if epoch_millis is None:
        raise ValueError("epoch_millis cannot be None")

    if not isinstance(epoch_millis, int):
        raise TypeError(f"epoch_millis must be an integer, got {type(epoch_millis)}")

    # Convert milliseconds to seconds
    epoch_seconds = epoch_millis / 1000.0

    # Create datetime object
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)

    # Format and return
    return dt.strftime(date_format)
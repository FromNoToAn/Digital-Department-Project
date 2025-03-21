import time


def validate_unix_timestamp(timestamp: int | str):
    """
    Validates whether the given timestamp is a valid Unix timestamp.

    Parameters
    ----------
    timestamp : int or str
        The timestamp to validate. It can be an integer or a string
        representing a Unix timestamp.

    Returns
    -------
    bool
        Returns `True` if the timestamp is valid and falls within
        the sensible range of years (1970-3000). Otherwise, returns `False`.

    Notes
    -----
    - The function ensures the timestamp is a valid integer and can be
      converted to a `struct_time` object using `time.gmtime()`.
    - Timestamps outside the Unix epoch range or exceeding system capabilities
      (e.g., year < 1970 or year > 3000) are considered invalid.
    """
    try:
        # Ensure the timestamp is an integer
        timestamp = int(timestamp)

        # Convert the timestamp to a struct_time object
        time_obj = time.gmtime(timestamp)

        # Optionally, check if the year is within a sensible range
        if time_obj.tm_year < 1970 or time_obj.tm_year > 3000:
            return False
        return True

    except (ValueError, OverflowError, OSError):
        # If it's not a valid integer or out of range, it's not valid
        return False

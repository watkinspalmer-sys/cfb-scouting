def parse_timecode(value: str) -> float:
    """Convert SS, MM:SS, or HH:MM:SS text into seconds."""
    text = value.strip()
    if not text:
        raise ValueError("Enter a timecode.")

    parts = text.split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("Timecode must look like 90, 12:34, or 1:02:03.") from exc

    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
        return hours * 3600 + minutes * 60 + seconds

    raise ValueError("Timecode must look like 90, 12:34, or 1:02:03.")


def format_timecode(total_seconds: float) -> str:
    total_seconds = max(0, int(round(total_seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"

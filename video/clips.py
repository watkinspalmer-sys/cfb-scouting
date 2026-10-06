from pathlib import Path
import subprocess


def extract_clip(
    source_video: str,
    output_path: str,
    start_seconds: float,
    end_seconds: float,
) -> str:
    """Extract a short play clip with ffmpeg.

    ffmpeg must be installed on the machine running the app.
    The clip is re-encoded for broad compatibility and predictable file size.
    """
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")

    source = Path(source_video)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    duration = end_seconds - start_seconds
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        str(source),
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(destination),
    ]

    subprocess.run(command, check=True, capture_output=True)
    return str(destination)

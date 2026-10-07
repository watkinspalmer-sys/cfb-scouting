from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def probe_duration(video_path: str | Path) -> float:
    """Return clip duration in seconds using ffprobe."""
    source = Path(video_path)
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "Unknown ffprobe error").strip()
        raise RuntimeError(detail)

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("Could not determine clip duration.") from exc

    if duration <= 0:
        raise RuntimeError("Clip duration must be greater than zero.")
    return duration


def extract_analysis_frames(
    clip_path: str | Path,
    output_dir: str | Path,
    frame_count: int = 8,
    width: int = 1280,
) -> list[tuple[Path, float]]:
    """
    Extract an ordered sequence of JPEG frames across a snap clip.

    Returns [(frame_path, seconds_from_clip_start), ...].
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    if frame_count < 4:
        raise ValueError("Use at least 4 frames for football snap analysis.")

    destination = Path(output_dir)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    duration = probe_duration(source)

    # Avoid the exact first/last frame, which are often transitions.
    start_fraction = 0.03
    end_fraction = 0.95
    if frame_count == 1:
        fractions = [0.5]
    else:
        step = (end_fraction - start_fraction) / (frame_count - 1)
        fractions = [start_fraction + (step * i) for i in range(frame_count)]

    extracted: list[tuple[Path, float]] = []
    for idx, fraction in enumerate(fractions, start=1):
        timestamp = min(max(duration * fraction, 0.0), max(duration - 0.02, 0.0))
        frame_path = destination / f"frame_{idx:02d}.jpg"

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                f"scale={int(width)}:-2",
                "-q:v",
                "3",
                str(frame_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "Unknown FFmpeg error").strip()
            raise RuntimeError(detail)

        extracted.append((frame_path, timestamp))

    return extracted

from __future__ import annotations

from math import ceil
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont


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

    # Avoid the exact first/last frame, which are often broadcast transitions.
    start_fraction = 0.03
    end_fraction = 0.95
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


def _font(size: int = 22):
    """Use Pillow's bundled default font without depending on a system font file."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _fit_frame(image: Image.Image, width: int, height: int) -> Image.Image:
    """Letterbox one frame into a fixed tile while preserving the broadcast aspect ratio."""
    source = image.convert("RGB")
    source.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "black")
    x = (width - source.width) // 2
    y = (height - source.height) // 2
    canvas.paste(source, (x, y))
    return canvas


def create_temporal_evidence_sheets(
    clip_path: str | Path,
    output_dir: str | Path,
    frame_count: int = 24,
    sheet_count: int = 3,
    columns: int = 2,
    tile_width: int = 640,
    tile_height: int = 360,
) -> dict:
    """
    Convert a snap clip into ordered, timestamped contact sheets for V3.

    The goal is to make temporal changes explicit to a cheap image model. Frames are
    numbered globally across the clip. Each sheet is read left-to-right, top-to-bottom.
    """
    if frame_count < 12:
        raise ValueError("V3 temporal evidence should use at least 12 frames.")
    if sheet_count < 1:
        raise ValueError("sheet_count must be at least 1.")
    if columns < 1:
        raise ValueError("columns must be at least 1.")

    destination = Path(output_dir)
    raw_dir = destination / "raw"
    sheets_dir = destination / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    frames = extract_analysis_frames(
        clip_path=clip_path,
        output_dir=raw_dir,
        frame_count=frame_count,
        width=max(tile_width, 960),
    )

    frames_per_sheet = ceil(frame_count / sheet_count)
    rows = ceil(frames_per_sheet / columns)
    label_height = 34
    tile_total_height = tile_height + label_height

    sheet_paths: list[Path] = []
    manifest: list[dict] = []
    font = _font(22)

    for sheet_index in range(sheet_count):
        start = sheet_index * frames_per_sheet
        end = min(start + frames_per_sheet, len(frames))
        chunk = frames[start:end]
        if not chunk:
            continue

        sheet = Image.new(
            "RGB",
            (columns * tile_width, rows * tile_total_height),
            "black",
        )
        draw = ImageDraw.Draw(sheet)

        for local_index, (frame_path, timestamp) in enumerate(chunk):
            global_index = start + local_index + 1
            row = local_index // columns
            col = local_index % columns
            x = col * tile_width
            y = row * tile_total_height

            with Image.open(frame_path) as image:
                tile = _fit_frame(image, tile_width, tile_height)
            sheet.paste(tile, (x, y))

            label = f"FRAME {global_index:02d}   t={timestamp:.2f}s"
            draw.rectangle(
                [x, y + tile_height, x + tile_width, y + tile_total_height],
                fill="black",
            )
            draw.text(
                (x + 10, y + tile_height + 6),
                label,
                fill="white",
                font=font,
            )

            manifest.append(
                {
                    "frame_index": global_index,
                    "timestamp_seconds": round(timestamp, 3),
                    "sheet_index": sheet_index + 1,
                    "position_on_sheet": local_index + 1,
                }
            )

        sheet_path = sheets_dir / f"timeline_{sheet_index + 1:02d}.jpg"
        sheet.save(sheet_path, format="JPEG", quality=84, optimize=True)
        sheet_paths.append(sheet_path)

    return {
        "clip_path": str(Path(clip_path)),
        "duration_seconds": probe_duration(clip_path),
        "frame_count": len(frames),
        "sheet_paths": [str(path) for path in sheet_paths],
        "manifest": manifest,
        "reading_order": "Sheets in numeric order; within each sheet read left-to-right, top-to-bottom.",
    }

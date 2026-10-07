from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Protocol

from openai import OpenAI

from models.schemas import FilmObservation
from video.frames import extract_analysis_frames


class VideoAnalyzer(Protocol):
    """Interface that any video model integration must implement."""

    def analyze_clip(self, clip_path: str, play_context: dict) -> FilmObservation:
        ...


def _nullable_enum(values: list[str]) -> dict:
    return {"type": ["string", "null"], "enum": values + [None]}


def football_chart_schema() -> dict:
    """Structured-output schema for one football snap."""
    properties = {
        "personnel": _nullable_enum(["10", "11", "12", "13", "20", "21", "22", "Other", "Unknown"]),
        "formation_family": _nullable_enum(["Gun", "Pistol", "Under Center", "Goalline", "Other", "Unknown"]),
        "initial_formation": _nullable_enum(["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]),
        "initial_backfield": _nullable_enum(["Split backs", "RB left", "RB right", "Pistol dot", "Empty", "Other", "Unknown"]),
        "final_formation": _nullable_enum(["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]),
        "final_backfield": _nullable_enum(["Split backs", "RB left", "RB right", "Pistol dot", "Empty", "Other", "Unknown"]),
        "formation_strength": _nullable_enum(["Left", "Right", "Balanced", "Boundary", "Field", "Unknown"]),
        "motion_present": {"type": ["boolean", "null"]},
        "motion_player": {"type": ["string", "null"]},
        "motion_type": _nullable_enum(["Across", "Jet", "Orbit", "Return", "Short", "Out to slot/wide", "Into backfield", "Trade", "Other", "Unknown"]),
        "motion_direction": {"type": ["string", "null"]},
        "motion_start_alignment": {"type": ["string", "null"]},
        "motion_end_alignment": {"type": ["string", "null"]},
        "shift_present": {"type": ["boolean", "null"]},
        "shift_description": {"type": ["string", "null"]},
        "film_play_type": _nullable_enum(["Run", "Pass", "RPO", "Scramble", "Sack", "Other", "Unknown"]),
        "run_concept": {"type": ["string", "null"]},
        "run_direction": _nullable_enum(["Left", "Right", "Middle", "Boundary", "Field", "Unknown"]),
        "pass_concept": {"type": ["string", "null"]},
        "rpo": {"type": ["boolean", "null"]},
        "play_action": {"type": ["boolean", "null"]},
        "defensive_personnel": {"type": ["string", "null"]},
        "front": {"type": ["string", "null"]},
        "initial_box_count": {"type": ["integer", "null"], "minimum": 0, "maximum": 11},
        "snap_box_count": {"type": ["integer", "null"], "minimum": 0, "maximum": 11},
        "shell": _nullable_enum(["1-High", "2-High", "0-High", "Unknown"]),
        "coverage": {"type": ["string", "null"]},
        "rushers": {"type": ["integer", "null"], "minimum": 0, "maximum": 11},
        "blitz": {"type": ["boolean", "null"]},
        "pressure_family": _nullable_enum(["Standard rush", "Blitz", "Sim pressure", "Creeper", "Zero pressure", "Unknown"]),
        "pressure_source": {"type": ["string", "null"]},
        "adjustment_trigger": _nullable_enum(["None", "Motion", "Shift", "Defensive stem", "Cadence/check", "Other", "Unknown"]),
        "adjustment_type": _nullable_enum(["None", "Bump", "Travel", "Safety rotation", "Front shift", "Box insert", "Box remove", "Other", "Unknown"]),
        "adjustment_player": {"type": ["string", "null"]},
        "adjustment_detail": {"type": ["string", "null"]},
        "overall_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "uncertain_fields": {"type": "array", "items": {"type": "string"}},
        "analysis_notes": {"type": "string"},
    }
    return {
        "type": "json_schema",
        "name": "football_snap_chart",
        "description": "Structured scouting chart for one American football snap.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
            "additionalProperties": False,
        },
    }


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _prompt(play_context: dict, frame_times: list[float]) -> str:
    context = {
        "chart_team": play_context.get("team"),
        "chart_team_role": play_context.get("chart_side"),
        "quarter": play_context.get("period"),
        "game_clock": play_context.get("clock"),
        "down": play_context.get("down"),
        "distance": play_context.get("distance"),
        "play_text": play_context.get("play_text"),
    }

    times = ", ".join(f"{seconds:.1f}s" for seconds in frame_times)
    return f"""
You are charting ONE college football snap for a scouting database.

The images are sequential frames from the same short broadcast clip, ordered from
earliest to latest. Approximate clip-relative frame times are: {times}.

Game context:
{json.dumps(context, indent=2)}

Chart BOTH the offense and defense from the film. The team named in chart_team is
identified only so you know which team is offense/defense on this snap.

Football charting rules:
- Personnel means WHO is on the field, not where they align. A RB motioning to
  receiver does not change 20 personnel into 10 personnel.
- initial_formation / initial_backfield describe the earliest settled offensive
  alignment visible before motion/shift.
- final_formation / final_backfield describe alignment at the snap.
- Motion means a player is moving immediately before/through the snap. A shift is
  a change of alignment followed by the offense becoming set.
- If the camera angle cannot support a label, use Unknown or null. Do not invent.
- Count the box from defenders structurally committed to the run box.
- A Blitz is normally 5+ rushers. Sim pressure/Creeper can have four rushers with
  a non-traditional rusher and a dropper.
- Coverage should be conservative. If post-snap evidence is insufficient, use
  "Unknown".
- Concepts should use common football terminology (inside zone, counter, power,
  stretch, pitch, mesh, hitches, four verts, screen, etc.) only when supported.
- Use the play_text as structured outcome context, but do NOT let it override
  what the film shows for personnel, formation, motion, front, shell, pressure,
  or concept.
- uncertain_fields should list fields you would want a human to review.
- overall_confidence is your confidence in the chart as a whole from 0 to 1.
"""


def analyze_clip_openai(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gpt-5",
    frame_count: int = 8,
) -> dict:
    """
    Analyze one snap by sampling frames and sending the ordered sequence to OpenAI.

    Full source broadcasts remain local; only sampled JPEG frames from the short
    extracted snap are sent to the model.
    """
    source = Path(clip_path)
    frame_dir = Path("tmp_ai_frames") / source.stem
    frames = extract_analysis_frames(source, frame_dir, frame_count=frame_count)

    content: list[dict] = [
        {
            "type": "input_text",
            "text": _prompt(play_context, [timestamp for _, timestamp in frames]),
        }
    ]
    for index, (frame_path, timestamp) in enumerate(frames, start=1):
        content.append(
            {
                "type": "input_text",
                "text": f"Frame {index} at approximately {timestamp:.1f} seconds:",
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": _image_data_url(frame_path),
                "detail": "high",
            }
        )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        text={"format": football_chart_schema()},
        store=False,
    )

    if not response.output_text:
        raise RuntimeError("The model returned no structured chart.")

    try:
        result = json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"The model response was not valid JSON: {response.output_text[:500]}"
        ) from exc

    result["_model"] = model
    result["_frame_count"] = len(frames)
    result["_frame_times"] = [timestamp for _, timestamp in frames]
    return result


def analyze_clip_stub(clip_path: str, play_context: dict) -> FilmObservation:
    """Legacy placeholder retained for provider-neutral architecture tests."""
    return FilmObservation(
        play_id=str(play_context.get("playId")) if play_context.get("playId") is not None else None,
        game_id=str(play_context.get("gameId")) if play_context.get("gameId") is not None else None,
        notes=f"Video analysis not configured yet for {clip_path}",
    )

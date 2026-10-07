from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal, Optional, Protocol

from google import genai
from pydantic import BaseModel, Field

from models.schemas import FilmObservation


class VideoAnalyzer(Protocol):
    """Interface that any video model integration must implement."""

    def analyze_clip(self, clip_path: str, play_context: dict) -> FilmObservation:
        ...


class FootballSnapChart(BaseModel):
    personnel: Optional[Literal["10", "11", "12", "13", "20", "21", "22", "Other", "Unknown"]] = None
    formation_family: Optional[Literal["Gun", "Pistol", "Under Center", "Goalline", "Other", "Unknown"]] = None
    initial_formation: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    initial_backfield: Optional[Literal["Split backs", "RB left", "RB right", "Pistol dot", "Empty", "Other", "Unknown"]] = None
    final_formation: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    final_backfield: Optional[Literal["Split backs", "RB left", "RB right", "Pistol dot", "Empty", "Other", "Unknown"]] = None
    formation_strength: Optional[Literal["Left", "Right", "Balanced", "Boundary", "Field", "Unknown"]] = None

    motion_present: Optional[bool] = None
    motion_player: Optional[str] = None
    motion_type: Optional[Literal["Across", "Jet", "Orbit", "Return", "Short", "Out to slot/wide", "Into backfield", "Trade", "Other", "Unknown"]] = None
    motion_direction: Optional[str] = None
    motion_start_alignment: Optional[str] = None
    motion_end_alignment: Optional[str] = None

    shift_present: Optional[bool] = None
    shift_description: Optional[str] = None

    film_play_type: Optional[Literal["Run", "Pass", "RPO", "Scramble", "Sack", "Other", "Unknown"]] = None
    run_concept: Optional[str] = None
    run_direction: Optional[Literal["Left", "Right", "Middle", "Boundary", "Field", "Unknown"]] = None
    pass_concept: Optional[str] = None
    rpo: Optional[bool] = None
    play_action: Optional[bool] = None

    defensive_personnel: Optional[str] = None
    front: Optional[str] = None
    initial_box_count: Optional[int] = Field(default=None, ge=0, le=11)
    snap_box_count: Optional[int] = Field(default=None, ge=0, le=11)
    shell: Optional[Literal["1-High", "2-High", "0-High", "Unknown"]] = None
    coverage: Optional[str] = None
    rushers: Optional[int] = Field(default=None, ge=0, le=11)
    blitz: Optional[bool] = None
    pressure_family: Optional[Literal["Standard rush", "Blitz", "Sim pressure", "Creeper", "Zero pressure", "Unknown"]] = None
    pressure_source: Optional[str] = None

    adjustment_trigger: Optional[Literal["None", "Motion", "Shift", "Defensive stem", "Cadence/check", "Other", "Unknown"]] = None
    adjustment_type: Optional[Literal["None", "Bump", "Travel", "Safety rotation", "Front shift", "Box insert", "Box remove", "Other", "Unknown"]] = None
    adjustment_player: Optional[str] = None
    adjustment_detail: Optional[str] = None

    overall_confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    analysis_notes: str = ""


def _prompt(play_context: dict) -> str:
    context = {
        "chart_team": play_context.get("team"),
        "chart_team_role": play_context.get("chart_side"),
        "quarter": play_context.get("period"),
        "game_clock": play_context.get("clock"),
        "down": play_context.get("down"),
        "distance": play_context.get("distance"),
        "play_text": play_context.get("play_text"),
    }

    return f"""
You are an expert college football film analyst charting ONE snap for a scouting database.

Game context:
{json.dumps(context, indent=2)}

Watch the supplied video clip from beginning to end and chart BOTH the offense and defense.

Charting rules:
- Personnel means WHO is on the field, not where players align. If a RB motions
  from the backfield to receiver, 20 personnel remains 20 personnel.
- initial_formation / initial_backfield describe the earliest settled offensive
  alignment visible before motion or shift.
- final_formation / final_backfield describe the alignment at the snap.
- Motion means a player is still moving immediately before/through the snap.
  A shift is a change of alignment followed by the offense becoming set.
- Distinguish field and boundary when the broadcast angle supports it.
- Count defenders structurally committed to the run box.
- Blitz normally means 5+ rushers. Sim pressure or creeper can rush four while
  bringing a non-traditional rusher and dropping a traditional rush player.
- Coverage must be conservative. Use Unknown if the broadcast angle or clip does
  not show enough of the secondary after the snap.
- Use common football terminology for concepts only when supported by the film:
  inside zone, outside zone/stretch, power, counter, duo, pitch, draw, screen,
  mesh, hitches, four verts, flood, smash, etc.
- The play-by-play is outcome context only. It must NOT override what the film
  shows for personnel, formation, motion, front, box, shell, pressure, or concept.
- Use Unknown/null instead of guessing.
- uncertain_fields must list fields that should receive human review.
- overall_confidence is confidence in the complete chart from 0 to 1.

Pay special attention to the several seconds immediately before the snap so that
motions, shifts, defensive bumps, safety rotations, and box changes are not missed.
"""


def analyze_clip_gemini(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 3.0,
) -> dict:
    """
    Upload one short snap clip to Gemini and return a structured football chart.

    The clip is temporarily uploaded to Google's Files API, analyzed using static
    video processing at the requested FPS, then deleted from the Files API.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    client = genai.Client(api_key=api_key)
    uploaded = None

    try:
        uploaded = client.files.upload(file=str(source))

        while not uploaded.state or uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)

        if uploaded.state and uploaded.state.name == "FAILED":
            raise RuntimeError("Gemini failed to process the uploaded video.")

        interaction = client.interactions.create(
            model=model,
            input=[
                {
                    "type": "video",
                    "uri": uploaded.uri,
                    "mime_type": uploaded.mime_type,
                    "processing": {
                        "type": "static",
                        "fps": float(fps),
                    },
                },
                {
                    "type": "text",
                    "text": _prompt(play_context),
                },
            ],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": FootballSnapChart.model_json_schema(),
            },
        )

        if not interaction.output_text:
            raise RuntimeError("Gemini returned no structured chart.")

        result = json.loads(interaction.output_text)
        result["_model"] = model
        result["_video_fps"] = float(fps)
        result["_provider"] = "google-gemini"
        return result

    finally:
        if uploaded is not None and uploaded.name:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def analyze_clip_stub(clip_path: str, play_context: dict) -> FilmObservation:
    """Legacy placeholder retained for provider-neutral architecture tests."""
    return FilmObservation(
        play_id=str(play_context.get("playId")) if play_context.get("playId") is not None else None,
        game_id=str(play_context.get("gameId")) if play_context.get("gameId") is not None else None,
        notes=f"Video analysis not configured yet for {clip_path}",
    )

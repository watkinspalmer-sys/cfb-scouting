from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Literal, Optional, Protocol

from google import genai
from pydantic import BaseModel, Field

from models.schemas import FilmObservation
from video.frames import create_temporal_evidence_sheets, extract_temporal_frames_v31, extract_snap_centered_frames_v34


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





class EligibleSkillPlayerV42(BaseModel):
    """One of the five non-QB, non-OL eligible offensive players."""

    player_label: str
    role: Literal["RB/FB", "TE/H-back", "WR", "Unknown"]
    evidence: str = ""


class FootballSnapChartV42(FootballSnapChart):
    """Full V1 chart plus mechanical offensive personnel accounting."""

    offensive_lineman_count: int = Field(ge=0, le=7)
    quarterback_count: int = Field(ge=0, le=2)
    eligible_players: list[EligibleSkillPlayerV42] = Field(min_length=5, max_length=5)
    personnel_accounting_notes: str = ""




class MechanicalPersonnelPassV43(BaseModel):
    """Personnel-only accounting pass, isolated from the full chart."""

    offensive_lineman_count: int = Field(ge=0, le=7)
    quarterback_count: int = Field(ge=0, le=2)
    eligible_players: list[EligibleSkillPlayerV42] = Field(min_length=5, max_length=5)
    confidence: float = Field(ge=0, le=1)
    notes: str = ""




class RosterEligiblePlayerV44(BaseModel):
    """One visible eligible offensive player whose jersey number is read from film."""

    player_label: str
    jersey_number: Optional[str] = None
    evidence: str = ""


class RosterPersonnelPassV44(BaseModel):
    """Personnel-only vision pass: identify players, do not infer their positions."""

    offensive_lineman_count: int = Field(ge=0, le=7)
    quarterback_count: int = Field(ge=0, le=2)
    eligible_players: list[RosterEligiblePlayerV44] = Field(min_length=5, max_length=5)
    confidence: float = Field(ge=0, le=1)
    notes: str = ""




class HybridEligiblePlayerV45(BaseModel):
    """Eligible-player evidence combining cautious jersey reading with whole-sequence role evidence."""

    player_label: str
    jersey_number: Optional[str] = None
    jersey_readability: Literal["Clear", "Partial", "Unreadable"]
    visual_role: Literal["RB/FB", "TE/H-back", "WR", "Unknown"]
    initial_alignment: str = ""
    final_alignment: str = ""
    identity_evidence: str = ""


class HybridPersonnelPassV45(BaseModel):
    """Personnel-only pass for hybrid roster/visual evidence."""

    offensive_lineman_count: int = Field(ge=0, le=7)
    quarterback_count: int = Field(ge=0, le=2)
    eligible_players: list[HybridEligiblePlayerV45] = Field(min_length=5, max_length=5)
    confidence: float = Field(ge=0, le=1)
    notes: str = ""




class ConsensusEligiblePlayerV46(BaseModel):
    """Eligible-player observation for deterministic visual-role consensus."""

    player_label: str
    initial_slot: Literal[
        "Backfield 1",
        "Backfield 2",
        "Left wide",
        "Left slot",
        "Left core/attached",
        "Right core/attached",
        "Right slot",
        "Right wide",
        "Other",
        "Unknown",
    ]
    jersey_number: Optional[str] = None
    jersey_readability: Literal["Clear", "Partial", "Unreadable"]
    visual_role: Literal["RB/FB", "TE/H-back", "WR", "Unknown"]
    initial_alignment: str = ""
    final_alignment: str = ""
    identity_evidence: str = ""


class PersonnelConsensusPassV46(BaseModel):
    """Personnel-only pass for deterministic three-run role consensus."""

    offensive_lineman_count: int = Field(ge=0, le=7)
    quarterback_count: int = Field(ge=0, le=2)
    eligible_players: list[ConsensusEligiblePlayerV46] = Field(min_length=5, max_length=5)
    confidence: float = Field(ge=0, le=1)
    notes: str = ""


class NativeCorrectionV4(BaseModel):
    """Narrow second native-video pass for fields where V1 historically underperforms."""

    personnel: Optional[Literal["10", "11", "12", "13", "20", "21", "22", "Other", "Unknown"]] = None
    back_count: Optional[int] = Field(default=None, ge=0, le=3)
    te_count: Optional[int] = Field(default=None, ge=0, le=4)

    initial_receiver_structure: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    initial_backfield: Optional[Literal["Split backs", "RB left", "RB right", "Pistol dot", "Empty", "Other", "Unknown"]] = None
    final_receiver_structure: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    final_backfield: Optional[Literal["Split backs", "RB left", "RB right", "Pistol dot", "Empty", "Other", "Unknown"]] = None

    motion_type: Optional[Literal["Across", "Jet", "Orbit", "Return", "Short", "Out to slot/wide", "Into backfield", "Trade", "Other", "Unknown"]] = None
    run_direction: Optional[Literal["Left", "Right", "Middle", "Boundary", "Field", "Unknown"]] = None

    confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    notes: str = ""


class TemporalEvidenceChart(BaseModel):
    """One-pass chart from timestamped chronological contact sheets."""
    snap_frame_index: Optional[int] = Field(default=None, ge=1)
    snap_timestamp_seconds: Optional[float] = Field(default=None, ge=0)
    back_count: Optional[int] = Field(default=None, ge=0, le=3)
    attached_te_count: Optional[int] = Field(default=None, ge=0, le=3)
    wing_hback_count: Optional[int] = Field(default=None, ge=0, le=3)
    flexed_te_count: Optional[int] = Field(default=None, ge=0, le=3)
    split_wr_count: Optional[int] = Field(default=None, ge=0, le=5)
    formation_family: Optional[Literal["Gun", "Pistol", "Under Center", "Goalline", "Other", "Unknown"]] = None
    initial_receiver_structure: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    final_receiver_structure: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    formation_strength: Optional[Literal["Left", "Right", "Balanced", "Boundary", "Field", "Unknown"]] = None
    rb_initial_side: Optional[Literal["Left", "Right", "Behind", "Multiple", "None", "Unknown"]] = None
    rb_final_side: Optional[Literal["Left", "Right", "Behind", "Multiple", "None", "Unknown"]] = None
    rb_changed_sides: Optional[bool] = None
    rb_set_before_snap: Optional[bool] = None
    any_player_moving_at_snap: Optional[bool] = None
    any_alignment_change_set_before_snap: Optional[bool] = None
    primary_mover: Optional[str] = None
    movement_type: Optional[Literal["Across", "Jet", "Orbit", "Return", "Short", "Out to slot/wide", "Into backfield", "Trade", "Other", "Unknown"]] = None
    movement_direction: Optional[str] = None
    movement_start_alignment: Optional[str] = None
    movement_end_alignment: Optional[str] = None
    defensive_personnel: Optional[str] = None
    front_family: Optional[Literal["Even", "Odd", "Bear", "Mint/Tite", "Other", "Unknown"]] = None
    initial_line_box_defenders: Optional[int] = Field(default=None, ge=0, le=11)
    initial_second_level_box_defenders: Optional[int] = Field(default=None, ge=0, le=11)
    snap_line_box_defenders: Optional[int] = Field(default=None, ge=0, le=11)
    snap_second_level_box_defenders: Optional[int] = Field(default=None, ge=0, le=11)
    shell: Optional[Literal["1-High", "2-High", "0-High", "Unknown"]] = None
    film_play_type: Optional[Literal["Run", "Pass", "RPO", "Scramble", "Sack", "Other", "Unknown"]] = None
    run_concept: Optional[str] = None
    run_direction: Optional[Literal["Left", "Right", "Middle", "Boundary", "Field", "Unknown"]] = None
    pass_concept: Optional[str] = None
    rpo: Optional[bool] = None
    play_action: Optional[bool] = None
    immediate_rushers_count: Optional[int] = Field(default=None, ge=0, le=11)
    delayed_rushers_count: Optional[int] = Field(default=0, ge=0, le=11)
    bluff_or_drop_count: Optional[int] = Field(default=None, ge=0, le=11)
    nontraditional_rushers_count: Optional[int] = Field(default=None, ge=0, le=11)
    pressure_family_observed: Optional[Literal["Standard rush", "Blitz", "Sim pressure", "Creeper", "Zero pressure", "Unknown"]] = None
    pressure_source: Optional[str] = None
    coverage: Optional[str] = None
    adjustment_trigger: Optional[Literal["None", "Motion", "Shift", "Defensive stem", "Cadence/check", "Other", "Unknown"]] = None
    adjustment_type: Optional[Literal["None", "Bump", "Travel", "Safety rotation", "Front shift", "Box insert", "Box remove", "Other", "Unknown"]] = None
    adjustment_player: Optional[str] = None
    adjustment_detail: Optional[str] = None
    confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    evidence_notes: str = ""




class TemporalDiagnosticV32(BaseModel):
    """Small required schema to test whether temporal still frames carry usable football evidence."""

    snap_frame_index: int = Field(ge=1, le=12)
    snap_timestamp_seconds: float = Field(ge=0)
    formation_family: Literal["Gun", "Pistol", "Under Center", "Goalline", "Other", "Unknown"]
    rb_initial_side: Literal["Left", "Right", "Behind", "Multiple", "None", "Unknown"]
    rb_final_side: Literal["Left", "Right", "Behind", "Multiple", "None", "Unknown"]
    rb_changed_sides: bool
    initial_box_count: int = Field(ge=0, le=11)
    snap_box_count: int = Field(ge=0, le=11)
    actual_rusher_count: int = Field(ge=0, le=11)
    confidence: float = Field(ge=0, le=1)
    evidence_notes: str






class SnapLocatorV34(BaseModel):
    """Minimal coarse pass used only to estimate the snap timestamp."""

    snap_frame_index: int = Field(ge=1, le=12)
    snap_timestamp_seconds: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    evidence_notes: str


class TemporalPreSnapV33(BaseModel):
    """Focused pre-snap temporal chart for V3.3."""

    snap_frame_index: int = Field(ge=1, le=12)
    snap_timestamp_seconds: float = Field(ge=0)

    back_count: int = Field(ge=0, le=3)
    attached_te_count: int = Field(ge=0, le=3)
    wing_hback_count: int = Field(ge=0, le=3)
    flexed_te_count: int = Field(ge=0, le=3)
    split_wr_count: int = Field(ge=0, le=5)

    formation_family: Literal["Gun", "Pistol", "Under Center", "Goalline", "Other", "Unknown"]
    initial_receiver_structure: Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]
    final_receiver_structure: Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]
    formation_strength: Literal["Left", "Right", "Balanced", "Boundary", "Field", "Unknown"]

    rb_initial_side: Literal["Left", "Right", "Behind", "Multiple", "None", "Unknown"]
    rb_final_side: Literal["Left", "Right", "Behind", "Multiple", "None", "Unknown"]
    rb_changed_sides: bool
    rb_set_before_snap: bool

    any_player_moving_at_snap: bool
    any_alignment_change_set_before_snap: bool
    primary_mover: str
    movement_type: Literal["Across", "Jet", "Orbit", "Return", "Short", "Out to slot/wide", "Into backfield", "Trade", "Other", "Unknown"]
    movement_direction: str

    defensive_personnel: str
    front_family: Literal["Even", "Odd", "Bear", "Mint/Tite", "Other", "Unknown"]
    initial_box_count: int = Field(ge=0, le=11)
    snap_box_count: int = Field(ge=0, le=11)
    shell: Literal["1-High", "2-High", "0-High", "Unknown"]

    adjustment_trigger: Literal["None", "Motion", "Shift", "Defensive stem", "Cadence/check", "Other", "Unknown"]
    adjustment_type: Literal["None", "Bump", "Travel", "Safety rotation", "Front shift", "Box insert", "Box remove", "Other", "Unknown"]

    confidence: float = Field(ge=0, le=1)
    evidence_notes: str


class TemporalPostSnapV33(BaseModel):
    """Focused post-snap chart for V3.3."""

    film_play_type: Literal["Run", "Pass", "RPO", "Scramble", "Sack", "Other", "Unknown"]
    run_concept: str
    run_direction: Literal["Left", "Right", "Middle", "Boundary", "Field", "Unknown"]
    pass_concept: str
    rpo: bool
    play_action: bool

    rusher_descriptions: list[str]
    bluff_or_drop_descriptions: list[str]
    pressure_family_observed: Literal["Standard rush", "Blitz", "Sim pressure", "Creeper", "Zero pressure", "Unknown"]
    pressure_source: str
    coverage: str

    confidence: float = Field(ge=0, le=1)
    evidence_notes: str


class StructurePass(BaseModel):
    """Mechanical pre-snap observations. Avoid football shorthand when possible."""

    rb_count: Optional[int] = Field(default=None, ge=0, le=4)
    te_count: Optional[int] = Field(default=None, ge=0, le=4)
    formation_family: Optional[Literal["Gun", "Pistol", "Under Center", "Goalline", "Other", "Unknown"]] = None
    initial_receiver_structure: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    initial_backfield: Optional[Literal["Split backs", "RB left", "RB right", "Pistol dot", "Empty", "Other", "Unknown"]] = None
    final_receiver_structure: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    final_backfield: Optional[Literal["Split backs", "RB left", "RB right", "Pistol dot", "Empty", "Other", "Unknown"]] = None
    formation_strength: Optional[Literal["Left", "Right", "Balanced", "Boundary", "Field", "Unknown"]] = None

    defensive_personnel: Optional[str] = None
    front_family: Optional[Literal["Even", "Odd", "Bear", "Mint/Tite", "Other", "Unknown"]] = None
    initial_box_count: Optional[int] = Field(default=None, ge=0, le=11)
    snap_box_count: Optional[int] = Field(default=None, ge=0, le=11)
    shell: Optional[Literal["1-High", "2-High", "0-High", "Unknown"]] = None

    confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    notes: str = ""


class MovementPass(BaseModel):
    """Track only movement from the earliest settled set through the snap."""

    motion_present: Optional[bool] = None
    moving_at_snap: Optional[bool] = None
    motion_player: Optional[str] = None
    motion_type: Optional[Literal["Across", "Jet", "Orbit", "Return", "Short", "Out to slot/wide", "Into backfield", "Trade", "Other", "Unknown"]] = None
    motion_direction: Optional[str] = None
    motion_start_alignment: Optional[str] = None
    motion_end_alignment: Optional[str] = None

    shift_present: Optional[bool] = None
    shift_player: Optional[str] = None
    shift_description: Optional[str] = None

    adjustment_trigger: Optional[Literal["None", "Motion", "Shift", "Defensive stem", "Cadence/check", "Other", "Unknown"]] = None
    adjustment_type: Optional[Literal["None", "Bump", "Travel", "Safety rotation", "Front shift", "Box insert", "Box remove", "Other", "Unknown"]] = None
    adjustment_player: Optional[str] = None
    adjustment_detail: Optional[str] = None

    confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    notes: str = ""


class PostSnapPass(BaseModel):
    """Post-snap result and defensive pressure/coverage observations."""

    film_play_type: Optional[Literal["Run", "Pass", "RPO", "Scramble", "Sack", "Other", "Unknown"]] = None
    run_concept: Optional[str] = None
    run_direction: Optional[Literal["Left", "Right", "Middle", "Boundary", "Field", "Unknown"]] = None
    pass_concept: Optional[str] = None
    rpo: Optional[bool] = None
    play_action: Optional[bool] = None

    rushers: Optional[int] = Field(default=None, ge=0, le=11)
    pressure_family_observed: Optional[Literal["Standard rush", "Blitz", "Sim pressure", "Creeper", "Zero pressure", "Unknown"]] = None
    pressure_source: Optional[str] = None
    coverage: Optional[str] = None

    confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    notes: str = ""


class StructurePassV21(BaseModel):
    """Mechanical pre-snap observations with mutually exclusive eligible-player buckets."""

    back_count: Optional[int] = Field(default=None, ge=0, le=3)
    attached_te_count: Optional[int] = Field(default=None, ge=0, le=3)
    wing_hback_count: Optional[int] = Field(default=None, ge=0, le=3)
    flexed_te_count: Optional[int] = Field(default=None, ge=0, le=3)
    split_wr_count: Optional[int] = Field(default=None, ge=0, le=5)

    formation_family: Optional[Literal["Gun", "Pistol", "Under Center", "Goalline", "Other", "Unknown"]] = None
    initial_receiver_structure: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    final_receiver_structure: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    initial_backfield_shape: Optional[Literal["Single back", "Split backs", "Pistol dot", "Empty", "Other", "Unknown"]] = None
    final_backfield_shape: Optional[Literal["Single back", "Split backs", "Pistol dot", "Empty", "Other", "Unknown"]] = None
    formation_strength: Optional[Literal["Left", "Right", "Balanced", "Boundary", "Field", "Unknown"]] = None

    defensive_personnel: Optional[str] = None
    front_family: Optional[Literal["Even", "Odd", "Bear", "Mint/Tite", "Other", "Unknown"]] = None
    initial_line_box_defenders: Optional[int] = Field(default=None, ge=0, le=11)
    initial_second_level_box_defenders: Optional[int] = Field(default=None, ge=0, le=11)
    snap_line_box_defenders: Optional[int] = Field(default=None, ge=0, le=11)
    snap_second_level_box_defenders: Optional[int] = Field(default=None, ge=0, le=11)
    shell: Optional[Literal["1-High", "2-High", "0-High", "Unknown"]] = None

    confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    notes: str = ""


class MovementPassV21(BaseModel):
    """Mechanical movement tracking from the first settled set to the instant before the snap."""

    any_player_moving_at_snap: Optional[bool] = None
    any_alignment_change_set_before_snap: Optional[bool] = None
    primary_mover: Optional[str] = None
    movement_type: Optional[Literal["Across", "Jet", "Orbit", "Return", "Short", "Out to slot/wide", "Into backfield", "Trade", "Other", "Unknown"]] = None
    movement_direction: Optional[str] = None
    movement_start_alignment: Optional[str] = None
    movement_end_alignment: Optional[str] = None

    rb_initial_side: Optional[Literal["Left", "Right", "Behind", "Multiple", "None", "Unknown"]] = None
    rb_final_side: Optional[Literal["Left", "Right", "Behind", "Multiple", "None", "Unknown"]] = None
    rb_changed_sides: Optional[bool] = None
    rb_set_before_snap: Optional[bool] = None

    adjustment_trigger: Optional[Literal["None", "Motion", "Shift", "Defensive stem", "Cadence/check", "Other", "Unknown"]] = None
    adjustment_type: Optional[Literal["None", "Bump", "Travel", "Safety rotation", "Front shift", "Box insert", "Box remove", "Other", "Unknown"]] = None
    adjustment_player: Optional[str] = None
    adjustment_detail: Optional[str] = None

    confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    notes: str = ""


class PostSnapPassV21(BaseModel):
    """Post-snap recognition with mechanical rusher accounting."""

    film_play_type: Optional[Literal["Run", "Pass", "RPO", "Scramble", "Sack", "Other", "Unknown"]] = None
    run_concept: Optional[str] = None
    run_direction: Optional[Literal["Left", "Right", "Middle", "Boundary", "Field", "Unknown"]] = None
    pass_concept: Optional[str] = None
    rpo: Optional[bool] = None
    play_action: Optional[bool] = None

    immediate_rushers_count: Optional[int] = Field(default=None, ge=0, le=11)
    delayed_rushers_count: Optional[int] = Field(default=0, ge=0, le=11)
    bluff_or_drop_count: Optional[int] = Field(default=None, ge=0, le=11)
    nontraditional_rushers_count: Optional[int] = Field(default=None, ge=0, le=11)
    pressure_family_observed: Optional[Literal["Standard rush", "Blitz", "Sim pressure", "Creeper", "Zero pressure", "Unknown"]] = None
    pressure_source: Optional[str] = None
    coverage: Optional[str] = None

    confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    notes: str = ""


def _json_safe(value):
    """Convert pandas/numpy scalar values into normal JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _context(play_context: dict) -> dict:
    return {
        "chart_team": _json_safe(play_context.get("team")),
        "chart_team_role": _json_safe(play_context.get("chart_side")),
        "quarter": _json_safe(play_context.get("period")),
        "game_clock": _json_safe(play_context.get("clock")),
        "down": _json_safe(play_context.get("down")),
        "distance": _json_safe(play_context.get("distance")),
        "play_text": _json_safe(play_context.get("play_text")),
    }


def _prompt(play_context: dict) -> str:
    context = _context(play_context)
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
- Use common football terminology for concepts only when supported by the film.
- The play-by-play is outcome context only. It must NOT override what the film
  shows for personnel, formation, motion, front, box, shell, pressure, or concept.
- Use Unknown/null instead of guessing.
- uncertain_fields must list fields that should receive human review.

Pay special attention to the several seconds immediately before the snap so that
motions, shifts, defensive bumps, safety rotations, and box changes are not missed.
"""




def _native_correction_prompt_v4(play_context: dict, foundation: dict) -> str:
    context = _context(play_context)
    foundation_subset = {
        "personnel": foundation.get("personnel"),
        "initial_formation": foundation.get("initial_formation"),
        "initial_backfield": foundation.get("initial_backfield"),
        "final_formation": foundation.get("final_formation"),
        "final_backfield": foundation.get("final_backfield"),
        "motion_present": foundation.get("motion_present"),
        "motion_type": foundation.get("motion_type"),
        "film_play_type": foundation.get("film_play_type"),
        "run_direction": foundation.get("run_direction"),
    }

    return f"""
You are the SECOND-PASS QUALITY-CONTROL analyst for ONE college football snap.

Game context:
{json.dumps(context, indent=2)}

A broad first-pass model produced these values for ONLY the fields you are checking:
{json.dumps(foundation_subset, indent=2)}

Watch the ENTIRE native video again from beginning to end. Independently re-chart ONLY the
specific weak fields below. Do not simply agree with the first pass.

1. PERSONNEL
Personnel means WHO is on the field, not where players line up at the snap.
Count RB/FB players as back_count and true TE/Y/H players as te_count.
A RB split wide is still a RB. A TE flexed into the slot is still a TE.
Use those identity counts across the entire pre-snap sequence, not just the final alignment.
Also return the personnel shorthand (10/11/12/13/20/21/22) that best matches those counts.
Do not default to 11 personnel just because it is common.

2. EARLIEST SETTLED OFFENSIVE ALIGNMENT
initial_receiver_structure and initial_backfield describe the EARLIEST clearly settled formation
visible before any shift or motion. Do not substitute the alignment at the snap.
For initial_backfield, track the primary RB relative to the QB.

3. FINAL ALIGNMENT AT THE SNAP
final_receiver_structure and final_backfield describe the offense immediately before the snap.

4. MOTION TYPE
Only chart motion_type when a player is actually moving immediately before/through the snap.
The broad first pass's motion_present value is shown above for context, but your job here is only
to classify the motion type when it is visibly present.

5. RUN DIRECTION
Only chart run_direction when the video clearly shows a run/RPO run path.
Use the runner's actual attack path, not the play-by-play wording.

IMPORTANT:
- This pass is NOT allowed to re-chart formation family, front, box count, shell, play type,
  RPO, play action, rushers, blitz, pressure, coverage, or defensive adjustments.
- The first-pass values are context, not ground truth.
- Use Unknown/null when a targeted field truly cannot be determined.
- uncertain_fields should list only these targeted correction fields that need human review.
- In notes, briefly explain any correction that differs from the first-pass value.
"""












def _personnel_consensus_prompt_v46(play_context: dict) -> str:
    context = _context(play_context)
    roster = play_context.get("offense_skill_roster") or []
    roster_reference = [
        {
            "jersey": item.get("jersey"),
            "name": item.get("name"),
        }
        for item in roster
    ]
    return f"""
You are doing ONLY offensive personnel identification for one college football snap.

Game context:
{json.dumps(context, indent=2)}

Optional offensive skill-player roster reference (jersey and name only):
{json.dumps(roster_reference, indent=2)}

IMPORTANT:
The FINAL personnel package will be derived in Python from three independent visual-role counts.
Jersey numbers are only corroborating evidence and can NEVER override the football-role count.

STEP 1 — ACCOUNT FOR THE OFFENSE
Identify/count the five true offensive linemen and return offensive_lineman_count.
Identify/count the quarterback and return quarterback_count.
Then return EXACTLY five eligible_players.

STEP 2 — TRACK THE SAME FIVE ELIGIBLE PLAYERS THROUGH THE ENTIRE PRE-SNAP SEQUENCE
For each eligible player, record:

initial_slot:
Use the EARLIEST SETTLED alignment and choose one canonical slot:
- Backfield 1
- Backfield 2
- Left wide
- Left slot
- Left core/attached
- Right core/attached
- Right slot
- Right wide
- Other
- Unknown

Use OFFENSIVE left/right, not where the player happens to finish after a shift.

visual_role:
Classify the player's football identity from the WHOLE pre-snap sequence:
- RB/FB
- TE/H-back
- WR
- Unknown

ROLE RULES:
- A player who begins beside/behind the QB as a true back remains RB/FB even if he shifts wide.
- A true TE/H-back who begins attached, wing, or core remains TE/H-back if he later flexes.
- A player who begins as a normal split receiver/slot and functions as a receiver remains WR.
- Final alignment does not change personnel identity.
- If role evidence is genuinely ambiguous, use Unknown instead of guessing.

jersey_number / jersey_readability:
- Read a jersey only if actually visible.
- Clear = genuinely readable digits.
- Partial = incomplete/uncertain digits.
- Unreadable = cannot responsibly identify.
- Do NOT infer a number because it exists on the roster.
- Do NOT change visual_role to make it agree with a jersey lookup.

initial_alignment / final_alignment:
Describe where the same player begins and ends before the snap.

identity_evidence:
Briefly explain why the visual_role was chosen.

Do NOT return personnel shorthand.
Do NOT chart formation, motion, play type, defense, pressure, or coverage.
Python will derive each run's personnel code from the five visual_role values, then take a 2-of-3
consensus across three runs. Jersey consistency is recorded only as a diagnostic.
"""


def _hybrid_personnel_prompt_v45(play_context: dict) -> str:
    context = _context(play_context)
    roster = play_context.get("offense_skill_roster") or []
    roster_reference = [
        {
            "jersey": item.get("jersey"),
            "name": item.get("name"),
        }
        for item in roster
    ]
    return f"""
You are doing ONLY offensive personnel identification for one college football snap.

Game context:
{json.dumps(context, indent=2)}

Offensive skill-player roster reference (jersey and name only):
{json.dumps(roster_reference, indent=2)}

Your job is to account for the offense and collect TWO INDEPENDENT kinds of evidence for each
eligible player:
1) jersey-number evidence, when the number is genuinely readable;
2) whole-sequence football-role evidence.

Do NOT chart formation, motion, play type, defense, pressure, or coverage in this pass.

STEP 1 — TRUE OFFENSIVE LINEMEN
Identify/count the five true offensive linemen: LT, LG, C, RG, RT.
Return offensive_lineman_count.
A WR who is on the line of scrimmage is not an offensive lineman.

STEP 2 — QUARTERBACK
Identify/count the quarterback and return quarterback_count.

STEP 3 — EXACTLY FIVE ELIGIBLE PLAYERS
Return exactly five eligible_players. Track the SAME five players through the entire pre-snap
sequence, even if one shifts or motions.

For each player return:

player_label:
- generic visual identity such as "back beside QB", "boundary WR", "field slot", "attached player".

jersey_number:
- return digits only when actually visible;
- otherwise null.
- Do NOT choose a jersey merely because a roster number seems plausible.

jersey_readability:
- Clear = the digits are genuinely readable from the video;
- Partial = you can see some number evidence but not enough to trust the full number;
- Unreadable = the number cannot be responsibly identified.
Be conservative. A guessed number is Unreadable, not Clear.

visual_role:
- RB/FB
- TE/H-back
- WR
- Unknown

IMPORTANT ROLE RULE:
Determine visual_role from the player's WHOLE pre-snap sequence, with extra weight on the
EARLIEST SETTLED ALIGNMENT before any shift or motion. Final alignment alone does NOT change
personnel identity.

Examples:
- A player who begins beside/behind the QB as a back, then shifts wide, remains RB/FB.
- A TE/H-back who begins attached/wing/core and later flexes out remains TE/H-back.
- A player who begins split wide/slot and functions as a receiver should remain WR even if a
  guessed jersey number would map to a different roster position.
- Do not call a wide receiver an RB merely because a roster lookup might suggest that jersey.
- If the visual identity itself is not supportable, return Unknown.

initial_alignment:
Describe where that player is in the earliest settled offensive picture.

final_alignment:
Describe where the same player is immediately before the snap.

identity_evidence:
Briefly explain WHY the visual_role was chosen, especially when initial and final alignment differ.

CRITICAL:
Jersey reading and visual_role are independent evidence streams. Do not force them to agree.
Python will decide how to reconcile them:
- a truly Clear jersey can use roster identity;
- unreadable/partial jerseys fall back to visual role;
- a Clear jersey that conflicts with a known visual role is flagged for review rather than forced.
"""


def _roster_personnel_prompt_v44(play_context: dict) -> str:
    context = _context(play_context)
    roster = play_context.get("offense_skill_roster") or []
    roster_reference = [
        {
            "jersey": item.get("jersey"),
            "name": item.get("name"),
        }
        for item in roster
    ]
    return f"""
You are doing ONLY player identification for offensive personnel on one college football snap.

Game context:
{json.dumps(context, indent=2)}

Offensive skill-player roster reference (jersey and name only):
{json.dumps(roster_reference, indent=2)}

IMPORTANT: Do NOT infer whether a player is RB, TE, or WR from alignment/body type.
Python already has roster positions and will map the jersey number to the player's roster identity.
Your job is only to identify WHICH offensive players are on the field.

STEP 1 — TRUE OFFENSIVE LINEMEN
Identify/count the five true offensive linemen (LT, LG, C, RG, RT).
Return offensive_lineman_count.
A WR who is legally on the line of scrimmage is still not an offensive lineman.

STEP 2 — QUARTERBACK
Identify/count the quarterback and return quarterback_count.

STEP 3 — FIVE ELIGIBLE OFFENSIVE PLAYERS
Return exactly five eligible_players.
For each one:
- give a generic visual label (field WR, boundary slot, back beside QB, attached player, etc.)
- read the jersey number when visible
- use the roster reference to help distinguish plausible jersey numbers
- if the number is not actually readable, return null rather than inventing one

Track players through the whole pre-snap sequence. A player who shifts or motions remains the
same player even after changing alignment.

Do NOT return personnel shorthand.
Do NOT classify anyone as RB/TE/WR.
Do NOT use alignment to guess roster identity.
Python will match jersey numbers to the roster and derive personnel.
"""


def _mechanical_personnel_prompt_v43(play_context: dict) -> str:
    context = _context(play_context)
    return f"""
You are doing ONLY offensive personnel accounting for one college football snap.

Game context:
{json.dumps(context, indent=2)}

Do NOT chart formation, motion, shift, play type, defense, pressure, or coverage.
Your only job is to account for the 11 offensive players mechanically.

STEP 1 — TRUE OFFENSIVE LINEMEN
Identify the five true offensive linemen by role/body/alignment:
LT, LG, C, RG, RT.
Return offensive_lineman_count.
Do NOT count a WR who is on the line of scrimmage as an offensive lineman.

STEP 2 — QUARTERBACK
Identify the quarterback and return quarterback_count.
Normally this should be exactly 1.

STEP 3 — THE FIVE REMAINING ELIGIBLE PLAYERS
After the OL and QB are accounted for, return exactly five eligible_players.
Classify each one as exactly one of:
- RB/FB
- TE/H-back
- WR
- Unknown

Personnel identity follows the PLAYER, not his final alignment.
Examples:
- A RB who starts beside the QB and shifts to the slot is still RB/FB.
- A TE who flexes wide is still TE/H-back.
- A WR aligned on the LOS is still WR.
- A wing/H-back with TE body/role is TE/H-back.

Watch the full pre-snap sequence so you can track a player who changes alignment.

Use generic labels such as "RB #1", "attached Y", "field slot", "boundary WR".
If one player's identity is genuinely unclear, mark only that player Unknown.

The correct accounting should be:
5 OL + 1 QB + 5 eligible players = 11 offensive players.

Do not return personnel shorthand. Python will derive it from the five eligible-player identities.
"""


def _prompt_v42_mechanical_personnel(play_context: dict) -> str:
    base = _prompt(play_context)
    return base + """
    
V4.2 PERSONNEL ACCOUNTING — OVERRIDES THE GENERAL PERSONNEL INSTRUCTION ABOVE:

Do NOT decide the offensive personnel package by pattern recognition or by the final formation.
The returned personnel field itself is not trusted; Python will derive the package.

Instead perform this accounting process over the full pre-snap video:

STEP 1 — OFFENSIVE LINE
Identify the five true offensive linemen by role/body/alignment:
LT, LG, C, RG, RT.
Return offensive_lineman_count.
Do NOT count a WR aligned on the line of scrimmage as an offensive lineman.

STEP 2 — QUARTERBACK
Identify the quarterback and return quarterback_count.
Normally this should be exactly 1.

STEP 3 — FIVE REMAINING ELIGIBLE PLAYERS
After accounting for 5 OL and 1 QB, classify EACH of the five remaining offensive players.
Return exactly five entries in eligible_players.

For each player use one identity:
- RB/FB
- TE/H-back
- WR
- Unknown

Use player identity/role across the ENTIRE pre-snap sequence, not only where the player stands
at the snap.

Examples:
- A RB who starts beside the QB and shifts to the slot is still RB/FB.
- A TE who flexes wide is still TE/H-back.
- A WR aligned on the LOS is still WR, not an offensive lineman.
- An H-back/wing who is a TE-type player is TE/H-back, not RB merely because he is in the backfield.

Use generic labels such as "RB #1", "left slot", "attached Y", "boundary WR" when jersey numbers
are unclear.

STEP 4 — SANITY CHECK
The accounting should be:
5 offensive linemen + 1 quarterback + 5 eligible players = 11 offensive players.

If you cannot account for the offense cleanly, mark uncertain eligible players as Unknown.
Do NOT fill missing identities by defaulting to 11 personnel.

Python will derive personnel only when:
- offensive_lineman_count == 5
- quarterback_count == 1
- all five eligible players have known roles
- the five eligible classifications sum to exactly five players

The personnel code is then RB count followed by TE count:
1 RB + 0 TE = 10
1 RB + 1 TE = 11
1 RB + 2 TE = 12
2 RB + 0 TE = 20
2 RB + 1 TE = 21
etc.

In personnel_accounting_notes, briefly explain which five players you classified and why.
"""


def _structure_prompt(play_context: dict) -> str:
    context = _context(play_context)
    return f"""
You are doing ONLY the PRE-SNAP STRUCTURE pass for one college football snap.

Game context:
{json.dumps(context, indent=2)}

Do not identify a play concept. Do not infer personnel shorthand such as 11 or 20.
Instead make mechanical visual observations.

OFFENSE:
1. Count RB/FB/H-back type backfield personnel on the field as rb_count.
   Count who is on the field, not where they are aligned at the snap.
2. Count true TE/Y/H players on the field as te_count, including a TE flexed out.
3. Identify QB alignment: Gun, Pistol, Under Center, etc.
4. Identify receiver distribution in the earliest settled set and again at the snap.
5. Identify RB/backfield alignment in the earliest settled set and at the snap.
6. Only call Field/Boundary strength when the broadcast view supports it.

DEFENSE:
1. Identify defensive personnel if visually supportable.
2. Classify the front only as Even, Odd, Bear, Mint/Tite, Other, or Unknown.
3. Count defenders structurally in the run box in the initial settled picture and at the snap.
4. Identify safety shell only as 0-High, 1-High, 2-High, or Unknown.

Important:
- A RB who shifts or motions out still counts as a RB for rb_count.
- Do not default to common personnel. If you cannot distinguish a TE from a WR, use null
  and list the field as uncertain.
- Do not use the play-by-play to infer formation or personnel.
- Prefer Unknown/null over guessing.
"""


def _movement_prompt(play_context: dict) -> str:
    context = _context(play_context)
    return f"""
You are doing ONLY the PRE-SNAP MOVEMENT pass for one college football snap.

Game context:
{json.dumps(context, indent=2)}

Watch from the first settled offensive alignment through the snap. Ignore the result
of the play except to identify the moment of the snap.

Definitions:
- MOTION: a player changes location and is still moving immediately before or through the snap.
- SHIFT: one or more offensive players change alignment, then the offense becomes set before the snap.
- A back moving from the QB's right side to the QB's left side and becoming set is a SHIFT,
  not motion.
- If both occur, mark both.

For any movement, describe the player generically when jersey/name is unclear:
RB, TE, slot, outside WR, etc. Record start and end alignment whenever visible.

Also watch how the defense reacts before the snap:
- bump
- travel
- safety rotation
- front shift
- box insert/remove
- other stem/check

Do not classify personnel, formation family, run/pass concept, or coverage in this pass.
Use Unknown/null instead of inventing movement that is not visible.
"""


def _postsnap_prompt(play_context: dict) -> str:
    context = _context(play_context)
    return f"""
You are doing ONLY the POST-SNAP pass for one college football snap.

Game context:
{json.dumps(context, indent=2)}

Determine:
- run/pass/RPO/scramble/sack
- run concept and direction only when the blocking/action clearly supports it
- pass concept only when the route structure is visible enough
- play action and RPO
- exact number of defenders who rush the passer
- observed pressure family and source
- coverage only when the broadcast angle gives enough post-snap evidence

Pressure rules:
- Five or more rushers is a blitz for our deterministic charting logic.
- Four rushers with a non-traditional rusher replacing a dropping DL can be sim pressure/creeper.
- Do not call a blitz merely because the QB is pressured.
- Count actual rushers after the snap, not defenders threatening at the line.

Coverage rules:
- Do not guess from the pre-snap shell alone.
- If deep defenders leave the broadcast frame before the coverage can be confirmed, use Unknown.
- Man/zone and exact coverage labels should be conservative.

The play-by-play may help with outcome context, but it must not decide concept, pressure,
rush count, or coverage for you. Prefer Unknown/null over a confident guess.
"""


def _structure_prompt_v21(play_context: dict) -> str:
    context = _context(play_context)
    return f"""
You are doing ONLY a MECHANICAL PRE-SNAP STRUCTURE count for one college football snap.

Game context:
{json.dumps(context, indent=2)}

Do NOT return personnel shorthand such as 10/11/20. Count the five non-QB, non-offensive-line
eligible skill players using MUTUALLY EXCLUSIVE buckets. Every eligible player should belong to
exactly one bucket:

- back_count: RB/FB players by body type/role, even if one is split wide.
- attached_te_count: true TE/Y aligned on or immediately next to the offensive line.
- wing_hback_count: TE/H-back aligned as a wing/off the tackle. Do NOT also count him as a back.
- flexed_te_count: true TE body/role detached into slot or wide alignment.
- split_wr_count: true WRs. Do not put a TE in this bucket merely because he is split out.

SANITY CHECK:
back_count + attached_te_count + wing_hback_count + flexed_te_count + split_wr_count should equal 5.
If you cannot visually distinguish TE vs WR, use null for the uncertain bucket rather than defaulting
to 11 personnel. The play-by-play must NOT be used to decide player positions.

Also chart:
- QB formation family.
- receiver distribution at the earliest settled alignment and again immediately before the snap.
- whether the backfield is single-back, split-backs, pistol-dot, or empty at those two moments.
- formation strength only if clearly supported.

DEFENSIVE BOX COUNT:
Do NOT give one intuitive 'box count'. Count two layers separately at the earliest settled picture
and again immediately before the snap:
1. line_box_defenders = defenders on/near the LOS who are structurally part of the run front,
   including edge defenders aligned tight enough to be a run-box player.
2. second_level_box_defenders = linebackers/safeties between roughly 2-6 yards of the LOS and
   inside the offensive core who are structurally committed to the box.

The Python code will add those layers to create the box total. Do not omit an edge defender merely
because he is just outside the offensive tackle.

Finally identify defensive personnel when visible, front family (Even/Odd/Bear/Mint-Tite/etc.),
and the PRE-SNAP safety shell only. Prefer Unknown/null to a default guess.
"""


def _movement_prompt_v21(play_context: dict) -> str:
    context = _context(play_context)
    return f"""
You are doing ONLY a MECHANICAL PRE-SNAP MOVEMENT TRACK for one college football snap.

Game context:
{json.dumps(context, indent=2)}

Your most important job is to compare TWO moments:
A) the earliest settled offensive alignment visible in the clip
B) the final instant immediately before the snap

For the RB/back specifically:
- record rb_initial_side relative to the QB: Left, Right, Behind, Multiple, None, or Unknown.
- record rb_final_side using the same choices.
- explicitly answer rb_changed_sides.
- explicitly answer rb_set_before_snap.

Example: if the RB starts to the QB's RIGHT, moves to the QB's LEFT, then becomes stationary
before the snap:
rb_initial_side=Right, rb_final_side=Left, rb_changed_sides=true,
rb_set_before_snap=true, any_alignment_change_set_before_snap=true,
any_player_moving_at_snap=false.

Definitions used by our code:
- any_player_moving_at_snap=true means MOTION.
- any_alignment_change_set_before_snap=true means a SHIFT occurred.
Both may be true on a play if a shift occurs and a different player later motions.

Also describe the primary mover's generic position, start/end alignment, direction and movement type.
Do not infer movement from the play result. Watch the pre-snap sequence itself frame by frame.

For the defense, record any pre-snap response: bump, travel, safety rotation, front shift,
box insert/remove, or other stem/check. Prefer Unknown/null over inventing movement.
"""


def _postsnap_prompt_v21(play_context: dict) -> str:
    context = _context(play_context)
    return f"""
You are doing ONLY the POST-SNAP pass for one college football snap.

Game context:
{json.dumps(context, indent=2)}

Chart run/pass/RPO/scramble/sack, concept only when clearly supported, play action, pressure,
and coverage. Keep the existing conservative coverage approach: pre-snap shell does NOT determine
post-snap coverage, and a defense may show 2-High then rotate to Cover 1.

RUSHER ACCOUNTING IS THE PRIORITY:
At the snap, track each defender who threatens the line.

- immediate_rushers_count: defenders whose first post-snap action commits them into the rush,
  attacking the LOS/backfield/pass protection.
- delayed_rushers_count: defenders who initially hesitate/fit/cover, then clearly add to the rush.
- bluff_or_drop_count: defenders who threaten pressure pre-snap but actually drop/cover.
- nontraditional_rushers_count: rushing DB/LB players replacing a traditional DL/edge who drops.

Do not count a defender as a rusher because he merely shows pressure before the snap.
Do not stop counting at the first four rushers. Follow the first several seconds after the snap.
The Python code will derive total rushers as immediate + delayed and will derive blitz=true at 5+.

Pressure family:
- 5+ actual rushers: normally Blitz (or Zero pressure if clearly zero-man pressure).
- 4 rushers with a nontraditional rusher replacing a dropping traditional rusher can be Sim/Creeper.
- QB pressure alone does not equal blitz.

Coverage:
- identify the actual post-snap coverage when the broadcast angle supports it.
- 2-High pre-snap rotating to Cover 1 post-snap is valid.
- use Unknown when deep coverage leaves the broadcast frame too early.

The play-by-play may provide outcome context only; it must not decide rush count, pressure, concept,
or coverage. Prefer Unknown/null over a confident guess.
"""



def _temporal_prompt(play_context: dict, evidence: dict) -> str:
    context = _context(play_context)
    manifest = evidence.get("manifest", [])
    frame_map = ", ".join(
        f"{item['frame_index']}={item['timestamp_seconds']:.2f}s"
        for item in manifest
    )
    return f"""
You are charting ONE college football snap from ORDERED TEMPORAL EVIDENCE SHEETS.

Game context:
{json.dumps(context, indent=2)}

The attached images are contact sheets from the SAME clip. Read sheet 1, then 2, then 3.
Within each sheet read left-to-right, top-to-bottom. Each tile is labeled FRAME ## and time.
Global timeline:
{frame_map}

First locate the snap and return the nearest frame number and time.
Then reason separately about EARLY PRE-SNAP, FINAL PRE-SNAP, and POST-SNAP.
Do not overwrite an early alignment with the final alignment.

PERSONNEL: Count the five non-QB/non-OL eligible players in mutually exclusive buckets:
back_count, attached_te_count, wing_hback_count, flexed_te_count, split_wr_count.
They should sum to 5. A player keeps his personnel identity after motion/shift. Use null if TE vs WR
cannot be distinguished instead of defaulting to 11 personnel.

RB TRACKING: Explicitly compare RB side relative to QB early vs final pre-snap.
If side changes and he becomes set, that is a SHIFT. If still moving at the snap, that is MOTION.

FORMATION: Chart QB formation family and early/final receiver distribution. Only use Field/Boundary
strength when broadcast geometry supports it.

DEFENSIVE BOX: At early and final pre-snap separately count line_box_defenders and
second_level_box_defenders. Include tight edge defenders. Python will add the layers.

SHELL/COVERAGE: shell is PRE-SNAP only; coverage is actual POST-SNAP coverage.
A 2-High shell rotating to Cover 1 is valid.

RUSHERS: Count immediate rushers, delayed rushers, bluff/drop players, and nontraditional rushers.
Do not stop at four. Python derives total rushers and blitz at 5+.

PLAY: Chart run/pass/RPO/scramble/sack and only name concepts when evidence is sufficient.

DEFENSIVE ADJUSTMENT: Record visible bump, travel, safety rotation, front shift, box insert/remove,
or other response.

CFBD play-by-play is outcome context only and cannot decide film labels.
Prefer Unknown/null over football-default guessing. In evidence_notes cite useful FRAME numbers.
"""




def _temporal_prompt_v31(play_context: dict, evidence: dict) -> str:
    context = _context(play_context)
    frame_map = "\n".join(
        f"FRAME {item['frame_index']:02d} = {item['timestamp_seconds']:.2f}s"
        for item in evidence.get("frames", [])
    )

    return f"""
You are charting ONE college football snap from INDIVIDUAL HIGH-RESOLUTION FRAMES.

Game context:
{json.dumps(context, indent=2)}

The attached images are separate chronological frames from the same clip.
Read them strictly in ascending order: FRAME 01 through FRAME 12.

Timeline:
{frame_map}

FIRST: locate the frame nearest the actual SNAP and return its frame number and timestamp.

THEN reason about three distinct temporal phases:
1. EARLY PRE-SNAP = the earliest settled offensive alignment visible before movement.
2. FINAL PRE-SNAP = the last settled/aligned state immediately before the ball is snapped.
3. POST-SNAP = frames after the snap.

Never overwrite an EARLY PRE-SNAP observation with the FINAL PRE-SNAP alignment.

OFFENSIVE PERSONNEL
Count the five non-QB, non-offensive-line eligible skill players using mutually exclusive buckets:
- back_count = RB/FB
- attached_te_count = true TE attached to the offensive line
- wing_hback_count = TE/H-back aligned as a wing/off tackle
- flexed_te_count = true TE detached into slot/wide alignment
- split_wr_count = true WR

The five buckets should sum to 5 when all players are visible.
A player keeps his PERSONNEL identity if he later shifts or motions.
If TE vs WR cannot be distinguished from the images, use null rather than defaulting to 11 personnel.

RB TEMPORAL TRACKING
Explicitly compare the RB relative to the QB in EARLY PRE-SNAP versus FINAL PRE-SNAP:
- rb_initial_side
- rb_final_side
- rb_changed_sides
- rb_set_before_snap

If the RB changes sides and becomes stationary before the snap, that is a SHIFT.
If a player is still moving at the snap, that is MOTION.
Do not infer the early state from the final state.

FORMATION
Chart:
- QB formation family
- receiver distribution in EARLY PRE-SNAP
- receiver distribution in FINAL PRE-SNAP
- formation strength only if field/boundary geometry is visually clear

DEFENSIVE BOX
At EARLY PRE-SNAP and FINAL PRE-SNAP separately count:
- line_box_defenders = LOS/edge defenders structurally in the run front
- second_level_box_defenders = LBs/safeties roughly 2-6 yards deep inside the offensive core
Include tight edge defenders. Python will add these layers to derive the box total.

SHELL AND COVERAGE
- shell = PRE-SNAP safety shell only
- coverage = actual POST-SNAP coverage
A 2-High shell rotating into Cover 1 is valid. Do not force pre-snap shell and post-snap coverage to agree.

RUSHER ACCOUNTING
Use POST-SNAP frames:
- immediate_rushers_count = defenders whose first post-snap action attacks the LOS/backfield/pass protection
- delayed_rushers_count = defenders who add to the rush later
- bluff_or_drop_count = defenders who threaten pressure but actually drop
- nontraditional_rushers_count = DB/LB rushers replacing a traditional rusher who drops

Do not stop counting when you reach four. Python will derive total rushers and blitz at 5+.

PLAY CLASSIFICATION
Chart run/pass/RPO/scramble/sack. Name run/pass concepts only when the evidence is sufficient.
Prefer Unknown/null instead of guessing.

DEFENSIVE ADJUSTMENT
Record visible bump, travel, safety rotation, front shift, box insert/remove, or other pre-snap response.

IMPORTANT
- CFBD play-by-play is OUTCOME CONTEXT only. It must not decide personnel, formation, movement,
  box, shell, rusher count, pressure, concept, or coverage.
- Prefer Unknown/null over football-default guessing.
- In evidence_notes, cite useful FRAME numbers for the snap, initial alignment, final alignment,
  movement/shift, rush count, and coverage when those observations are visible.
"""




def _temporal_diagnostic_prompt_v32(play_context: dict, evidence: dict) -> str:
    context = _context(play_context)
    frame_map = "\n".join(
        f"FRAME {item['frame_index']:02d} = {item['timestamp_seconds']:.2f}s"
        for item in evidence.get("frames", [])
    )

    return f"""
You are performing a SMALL DIAGNOSTIC TEST on one college football snap.

Game context:
{json.dumps(context, indent=2)}

You receive 12 individual high-resolution images from the SAME play in chronological order.
Read FRAME 01 through FRAME 12 in order.

Timeline:
{frame_map}

Return ONLY these observations. Every field is required.

1. snap_frame_index
   Pick the frame nearest the instant the ball is snapped.

2. snap_timestamp_seconds
   Use the supplied timestamp for that frame.

3. formation_family
   Choose Gun, Pistol, Under Center, Goalline, Other, or Unknown.

4. rb_initial_side
   In the EARLIEST settled pre-snap alignment, identify the primary RB relative to the QB:
   Left, Right, Behind, Multiple, None, or Unknown.

5. rb_final_side
   In the LAST pre-snap alignment before the snap, identify the primary RB relative to the QB
   using the same choices.

6. rb_changed_sides
   True only if the chronological frames show the RB changing from one side/alignment to another.

7. initial_box_count
   Best visual count of defenders structurally in the run box in the EARLIEST settled pre-snap frame.
   Count LOS/edge defenders plus second-level defenders clearly committed to the box.

8. snap_box_count
   Best visual count of defenders structurally in the run box immediately before the snap.

9. actual_rusher_count
   Use the POST-SNAP frames. Count defenders whose actual post-snap action attacks the LOS,
   backfield, QB, or pass protection. Do not count bluffing defenders who drop into coverage.
   Do not stop counting at four.

DIAGNOSTIC RULES:
- This is intentionally a narrow visual task. Do not analyze personnel, coverage, concepts, shell,
  motion type, pressure family, or defensive personnel.
- Compare EARLY PRE-SNAP and FINAL PRE-SNAP as different moments.
- Make the best visual count for box defenders and rushers instead of leaving those counts blank.
- Use Unknown for an RB side or formation only when the images truly do not support a choice.
- CFBD play-by-play is outcome context only and must not determine these visual observations.
- evidence_notes should briefly cite the FRAME numbers that support the snap, RB change/no-change,
  box counts, and rusher count.
"""




def _temporal_presnap_prompt_v33(play_context: dict, evidence: dict) -> str:
    context = _context(play_context)
    frame_map = "\n".join(
        f"FRAME {item['frame_index']:02d} = {item['timestamp_seconds']:.2f}s"
        for item in evidence.get("frames", [])
    )
    return f"""
You are performing the PRE-SNAP pass for one college football snap using 12 individual,
high-resolution chronological frames.

Game context:
{json.dumps(context, indent=2)}

Timeline:
{frame_map}

Read FRAME 01 through FRAME 12 in order.

FIRST identify the frame nearest the actual snap. Then keep EARLY PRE-SNAP and FINAL PRE-SNAP
as separate moments. Do not replace the early alignment with the final alignment.

PERSONNEL — make a best visual count and return integers for all five buckets:
- back_count = RB/FB
- attached_te_count = true TE attached to line
- wing_hback_count = TE/H-back aligned as wing/off tackle
- flexed_te_count = true TE detached into slot/wide
- split_wr_count = true WR

These mutually exclusive buckets should sum to 5. A player keeps his personnel identity after
shifting or motioning. Do not default to 11 personnel simply because a player is hard to identify.

FORMATION AND BACKFIELD:
- formation_family = QB alignment
- initial_receiver_structure = earliest settled distribution
- final_receiver_structure = last pre-snap distribution
- rb_initial_side = RB relative to QB in earliest settled state
- rb_final_side = RB relative to QB immediately before snap
- rb_changed_sides = compare those two states literally
- rb_set_before_snap = whether the RB was stationary before the snap

MOVEMENT:
- any_player_moving_at_snap = true only for a player still moving at the snap
- any_alignment_change_set_before_snap = true for a shift that finished and became set
- identify the primary mover and movement type when visible

DEFENSE:
- defensive personnel if visually supportable; otherwise return "Unknown"
- front family
- initial_box_count = best count in earliest settled picture
- snap_box_count = best count immediately before snap
- shell = pre-snap 0/1/2-high
- adjustment trigger/type for visible bump, travel, safety rotation, front shift, box insert/remove

BOX RULE:
Count LOS/edge defenders structurally in the run front plus second-level defenders clearly committed
to the box. Include a tight edge defender just outside the offensive tackle.

Use Unknown for categorical uncertainty, but DO NOT leave required counts blank.
CFBD play-by-play is outcome context only. It cannot decide these visual fields.
In evidence_notes cite the FRAME numbers that support snap timing, RB start/end, box counts,
and any shift/motion.
"""


def _temporal_postsnap_prompt_v33(play_context: dict, evidence: dict, snap_frame_index: int) -> str:
    context = _context(play_context)
    frame_map = "\n".join(
        f"FRAME {item['frame_index']:02d} = {item['timestamp_seconds']:.2f}s"
        for item in evidence.get("frames", [])
    )
    return f"""
You are performing the POST-SNAP pass for one college football snap using individual,
high-resolution chronological frames.

Game context:
{json.dumps(context, indent=2)}

The pre-snap pass estimated the snap nearest FRAME {snap_frame_index:02d}.
The images supplied to you begin at or just before that point and continue after the snap.

Timeline of supplied frames:
{frame_map}

PLAY:
- classify Run, Pass, RPO, Scramble, Sack, Other, or Unknown
- identify run concept/direction only when blocking and ball action support it
- identify pass concept only when route structure is visible enough
- answer RPO and play_action from the post-snap action

RUSHER ENUMERATION — this is the priority:
Create rusher_descriptions with EXACTLY ONE entry for EACH UNIQUE defender whose actual post-snap
action attacks the LOS/backfield/QB/pass protection. Use generic labels such as:
"left edge", "left DT", "right DT", "right edge", "Mike LB", "nickel", "safety".

Do not write the same defender twice across frames.
Do not include a defender merely because he threatened pressure before the snap.
Do not stop after four rushers. Follow later frames for delayed add-on rushers.
If five unique defenders rush, the list must contain five entries.

Create bluff_or_drop_descriptions separately for defenders who threatened pressure but actually
dropped into coverage.

PRESSURE AND COVERAGE:
- pressure_family_observed should reflect the actual rush structure
- pressure_source should summarize unusual/nontraditional pressure sources
- coverage is the ACTUAL post-snap coverage, not the pre-snap shell
- a 2-high pre-snap look rotating to Cover 1 post-snap is valid

Make a best visual football judgment rather than returning blank fields.
CFBD play-by-play is outcome context only and cannot decide concept, rushers, pressure, or coverage.
In evidence_notes cite FRAME numbers supporting the play type, each rusher, and coverage rotation.
"""




def _snap_locator_prompt_v34(play_context: dict, evidence: dict) -> str:
    context = _context(play_context)
    frame_map = "\n".join(
        f"FRAME {item['frame_index']:02d} = {item['timestamp_seconds']:.2f}s"
        for item in evidence.get("frames", [])
    )
    return f"""
You are locating ONLY the snap in one college football play.

Game context:
{json.dumps(context, indent=2)}

You receive 12 chronological coarse frames from the same clip.

Timeline:
{frame_map}

Your only task:
- identify the frame nearest the instant the center snaps the ball
- return that frame number
- return the exact timestamp shown for that frame
- give a confidence score
- briefly cite the frames around the snap in evidence_notes

Do not analyze personnel, formation, motion, defense, pressure, coverage, or play concept.
"""


def _temporal_presnap_prompt_v34(play_context: dict, evidence: dict) -> str:
    context = _context(play_context)
    frame_map = "\n".join(
        (
            f"FRAME {item['frame_index']:02d} = {item['timestamp_seconds']:.2f}s "
            f"({item.get('relative_to_snap_seconds', 0):+.2f}s from snap)"
        )
        for item in evidence.get("frames", [])
        if item.get("phase") == "pre_snap"
    )
    return f"""
You are performing the PRE-SNAP pass for one college football snap.

Game context:
{json.dumps(context, indent=2)}

These frames were re-extracted relative to an estimated snap time.
Use only the PRE-SNAP frames listed below.

Timeline:
{frame_map}

IMPORTANT: the timestamps are football-relative now.
A frame at -5.00s is much earlier than a frame at -0.20s.

PERSONNEL:
Return best visual counts for:
- back_count
- attached_te_count
- wing_hback_count
- flexed_te_count
- split_wr_count
These buckets should sum to 5 when all players are visible.

FORMATION / RB TIMELINE:
- earliest settled frame: use the earliest frame that clearly shows a settled offensive formation
- final pre-snap frame: use the last clear frame before the snap
- compare rb_initial_side to rb_final_side literally
- if the RB changes sides and becomes stationary before snap, that is a shift
- if a player is still moving immediately before snap, that is motion

Also chart:
- formation family
- initial/final receiver structure
- formation strength
- defensive personnel
- front family
- initial box count
- snap box count
- safety shell
- defensive adjustment trigger/type

BOX RULE:
Count LOS/edge defenders in the run front plus second-level defenders clearly committed to the box.
Use the earliest settled frame for initial_box_count and the final clear pre-snap frame for snap_box_count.

CFBD play-by-play is outcome context only.
In evidence_notes cite exact relative times that support RB start/end, shift/motion, and box counts.
"""


def _temporal_postsnap_prompt_v34(play_context: dict, evidence: dict) -> str:
    context = _context(play_context)
    frame_map = "\n".join(
        (
            f"FRAME {item['frame_index']:02d} = {item['timestamp_seconds']:.2f}s "
            f"({item.get('relative_to_snap_seconds', 0):+.2f}s from snap)"
        )
        for item in evidence.get("frames", [])
        if item.get("phase") == "post_snap"
    )
    return f"""
You are performing the POST-SNAP pass for one college football snap.

Game context:
{json.dumps(context, indent=2)}

These frames were re-extracted relative to the estimated snap time.
Use only the POST-SNAP frames listed below.

Timeline:
{frame_map}

PLAY:
- classify Run, Pass, RPO, Scramble, Sack, Other, or Unknown
- identify run concept/direction only when blocking and ball action support it
- identify pass concept only when route structure is visible enough
- answer RPO and play_action from actual post-snap action

RUSHER ENUMERATION — highest priority:
Create rusher_descriptions with EXACTLY ONE entry for EACH UNIQUE defender whose actual action
attacks the LOS/backfield/QB/pass protection.

Use generic identities such as:
"left edge", "left DT", "right DT", "right edge", "Mike LB", "nickel", "safety".

Track defenders across the dense +0.20s, +0.40s, +0.70s, +1.00s, +1.50s, +2.00s, +3.00s frames.
Do not stop at four.
Do not count the same defender twice.
Put threatened-but-dropping defenders in bluff_or_drop_descriptions instead.

PRESSURE / COVERAGE:
- pressure_family_observed must reflect the actual rush
- pressure_source summarizes unusual/nontraditional rushers
- coverage is actual post-snap coverage, not the pre-snap shell

CFBD play-by-play is outcome context only.
In evidence_notes cite exact relative times for each rusher and for any coverage rotation.
"""


def _derive_personnel(rb_count: Optional[int], te_count: Optional[int]) -> Optional[str]:
    if rb_count is None or te_count is None:
        return "Unknown"
    mapping = {
        (1, 0): "10",
        (1, 1): "11",
        (1, 2): "12",
        (1, 3): "13",
        (2, 0): "20",
        (2, 1): "21",
        (2, 2): "22",
    }
    return mapping.get((rb_count, te_count), "Other")


def _derive_pressure(rushers: Optional[int], observed: Optional[str]) -> tuple[Optional[bool], Optional[str]]:
    if rushers is None:
        return None, observed or "Unknown"

    if rushers >= 5:
        return True, "Zero pressure" if observed == "Zero pressure" else "Blitz"

    if rushers == 4 and observed in {"Sim pressure", "Creeper"}:
        return False, observed

    return False, observed if observed not in {None, "Blitz"} else "Standard rush"


def _derive_personnel_v21(
    back_count: Optional[int],
    attached_te_count: Optional[int],
    wing_hback_count: Optional[int],
    flexed_te_count: Optional[int],
    split_wr_count: Optional[int],
) -> tuple[str, Optional[int], Optional[int]]:
    counts = [back_count, attached_te_count, wing_hback_count, flexed_te_count, split_wr_count]
    if any(value is None for value in counts):
        return "Unknown", None, None

    te_count = int(attached_te_count) + int(wing_hback_count) + int(flexed_te_count)
    eligible_total = int(back_count) + te_count + int(split_wr_count)
    if eligible_total != 5:
        return "Unknown", te_count, eligible_total

    personnel = _derive_personnel(int(back_count), te_count)
    return personnel or "Unknown", te_count, eligible_total


def _sum_optional(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None or b is None:
        return None
    return int(a) + int(b)


def _backfield_from_shape_and_side(shape: Optional[str], side: Optional[str]) -> Optional[str]:
    if shape in {"Split backs", "Pistol dot", "Empty"}:
        return shape
    if side == "Right":
        return "RB right"
    if side == "Left":
        return "RB left"
    if side == "Behind":
        return "Pistol dot"
    if side == "Multiple":
        return "Split backs"
    if side == "None":
        return "Empty"
    if shape in {"Other", "Unknown"}:
        return shape
    return "Unknown"


def _derive_movement_v21(movement: MovementPassV21) -> tuple[Optional[bool], Optional[bool]]:
    motion_present = movement.any_player_moving_at_snap
    shift_present = movement.any_alignment_change_set_before_snap

    if movement.rb_changed_sides is True and movement.rb_set_before_snap is True:
        shift_present = True
    if movement.rb_changed_sides is True and movement.rb_set_before_snap is False:
        motion_present = True

    return motion_present, shift_present


def _derive_rushers_v21(post: PostSnapPassV21) -> Optional[int]:
    if post.immediate_rushers_count is None:
        return None
    delayed = post.delayed_rushers_count or 0
    return int(post.immediate_rushers_count) + int(delayed)




def _usable_correction(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in {"", "unknown", "none", "null"}:
        return False
    return True


def _v4_personnel_from_correction(correction: NativeCorrectionV4) -> Optional[str]:
    if correction.back_count is not None and correction.te_count is not None:
        derived = _derive_personnel(correction.back_count, correction.te_count)
        if derived not in {None, "Unknown", "Other"}:
            return derived
    if _usable_correction(correction.personnel):
        return correction.personnel
    return None


def _merge_uncertain(*passes: BaseModel) -> list[str]:
    seen = []
    for item in passes:
        for field in getattr(item, "uncertain_fields", []) or []:
            if field not in seen:
                seen.append(field)
    return seen


def _pass_confidence_average(*passes: BaseModel) -> float:
    values = [float(getattr(item, "confidence")) for item in passes if getattr(item, "confidence", None) is not None]
    return sum(values) / len(values) if values else 0.0










def _normalize_jersey_v44(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "none", "null"}:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    try:
        return str(int(digits))
    except ValueError:
        return None






def _derive_visual_personnel_v46(
    personnel_pass: PersonnelConsensusPassV46,
    roster: list[dict],
) -> tuple[str, dict]:
    """
    Derive personnel from visual roles only.

    Jersey/roster evidence is captured for diagnostics but never changes the role used
    for personnel. A run is valid only with 5 OL, 1 QB, five known visual roles, and
    internally complete eligible-player accounting.
    """
    by_jersey: dict[str, list[dict]] = {}
    for item in roster:
        jersey = _normalize_jersey_v44(item.get("jersey"))
        role = item.get("role")
        if jersey and role in {"RB/FB", "TE/H-back", "WR"}:
            by_jersey.setdefault(jersey, []).append(item)

    observed_players = []
    for visible in personnel_pass.eligible_players:
        jersey = _normalize_jersey_v44(visible.jersey_number)
        matches = by_jersey.get(jersey, []) if jersey else []
        roster_roles = sorted({item.get("role") for item in matches if item.get("role")})
        roster_role = roster_roles[0] if len(roster_roles) == 1 else None
        matched_name = matches[0].get("name") if len(matches) == 1 else None
        matched_position = matches[0].get("position") if len(matches) == 1 else None

        observed_players.append(
            {
                "player_label": visible.player_label,
                "initial_slot": visible.initial_slot,
                "jersey": jersey,
                "jersey_readability": visible.jersey_readability,
                "visual_role": visible.visual_role,
                "roster_role": roster_role,
                "jersey_visual_agree": (
                    roster_role == visible.visual_role
                    if roster_role and visible.visual_role != "Unknown"
                    else None
                ),
                "matched_name": matched_name,
                "roster_position": matched_position,
                "roster_match_count": len(matches),
                "initial_alignment": visible.initial_alignment,
                "final_alignment": visible.final_alignment,
                "identity_evidence": visible.identity_evidence,
            }
        )

    roles = [item["visual_role"] for item in observed_players]
    rb_count = sum(role == "RB/FB" for role in roles)
    te_count = sum(role == "TE/H-back" for role in roles)
    wr_count = sum(role == "WR" for role in roles)
    unknown_count = sum(role == "Unknown" for role in roles)

    valid_accounting = (
        personnel_pass.offensive_lineman_count == 5
        and personnel_pass.quarterback_count == 1
        and len(observed_players) == 5
        and unknown_count == 0
        and (rb_count + te_count + wr_count) == 5
    )

    if valid_accounting:
        code = f"{rb_count}{te_count}"
        standard = {"00", "01", "02", "10", "11", "12", "13", "20", "21", "22", "23", "30", "31", "32"}
        personnel = code if code in standard else "Other"
    else:
        personnel = "Unknown"

    audit = {
        "offensive_lineman_count": personnel_pass.offensive_lineman_count,
        "quarterback_count": personnel_pass.quarterback_count,
        "eligible_player_count": len(observed_players),
        "rb_fb_count": rb_count,
        "te_hback_count": te_count,
        "wr_count": wr_count,
        "unknown_eligible_count": unknown_count,
        "valid_accounting": valid_accounting,
        "eligible_players": observed_players,
        "notes": personnel_pass.notes,
        "confidence": personnel_pass.confidence,
    }
    return personnel, audit


def _jersey_corroboration_v46(audits: list[dict]) -> list[dict]:
    """
    Show whether a jersey read is corroborated at the same canonical initial slot.

    This is diagnostic only; it does not affect personnel.
    """
    by_slot: dict[str, list[dict]] = {}
    for run_index, audit in enumerate(audits, start=1):
        for player in audit.get("eligible_players") or []:
            slot = player.get("initial_slot") or "Unknown"
            by_slot.setdefault(slot, []).append(
                {
                    "run": run_index,
                    "jersey": player.get("jersey"),
                    "readability": player.get("jersey_readability"),
                    "visual_role": player.get("visual_role"),
                }
            )

    rows = []
    for slot, observations in sorted(by_slot.items()):
        clear_jerseys = [
            item["jersey"]
            for item in observations
            if item.get("readability") == "Clear" and item.get("jersey")
        ]
        counts = {}
        for jersey in clear_jerseys:
            counts[jersey] = counts.get(jersey, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        corroborated = ranked[0][0] if ranked and ranked[0][1] >= 2 else None
        rows.append(
            {
                "initial_slot": slot,
                "corroborated_jersey": corroborated,
                "clear_jersey_votes": ranked,
                "observations": observations,
            }
        )
    return rows


def _derive_hybrid_personnel_v45(
    personnel_pass: HybridPersonnelPassV45,
    roster: list[dict],
) -> tuple[str, dict]:
    """
    Resolve each eligible player's role using an evidence hierarchy.

    Clear jersey + unique roster role is trusted only when it does not conflict with
    an independently observed visual role. Partial/unreadable jerseys never drive role
    assignment; they fall back to whole-sequence visual identity.
    """
    by_jersey: dict[str, list[dict]] = {}
    for item in roster:
        jersey = _normalize_jersey_v44(item.get("jersey"))
        role = item.get("role")
        if jersey and role in {"RB/FB", "TE/H-back", "WR"}:
            by_jersey.setdefault(jersey, []).append(item)

    resolved_players = []
    clear_jerseys = []

    for visible in personnel_pass.eligible_players:
        jersey = _normalize_jersey_v44(visible.jersey_number)
        readability = visible.jersey_readability
        visual_role = visible.visual_role
        matches = by_jersey.get(jersey, []) if jersey else []
        roster_roles = sorted({item.get("role") for item in matches if item.get("role")})
        roster_role = roster_roles[0] if len(roster_roles) == 1 else None

        matched_name = matches[0].get("name") if len(matches) == 1 else None
        matched_position = matches[0].get("position") if len(matches) == 1 else None

        resolved_role = "Unknown"
        source = "unresolved"
        conflict = False

        if readability == "Clear":
            if jersey:
                clear_jerseys.append(jersey)

            if roster_role:
                if visual_role not in {"Unknown", roster_role}:
                    # Strong evidence streams disagree: do not force a package.
                    conflict = True
                    source = "jersey-visual-conflict"
                else:
                    resolved_role = roster_role
                    source = "clear-jersey+roster"
            else:
                source = "clear-jersey-no-unique-roster-match"
        else:
            # Partial/unreadable digits cannot override football-role evidence.
            if visual_role != "Unknown":
                resolved_role = visual_role
                source = "whole-sequence-visual-role"
            else:
                source = "no-trustworthy-evidence"

        resolved_players.append(
            {
                "player_label": visible.player_label,
                "jersey": jersey,
                "jersey_readability": readability,
                "visual_role": visual_role,
                "roster_role": roster_role,
                "resolved_role": resolved_role,
                "resolution_source": source,
                "conflict": conflict,
                "matched_name": matched_name,
                "roster_position": matched_position,
                "roster_match_count": len(matches),
                "initial_alignment": visible.initial_alignment,
                "final_alignment": visible.final_alignment,
                "identity_evidence": visible.identity_evidence,
            }
        )

    roles = [item["resolved_role"] for item in resolved_players]
    rb_count = sum(role == "RB/FB" for role in roles)
    te_count = sum(role == "TE/H-back" for role in roles)
    wr_count = sum(role == "WR" for role in roles)
    unknown_count = sum(role == "Unknown" for role in roles)
    conflict_count = sum(bool(item["conflict"]) for item in resolved_players)
    duplicate_clear_jerseys = len(clear_jerseys) != len(set(clear_jerseys))

    valid_accounting = (
        personnel_pass.offensive_lineman_count == 5
        and personnel_pass.quarterback_count == 1
        and len(resolved_players) == 5
        and unknown_count == 0
        and conflict_count == 0
        and not duplicate_clear_jerseys
        and (rb_count + te_count + wr_count) == 5
    )

    if valid_accounting:
        code = f"{rb_count}{te_count}"
        standard = {"00", "01", "02", "10", "11", "12", "13", "20", "21", "22", "23", "30", "31", "32"}
        personnel = code if code in standard else "Other"
    else:
        personnel = "Unknown"

    audit = {
        "offensive_lineman_count": personnel_pass.offensive_lineman_count,
        "quarterback_count": personnel_pass.quarterback_count,
        "eligible_player_count": len(resolved_players),
        "rb_fb_count": rb_count,
        "te_hback_count": te_count,
        "wr_count": wr_count,
        "unknown_eligible_count": unknown_count,
        "conflict_count": conflict_count,
        "duplicate_clear_jerseys": duplicate_clear_jerseys,
        "valid_accounting": valid_accounting,
        "eligible_players": resolved_players,
        "notes": personnel_pass.notes,
        "confidence": personnel_pass.confidence,
    }
    return personnel, audit


def _derive_roster_personnel_v44(
    personnel_pass: RosterPersonnelPassV44,
    roster: list[dict],
) -> tuple[str, dict]:
    """
    Map visible jersey numbers to roster positions, then derive personnel in Python.

    Any unreadable jersey, missing roster match, duplicate/ambiguous roster match, or
    incomplete 11-man accounting makes the personnel result Unknown.
    """
    by_jersey: dict[str, list[dict]] = {}
    for item in roster:
        jersey = _normalize_jersey_v44(item.get("jersey"))
        role = item.get("role")
        if jersey and role in {"RB/FB", "TE/H-back", "WR"}:
            by_jersey.setdefault(jersey, []).append(item)

    mapped_players = []
    for visible in personnel_pass.eligible_players:
        jersey = _normalize_jersey_v44(visible.jersey_number)
        matches = by_jersey.get(jersey, []) if jersey else []
        unique_roles = sorted({item.get("role") for item in matches if item.get("role")})
        role = unique_roles[0] if len(unique_roles) == 1 else "Unknown"
        matched_name = None
        matched_position = None
        if len(matches) == 1:
            matched_name = matches[0].get("name")
            matched_position = matches[0].get("position")

        mapped_players.append(
            {
                "player_label": visible.player_label,
                "jersey": jersey,
                "role": role,
                "matched_name": matched_name,
                "roster_position": matched_position,
                "roster_match_count": len(matches),
                "evidence": visible.evidence,
            }
        )

    roles = [item["role"] for item in mapped_players]
    rb_count = sum(role == "RB/FB" for role in roles)
    te_count = sum(role == "TE/H-back" for role in roles)
    wr_count = sum(role == "WR" for role in roles)
    unknown_count = sum(role == "Unknown" for role in roles)

    jerseys = [item["jersey"] for item in mapped_players if item["jersey"]]
    duplicate_visible_jerseys = len(jerseys) != len(set(jerseys))

    valid_accounting = (
        personnel_pass.offensive_lineman_count == 5
        and personnel_pass.quarterback_count == 1
        and len(mapped_players) == 5
        and unknown_count == 0
        and not duplicate_visible_jerseys
        and (rb_count + te_count + wr_count) == 5
    )

    if valid_accounting:
        code = f"{rb_count}{te_count}"
        standard = {"00", "01", "02", "10", "11", "12", "13", "20", "21", "22", "23", "30", "31", "32"}
        personnel = code if code in standard else "Other"
    else:
        personnel = "Unknown"

    audit = {
        "offensive_lineman_count": personnel_pass.offensive_lineman_count,
        "quarterback_count": personnel_pass.quarterback_count,
        "eligible_player_count": len(mapped_players),
        "rb_fb_count": rb_count,
        "te_hback_count": te_count,
        "wr_count": wr_count,
        "unknown_eligible_count": unknown_count,
        "duplicate_visible_jerseys": duplicate_visible_jerseys,
        "valid_accounting": valid_accounting,
        "eligible_players": mapped_players,
        "notes": personnel_pass.notes,
        "confidence": personnel_pass.confidence,
    }
    return personnel, audit


def _derive_personnel_from_accounting_v43(
    offensive_lineman_count: int,
    quarterback_count: int,
    eligible_players: list[EligibleSkillPlayerV42],
    notes: str = "",
) -> tuple[str, dict]:
    roles = [player.role for player in eligible_players]
    rb_count = sum(role == "RB/FB" for role in roles)
    te_count = sum(role == "TE/H-back" for role in roles)
    wr_count = sum(role == "WR" for role in roles)
    unknown_count = sum(role == "Unknown" for role in roles)

    valid_accounting = (
        offensive_lineman_count == 5
        and quarterback_count == 1
        and len(roles) == 5
        and unknown_count == 0
        and (rb_count + te_count + wr_count) == 5
    )

    if valid_accounting:
        code = f"{rb_count}{te_count}"
        standard = {"00", "01", "02", "10", "11", "12", "13", "20", "21", "22", "23", "30", "31", "32"}
        personnel = code if code in standard else "Other"
    else:
        personnel = "Unknown"

    audit = {
        "offensive_lineman_count": offensive_lineman_count,
        "quarterback_count": quarterback_count,
        "eligible_player_count": len(roles),
        "rb_fb_count": rb_count,
        "te_hback_count": te_count,
        "wr_count": wr_count,
        "unknown_eligible_count": unknown_count,
        "valid_accounting": valid_accounting,
        "eligible_players": [player.model_dump() for player in eligible_players],
        "notes": notes,
    }
    return personnel, audit


def _derive_personnel_v42(chart: FootballSnapChartV42) -> tuple[str, dict]:
    """
    Derive personnel only from complete 11-man accounting.

    Returns (personnel, audit_dict). Unknown is returned whenever the accounting
    is incomplete or internally inconsistent.
    """
    roles = [player.role for player in chart.eligible_players]
    rb_count = sum(role == "RB/FB" for role in roles)
    te_count = sum(role == "TE/H-back" for role in roles)
    wr_count = sum(role == "WR" for role in roles)
    unknown_count = sum(role == "Unknown" for role in roles)

    valid_accounting = (
        chart.offensive_lineman_count == 5
        and chart.quarterback_count == 1
        and len(roles) == 5
        and unknown_count == 0
        and (rb_count + te_count + wr_count) == 5
    )

    if valid_accounting:
        code = f"{rb_count}{te_count}"
        standard = {"00", "01", "02", "10", "11", "12", "13", "20", "21", "22", "23", "30", "31", "32"}
        personnel = code if code in standard else "Other"
    else:
        personnel = "Unknown"

    audit = {
        "offensive_lineman_count": chart.offensive_lineman_count,
        "quarterback_count": chart.quarterback_count,
        "eligible_player_count": len(roles),
        "rb_fb_count": rb_count,
        "te_hback_count": te_count,
        "wr_count": wr_count,
        "unknown_eligible_count": unknown_count,
        "valid_accounting": valid_accounting,
        "eligible_players": [player.model_dump() for player in chart.eligible_players],
        "notes": chart.personnel_accounting_notes,
    }
    return personnel, audit


def _repeatability_vote_key_v41(value):
    """Create a stable equality key for repeated structured outputs."""
    if value is None:
        return ("none", "")
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ("number", float(value))
    if isinstance(value, str):
        normalized = " ".join(value.strip().lower().split())
        return ("string", normalized)
    return ("json", json.dumps(value, sort_keys=True, ensure_ascii=False))


def _majority_vote_v41(values: list):
    """
    Return (consensus, top_count, vote_counts, has_majority).

    A true five-run majority requires at least 3 agreeing observations. When no
    value reaches 3/5, consensus is None so downstream logic treats the field as
    unresolved rather than arbitrarily breaking a 2-2-1 split.
    """
    groups = {}
    representatives = {}
    order = []
    for value in values:
        key = _repeatability_vote_key_v41(value)
        if key not in groups:
            groups[key] = 0
            representatives[key] = value
            order.append(key)
        groups[key] += 1

    ranked = sorted(order, key=lambda key: groups[key], reverse=True)
    winner = ranked[0]
    top_count = groups[winner]
    has_majority = top_count >= 3

    vote_counts = []
    for key in ranked:
        representative = representatives[key]
        vote_counts.append(
            {
                "value": representative,
                "count": groups[key],
            }
        )

    return (
        representatives[winner] if has_majority else None,
        top_count,
        vote_counts,
        has_majority,
    )


def _call_structured(
    client,
    uploaded,
    model: str,
    fps: float,
    prompt: str,
    schema: type[BaseModel],
) -> BaseModel:
    interaction = None
    for attempt in range(4):
        try:
            interaction = client.interactions.create(
                model=model,
                input=[
                    {
                        "type": "video",
                        "uri": uploaded.uri,
                        "mime_type": uploaded.mime_type,
                        "processing": {"type": "static", "fps": float(fps)},
                    },
                    {"type": "text", "text": prompt},
                ],
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema.model_json_schema(),
                },
            )
            break
        except Exception as exc:
            message = str(exc).lower()
            retryable = any(
                token in message
                for token in (
                    "503",
                    "service_unavailable",
                    "high demand",
                    "429",
                    "resource_exhausted",
                    "rate limit",
                )
            )
            if not retryable or attempt == 3:
                raise
            time.sleep(2 * (2 ** attempt))

    if interaction is None or not interaction.output_text:
        raise RuntimeError("Gemini returned no structured chart.")

    return schema.model_validate(json.loads(interaction.output_text))


def _upload_video(client, source: Path):
    uploaded = client.files.upload(file=str(source))
    while not uploaded.state or uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state and uploaded.state.name == "FAILED":
        raise RuntimeError("Gemini failed to process the uploaded video.")
    return uploaded


def analyze_clip_gemini(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 3.0,
) -> dict:
    """V1: one broad model pass over one snap."""
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = _upload_video(client, source)
        chart = _call_structured(client, uploaded, model, fps, _prompt(play_context), FootballSnapChart)
        result = chart.model_dump()
        result["_model"] = model
        result["_video_fps"] = float(fps)
        result["_provider"] = "google-gemini"
        result["_analyzer_version"] = "v1-single-pass"
        return result
    finally:
        if uploaded is not None and uploaded.name:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


















def analyze_clip_gemini_v46_personnel_only(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V4.6 personnel-only validator: three visual-role personnel observations, no full-chart calls.

    Each run derives personnel from five visual player roles. Final personnel is a true
    2-of-3 vote. Jersey/roster evidence is retained only as a diagnostic.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    roster = play_context.get("offense_skill_roster") or []

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = _upload_video(client, source)

        personnel_runs = []
        personnel_audits = []
        confidences = []

        for run_index in range(3):
            personnel_pass = _call_structured(
                client,
                uploaded,
                model,
                fps,
                _personnel_consensus_prompt_v46(play_context),
                PersonnelConsensusPassV46,
            )
            personnel, audit = _derive_visual_personnel_v46(personnel_pass, roster)
            personnel_runs.append(personnel)
            personnel_audits.append(audit)
            confidences.append(float(personnel_pass.confidence))
            if run_index < 2:
                time.sleep(2)

        counts = {}
        first_order = []
        for value in personnel_runs:
            key = _repeatability_vote_key_v41(value)
            if key not in counts:
                counts[key] = {"count": 0, "value": value}
                first_order.append(key)
            counts[key]["count"] += 1

        ranked_keys = sorted(first_order, key=lambda key: counts[key]["count"], reverse=True)
        personnel_votes = [
            {"value": counts[key]["value"], "count": counts[key]["count"]}
            for key in ranked_keys
        ]
        top = personnel_votes[0] if personnel_votes else None

        if top and top["count"] >= 2 and top["value"] not in {None, "Unknown"}:
            personnel_consensus = top["value"]
            personnel_has_majority = True
        else:
            personnel_consensus = None
            personnel_has_majority = False

        valid_personnel_runs = sum(
            bool(audit.get("valid_accounting")) for audit in personnel_audits
        )

        return {
            "personnel": personnel_consensus,
            "overall_confidence": (
                sum(confidences) / len(confidences) if confidences else 0.0
            ),
            "uncertain_fields": [] if personnel_has_majority else ["personnel"],
            "analysis_notes": (
                "V4.6 personnel-only validation: three visual-role observations; "
                f"{valid_personnel_runs}/3 runs had complete accounting. Final personnel uses "
                "a true 2-of-3 vote. Jersey/roster evidence is diagnostic only."
            ),
            "_model": model,
            "_video_fps": float(fps),
            "_provider": "google-gemini",
            "_analyzer_version": "v4.6-personnel-only",
            "_personnel_accounting": {
                "method": "Three-run visual-role personnel consensus; jersey/roster is diagnostic only",
                "run_count": 3,
                "consensus_rule": "At least 2 of 3 visual personnel codes",
                "has_majority": personnel_has_majority,
                "valid_runs": valid_personnel_runs,
                "personnel_runs": personnel_runs,
                "votes": personnel_votes,
                "jersey_corroboration": _jersey_corroboration_v46(personnel_audits),
                "roster": roster,
                "runs": personnel_audits,
            },
        }

    finally:
        if uploaded is not None and uploaded.name:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def analyze_clip_gemini_v46_personnel_consensus(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V4.6: five unchanged V1 full-chart calls plus deterministic 3-run visual-personnel consensus.

    Each specialist run derives a personnel code from five visual player roles only. Final personnel
    is a 2-of-3 vote across those codes. Jersey/roster evidence is diagnostic and cannot override roles.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    roster = play_context.get("offense_skill_roster") or []

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = _upload_video(client, source)

        full_runs: list[dict] = []
        for run_index in range(5):
            chart = _call_structured(
                client,
                uploaded,
                model,
                fps,
                _prompt(play_context),
                FootballSnapChart,
            )
            full_runs.append(chart.model_dump())
            if run_index < 4:
                time.sleep(2)

        chart_fields = [
            name
            for name in FootballSnapChart.model_fields
            if name not in {"overall_confidence", "uncertain_fields", "analysis_notes"}
        ]

        result = {}
        stability = {}
        no_majority_fields = []
        unanimous_fields = []
        strong_majority_fields = []

        for field in chart_fields:
            values = [run.get(field) for run in full_runs]
            consensus, top_count, vote_counts, has_majority = _majority_vote_v41(values)
            result[field] = consensus
            stability[field] = {
                "top_count": top_count,
                "run_count": len(full_runs),
                "agreement": top_count / len(full_runs),
                "has_majority": has_majority,
                "vote_counts": vote_counts,
                "source": "v1-full-chart",
            }
            if top_count == len(full_runs):
                unanimous_fields.append(field)
            elif top_count >= 4:
                strong_majority_fields.append(field)
            if not has_majority:
                no_majority_fields.append(field)

        personnel_runs = []
        personnel_audits = []
        for run_index in range(3):
            personnel_pass = _call_structured(
                client,
                uploaded,
                model,
                fps,
                _personnel_consensus_prompt_v46(play_context),
                PersonnelConsensusPassV46,
            )
            personnel, audit = _derive_visual_personnel_v46(personnel_pass, roster)
            personnel_runs.append(personnel)
            personnel_audits.append(audit)
            if run_index < 2:
                time.sleep(2)

        # True 2-of-3 vote on the per-run visual personnel codes.
        counts = {}
        first_order = []
        for value in personnel_runs:
            key = _repeatability_vote_key_v41(value)
            if key not in counts:
                counts[key] = {"count": 0, "value": value}
                first_order.append(key)
            counts[key]["count"] += 1

        ranked_keys = sorted(first_order, key=lambda key: counts[key]["count"], reverse=True)
        personnel_votes = [
            {"value": counts[key]["value"], "count": counts[key]["count"]}
            for key in ranked_keys
        ]
        top = personnel_votes[0] if personnel_votes else None
        if top and top["count"] >= 2 and top["value"] not in {None, "Unknown"}:
            personnel_consensus = top["value"]
            personnel_has_majority = True
        else:
            personnel_consensus = None
            personnel_has_majority = False

        result["personnel"] = personnel_consensus
        stability["personnel"] = {
            "top_count": top["count"] if top else 0,
            "run_count": 3,
            "agreement": (top["count"] / 3) if top else 0.0,
            "has_majority": personnel_has_majority,
            "vote_counts": personnel_votes,
            "source": "three-run-visual-personnel-consensus",
        }

        if "personnel" in no_majority_fields:
            no_majority_fields.remove("personnel")
        if not personnel_has_majority:
            no_majority_fields.append("personnel")

        confidences = [
            float(run["overall_confidence"])
            for run in full_runs
            if run.get("overall_confidence") is not None
        ]
        result["overall_confidence"] = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        uncertain = []
        for run in full_runs:
            for field in run.get("uncertain_fields") or []:
                if field not in uncertain:
                    uncertain.append(field)
        for field in no_majority_fields:
            if field not in uncertain:
                uncertain.append(field)
        if result.get("personnel") is None and "personnel" not in uncertain:
            uncertain.append("personnel")
        result["uncertain_fields"] = uncertain

        valid_personnel_runs = sum(
            bool(audit.get("valid_accounting")) for audit in personnel_audits
        )
        jersey_corroboration = _jersey_corroboration_v46(personnel_audits)

        result["analysis_notes"] = (
            "V4.6 uses five unchanged V1 full-chart calls plus three personnel-only visual-role "
            f"observations. {valid_personnel_runs}/3 specialist runs had complete visual accounting. "
            "Final personnel is a 2-of-3 vote on visual personnel codes; jersey reads are diagnostics only."
        )

        result["_model"] = model
        result["_video_fps"] = float(fps)
        result["_provider"] = "google-gemini"
        result["_analyzer_version"] = "v4.6-personnel-consensus"
        result["_repeatability"] = {
            "run_count": 5,
            "consensus_rule": "At least 3 of 5 identical values for full-chart fields",
            "runs": full_runs,
            "stability": stability,
            "unanimous_fields": unanimous_fields,
            "strong_majority_fields": strong_majority_fields,
            "no_majority_fields": no_majority_fields,
        }
        result["_personnel_accounting"] = {
            "method": "Three-run visual-role personnel consensus; jersey/roster is corroborating diagnostic only",
            "run_count": 3,
            "consensus_rule": "At least 2 of 3 visual personnel codes",
            "valid_runs": valid_personnel_runs,
            "personnel_runs": personnel_runs,
            "votes": personnel_votes,
            "jersey_corroboration": jersey_corroboration,
            "roster": roster,
            "runs": personnel_audits,
        }
        return result

    finally:
        if uploaded is not None and uploaded.name:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def analyze_clip_gemini_v45_hybrid_personnel(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V4.5: five unchanged V1 full-chart calls plus three hybrid personnel-evidence calls.

    Personnel uses clear jersey+roster identity when trustworthy, otherwise whole-sequence
    visual role. Conflicting strong evidence is left Unknown/review rather than forced.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    roster = play_context.get("offense_skill_roster") or []
    if not roster:
        raise ValueError("V4.5 requires an offensive skill-player roster in play_context.")

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = _upload_video(client, source)

        # Preserve V4.4/V4.1 full-chart behavior exactly.
        full_runs: list[dict] = []
        for run_index in range(5):
            chart = _call_structured(
                client,
                uploaded,
                model,
                fps,
                _prompt(play_context),
                FootballSnapChart,
            )
            full_runs.append(chart.model_dump())
            if run_index < 4:
                time.sleep(2)

        chart_fields = [
            name
            for name in FootballSnapChart.model_fields
            if name not in {"overall_confidence", "uncertain_fields", "analysis_notes"}
        ]

        result = {}
        stability = {}
        no_majority_fields = []
        unanimous_fields = []
        strong_majority_fields = []

        for field in chart_fields:
            values = [run.get(field) for run in full_runs]
            consensus, top_count, vote_counts, has_majority = _majority_vote_v41(values)
            result[field] = consensus
            stability[field] = {
                "top_count": top_count,
                "run_count": len(full_runs),
                "agreement": top_count / len(full_runs),
                "has_majority": has_majority,
                "vote_counts": vote_counts,
                "source": "v1-full-chart",
            }
            if top_count == len(full_runs):
                unanimous_fields.append(field)
            elif top_count >= 4:
                strong_majority_fields.append(field)
            if not has_majority:
                no_majority_fields.append(field)

        personnel_runs = []
        personnel_audits = []
        for run_index in range(3):
            personnel_pass = _call_structured(
                client,
                uploaded,
                model,
                fps,
                _hybrid_personnel_prompt_v45(play_context),
                HybridPersonnelPassV45,
            )
            personnel, audit = _derive_hybrid_personnel_v45(personnel_pass, roster)
            personnel_runs.append(personnel)
            personnel_audits.append(audit)
            if run_index < 2:
                time.sleep(2)

        _, personnel_top_count, personnel_votes, _ = _majority_vote_v41(personnel_runs)
        if personnel_top_count >= 2 and personnel_votes:
            personnel_consensus = personnel_votes[0]["value"]
            personnel_has_majority = True
        else:
            personnel_consensus = None
            personnel_has_majority = False

        result["personnel"] = personnel_consensus
        stability["personnel"] = {
            "top_count": personnel_top_count,
            "run_count": 3,
            "agreement": personnel_top_count / 3,
            "has_majority": personnel_has_majority,
            "vote_counts": personnel_votes,
            "source": "hybrid-personnel-evidence",
        }

        if "personnel" in no_majority_fields:
            no_majority_fields.remove("personnel")
        if not personnel_has_majority or result.get("personnel") in {None, "Unknown"}:
            no_majority_fields.append("personnel")

        confidences = [
            float(run["overall_confidence"])
            for run in full_runs
            if run.get("overall_confidence") is not None
        ]
        result["overall_confidence"] = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        uncertain = []
        for run in full_runs:
            for field in run.get("uncertain_fields") or []:
                if field not in uncertain:
                    uncertain.append(field)
        for field in no_majority_fields:
            if field not in uncertain:
                uncertain.append(field)
        if result.get("personnel") in {None, "Unknown"} and "personnel" not in uncertain:
            uncertain.append("personnel")
        result["uncertain_fields"] = uncertain

        valid_personnel_runs = sum(
            bool(audit.get("valid_accounting")) for audit in personnel_audits
        )
        conflict_runs = sum(
            int(audit.get("conflict_count", 0) > 0) for audit in personnel_audits
        )

        result["analysis_notes"] = (
            "V4.5 uses five unchanged V1 full-chart calls plus three hybrid personnel calls. "
            f"{valid_personnel_runs}/3 personnel runs produced complete accounting; "
            f"{conflict_runs}/3 had at least one clear-jersey vs visual-role conflict."
        )

        result["_model"] = model
        result["_video_fps"] = float(fps)
        result["_provider"] = "google-gemini"
        result["_analyzer_version"] = "v4.5-hybrid-personnel"
        result["_repeatability"] = {
            "run_count": 5,
            "consensus_rule": "At least 3 of 5 identical values for full-chart fields",
            "runs": full_runs,
            "stability": stability,
            "unanimous_fields": unanimous_fields,
            "strong_majority_fields": strong_majority_fields,
            "no_majority_fields": no_majority_fields,
        }
        result["_personnel_accounting"] = {
            "method": "Clear jersey+roster when corroborated; otherwise whole-sequence visual role; conflicts -> review",
            "run_count": 3,
            "consensus_rule": "At least 2 of 3 derived personnel codes",
            "valid_runs": valid_personnel_runs,
            "conflict_runs": conflict_runs,
            "personnel_runs": personnel_runs,
            "votes": personnel_votes,
            "roster": roster,
            "runs": personnel_audits,
        }
        return result

    finally:
        if uploaded is not None and uploaded.name:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def analyze_clip_gemini_v44_roster_personnel(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V4.4: five normal V1 full-chart calls plus three roster-assisted jersey-ID passes.

    The video model identifies the five eligible offensive jersey numbers. Python maps
    those numbers to roster positions and derives personnel. No visual TE-vs-WR guess is used.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    roster = play_context.get("offense_skill_roster") or []
    if not roster:
        raise ValueError("V4.4 requires an offensive skill-player roster in play_context.")

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = _upload_video(client, source)

        # Full chart remains identical to V4.1/V4.3.
        full_runs: list[dict] = []
        for run_index in range(5):
            chart = _call_structured(
                client,
                uploaded,
                model,
                fps,
                _prompt(play_context),
                FootballSnapChart,
            )
            full_runs.append(chart.model_dump())
            if run_index < 4:
                time.sleep(2)

        chart_fields = [
            name
            for name in FootballSnapChart.model_fields
            if name not in {"overall_confidence", "uncertain_fields", "analysis_notes"}
        ]

        result = {}
        stability = {}
        no_majority_fields = []
        unanimous_fields = []
        strong_majority_fields = []

        for field in chart_fields:
            values = [run.get(field) for run in full_runs]
            consensus, top_count, vote_counts, has_majority = _majority_vote_v41(values)
            result[field] = consensus
            stability[field] = {
                "top_count": top_count,
                "run_count": len(full_runs),
                "agreement": top_count / len(full_runs),
                "has_majority": has_majority,
                "vote_counts": vote_counts,
                "source": "v1-full-chart",
            }
            if top_count == len(full_runs):
                unanimous_fields.append(field)
            elif top_count >= 4:
                strong_majority_fields.append(field)
            if not has_majority:
                no_majority_fields.append(field)

        # Three narrow jersey-identification passes.
        personnel_runs = []
        personnel_audits = []
        for run_index in range(3):
            personnel_pass = _call_structured(
                client,
                uploaded,
                model,
                fps,
                _roster_personnel_prompt_v44(play_context),
                RosterPersonnelPassV44,
            )
            personnel, audit = _derive_roster_personnel_v44(personnel_pass, roster)
            personnel_runs.append(personnel)
            personnel_audits.append(audit)
            if run_index < 2:
                time.sleep(2)

        _, personnel_top_count, personnel_votes, _ = _majority_vote_v41(personnel_runs)
        if personnel_top_count >= 2 and personnel_votes:
            personnel_consensus = personnel_votes[0]["value"]
            personnel_has_majority = True
        else:
            personnel_consensus = None
            personnel_has_majority = False

        result["personnel"] = personnel_consensus
        stability["personnel"] = {
            "top_count": personnel_top_count,
            "run_count": 3,
            "agreement": personnel_top_count / 3,
            "has_majority": personnel_has_majority,
            "vote_counts": personnel_votes,
            "source": "roster-assisted-jersey-id",
        }

        if "personnel" in no_majority_fields:
            no_majority_fields.remove("personnel")
        if not personnel_has_majority:
            no_majority_fields.append("personnel")

        confidences = [
            float(run["overall_confidence"])
            for run in full_runs
            if run.get("overall_confidence") is not None
        ]
        result["overall_confidence"] = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        uncertain = []
        for run in full_runs:
            for field in run.get("uncertain_fields") or []:
                if field not in uncertain:
                    uncertain.append(field)
        for field in no_majority_fields:
            if field not in uncertain:
                uncertain.append(field)
        if result.get("personnel") in {None, "Unknown"} and "personnel" not in uncertain:
            uncertain.append("personnel")
        result["uncertain_fields"] = uncertain

        valid_personnel_runs = sum(
            bool(audit.get("valid_accounting")) for audit in personnel_audits
        )
        result["analysis_notes"] = (
            "V4.4 uses five unchanged V1 full-chart calls plus three roster-assisted jersey-ID "
            f"personnel calls. {valid_personnel_runs}/3 personnel runs matched all five visible "
            "eligible players to unique offensive skill-player roster identities."
        )

        result["_model"] = model
        result["_video_fps"] = float(fps)
        result["_provider"] = "google-gemini"
        result["_analyzer_version"] = "v4.4-roster-assisted-personnel"
        result["_repeatability"] = {
            "run_count": 5,
            "consensus_rule": "At least 3 of 5 identical values for full-chart fields",
            "runs": full_runs,
            "stability": stability,
            "unanimous_fields": unanimous_fields,
            "strong_majority_fields": strong_majority_fields,
            "no_majority_fields": no_majority_fields,
        }
        result["_personnel_accounting"] = {
            "method": "Video jersey ID -> roster position mapping -> Python personnel derivation",
            "run_count": 3,
            "consensus_rule": "At least 2 of 3 derived personnel codes",
            "valid_runs": valid_personnel_runs,
            "personnel_runs": personnel_runs,
            "votes": personnel_votes,
            "roster": roster,
            "runs": personnel_audits,
        }
        return result

    finally:
        if uploaded is not None and uploaded.name:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def analyze_clip_gemini_v43_repeatability_with_personnel_pass(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V4.3: five normal V1 full-chart calls plus three isolated mechanical-personnel calls.

    Non-personnel fields come only from the five-run V1 consensus. Personnel is derived
    only from the separate accounting pass, preventing the personnel prompt from degrading
    the rest of the chart.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = _upload_video(client, source)

        # Full-chart repeatability: unchanged V1 behavior.
        full_runs: list[dict] = []
        for run_index in range(5):
            chart = _call_structured(
                client,
                uploaded,
                model,
                fps,
                _prompt(play_context),
                FootballSnapChart,
            )
            full_runs.append(chart.model_dump())
            if run_index < 4:
                time.sleep(2)

        chart_fields = [
            name
            for name in FootballSnapChart.model_fields
            if name not in {"overall_confidence", "uncertain_fields", "analysis_notes"}
        ]

        result = {}
        stability = {}
        no_majority_fields = []
        unanimous_fields = []
        strong_majority_fields = []

        for field in chart_fields:
            # Personnel will be overwritten by the isolated accounting consensus below.
            values = [run.get(field) for run in full_runs]
            consensus, top_count, vote_counts, has_majority = _majority_vote_v41(values)
            result[field] = consensus
            agreement = top_count / len(full_runs)
            stability[field] = {
                "top_count": top_count,
                "run_count": len(full_runs),
                "agreement": agreement,
                "has_majority": has_majority,
                "vote_counts": vote_counts,
                "source": "v1-full-chart",
            }
            if top_count == len(full_runs):
                unanimous_fields.append(field)
            elif top_count >= 4:
                strong_majority_fields.append(field)
            if not has_majority:
                no_majority_fields.append(field)

        # Isolated personnel accounting: three independent narrow passes.
        personnel_runs = []
        personnel_audits = []
        for run_index in range(3):
            personnel_pass = _call_structured(
                client,
                uploaded,
                model,
                fps,
                _mechanical_personnel_prompt_v43(play_context),
                MechanicalPersonnelPassV43,
            )
            personnel, audit = _derive_personnel_from_accounting_v43(
                personnel_pass.offensive_lineman_count,
                personnel_pass.quarterback_count,
                personnel_pass.eligible_players,
                personnel_pass.notes,
            )
            personnel_runs.append(personnel)
            audit["confidence"] = personnel_pass.confidence
            personnel_audits.append(audit)
            if run_index < 2:
                time.sleep(2)

        personnel_consensus, personnel_top_count, personnel_votes, personnel_has_majority = _majority_vote_v41(
            personnel_runs
        )
        # V4.1's helper is intentionally hard-coded to 3-of-5. For these three
        # personnel-only observations, a true majority is 2-of-3.
        if personnel_top_count >= 2 and personnel_votes:
            personnel_consensus = personnel_votes[0]["value"]
            personnel_has_majority = True
        else:
            personnel_consensus = None
            personnel_has_majority = False

        result["personnel"] = personnel_consensus
        stability["personnel"] = {
            "top_count": personnel_top_count,
            "run_count": 3,
            "agreement": personnel_top_count / 3,
            "has_majority": personnel_has_majority,
            "vote_counts": personnel_votes,
            "source": "isolated-mechanical-personnel",
        }

        if "personnel" in no_majority_fields:
            no_majority_fields.remove("personnel")
        if not personnel_has_majority:
            no_majority_fields.append("personnel")

        confidences = [
            float(run["overall_confidence"])
            for run in full_runs
            if run.get("overall_confidence") is not None
        ]
        result["overall_confidence"] = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        uncertain = []
        for run in full_runs:
            for field in run.get("uncertain_fields") or []:
                if field not in uncertain:
                    uncertain.append(field)
        for field in no_majority_fields:
            if field not in uncertain:
                uncertain.append(field)
        if result.get("personnel") in {None, "Unknown"} and "personnel" not in uncertain:
            uncertain.append("personnel")
        result["uncertain_fields"] = uncertain

        valid_personnel_runs = sum(
            bool(audit.get("valid_accounting")) for audit in personnel_audits
        )

        result["analysis_notes"] = (
            "V4.3 separates full-chart repeatability from personnel accounting. "
            "Five unchanged V1 calls determine all non-personnel fields; three isolated "
            f"personnel-only calls determine personnel. {valid_personnel_runs}/3 personnel "
            "runs had complete 5 OL + 1 QB + 5 eligible accounting."
        )

        result["_model"] = model
        result["_video_fps"] = float(fps)
        result["_provider"] = "google-gemini"
        result["_analyzer_version"] = "v4.3-repeatability-separated-personnel"
        result["_repeatability"] = {
            "run_count": 5,
            "consensus_rule": "At least 3 of 5 identical values for full-chart fields",
            "runs": full_runs,
            "stability": stability,
            "unanimous_fields": unanimous_fields,
            "strong_majority_fields": strong_majority_fields,
            "no_majority_fields": no_majority_fields,
        }
        result["_personnel_accounting"] = {
            "method": "Separate personnel-only pass: 5 OL + 1 QB + 5 eligible identities",
            "run_count": 3,
            "consensus_rule": "At least 2 of 3 derived personnel codes",
            "valid_runs": valid_personnel_runs,
            "personnel_runs": personnel_runs,
            "votes": personnel_votes,
            "runs": personnel_audits,
        }
        return result

    finally:
        if uploaded is not None and uploaded.name:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def analyze_clip_gemini_v42_personnel_repeatability(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V4.2: five independent V1-style native-video charts with mechanical personnel accounting.

    Each run must account for 5 OL + 1 QB + 5 eligible skill players. Python derives
    personnel from the eligible-player identities before the five-run consensus vote.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = _upload_video(client, source)
        runs: list[dict] = []
        personnel_audits: list[dict] = []

        for run_index in range(5):
            chart = _call_structured(
                client,
                uploaded,
                model,
                fps,
                _prompt_v42_mechanical_personnel(play_context),
                FootballSnapChartV42,
            )
            run = chart.model_dump()

            personnel, audit = _derive_personnel_v42(chart)
            run["personnel"] = personnel
            run["_personnel_accounting"] = audit
            personnel_audits.append(audit)
            runs.append(run)

            if run_index < 4:
                time.sleep(2)

        chart_fields = [
            name
            for name in FootballSnapChart.model_fields
            if name not in {"overall_confidence", "uncertain_fields", "analysis_notes"}
        ]

        result = {}
        stability = {}
        no_majority_fields = []
        unanimous_fields = []
        strong_majority_fields = []

        for field in chart_fields:
            values = [run.get(field) for run in runs]
            consensus, top_count, vote_counts, has_majority = _majority_vote_v41(values)
            result[field] = consensus
            agreement = top_count / len(runs)

            stability[field] = {
                "top_count": top_count,
                "run_count": len(runs),
                "agreement": agreement,
                "has_majority": has_majority,
                "vote_counts": vote_counts,
            }

            if top_count == len(runs):
                unanimous_fields.append(field)
            elif top_count >= 4:
                strong_majority_fields.append(field)

            if not has_majority:
                no_majority_fields.append(field)

        confidences = [
            float(run["overall_confidence"])
            for run in runs
            if run.get("overall_confidence") is not None
        ]
        result["overall_confidence"] = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        uncertain = []
        for run in runs:
            for field in run.get("uncertain_fields") or []:
                if field not in uncertain:
                    uncertain.append(field)
        for field in no_majority_fields:
            if field not in uncertain:
                uncertain.append(field)
        if result.get("personnel") in {None, "Unknown"} and "personnel" not in uncertain:
            uncertain.append("personnel")
        result["uncertain_fields"] = uncertain

        valid_personnel_runs = sum(
            bool(audit.get("valid_accounting")) for audit in personnel_audits
        )
        result["analysis_notes"] = (
            "V4.2 repeatability consensus from five native-video calls with mechanical personnel "
            f"accounting. {valid_personnel_runs}/5 runs produced complete 5 OL + 1 QB + 5 eligible "
            f"accounting. {len(unanimous_fields)} chart fields were unanimous; "
            f"{len(no_majority_fields)} had no 3-of-5 majority."
        )

        result["_model"] = model
        result["_video_fps"] = float(fps)
        result["_provider"] = "google-gemini"
        result["_analyzer_version"] = "v4.2-mechanical-personnel-repeatability"
        result["_repeatability"] = {
            "run_count": 5,
            "consensus_rule": "At least 3 of 5 identical normalized values",
            "runs": runs,
            "stability": stability,
            "unanimous_fields": unanimous_fields,
            "strong_majority_fields": strong_majority_fields,
            "no_majority_fields": no_majority_fields,
        }
        result["_personnel_accounting"] = {
            "method": "5 OL + 1 QB + 5 eligible players; personnel derived from RB/FB and TE/H-back identities",
            "valid_runs": valid_personnel_runs,
            "runs": personnel_audits,
        }
        return result

    finally:
        if uploaded is not None and uploaded.name:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def analyze_clip_gemini_v41_repeatability(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V4.1 diagnostic: five independent V1 native-video observations plus majority consensus.

    The clip uploads once. The same broad V1 prompt/schema is then called five times.
    A field receives a consensus value only when at least 3 of 5 runs agree.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = _upload_video(client, source)
        runs: list[dict] = []

        for run_index in range(5):
            chart = _call_structured(
                client,
                uploaded,
                model,
                fps,
                _prompt(play_context),
                FootballSnapChart,
            )
            run = chart.model_dump()
            runs.append(run)
            if run_index < 4:
                time.sleep(2)

        chart_fields = [
            name
            for name in FootballSnapChart.model_fields
            if name not in {"overall_confidence", "uncertain_fields", "analysis_notes"}
        ]

        result = {}
        stability = {}
        no_majority_fields = []
        unanimous_fields = []
        strong_majority_fields = []

        for field in chart_fields:
            values = [run.get(field) for run in runs]
            consensus, top_count, vote_counts, has_majority = _majority_vote_v41(values)
            result[field] = consensus
            agreement = top_count / len(runs)

            stability[field] = {
                "top_count": top_count,
                "run_count": len(runs),
                "agreement": agreement,
                "has_majority": has_majority,
                "vote_counts": vote_counts,
            }

            if top_count == len(runs):
                unanimous_fields.append(field)
            elif top_count >= 4:
                strong_majority_fields.append(field)

            if not has_majority:
                no_majority_fields.append(field)

        confidences = [
            float(run["overall_confidence"])
            for run in runs
            if run.get("overall_confidence") is not None
        ]
        result["overall_confidence"] = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        uncertain = []
        for run in runs:
            for field in run.get("uncertain_fields") or []:
                if field not in uncertain:
                    uncertain.append(field)
        for field in no_majority_fields:
            if field not in uncertain:
                uncertain.append(field)
        result["uncertain_fields"] = uncertain

        result["analysis_notes"] = (
            "V4.1 repeatability consensus from five independent V1 native-video calls. "
            f"{len(unanimous_fields)} chart fields were unanimous; "
            f"{len(no_majority_fields)} had no 3-of-5 majority and were left unresolved."
        )

        result["_model"] = model
        result["_video_fps"] = float(fps)
        result["_provider"] = "google-gemini"
        result["_analyzer_version"] = "v4.1-repeatability"
        result["_repeatability"] = {
            "run_count": 5,
            "consensus_rule": "At least 3 of 5 identical normalized values",
            "runs": runs,
            "stability": stability,
            "unanimous_fields": unanimous_fields,
            "strong_majority_fields": strong_majority_fields,
            "no_majority_fields": no_majority_fields,
        }
        return result

    finally:
        if uploaded is not None and uploaded.name:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def analyze_clip_gemini_v4(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V4: V1 native-video foundation plus one narrow corrective native-video pass.

    The second pass may overwrite only historically weak fields. Strong V1 fields remain
    untouched. The video is uploaded once and reused for both calls.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = _upload_video(client, source)

        foundation_chart = _call_structured(
            client,
            uploaded,
            model,
            fps,
            _prompt(play_context),
            FootballSnapChart,
        )
        foundation = foundation_chart.model_dump()

        time.sleep(1)

        correction = _call_structured(
            client,
            uploaded,
            model,
            fps,
            _native_correction_prompt_v4(play_context, foundation),
            NativeCorrectionV4,
        )

        result = dict(foundation)
        field_sources = {key: "v1-foundation" for key in foundation.keys()}

        personnel_correction = _v4_personnel_from_correction(correction)
        if _usable_correction(personnel_correction):
            result["personnel"] = personnel_correction
            field_sources["personnel"] = "v4-correction"

        correction_map = {
            "initial_formation": correction.initial_receiver_structure,
            "initial_backfield": correction.initial_backfield,
            "final_formation": correction.final_receiver_structure,
            "final_backfield": correction.final_backfield,
        }
        for key, value in correction_map.items():
            if _usable_correction(value):
                result[key] = value
                field_sources[key] = "v4-correction"

        # Preserve V1's motion-present judgment. The correction pass only labels the type
        # when V1 already says motion occurred.
        if result.get("motion_present") is True and _usable_correction(correction.motion_type):
            result["motion_type"] = correction.motion_type
            field_sources["motion_type"] = "v4-correction"

        # Preserve V1 play classification. Only correct direction when V1 says the play
        # involved a run/RPO and the correction pass can actually see a direction.
        if result.get("film_play_type") in {"Run", "RPO"} and _usable_correction(correction.run_direction):
            result["run_direction"] = correction.run_direction
            field_sources["run_direction"] = "v4-correction"

        foundation_notes = str(result.get("analysis_notes") or "").strip()
        correction_notes = str(correction.notes or "").strip()
        result["analysis_notes"] = " | ".join(
            part for part in (
                foundation_notes,
                f"V4 correction: {correction_notes}" if correction_notes else "",
            )
            if part
        )

        foundation_uncertain = list(result.get("uncertain_fields") or [])
        correction_uncertain = list(correction.uncertain_fields or [])
        result["uncertain_fields"] = list(dict.fromkeys(foundation_uncertain + correction_uncertain))

        result["_model"] = model
        result["_video_fps"] = float(fps)
        result["_provider"] = "google-gemini"
        result["_analyzer_version"] = "v4-native-corrective"
        result["_field_sources"] = field_sources
        result["_pass_confidence"] = {
            "v1_foundation": foundation.get("overall_confidence"),
            "targeted_correction": correction.confidence,
        }
        result["_passes"] = {
            "v1_foundation": foundation,
            "targeted_correction": correction.model_dump(),
        }
        result["_correction_targets"] = [
            "personnel",
            "initial_formation",
            "initial_backfield",
            "final_formation",
            "final_backfield",
            "motion_type",
            "run_direction",
        ]
        result["_correction_mechanics"] = {
            "personnel_back_count": correction.back_count,
            "personnel_te_count": correction.te_count,
            "personnel_direct": correction.personnel,
        }
        return result

    finally:
        if uploaded is not None and uploaded.name:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def analyze_clip_gemini_v2(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V2: upload a snap once, then run three narrow visual tasks against the same video.

    Structure and movement are kept separate from post-snap concept recognition. Football
    shorthand that can be derived deterministically (personnel and blitz) is calculated
    in Python rather than guessed by the model.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = _upload_video(client, source)

        structure = _call_structured(
            client, uploaded, model, fps, _structure_prompt(play_context), StructurePass
        )
        time.sleep(1)
        movement = _call_structured(
            client, uploaded, model, fps, _movement_prompt(play_context), MovementPass
        )
        time.sleep(1)
        post = _call_structured(
            client, uploaded, model, fps, _postsnap_prompt(play_context), PostSnapPass
        )

        blitz, pressure_family = _derive_pressure(post.rushers, post.pressure_family_observed)
        personnel = _derive_personnel(structure.rb_count, structure.te_count)

        result = {
            "personnel": personnel,
            "formation_family": structure.formation_family,
            "initial_formation": structure.initial_receiver_structure,
            "initial_backfield": structure.initial_backfield,
            "final_formation": structure.final_receiver_structure,
            "final_backfield": structure.final_backfield,
            "formation_strength": structure.formation_strength,

            "motion_present": movement.motion_present,
            "motion_player": movement.motion_player,
            "motion_type": movement.motion_type,
            "motion_direction": movement.motion_direction,
            "motion_start_alignment": movement.motion_start_alignment,
            "motion_end_alignment": movement.motion_end_alignment,
            "shift_present": movement.shift_present,
            "shift_description": movement.shift_description,

            "film_play_type": post.film_play_type,
            "run_concept": post.run_concept,
            "run_direction": post.run_direction,
            "pass_concept": post.pass_concept,
            "rpo": post.rpo,
            "play_action": post.play_action,

            "defensive_personnel": structure.defensive_personnel,
            "front": structure.front_family,
            "initial_box_count": structure.initial_box_count,
            "snap_box_count": structure.snap_box_count,
            "shell": structure.shell,
            "coverage": post.coverage,
            "rushers": post.rushers,
            "blitz": blitz,
            "pressure_family": pressure_family,
            "pressure_source": post.pressure_source,

            "adjustment_trigger": movement.adjustment_trigger,
            "adjustment_type": movement.adjustment_type,
            "adjustment_player": movement.adjustment_player,
            "adjustment_detail": movement.adjustment_detail,

            "overall_confidence": _pass_confidence_average(structure, movement, post),
            "uncertain_fields": _merge_uncertain(structure, movement, post),
            "analysis_notes": " | ".join(
                note for note in (structure.notes, movement.notes, post.notes) if note
            ),

            "_model": model,
            "_video_fps": float(fps),
            "_provider": "google-gemini",
            "_analyzer_version": "v2-specialized",
            "_derived": {
                "rb_count": structure.rb_count,
                "te_count": structure.te_count,
                "personnel_rule": "RB/TE counts -> personnel grouping",
                "blitz_rule": "5+ actual rushers -> blitz",
            },
            "_pass_confidence": {
                "structure": structure.confidence,
                "movement": movement.confidence,
                "post_snap": post.confidence,
            },
            "_passes": {
                "structure": structure.model_dump(),
                "movement": movement.model_dump(),
                "post_snap": post.model_dump(),
            },
        }
        return result
    finally:
        if uploaded is not None and uploaded.name:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def analyze_clip_gemini_v21(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V2.1: three specialized calls, but with mechanical counting and deterministic derivation.

    The video is uploaded once. Personnel, backfield changes, box totals and blitz are calculated
    from lower-level visual observations instead of asking Gemini for the final shorthand directly.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = _upload_video(client, source)

        structure = _call_structured(
            client, uploaded, model, fps, _structure_prompt_v21(play_context), StructurePassV21
        )
        time.sleep(1)
        movement = _call_structured(
            client, uploaded, model, fps, _movement_prompt_v21(play_context), MovementPassV21
        )
        time.sleep(1)
        post = _call_structured(
            client, uploaded, model, fps, _postsnap_prompt_v21(play_context), PostSnapPassV21
        )

        personnel, te_count, eligible_total = _derive_personnel_v21(
            structure.back_count,
            structure.attached_te_count,
            structure.wing_hback_count,
            structure.flexed_te_count,
            structure.split_wr_count,
        )

        initial_box_count = _sum_optional(
            structure.initial_line_box_defenders,
            structure.initial_second_level_box_defenders,
        )
        snap_box_count = _sum_optional(
            structure.snap_line_box_defenders,
            structure.snap_second_level_box_defenders,
        )

        initial_backfield = _backfield_from_shape_and_side(
            structure.initial_backfield_shape,
            movement.rb_initial_side,
        )
        final_backfield = _backfield_from_shape_and_side(
            structure.final_backfield_shape,
            movement.rb_final_side,
        )

        motion_present, shift_present = _derive_movement_v21(movement)
        rushers = _derive_rushers_v21(post)
        blitz, pressure_family = _derive_pressure(rushers, post.pressure_family_observed)

        shift_description = None
        if shift_present:
            if movement.rb_changed_sides:
                shift_description = (
                    f"RB changed from {movement.rb_initial_side} to {movement.rb_final_side} "
                    f"and {'set' if movement.rb_set_before_snap else 'did not set'} before snap"
                )
            elif movement.primary_mover:
                shift_description = (
                    f"{movement.primary_mover}: {movement.movement_start_alignment} -> "
                    f"{movement.movement_end_alignment}"
                )

        result = {
            "personnel": personnel,
            "formation_family": structure.formation_family,
            "initial_formation": structure.initial_receiver_structure,
            "initial_backfield": initial_backfield,
            "final_formation": structure.final_receiver_structure,
            "final_backfield": final_backfield,
            "formation_strength": structure.formation_strength,

            "motion_present": motion_present,
            "motion_player": movement.primary_mover if motion_present else None,
            "motion_type": movement.movement_type if motion_present else None,
            "motion_direction": movement.movement_direction if motion_present else None,
            "motion_start_alignment": movement.movement_start_alignment if motion_present else None,
            "motion_end_alignment": movement.movement_end_alignment if motion_present else None,
            "shift_present": shift_present,
            "shift_description": shift_description,

            "film_play_type": post.film_play_type,
            "run_concept": post.run_concept,
            "run_direction": post.run_direction,
            "pass_concept": post.pass_concept,
            "rpo": post.rpo,
            "play_action": post.play_action,

            "defensive_personnel": structure.defensive_personnel,
            "front": structure.front_family,
            "initial_box_count": initial_box_count,
            "snap_box_count": snap_box_count,
            "shell": structure.shell,
            "coverage": post.coverage,
            "rushers": rushers,
            "blitz": blitz,
            "pressure_family": pressure_family,
            "pressure_source": post.pressure_source,

            "adjustment_trigger": movement.adjustment_trigger,
            "adjustment_type": movement.adjustment_type,
            "adjustment_player": movement.adjustment_player,
            "adjustment_detail": movement.adjustment_detail,

            "overall_confidence": _pass_confidence_average(structure, movement, post),
            "uncertain_fields": _merge_uncertain(structure, movement, post),
            "analysis_notes": " | ".join(
                note for note in (structure.notes, movement.notes, post.notes) if note
            ),

            "_model": model,
            "_video_fps": float(fps),
            "_provider": "google-gemini",
            "_analyzer_version": "v2.1-mechanical",
            "_derived": {
                "back_count": structure.back_count,
                "attached_te_count": structure.attached_te_count,
                "wing_hback_count": structure.wing_hback_count,
                "flexed_te_count": structure.flexed_te_count,
                "te_count": te_count,
                "split_wr_count": structure.split_wr_count,
                "eligible_total": eligible_total,
                "personnel_rule": "5 eligible-player buckets -> RB/TE personnel grouping",
                "rb_initial_side": movement.rb_initial_side,
                "rb_final_side": movement.rb_final_side,
                "rb_changed_sides": movement.rb_changed_sides,
                "rb_set_before_snap": movement.rb_set_before_snap,
                "initial_line_box_defenders": structure.initial_line_box_defenders,
                "initial_second_level_box_defenders": structure.initial_second_level_box_defenders,
                "snap_line_box_defenders": structure.snap_line_box_defenders,
                "snap_second_level_box_defenders": structure.snap_second_level_box_defenders,
                "immediate_rushers_count": post.immediate_rushers_count,
                "delayed_rushers_count": post.delayed_rushers_count,
                "bluff_or_drop_count": post.bluff_or_drop_count,
                "nontraditional_rushers_count": post.nontraditional_rushers_count,
                "blitz_rule": "immediate + delayed actual rushers; 5+ -> blitz",
            },
            "_pass_confidence": {
                "structure": structure.confidence,
                "movement": movement.confidence,
                "post_snap": post.confidence,
            },
            "_passes": {
                "structure": structure.model_dump(),
                "movement": movement.model_dump(),
                "post_snap": post.model_dump(),
            },
        }
        return result
    finally:
        if uploaded is not None and uploaded.name:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass



def _call_structured_images(client, model: str, image_paths: list[str | Path], prompt: str, schema: type[BaseModel]) -> BaseModel:
    input_items = [{"type": "text", "text": prompt}]
    for path in image_paths:
        image_bytes = Path(path).read_bytes()
        input_items.append({
            "type": "image",
            "data": base64.b64encode(image_bytes).decode("utf-8"),
            "mime_type": "image/jpeg",
        })

    interaction = None
    for attempt in range(4):
        try:
            interaction = client.interactions.create(
                model=model,
                input=input_items,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema.model_json_schema(),
                },
            )
            break
        except Exception as exc:
            message = str(exc).lower()
            retryable = any(token in message for token in (
                "503", "service_unavailable", "high demand", "429",
                "resource_exhausted", "rate limit",
            ))
            if not retryable or attempt == 3:
                raise
            time.sleep(2 * (2 ** attempt))

    if interaction is None or not interaction.output_text:
        raise RuntimeError("Gemini returned no structured chart from temporal evidence.")
    return schema.model_validate(json.loads(interaction.output_text))


def analyze_clip_gemini_v3(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """V3: local timestamped evidence sheets plus one Gemini image-analysis call."""
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    evidence_dir = Path("local_data") / "temporal_evidence" / source.stem
    evidence = create_temporal_evidence_sheets(
        clip_path=source,
        output_dir=evidence_dir,
        frame_count=24,
        sheet_count=3,
        columns=2,
        tile_width=640,
        tile_height=360,
    )

    client = genai.Client(api_key=api_key)
    chart = _call_structured_images(
        client, model, evidence["sheet_paths"],
        _temporal_prompt(play_context, evidence), TemporalEvidenceChart,
    )

    personnel, te_count, eligible_total = _derive_personnel_v21(
        chart.back_count, chart.attached_te_count, chart.wing_hback_count,
        chart.flexed_te_count, chart.split_wr_count,
    )
    initial_box_count = _sum_optional(chart.initial_line_box_defenders, chart.initial_second_level_box_defenders)
    snap_box_count = _sum_optional(chart.snap_line_box_defenders, chart.snap_second_level_box_defenders)
    initial_backfield = _backfield_from_shape_and_side(None, chart.rb_initial_side)
    final_backfield = _backfield_from_shape_and_side(None, chart.rb_final_side)

    motion_present = chart.any_player_moving_at_snap
    shift_present = chart.any_alignment_change_set_before_snap
    if chart.rb_changed_sides is True and chart.rb_set_before_snap is True:
        shift_present = True
    if chart.rb_changed_sides is True and chart.rb_set_before_snap is False:
        motion_present = True

    rushers = None
    if chart.immediate_rushers_count is not None:
        rushers = int(chart.immediate_rushers_count) + int(chart.delayed_rushers_count or 0)
    blitz, pressure_family = _derive_pressure(rushers, chart.pressure_family_observed)

    shift_description = None
    if shift_present:
        if chart.rb_changed_sides:
            shift_description = (
                f"RB changed from {chart.rb_initial_side} to {chart.rb_final_side} "
                f"and {'set' if chart.rb_set_before_snap else 'did not set'} before snap"
            )
        elif chart.primary_mover:
            shift_description = f"{chart.primary_mover}: {chart.movement_start_alignment} -> {chart.movement_end_alignment}"

    result = {
        "personnel": personnel,
        "formation_family": chart.formation_family,
        "initial_formation": chart.initial_receiver_structure,
        "initial_backfield": initial_backfield,
        "final_formation": chart.final_receiver_structure,
        "final_backfield": final_backfield,
        "formation_strength": chart.formation_strength,
        "motion_present": motion_present,
        "motion_player": chart.primary_mover if motion_present else None,
        "motion_type": chart.movement_type if motion_present else None,
        "motion_direction": chart.movement_direction if motion_present else None,
        "motion_start_alignment": chart.movement_start_alignment if motion_present else None,
        "motion_end_alignment": chart.movement_end_alignment if motion_present else None,
        "shift_present": shift_present,
        "shift_description": shift_description,
        "film_play_type": chart.film_play_type,
        "run_concept": chart.run_concept,
        "run_direction": chart.run_direction,
        "pass_concept": chart.pass_concept,
        "rpo": chart.rpo,
        "play_action": chart.play_action,
        "defensive_personnel": chart.defensive_personnel,
        "front": chart.front_family,
        "initial_box_count": initial_box_count,
        "snap_box_count": snap_box_count,
        "shell": chart.shell,
        "coverage": chart.coverage,
        "rushers": rushers,
        "blitz": blitz,
        "pressure_family": pressure_family,
        "pressure_source": chart.pressure_source,
        "adjustment_trigger": chart.adjustment_trigger,
        "adjustment_type": chart.adjustment_type,
        "adjustment_player": chart.adjustment_player,
        "adjustment_detail": chart.adjustment_detail,
        "overall_confidence": chart.confidence,
        "uncertain_fields": chart.uncertain_fields,
        "analysis_notes": chart.evidence_notes,
        "_model": model,
        "_provider": "google-gemini",
        "_analyzer_version": "v3-temporal-evidence",
        "_temporal_evidence": {
            "duration_seconds": evidence["duration_seconds"],
            "frame_count": evidence["frame_count"],
            "sheet_paths": evidence["sheet_paths"],
            "snap_frame_index": chart.snap_frame_index,
            "snap_timestamp_seconds": chart.snap_timestamp_seconds,
        },
        "_derived": {
            "back_count": chart.back_count,
            "attached_te_count": chart.attached_te_count,
            "wing_hback_count": chart.wing_hback_count,
            "flexed_te_count": chart.flexed_te_count,
            "te_count": te_count,
            "split_wr_count": chart.split_wr_count,
            "eligible_total": eligible_total,
            "rb_initial_side": chart.rb_initial_side,
            "rb_final_side": chart.rb_final_side,
            "rb_changed_sides": chart.rb_changed_sides,
            "rb_set_before_snap": chart.rb_set_before_snap,
            "initial_line_box_defenders": chart.initial_line_box_defenders,
            "initial_second_level_box_defenders": chart.initial_second_level_box_defenders,
            "snap_line_box_defenders": chart.snap_line_box_defenders,
            "snap_second_level_box_defenders": chart.snap_second_level_box_defenders,
            "immediate_rushers_count": chart.immediate_rushers_count,
            "delayed_rushers_count": chart.delayed_rushers_count,
            "bluff_or_drop_count": chart.bluff_or_drop_count,
            "nontraditional_rushers_count": chart.nontraditional_rushers_count,
        },
        "_passes": {"temporal_evidence": chart.model_dump()},
    }
    return result




def _call_structured_images_v31(
    client,
    model: str,
    evidence: dict,
    prompt: str,
    schema: type[BaseModel],
) -> BaseModel:
    """Send labeled full-resolution chronological frames in one Gemini request."""
    input_items = [{"type": "text", "text": prompt}]

    for item in evidence.get("frames", []):
        input_items.append(
            {
                "type": "text",
                "text": (
                    f"FRAME {int(item['frame_index']):02d} — "
                    f"{float(item['timestamp_seconds']):.2f} seconds"
                ),
            }
        )
        image_bytes = Path(item["path"]).read_bytes()
        input_items.append(
            {
                "type": "image",
                "data": base64.b64encode(image_bytes).decode("utf-8"),
                "mime_type": "image/jpeg",
            }
        )

    interaction = None
    for attempt in range(4):
        try:
            interaction = client.interactions.create(
                model=model,
                input=input_items,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema.model_json_schema(),
                },
            )
            break
        except Exception as exc:
            message = str(exc).lower()
            retryable = any(
                token in message
                for token in (
                    "503",
                    "service_unavailable",
                    "high demand",
                    "429",
                    "resource_exhausted",
                    "rate limit",
                )
            )
            if not retryable or attempt == 3:
                raise
            time.sleep(2 * (2 ** attempt))

    if interaction is None or not interaction.output_text:
        raise RuntimeError("Gemini returned no V3.1 temporal-frame chart.")

    return schema.model_validate(json.loads(interaction.output_text))


def analyze_clip_gemini_v31(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V3.1: extract 12 high-resolution chronological frames locally and make one Gemini call.

    The fps argument is retained for compatibility with the benchmark UI; V3.1 ignores it.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    evidence_dir = Path("local_data") / "temporal_evidence_v31" / source.stem
    evidence = extract_temporal_frames_v31(
        clip_path=source,
        output_dir=evidence_dir,
        width=1280,
    )

    client = genai.Client(api_key=api_key)
    chart = _call_structured_images_v31(
        client=client,
        model=model,
        evidence=evidence,
        prompt=_temporal_prompt_v31(play_context, evidence),
        schema=TemporalEvidenceChart,
    )

    personnel, te_count, eligible_total = _derive_personnel_v21(
        chart.back_count,
        chart.attached_te_count,
        chart.wing_hback_count,
        chart.flexed_te_count,
        chart.split_wr_count,
    )

    initial_box_count = _sum_optional(
        chart.initial_line_box_defenders,
        chart.initial_second_level_box_defenders,
    )
    snap_box_count = _sum_optional(
        chart.snap_line_box_defenders,
        chart.snap_second_level_box_defenders,
    )

    initial_backfield = _backfield_from_shape_and_side(None, chart.rb_initial_side)
    final_backfield = _backfield_from_shape_and_side(None, chart.rb_final_side)

    motion_present = chart.any_player_moving_at_snap
    shift_present = chart.any_alignment_change_set_before_snap
    if chart.rb_changed_sides is True and chart.rb_set_before_snap is True:
        shift_present = True
    if chart.rb_changed_sides is True and chart.rb_set_before_snap is False:
        motion_present = True

    rushers = None
    if chart.immediate_rushers_count is not None:
        rushers = int(chart.immediate_rushers_count) + int(chart.delayed_rushers_count or 0)
    blitz, pressure_family = _derive_pressure(rushers, chart.pressure_family_observed)

    shift_description = None
    if shift_present:
        if chart.rb_changed_sides:
            shift_description = (
                f"RB changed from {chart.rb_initial_side} to {chart.rb_final_side} "
                f"and {'set' if chart.rb_set_before_snap else 'did not set'} before snap"
            )
        elif chart.primary_mover:
            shift_description = (
                f"{chart.primary_mover}: {chart.movement_start_alignment} -> "
                f"{chart.movement_end_alignment}"
            )

    result = {
        "personnel": personnel,
        "formation_family": chart.formation_family,
        "initial_formation": chart.initial_receiver_structure,
        "initial_backfield": initial_backfield,
        "final_formation": chart.final_receiver_structure,
        "final_backfield": final_backfield,
        "formation_strength": chart.formation_strength,

        "motion_present": motion_present,
        "motion_player": chart.primary_mover if motion_present else None,
        "motion_type": chart.movement_type if motion_present else None,
        "motion_direction": chart.movement_direction if motion_present else None,
        "motion_start_alignment": chart.movement_start_alignment if motion_present else None,
        "motion_end_alignment": chart.movement_end_alignment if motion_present else None,
        "shift_present": shift_present,
        "shift_description": shift_description,

        "film_play_type": chart.film_play_type,
        "run_concept": chart.run_concept,
        "run_direction": chart.run_direction,
        "pass_concept": chart.pass_concept,
        "rpo": chart.rpo,
        "play_action": chart.play_action,

        "defensive_personnel": chart.defensive_personnel,
        "front": chart.front_family,
        "initial_box_count": initial_box_count,
        "snap_box_count": snap_box_count,
        "shell": chart.shell,
        "coverage": chart.coverage,
        "rushers": rushers,
        "blitz": blitz,
        "pressure_family": pressure_family,
        "pressure_source": chart.pressure_source,

        "adjustment_trigger": chart.adjustment_trigger,
        "adjustment_type": chart.adjustment_type,
        "adjustment_player": chart.adjustment_player,
        "adjustment_detail": chart.adjustment_detail,

        "overall_confidence": chart.confidence,
        "uncertain_fields": chart.uncertain_fields,
        "analysis_notes": chart.evidence_notes,

        "_model": model,
        "_provider": "google-gemini",
        "_analyzer_version": "v3.1-temporal-frames",
        "_temporal_evidence": {
            "duration_seconds": evidence["duration_seconds"],
            "frame_count": evidence["frame_count"],
            "frames": evidence["frames"],
            "reading_order": evidence["reading_order"],
            "snap_frame_index": chart.snap_frame_index,
            "snap_timestamp_seconds": chart.snap_timestamp_seconds,
        },
        "_derived": {
            "back_count": chart.back_count,
            "attached_te_count": chart.attached_te_count,
            "wing_hback_count": chart.wing_hback_count,
            "flexed_te_count": chart.flexed_te_count,
            "te_count": te_count,
            "split_wr_count": chart.split_wr_count,
            "eligible_total": eligible_total,
            "rb_initial_side": chart.rb_initial_side,
            "rb_final_side": chart.rb_final_side,
            "rb_changed_sides": chart.rb_changed_sides,
            "rb_set_before_snap": chart.rb_set_before_snap,
            "initial_line_box_defenders": chart.initial_line_box_defenders,
            "initial_second_level_box_defenders": chart.initial_second_level_box_defenders,
            "snap_line_box_defenders": chart.snap_line_box_defenders,
            "snap_second_level_box_defenders": chart.snap_second_level_box_defenders,
            "immediate_rushers_count": chart.immediate_rushers_count,
            "delayed_rushers_count": chart.delayed_rushers_count,
            "bluff_or_drop_count": chart.bluff_or_drop_count,
            "nontraditional_rushers_count": chart.nontraditional_rushers_count,
            "personnel_rule": "5 eligible-player buckets -> RB/TE personnel grouping",
            "blitz_rule": "immediate + delayed actual rushers; 5+ -> blitz",
        },
        "_passes": {
            "temporal_frames": chart.model_dump(),
        },
    }
    return result




def analyze_clip_gemini_v32_diagnostic(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V3.2 diagnostic: same 12 high-resolution frames as V3.1, but only a tiny required schema.

    This is not intended as the final scouting chart. It tests whether Flash-Lite can recover
    temporal football evidence from individual still frames before more fields are added back.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    evidence_dir = Path("local_data") / "temporal_evidence_v32" / source.stem
    evidence = extract_temporal_frames_v31(
        clip_path=source,
        output_dir=evidence_dir,
        width=1280,
    )

    client = genai.Client(api_key=api_key)
    chart = _call_structured_images_v31(
        client=client,
        model=model,
        evidence=evidence,
        prompt=_temporal_diagnostic_prompt_v32(play_context, evidence),
        schema=TemporalDiagnosticV32,
    )

    initial_backfield = _backfield_from_shape_and_side(None, chart.rb_initial_side)
    final_backfield = _backfield_from_shape_and_side(None, chart.rb_final_side)
    blitz, pressure_family = _derive_pressure(chart.actual_rusher_count, None)

    # Map the diagnostic observations onto the normal prediction keys where possible so
    # the existing human-vs-AI benchmark table can still display relevant matches.
    result = {
        "personnel": None,
        "formation_family": chart.formation_family,
        "initial_formation": None,
        "initial_backfield": initial_backfield,
        "final_formation": None,
        "final_backfield": final_backfield,
        "formation_strength": None,

        "motion_present": None,
        "motion_player": None,
        "motion_type": None,
        "motion_direction": None,
        "motion_start_alignment": None,
        "motion_end_alignment": None,
        "shift_present": chart.rb_changed_sides,
        "shift_description": (
            f"Diagnostic RB side change: {chart.rb_initial_side} -> {chart.rb_final_side}"
            if chart.rb_changed_sides else "Diagnostic saw no RB side change"
        ),

        "film_play_type": None,
        "run_concept": None,
        "run_direction": None,
        "pass_concept": None,
        "rpo": None,
        "play_action": None,

        "defensive_personnel": None,
        "front": None,
        "initial_box_count": chart.initial_box_count,
        "snap_box_count": chart.snap_box_count,
        "shell": None,
        "coverage": None,
        "rushers": chart.actual_rusher_count,
        "blitz": blitz,
        "pressure_family": pressure_family,
        "pressure_source": None,

        "adjustment_trigger": None,
        "adjustment_type": None,
        "adjustment_player": None,
        "adjustment_detail": None,

        "overall_confidence": chart.confidence,
        "uncertain_fields": [],
        "analysis_notes": chart.evidence_notes,

        "_model": model,
        "_provider": "google-gemini",
        "_analyzer_version": "v3.2-temporal-diagnostic",
        "_diagnostic_only": True,
        "_temporal_evidence": {
            "duration_seconds": evidence["duration_seconds"],
            "frame_count": evidence["frame_count"],
            "frames": evidence["frames"],
            "reading_order": evidence["reading_order"],
            "snap_frame_index": chart.snap_frame_index,
            "snap_timestamp_seconds": chart.snap_timestamp_seconds,
        },
        "_diagnostic": {
            "snap_frame_index": chart.snap_frame_index,
            "snap_timestamp_seconds": chart.snap_timestamp_seconds,
            "formation_family": chart.formation_family,
            "rb_initial_side": chart.rb_initial_side,
            "rb_final_side": chart.rb_final_side,
            "rb_changed_sides": chart.rb_changed_sides,
            "initial_box_count": chart.initial_box_count,
            "snap_box_count": chart.snap_box_count,
            "actual_rusher_count": chart.actual_rusher_count,
            "confidence": chart.confidence,
            "evidence_notes": chart.evidence_notes,
        },
    }
    return result




def _temporal_post_evidence_v33(evidence: dict, snap_frame_index: int) -> dict:
    """Keep one frame before the estimated snap plus every available post-snap frame."""
    frames = evidence.get("frames", [])
    start_index = max(1, int(snap_frame_index) - 1)
    selected = [item for item in frames if int(item["frame_index"]) >= start_index]
    if len(selected) < 4:
        selected = frames[-6:]
    return {
        "clip_path": evidence.get("clip_path"),
        "duration_seconds": evidence.get("duration_seconds"),
        "frame_count": len(selected),
        "frames": selected,
        "reading_order": "Read the supplied frames in ascending frame order.",
    }


def _unique_description_count(items: list[str]) -> int:
    """Count unique non-empty rusher descriptions while ignoring capitalization/spacing."""
    seen = set()
    for item in items or []:
        normalized = " ".join(str(item).strip().lower().split())
        if normalized:
            seen.add(normalized)
    return len(seen)


def analyze_clip_gemini_v33(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V3.3: two focused image passes over one locally extracted 12-frame timeline.

    Pass 1 handles pre-snap structure/movement. Pass 2 handles post-snap play/pressure/coverage.
    The frames are extracted once and reused. Personnel and blitz remain deterministic.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    evidence_dir = Path("local_data") / "temporal_evidence_v33" / source.stem
    evidence = extract_temporal_frames_v31(
        clip_path=source,
        output_dir=evidence_dir,
        width=1280,
    )

    client = genai.Client(api_key=api_key)

    pre = _call_structured_images_v31(
        client=client,
        model=model,
        evidence=evidence,
        prompt=_temporal_presnap_prompt_v33(play_context, evidence),
        schema=TemporalPreSnapV33,
    )

    time.sleep(1)

    post_evidence = _temporal_post_evidence_v33(evidence, pre.snap_frame_index)
    post = _call_structured_images_v31(
        client=client,
        model=model,
        evidence=post_evidence,
        prompt=_temporal_postsnap_prompt_v33(
            play_context,
            post_evidence,
            pre.snap_frame_index,
        ),
        schema=TemporalPostSnapV33,
    )

    personnel, te_count, eligible_total = _derive_personnel_v21(
        pre.back_count,
        pre.attached_te_count,
        pre.wing_hback_count,
        pre.flexed_te_count,
        pre.split_wr_count,
    )

    initial_backfield = _backfield_from_shape_and_side(None, pre.rb_initial_side)
    final_backfield = _backfield_from_shape_and_side(None, pre.rb_final_side)

    motion_present = pre.any_player_moving_at_snap
    shift_present = pre.any_alignment_change_set_before_snap
    if pre.rb_changed_sides and pre.rb_set_before_snap:
        shift_present = True
    if pre.rb_changed_sides and not pre.rb_set_before_snap:
        motion_present = True

    rusher_count = _unique_description_count(post.rusher_descriptions)
    blitz, pressure_family = _derive_pressure(
        rusher_count,
        post.pressure_family_observed,
    )

    shift_description = None
    if shift_present:
        if pre.rb_changed_sides:
            shift_description = (
                f"RB changed from {pre.rb_initial_side} to {pre.rb_final_side} "
                f"and {'set' if pre.rb_set_before_snap else 'did not set'} before snap"
            )
        elif pre.primary_mover and pre.primary_mover.lower() != "unknown":
            shift_description = f"{pre.primary_mover} changed alignment and became set before snap"

    evidence_notes = " | ".join(
        note for note in (pre.evidence_notes, post.evidence_notes) if note
    )

    return {
        "personnel": personnel,
        "formation_family": pre.formation_family,
        "initial_formation": pre.initial_receiver_structure,
        "initial_backfield": initial_backfield,
        "final_formation": pre.final_receiver_structure,
        "final_backfield": final_backfield,
        "formation_strength": pre.formation_strength,

        "motion_present": motion_present,
        "motion_player": pre.primary_mover if motion_present else None,
        "motion_type": pre.movement_type if motion_present else None,
        "motion_direction": pre.movement_direction if motion_present else None,
        "motion_start_alignment": None,
        "motion_end_alignment": None,
        "shift_present": shift_present,
        "shift_description": shift_description,

        "film_play_type": post.film_play_type,
        "run_concept": post.run_concept,
        "run_direction": post.run_direction,
        "pass_concept": post.pass_concept,
        "rpo": post.rpo,
        "play_action": post.play_action,

        "defensive_personnel": pre.defensive_personnel,
        "front": pre.front_family,
        "initial_box_count": pre.initial_box_count,
        "snap_box_count": pre.snap_box_count,
        "shell": pre.shell,
        "coverage": post.coverage,
        "rushers": rusher_count,
        "blitz": blitz,
        "pressure_family": pressure_family,
        "pressure_source": post.pressure_source,

        "adjustment_trigger": pre.adjustment_trigger,
        "adjustment_type": pre.adjustment_type,
        "adjustment_player": None,
        "adjustment_detail": None,

        "overall_confidence": (float(pre.confidence) + float(post.confidence)) / 2.0,
        "uncertain_fields": [],
        "analysis_notes": evidence_notes,

        "_model": model,
        "_provider": "google-gemini",
        "_analyzer_version": "v3.3-two-pass-temporal",
        "_temporal_evidence": {
            "duration_seconds": evidence["duration_seconds"],
            "frame_count": evidence["frame_count"],
            "frames": evidence["frames"],
            "post_frame_count": post_evidence["frame_count"],
            "post_frames": post_evidence["frames"],
            "snap_frame_index": pre.snap_frame_index,
            "snap_timestamp_seconds": pre.snap_timestamp_seconds,
        },
        "_derived": {
            "back_count": pre.back_count,
            "attached_te_count": pre.attached_te_count,
            "wing_hback_count": pre.wing_hback_count,
            "flexed_te_count": pre.flexed_te_count,
            "te_count": te_count,
            "split_wr_count": pre.split_wr_count,
            "eligible_total": eligible_total,
            "rb_initial_side": pre.rb_initial_side,
            "rb_final_side": pre.rb_final_side,
            "rb_changed_sides": pre.rb_changed_sides,
            "rb_set_before_snap": pre.rb_set_before_snap,
            "initial_box_count": pre.initial_box_count,
            "snap_box_count": pre.snap_box_count,
            "rusher_descriptions": post.rusher_descriptions,
            "bluff_or_drop_descriptions": post.bluff_or_drop_descriptions,
            "rusher_count_rule": "unique non-empty rusher descriptions",
            "blitz_rule": "5+ unique actual rushers -> blitz",
        },
        "_pass_confidence": {
            "pre_snap": pre.confidence,
            "post_snap": post.confidence,
        },
        "_passes": {
            "pre_snap": pre.model_dump(),
            "post_snap": post.model_dump(),
        },
    }




def _filter_phase_evidence_v34(evidence: dict, phase: str) -> dict:
    selected = [item for item in evidence.get("frames", []) if item.get("phase") == phase]
    return {
        "clip_path": evidence.get("clip_path"),
        "duration_seconds": evidence.get("duration_seconds"),
        "frame_count": len(selected),
        "frames": selected,
        "estimated_snap_timestamp_seconds": evidence.get("estimated_snap_timestamp_seconds"),
        "reading_order": "Read frames in ascending frame_index.",
    }


def analyze_clip_gemini_v34(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V3.4: coarse snap locator -> dense snap-centered extraction -> focused pre/post passes.

    Three Gemini requests are used:
    1) locate snap from coarse frames,
    2) analyze dense pre-snap frames,
    3) analyze dense post-snap frames.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    base_dir = Path("local_data") / "temporal_evidence_v34" / source.stem
    coarse_dir = base_dir / "coarse"
    centered_dir = base_dir / "snap_centered"

    coarse = extract_temporal_frames_v31(
        clip_path=source,
        output_dir=coarse_dir,
        width=1280,
    )

    client = genai.Client(api_key=api_key)

    locator = _call_structured_images_v31(
        client=client,
        model=model,
        evidence=coarse,
        prompt=_snap_locator_prompt_v34(play_context, coarse),
        schema=SnapLocatorV34,
    )

    centered = extract_snap_centered_frames_v34(
        clip_path=source,
        output_dir=centered_dir,
        snap_timestamp_seconds=locator.snap_timestamp_seconds,
        width=1280,
    )
    pre_evidence = _filter_phase_evidence_v34(centered, "pre_snap")
    post_evidence = _filter_phase_evidence_v34(centered, "post_snap")

    time.sleep(1)

    pre = _call_structured_images_v31(
        client=client,
        model=model,
        evidence=pre_evidence,
        prompt=_temporal_presnap_prompt_v34(play_context, centered),
        schema=TemporalPreSnapV33,
    )

    time.sleep(1)

    post = _call_structured_images_v31(
        client=client,
        model=model,
        evidence=post_evidence,
        prompt=_temporal_postsnap_prompt_v34(play_context, centered),
        schema=TemporalPostSnapV33,
    )

    personnel, te_count, eligible_total = _derive_personnel_v21(
        pre.back_count,
        pre.attached_te_count,
        pre.wing_hback_count,
        pre.flexed_te_count,
        pre.split_wr_count,
    )

    initial_backfield = _backfield_from_shape_and_side(None, pre.rb_initial_side)
    final_backfield = _backfield_from_shape_and_side(None, pre.rb_final_side)

    motion_present = pre.any_player_moving_at_snap
    shift_present = pre.any_alignment_change_set_before_snap
    if pre.rb_changed_sides and pre.rb_set_before_snap:
        shift_present = True
    if pre.rb_changed_sides and not pre.rb_set_before_snap:
        motion_present = True

    rusher_count = _unique_description_count(post.rusher_descriptions)
    blitz, pressure_family = _derive_pressure(
        rusher_count,
        post.pressure_family_observed,
    )

    shift_description = None
    if shift_present:
        if pre.rb_changed_sides:
            shift_description = (
                f"RB changed from {pre.rb_initial_side} to {pre.rb_final_side} "
                f"and {'set' if pre.rb_set_before_snap else 'did not set'} before snap"
            )
        elif pre.primary_mover and pre.primary_mover.lower() != "unknown":
            shift_description = f"{pre.primary_mover} changed alignment and became set before snap"

    evidence_notes = " | ".join(
        note
        for note in (
            f"Snap locator: {locator.evidence_notes}",
            pre.evidence_notes,
            post.evidence_notes,
        )
        if note
    )

    return {
        "personnel": personnel,
        "formation_family": pre.formation_family,
        "initial_formation": pre.initial_receiver_structure,
        "initial_backfield": initial_backfield,
        "final_formation": pre.final_receiver_structure,
        "final_backfield": final_backfield,
        "formation_strength": pre.formation_strength,

        "motion_present": motion_present,
        "motion_player": pre.primary_mover if motion_present else None,
        "motion_type": pre.movement_type if motion_present else None,
        "motion_direction": pre.movement_direction if motion_present else None,
        "motion_start_alignment": None,
        "motion_end_alignment": None,
        "shift_present": shift_present,
        "shift_description": shift_description,

        "film_play_type": post.film_play_type,
        "run_concept": post.run_concept,
        "run_direction": post.run_direction,
        "pass_concept": post.pass_concept,
        "rpo": post.rpo,
        "play_action": post.play_action,

        "defensive_personnel": pre.defensive_personnel,
        "front": pre.front_family,
        "initial_box_count": pre.initial_box_count,
        "snap_box_count": pre.snap_box_count,
        "shell": pre.shell,
        "coverage": post.coverage,
        "rushers": rusher_count,
        "blitz": blitz,
        "pressure_family": pressure_family,
        "pressure_source": post.pressure_source,

        "adjustment_trigger": pre.adjustment_trigger,
        "adjustment_type": pre.adjustment_type,
        "adjustment_player": None,
        "adjustment_detail": None,

        "overall_confidence": (
            float(locator.confidence) + float(pre.confidence) + float(post.confidence)
        ) / 3.0,
        "uncertain_fields": [],
        "analysis_notes": evidence_notes,

        "_model": model,
        "_provider": "google-gemini",
        "_analyzer_version": "v3.4-snap-centered",
        "_temporal_evidence": {
            "coarse_frame_count": coarse["frame_count"],
            "coarse_frames": coarse["frames"],
            "locator_snap_frame_index": locator.snap_frame_index,
            "locator_snap_timestamp_seconds": locator.snap_timestamp_seconds,
            "centered_frame_count": centered["frame_count"],
            "centered_frames": centered["frames"],
            "estimated_snap_timestamp_seconds": centered["estimated_snap_timestamp_seconds"],
            "pre_frame_count": pre_evidence["frame_count"],
            "post_frame_count": post_evidence["frame_count"],
        },
        "_derived": {
            "back_count": pre.back_count,
            "attached_te_count": pre.attached_te_count,
            "wing_hback_count": pre.wing_hback_count,
            "flexed_te_count": pre.flexed_te_count,
            "te_count": te_count,
            "split_wr_count": pre.split_wr_count,
            "eligible_total": eligible_total,
            "rb_initial_side": pre.rb_initial_side,
            "rb_final_side": pre.rb_final_side,
            "rb_changed_sides": pre.rb_changed_sides,
            "rb_set_before_snap": pre.rb_set_before_snap,
            "initial_box_count": pre.initial_box_count,
            "snap_box_count": pre.snap_box_count,
            "rusher_descriptions": post.rusher_descriptions,
            "bluff_or_drop_descriptions": post.bluff_or_drop_descriptions,
            "rusher_count_rule": "unique non-empty rusher descriptions",
            "blitz_rule": "5+ unique actual rushers -> blitz",
        },
        "_pass_confidence": {
            "snap_locator": locator.confidence,
            "pre_snap": pre.confidence,
            "post_snap": post.confidence,
        },
        "_passes": {
            "snap_locator": locator.model_dump(),
            "pre_snap": pre.model_dump(),
            "post_snap": post.model_dump(),
        },
    }


def analyze_clip_stub(clip_path: str, play_context: dict) -> FilmObservation:
    """Legacy placeholder retained for provider-neutral architecture tests."""
    return FilmObservation(
        play_id=str(play_context.get("playId")) if play_context.get("playId") is not None else None,
        game_id=str(play_context.get("gameId")) if play_context.get("gameId") is not None else None,
        notes=f"Video analysis not configured yet for {clip_path}",
    )

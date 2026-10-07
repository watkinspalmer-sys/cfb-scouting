from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FilmObservation:
    """Structured fields produced by human charting or a future video model."""

    play_id: Optional[str] = None
    game_id: Optional[str] = None

    # Offense
    personnel: Optional[str] = None
    formation_family: Optional[str] = None

    # Formation evolution from huddle/set through the snap.
    initial_formation: Optional[str] = None
    initial_backfield: Optional[str] = None
    formation: Optional[str] = None  # Backward-compatible alias for final_formation.
    final_formation: Optional[str] = None
    final_backfield: Optional[str] = None
    formation_strength: Optional[str] = None

    # Motion and shift details.
    motion_present: Optional[bool] = None
    motion_player: Optional[str] = None
    motion_type: Optional[str] = None
    motion_direction: Optional[str] = None
    motion_start_alignment: Optional[str] = None
    motion_end_alignment: Optional[str] = None
    motion: Optional[str] = None  # Backward-compatible alias for motion_type.
    shift_present: Optional[bool] = None
    shift_description: Optional[str] = None
    shift: Optional[str] = None  # Backward-compatible alias for shift_description.
    play_type: Optional[str] = None
    run_concept: Optional[str] = None
    run_direction: Optional[str] = None
    pass_concept: Optional[str] = None
    rpo: Optional[bool] = None
    play_action: Optional[bool] = None

    # Defense
    defensive_personnel: Optional[str] = None
    front: Optional[str] = None
    pre_motion_box_count: Optional[int] = None
    post_motion_box_count: Optional[int] = None
    box_count: Optional[int] = None  # Backward-compatible alias for box count at snap.
    shell: Optional[str] = None
    coverage: Optional[str] = None
    rushers: Optional[int] = None
    blitz: Optional[bool] = None
    pressure_type: Optional[str] = None
    motion_response_type: Optional[str] = None
    motion_response_player: Optional[str] = None
    motion_response: Optional[str] = None

    # Video linkage
    video_start_seconds: Optional[float] = None
    video_end_seconds: Optional[float] = None

    # Model / review metadata
    playbook_match: Optional[str] = None
    match_confidence: Optional[float] = None
    confidence: dict = field(default_factory=dict)
    reviewed: bool = False
    notes: Optional[str] = None

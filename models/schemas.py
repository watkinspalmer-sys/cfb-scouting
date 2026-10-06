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
    formation: Optional[str] = None
    formation_strength: Optional[str] = None
    motion: Optional[str] = None
    shift: Optional[str] = None
    play_type: Optional[str] = None
    run_concept: Optional[str] = None
    run_direction: Optional[str] = None
    pass_concept: Optional[str] = None
    rpo: Optional[bool] = None
    play_action: Optional[bool] = None

    # Defense
    defensive_personnel: Optional[str] = None
    front: Optional[str] = None
    box_count: Optional[int] = None
    shell: Optional[str] = None
    coverage: Optional[str] = None
    rushers: Optional[int] = None
    blitz: Optional[bool] = None
    pressure_type: Optional[str] = None
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

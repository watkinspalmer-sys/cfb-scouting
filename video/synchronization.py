from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoAnchor:
    """Known mapping between a game-clock moment and a video timestamp."""

    quarter: int
    game_clock_seconds: int
    video_seconds: float
    note: Optional[str] = None


def clock_to_seconds(minutes: int, seconds: int) -> int:
    return (minutes * 60) + seconds


def estimate_video_time(
    anchor: VideoAnchor,
    quarter: int,
    game_clock_seconds: int,
) -> float:
    """Rough estimate from a known anchor.

    This intentionally handles only same-quarter linear estimation for v1.
    Broadcast stoppages make cross-quarter extrapolation unreliable.
    """
    if quarter != anchor.quarter:
        raise ValueError("v1 estimation only supports plays in the same quarter as the anchor")

    elapsed_game_seconds = anchor.game_clock_seconds - game_clock_seconds
    return anchor.video_seconds + elapsed_game_seconds

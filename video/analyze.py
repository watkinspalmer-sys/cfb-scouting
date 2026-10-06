from typing import Protocol

from models.schemas import FilmObservation


class VideoAnalyzer(Protocol):
    """Interface that any future video model integration must implement."""

    def analyze_clip(self, clip_path: str, play_context: dict) -> FilmObservation:
        ...


def analyze_clip_stub(clip_path: str, play_context: dict) -> FilmObservation:
    """Placeholder used until a video-model provider is selected."""
    return FilmObservation(
        play_id=str(play_context.get("playId")) if play_context.get("playId") is not None else None,
        game_id=str(play_context.get("gameId")) if play_context.get("gameId") is not None else None,
        notes=f"Video analysis not configured yet for {clip_path}",
    )

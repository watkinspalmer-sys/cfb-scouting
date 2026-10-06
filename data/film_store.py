from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_FILM_CHART = Path("local_data/film_chart.csv")


def load_film_chart(path: str | Path = DEFAULT_FILM_CHART) -> pd.DataFrame:
    """Load locally reviewed film observations."""
    target = Path(path)
    if not target.exists():
        return pd.DataFrame()
    return pd.read_csv(target)


def save_film_observation(
    observation: dict,
    path: str | Path = DEFAULT_FILM_CHART,
) -> Path:
    """Insert or replace one reviewed observation keyed by game_id + play_id."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    existing = load_film_chart(target)
    new_row = pd.DataFrame([observation])

    if not existing.empty and {"game_id", "play_id"}.issubset(existing.columns):
        game_id = str(observation.get("game_id", ""))
        play_id = str(observation.get("play_id", ""))
        keep = ~(
            existing["game_id"].astype(str).eq(game_id)
            & existing["play_id"].astype(str).eq(play_id)
        )
        existing = existing.loc[keep]

    combined = pd.concat([existing, new_row], ignore_index=True)
    combined.to_csv(target, index=False)
    return target

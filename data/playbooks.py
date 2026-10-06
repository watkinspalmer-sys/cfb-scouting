import pandas as pd


PLAYBOOK_COLUMNS = [
    "team",
    "formation_family",
    "formation",
    "play_name",
    "play_type",
    "concept_family",
]


def empty_playbook() -> pd.DataFrame:
    """Return the canonical playbook table shape used by the scouting app."""
    return pd.DataFrame(columns=PLAYBOOK_COLUMNS)


def candidate_matches(
    playbook: pd.DataFrame,
    formation_family: str | None = None,
    formation: str | None = None,
    play_type: str | None = None,
) -> pd.DataFrame:
    """Simple deterministic filter used before future similarity scoring."""
    matches = playbook.copy()

    if formation_family:
        matches = matches[matches["formation_family"].eq(formation_family)]
    if formation:
        matches = matches[matches["formation"].eq(formation)]
    if play_type:
        matches = matches[matches["play_type"].eq(play_type)]

    return matches.reset_index(drop=True)

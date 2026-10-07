from __future__ import annotations

import math
import re

import pandas as pd


FIELD_MAP = [
    ("Personnel", "personnel", "personnel", True),
    ("Formation family", "formation_family", "formation_family", True),
    ("Initial receiver structure", "initial_formation", "initial_formation", True),
    ("Initial backfield", "initial_backfield", "initial_backfield", True),
    ("Receiver structure at snap", "final_formation", "final_formation", True),
    ("Backfield at snap", "final_backfield", "final_backfield", True),
    ("Formation strength", "formation_strength", "formation_strength", True),
    ("Motion present", "motion_present", "motion_present", True),
    ("Motion player", "motion_player", "motion_player", False),
    ("Motion type", "motion_type", "motion_type", True),
    ("Motion direction", "motion_direction", "motion_direction", False),
    ("Shift present", "shift_present", "shift_present", True),
    ("Shift description", "shift_description", "shift_description", False),
    ("Film play type", "film_play_type", "film_play_type", True),
    ("Run concept", "run_concept", "run_concept", False),
    ("Run direction", "run_direction", "run_direction", True),
    ("Pass concept", "pass_concept", "pass_concept", False),
    ("RPO", "rpo", "rpo", True),
    ("Play action", "play_action", "play_action", True),
    ("Defensive personnel", "defensive_personnel", "defensive_personnel", True),
    ("Front", "front", "front", True),
    ("Initial box count", "pre_motion_box_count", "initial_box_count", True),
    ("Box count at snap", "post_motion_box_count", "snap_box_count", True),
    ("Shell", "shell", "shell", True),
    ("Coverage", "coverage", "coverage", False),
    ("Rushers", "rushers", "rushers", True),
    ("Blitz", "blitz", "blitz", True),
    ("Pressure family", "pressure_family", "pressure_family", True),
    ("Pressure source", "pressure_source", "pressure_source", False),
    ("Adjustment trigger", "adjustment_trigger", "adjustment_trigger", True),
    ("Adjustment type", "adjustment_type", "adjustment_type", True),
    ("Adjustment player", "adjustment_player", "adjustment_player", False),
    ("Adjustment detail", "adjustment_detail", "adjustment_detail", False),
]


FIELD_CATEGORIES = {
    "Personnel": "Offensive structure",
    "Formation family": "Offensive structure",
    "Initial receiver structure": "Offensive structure",
    "Initial backfield": "Offensive structure",
    "Receiver structure at snap": "Offensive structure",
    "Backfield at snap": "Offensive structure",
    "Formation strength": "Offensive structure",
    "Motion present": "Pre-snap movement",
    "Motion player": "Pre-snap movement",
    "Motion type": "Pre-snap movement",
    "Motion direction": "Pre-snap movement",
    "Shift present": "Pre-snap movement",
    "Shift description": "Pre-snap movement",
    "Film play type": "Play classification",
    "Run concept": "Play classification",
    "Run direction": "Play classification",
    "Pass concept": "Play classification",
    "RPO": "Play classification",
    "Play action": "Play classification",
    "Defensive personnel": "Defensive structure",
    "Front": "Defensive structure",
    "Initial box count": "Defensive structure",
    "Box count at snap": "Defensive structure",
    "Shell": "Defensive structure",
    "Coverage": "Coverage",
    "Rushers": "Pressure",
    "Blitz": "Pressure",
    "Pressure family": "Pressure",
    "Pressure source": "Pressure",
    "Adjustment trigger": "Defensive adjustment",
    "Adjustment type": "Defensive adjustment",
    "Adjustment player": "Defensive adjustment",
    "Adjustment detail": "Defensive adjustment",
}


ALIASES = {
    "two high": "2 high",
    "2-high": "2 high",
    "one high": "1 high",
    "1-high": "1 high",
    "zero high": "0 high",
    "0-high": "0 high",
    "split back": "split backs",
    "split-backs": "split backs",
    "simulated pressure": "sim pressure",
}


def _missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null"}


BOOLEAN_FIELDS = {"Motion present", "Shift present", "RPO", "Play action", "Blitz"}


def _normalize(value, field: str | None = None):
    if _missing(value):
        return None

    if field in BOOLEAN_FIELDS:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if number == 1:
                return "yes"
            if number == 0:
                return "no"
        text = str(value).strip().lower()
        if text in {"true", "yes", "1", "1.0"}:
            return "yes"
        if text in {"false", "no", "0", "0.0"}:
            return "no"

    if isinstance(value, bool):
        return "yes" if value else "no"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)

    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9+]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = ALIASES.get(text, text)

    if field in {"Initial receiver structure", "Receiver structure at snap"}:
        for structure in ("2x2", "3x1", "2x1", "3x2", "quads", "unbalanced"):
            if structure in text:
                return structure

    if field == "Front":
        if any(token in text for token in ("4 2", "4 down", "even")):
            return "even"
        if any(token in text for token in ("3 3", "3 down", "odd")):
            return "odd"

    if field in {"Initial backfield", "Backfield at snap"}:
        if "back to the right" in text or text == "rb right":
            return "rb right"
        if "back to the left" in text or text == "rb left":
            return "rb left"
        if "split backs" in text:
            return "split backs"
        if text in {"pistol", "pistol dot"}:
            return "pistol dot"

    return text


def compare_prediction(human: dict, prediction: dict) -> pd.DataFrame:
    rows = []
    for label, human_key, ai_key, scored in FIELD_MAP:
        human_value = human.get(human_key)
        if human_key == "final_formation" and _missing(human_value):
            human_value = human.get("formation")
        if human_key == "adjustment_type" and _missing(human_value):
            human_value = human.get("motion_response_type")
        if human_key == "adjustment_player" and _missing(human_value):
            human_value = human.get("motion_response_player")
        if human_key == "adjustment_detail" and _missing(human_value):
            human_value = human.get("motion_response")

        ai_value = prediction.get(ai_key)

        eligible = scored and not _missing(human_value)
        match = None
        if eligible:
            match = _normalize(human_value, label) == _normalize(ai_value, label)

        rows.append(
            {
                "Field": label,
                "Category": FIELD_CATEGORIES.get(label, "Other"),
                "Human": None if _missing(human_value) else human_value,
                "AI": None if _missing(ai_value) else ai_value,
                "Scored": eligible,
                "Match": match,
            }
        )

    return pd.DataFrame(rows)


def score_prediction(comparison: pd.DataFrame) -> tuple[int, int, float | None]:
    scored = comparison[comparison["Scored"].eq(True)]
    if scored.empty:
        return 0, 0, None

    matches = int(scored["Match"].eq(True).sum())
    total = int(len(scored))
    return matches, total, matches / total


def category_summary(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exact-match accuracy by scouting category."""
    if comparisons.empty:
        return pd.DataFrame(columns=["Category", "Matches", "Scored fields", "Accuracy"])

    scored = comparisons[comparisons["Scored"].eq(True)].copy()
    if scored.empty:
        return pd.DataFrame(columns=["Category", "Matches", "Scored fields", "Accuracy"])

    grouped = (
        scored.groupby("Category", dropna=False)
        .agg(
            Matches=("Match", lambda s: int(s.eq(True).sum())),
            **{"Scored fields": ("Match", "size")},
        )
        .reset_index()
    )
    grouped["Accuracy"] = grouped["Matches"] / grouped["Scored fields"]
    return grouped.sort_values(["Accuracy", "Category"], ascending=[False, True]).reset_index(drop=True)


def field_summary(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exact-match accuracy by individual scored benchmark field."""
    columns = ["Field", "Category", "Matches", "Scored plays", "Accuracy"]
    if comparisons.empty:
        return pd.DataFrame(columns=columns)

    scored = comparisons[comparisons["Scored"].eq(True)].copy()
    if scored.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        scored.groupby(["Field", "Category"], dropna=False)
        .agg(
            Matches=("Match", lambda s: int(s.eq(True).sum())),
            **{"Scored plays": ("Match", "size")},
        )
        .reset_index()
    )
    grouped["Accuracy"] = grouped["Matches"] / grouped["Scored plays"]
    return grouped.sort_values(
        ["Category", "Field"],
        ascending=[True, True],
    ).reset_index(drop=True)


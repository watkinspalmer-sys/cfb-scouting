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


def _normalize(value):
    if _missing(value):
        return None

    if isinstance(value, bool):
        return "yes" if value else "no"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)

    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return "yes"
    if text in {"false", "no"}:
        return "no"

    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9+]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return ALIASES.get(text, text)


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
            match = _normalize(human_value) == _normalize(ai_value)

        rows.append(
            {
                "Field": label,
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

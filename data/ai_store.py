from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


DEFAULT_AI_PREDICTIONS = Path("local_data/ai_predictions.csv")


def load_ai_predictions(path: str | Path = DEFAULT_AI_PREDICTIONS) -> pd.DataFrame:
    target = Path(path)
    if not target.exists():
        return pd.DataFrame()
    return pd.read_csv(target)


def save_ai_prediction(
    game_id: str,
    play_id: str,
    model: str,
    chart_side: str,
    prediction: dict,
    score_matches: int | None = None,
    score_total: int | None = None,
    video_fps: float | None = None,
    use_play_text: bool | None = None,
    analyzer_version: str | None = None,
    path: str | Path = DEFAULT_AI_PREDICTIONS,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    version = analyzer_version or prediction.get("_analyzer_version") or "v1-single-pass"

    record = {
        "game_id": str(game_id),
        "play_id": str(play_id),
        "model": model,
        "analyzer_version": version,
        "chart_side": chart_side,
        "video_fps": video_fps,
        "use_play_text": use_play_text,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_confidence": prediction.get("overall_confidence"),
        "score_matches": score_matches,
        "score_total": score_total,
        "prediction_json": json.dumps(prediction, ensure_ascii=False),
    }

    existing = load_ai_predictions(target)
    if not existing.empty and "analyzer_version" not in existing.columns:
        existing["analyzer_version"] = "v1-single-pass"
    elif not existing.empty:
        existing["analyzer_version"] = existing["analyzer_version"].fillna("v1-single-pass")

    new_row = pd.DataFrame([record])

    key_columns = {
        "game_id",
        "play_id",
        "model",
        "analyzer_version",
        "video_fps",
        "use_play_text",
    }
    if not existing.empty and key_columns.issubset(existing.columns):
        keep = ~(
            existing["game_id"].astype(str).eq(str(game_id))
            & existing["play_id"].astype(str).eq(str(play_id))
            & existing["model"].astype(str).eq(str(model))
            & existing["analyzer_version"].astype(str).eq(str(version))
            & pd.to_numeric(existing["video_fps"], errors="coerce").eq(video_fps)
            & existing["use_play_text"].astype(str).str.lower().eq(str(use_play_text).lower())
        )
        existing = existing.loc[keep]

    pd.concat([existing, new_row], ignore_index=True).to_csv(target, index=False)
    return target

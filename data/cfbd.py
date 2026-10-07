import json
import re
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.collegefootballdata.com"
CACHE_DIR = Path("local_data") / "cfbd"

REQUIRED_COLUMNS = [
    "offense", "defense", "playType", "down", "distance",
    "yardsToGoal", "yardsGained", "offenseScore", "defenseScore"
]


def _cache_path(team, year, week):
    safe_team = re.sub(r"[^A-Za-z0-9_-]+", "_", str(team).strip())
    return CACHE_DIR / f"{safe_team}_{int(year)}_week_{int(week)}.json"


def fetch_week_plays(team, year, week, api_key, use_cache=True):
    """Load one team's plays for one week, preferring a persistent local cache."""
    cache_path = _cache_path(team, year, week)

    if use_cache and cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            frame = pd.DataFrame(data)
            if not frame.empty:
                frame["_requested_week"] = int(week)
            return frame
        except (OSError, json.JSONDecodeError, ValueError):
            # A bad cache should never make Film Lab unusable; refetch instead.
            pass

    response = requests.get(
        f"{BASE}/plays",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"year": int(year), "week": int(week), "team": team},
        timeout=60,
    )
    if response.status_code == 401:
        raise ValueError("The API key was rejected (401). Check the key in your app secrets.")
    if response.status_code == 429:
        raise ValueError("You have used up your monthly request allowance (429).")

    response.raise_for_status()
    data = response.json()

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle)

    frame = pd.DataFrame(data)
    if not frame.empty:
        frame["_requested_week"] = int(week)
    return frame


def fetch_plays(team, year, last_week, api_key):
    """Download one team's plays through last_week, reusing per-week disk caches."""
    frames = []
    for week in range(1, int(last_week) + 1):
        frame = fetch_week_plays(team, year, week, api_key)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

import requests
import pandas as pd

BASE = "https://api.collegefootballdata.com"

REQUIRED_COLUMNS = [
    "offense", "defense", "playType", "down", "distance",
    "yardsToGoal", "yardsGained", "offenseScore", "defenseScore"
]


def fetch_plays(team, year, last_week, api_key):
    """Download one team's plays week by week from CollegeFootballData."""
    frames = []
    for week in range(1, last_week + 1):
        response = requests.get(
            f"{BASE}/plays",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"year": year, "week": week, "team": team},
            timeout=60,
        )
        if response.status_code == 401:
            raise ValueError("The API key was rejected (401). Check the key in your app secrets.")
        if response.status_code == 429:
            raise ValueError("You have used up your monthly request allowance (429).")

        response.raise_for_status()
        data = response.json()
        if data:
            frame = pd.DataFrame(data)
            # CFBD play payloads do not always include the requested week.
            # Preserve it explicitly so downstream game/film pages can filter
            # the multi-week fetch reliably.
            frame["_requested_week"] = week
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

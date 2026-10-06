from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from data.cfbd import fetch_plays
from data.film_store import load_film_chart, save_film_observation
from video.clips import extract_clip
from video.timecode import format_timecode, parse_timecode


def get_key():
    try:
        return st.secrets["CFBD_API_KEY"]
    except Exception:
        return None


def _format_clock(value) -> str:
    """Normalize CFBD clock values such as {'minutes': 12, 'seconds': 59}."""
    if isinstance(value, dict):
        minutes = value.get("minutes")
        seconds = value.get("seconds")
        if minutes is not None and seconds is not None:
            return f"{int(minutes)}:{int(seconds):02d}"
    return str(value)


def _safe_text(row: pd.Series, *names: str, default: str = "") -> str:
    for name in names:
        if name in row.index:
            value = row[name]
            if name == "clock" and isinstance(value, dict):
                return _format_clock(value)
            try:
                if pd.notna(value):
                    return str(value)
            except (TypeError, ValueError):
                if value is not None:
                    return str(value)
    return default


def _safe_number(row: pd.Series, *names: str):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return None


def _play_label(row: pd.Series, idx: int) -> str:
    period = _safe_text(row, "period", default="?")
    clock = _safe_text(row, "clock", default="")
    down = _safe_text(row, "down", default="?")
    distance = _safe_text(row, "distance", default="?")
    play_type = _safe_text(row, "playType", default="")
    text = _safe_text(row, "playText", default="")
    if len(text) > 70:
        text = text[:67] + "..."
    return f"{idx + 1}. Q{period} {clock} | {down} & {distance} | {play_type} | {text}"


def main():
    st.set_page_config(page_title="Film Lab", layout="wide")
    st.title("Film Lab")
    st.caption(
        "Pair CFBD play-by-play with a local game video, extract snap clips, "
        "and save reviewed football tags."
    )

    key = get_key()
    if not key:
        st.error("Add CFBD_API_KEY to .streamlit/secrets.toml before using Film Lab.")
        st.stop()

    st.info(
        "This page is designed to run locally. Enter the full path to the game "
        "video on this computer; the video is not uploaded to GitHub."
    )

    with st.sidebar:
        st.header("Game")
        team = st.text_input("Team", value="Tulsa")
        year = st.number_input("Season", min_value=2004, max_value=2030, value=2026, step=1)
        week = st.number_input("Week", min_value=1, max_value=15, value=5, step=1)

        st.header("Video")
        video_path_text = st.text_input(
            "Local video path",
            placeholder=r"C:\Football\NorthTexas_Tulsa.mp4",
        )

    with st.spinner("Loading CFBD plays..."):
        try:
            plays = fetch_plays(team.strip(), int(year), int(week), key)
        except Exception as exc:
            st.error(f"Could not load plays: {exc}")
            st.stop()

    if plays.empty:
        st.warning("No plays returned for this team/week.")
        st.stop()

    # fetch_plays downloads weeks 1..selected week. Use the explicit marker
    # added by data.cfbd because CFBD play payloads may omit a week field.
    if "_requested_week" in plays.columns:
        plays = plays[
            pd.to_numeric(plays["_requested_week"], errors="coerce").eq(int(week))
        ].copy()
    elif "week" in plays.columns:
        plays = plays[
            pd.to_numeric(plays["week"], errors="coerce").eq(int(week))
        ].copy()

    if "offense" in plays.columns:
        side = st.radio("Chart", ["Tulsa offense", "Tulsa defense"], horizontal=True)
        if side == "Tulsa offense":
            filtered = plays[plays["offense"].astype(str).eq(team.strip())].copy()
        else:
            filtered = plays[plays["defense"].astype(str).eq(team.strip())].copy()
    else:
        filtered = plays.copy()

    # Film Lab v1 is for scrimmage scouting, not special teams.
    if "playType" in filtered.columns:
        scrimmage = filtered["playType"].astype(str).str.contains(
            r"Rush|Pass|Sack", case=False, na=False, regex=True
        )
        filtered = filtered[scrimmage].copy()

    if filtered.empty:
        st.warning("No matching plays found for this side of the ball.")
        st.stop()

    filtered = filtered.reset_index(drop=True)
    labels = [_play_label(row, idx) for idx, (_, row) in enumerate(filtered.iterrows())]
    selected_idx = st.selectbox(
        "Select play",
        options=list(range(len(filtered))),
        format_func=lambda i: labels[i],
    )
    play = filtered.iloc[selected_idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quarter", _safe_text(play, "period", default="?"))
    c2.metric("Clock", _safe_text(play, "clock", default="?"))
    c3.metric("Down", _safe_text(play, "down", default="?"))
    c4.metric("Distance", _safe_text(play, "distance", default="?"))

    st.write("**Play:**", _safe_text(play, "playText", default="No play text available"))

    st.divider()
    st.subheader("1. Extract a short clip")

    video_path = Path(video_path_text).expanduser() if video_path_text else None
    if video_path_text:
        if not video_path.exists():
            st.warning("That video path does not exist on this computer.")
        elif not video_path.is_file():
            st.warning(
                "That path points to a folder, not a video file. "
                "Paste the full path to the actual video, including its filename "
                "and extension such as .mp4, .mkv, or .mov."
            )

    t1, t2 = st.columns(2)
    start_text = t1.text_input(
        "Clip start",
        value="0:00",
        help="Enter SS, MM:SS, or HH:MM:SS from the downloaded broadcast.",
    )
    end_text = t2.text_input(
        "Clip end",
        value="0:20",
        help="Usually 10-25 seconds is enough for a snap.",
    )

    play_id = _safe_text(play, "id", "playId", default=f"row-{selected_idx}")
    game_id = _safe_text(play, "gameId", "game_id", default="unknown-game")

    clip_name = f"{game_id}_{play_id}".replace("/", "-").replace(" ", "_") + ".mp4"
    clip_path = Path("clips") / clip_name

    if st.button("Extract clip", type="primary"):
        if not video_path_text:
            st.error("Enter the local video path first.")
        elif not video_path.exists():
            st.error("The local video path could not be found.")
        elif not video_path.is_file():
            st.error(
                "The selected path is a folder. Choose the actual video file "
                "(for example C:\\Users\\palme\\Videos\\game.mp4)."
            )
        else:
            try:
                start_seconds = parse_timecode(start_text)
                end_seconds = parse_timecode(end_text)
                with st.spinner("Extracting clip with FFmpeg..."):
                    extract_clip(
                        str(video_path),
                        str(clip_path),
                        start_seconds,
                        end_seconds,
                    )
                st.success(
                    f"Clip created: {clip_path} "
                    f"({format_timecode(start_seconds)}-{format_timecode(end_seconds)})"
                )
            except Exception as exc:
                st.error(f"Could not extract clip: {exc}")

    if clip_path.exists():
        st.video(str(clip_path))

    st.divider()
    st.subheader("2. Chart the snap")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Offense")
        personnel = st.selectbox("Personnel", ["", "10", "11", "12", "13", "20", "21", "22", "Empty", "Other"])
        formation_family = st.selectbox("Formation family", ["", "Gun", "Pistol", "Under Center", "Goalline", "Other"])
        formation = st.text_input("Formation")
        formation_strength = st.selectbox("Formation strength", ["", "Left", "Right", "Balanced", "Boundary", "Field", "Unknown"])
        motion = st.text_input("Motion")
        shift = st.text_input("Shift")
        play_type = st.selectbox("Film play type", ["", "Run", "Pass", "RPO", "Scramble", "Sack", "Other"])
        run_concept = st.text_input("Run concept")
        run_direction = st.selectbox("Run direction", ["", "Left", "Right", "Middle", "Boundary", "Field", "Unknown"])
        pass_concept = st.text_input("Pass concept")
        rpo = st.selectbox("RPO?", ["Unknown", "No", "Yes"])
        play_action = st.selectbox("Play action?", ["Unknown", "No", "Yes"])

    with right:
        st.markdown("#### Defense")
        defensive_personnel = st.text_input("Defensive personnel", placeholder="4-2-5")
        front = st.text_input("Front", placeholder="Even / Odd / Mint / Bear")
        box_count = st.number_input("Box count", min_value=0, max_value=11, value=6, step=1)
        shell = st.selectbox("Shell", ["", "1-High", "2-High", "0-High", "Unknown"])
        coverage = st.text_input("Coverage", placeholder="Cover 1 / 3 / 4 / 6 / Match / Unknown")
        rushers = st.number_input("Rushers", min_value=0, max_value=11, value=4, step=1)
        blitz = st.selectbox("Blitz?", ["Unknown", "No", "Yes"])
        pressure_type = st.text_input("Pressure type", placeholder="LB / DB / Sim / Zero / Other")
        motion_response = st.text_input("Motion response")
        playbook_match = st.text_input("CFB 27 playbook match")
        match_confidence = st.slider("Playbook match confidence", 0, 100, 0, 5)
        notes = st.text_area("Notes")

    reviewed = st.checkbox("Reviewed / validated", value=True)

    if st.button("Save film chart"):
        try:
            start_seconds = parse_timecode(start_text)
            end_seconds = parse_timecode(end_text)
        except Exception:
            start_seconds = None
            end_seconds = None

        observation = {
            "game_id": game_id,
            "play_id": play_id,
            "team": team.strip(),
            "year": int(year),
            "week": int(week),
            "period": _safe_number(play, "period"),
            "clock": _safe_text(play, "clock"),
            "down": _safe_number(play, "down"),
            "distance": _safe_number(play, "distance"),
            "yards_gained": _safe_number(play, "yardsGained"),
            "play_text": _safe_text(play, "playText"),
            "video_source": str(video_path) if video_path else "",
            "video_start_seconds": start_seconds,
            "video_end_seconds": end_seconds,
            "personnel": personnel or None,
            "formation_family": formation_family or None,
            "formation": formation or None,
            "formation_strength": formation_strength or None,
            "motion": motion or None,
            "shift": shift or None,
            "film_play_type": play_type or None,
            "run_concept": run_concept or None,
            "run_direction": run_direction or None,
            "pass_concept": pass_concept or None,
            "rpo": None if rpo == "Unknown" else rpo == "Yes",
            "play_action": None if play_action == "Unknown" else play_action == "Yes",
            "defensive_personnel": defensive_personnel or None,
            "front": front or None,
            "box_count": int(box_count),
            "shell": shell or None,
            "coverage": coverage or None,
            "rushers": int(rushers),
            "blitz": None if blitz == "Unknown" else blitz == "Yes",
            "pressure_type": pressure_type or None,
            "motion_response": motion_response or None,
            "playbook_match": playbook_match or None,
            "match_confidence": match_confidence / 100 if match_confidence else None,
            "reviewed": reviewed,
            "notes": notes or None,
        }
        saved_path = save_film_observation(observation)
        st.success(f"Saved reviewed film data to {saved_path}")

    st.divider()
    st.subheader("3. Reviewed chart")

    chart = load_film_chart()
    if chart.empty:
        st.caption("No film observations saved yet.")
    else:
        current_game = chart[chart["game_id"].astype(str).eq(str(game_id))] if "game_id" in chart.columns else chart
        st.dataframe(current_game, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()

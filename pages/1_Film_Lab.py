from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from data.cfbd import fetch_week_plays
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


def _saved_observation(game_id: str, play_id: str) -> dict:
    """Return the saved chart row for a play, if one exists."""
    chart = load_film_chart()
    if chart.empty or not {"game_id", "play_id"}.issubset(chart.columns):
        return {}

    matches = chart[
        chart["game_id"].astype(str).eq(str(game_id))
        & chart["play_id"].astype(str).eq(str(play_id))
    ]
    if matches.empty:
        return {}

    row = matches.iloc[-1].to_dict()
    return {
        key: value
        for key, value in row.items()
        if not (isinstance(value, float) and pd.isna(value))
    }


def _saved_text(saved: dict, key: str, default: str = "") -> str:
    value = saved.get(key, default)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value)


def _saved_int(saved: dict, key: str, default: int) -> int:
    value = saved.get(key, default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _saved_yes_no(saved: dict, key: str, default: str = "Unknown") -> str:
    value = saved.get(key)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return "Yes"
        if normalized in {"false", "no", "0"}:
            return "No"
        return default
    return "Yes" if bool(value) else "No"


def _options_with_saved(options: list[str], saved_value) -> tuple[list[str], int]:
    value = "" if saved_value is None else str(saved_value)
    choices = list(options)
    if value and value not in choices:
        choices.append(value)
    try:
        index = choices.index(value)
    except ValueError:
        index = 0
    return choices, index


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

    api_error = None
    with st.spinner("Loading CFBD plays..."):
        try:
            # Film Lab needs only the selected game week. This call is persisted
            # to local_data/cfbd after the first successful download.
            plays = fetch_week_plays(team.strip(), int(year), int(week), key)
        except Exception as exc:
            api_error = str(exc)
            plays = pd.DataFrame()

    manual_mode = False
    if api_error:
        st.warning(
            f"CFBD is unavailable: {api_error} "
            "You can keep charting in Manual play mode below."
        )
        manual_mode = True
    elif plays.empty:
        st.warning(
            "No CFBD plays were returned for this team/week. "
            "You can keep charting in Manual play mode below."
        )
        manual_mode = True

    if not manual_mode:
        if "offense" in plays.columns:
            side = st.radio(
                "Chart",
                [f"{team.strip()} offense", f"{team.strip()} defense"],
                horizontal=True,
            )
            if side == f"{team.strip()} offense":
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
            st.warning(
                "No matching scrimmage plays were found. "
                "Switching to Manual play mode."
            )
            manual_mode = True

    if manual_mode:
        st.markdown("### Manual play entry")
        st.caption(
            "This keeps Film Lab usable without CFBD. Enter the game-state "
            "information from the broadcast or official play-by-play."
        )
        side = st.radio(
            "Chart",
            [f"{team.strip()} offense", f"{team.strip()} defense"],
            horizontal=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        manual_period = m1.number_input(
            "Quarter",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            key="manual_period",
        )
        manual_clock = m2.text_input(
            "Game clock",
            value="15:00",
            key="manual_clock",
        )
        manual_down = m3.number_input(
            "Down",
            min_value=1,
            max_value=4,
            value=1,
            step=1,
            key="manual_down",
        )
        manual_distance = m4.number_input(
            "Distance",
            min_value=0,
            max_value=99,
            value=10,
            step=1,
            key="manual_distance",
        )
        manual_play_text = st.text_input(
            "Play description",
            placeholder="Example: Alston rush right for 6 yards",
            key="manual_play_text",
        )

        manual_id = (
            f"manual-{int(year)}-{int(week)}-"
            f"Q{int(manual_period)}-{manual_clock}-"
            f"D{int(manual_down)}-{int(manual_distance)}"
        ).replace(":", "")
        play = pd.Series(
            {
                "id": manual_id,
                "gameId": f"{team.strip()}-{int(year)}-week-{int(week)}",
                "period": int(manual_period),
                "clock": manual_clock,
                "down": int(manual_down),
                "distance": int(manual_distance),
                "playText": manual_play_text,
                "playType": "",
            }
        )
        selected_idx = 0
    else:
        filtered = filtered.reset_index(drop=True)
        labels = [_play_label(row, idx) for idx, (_, row) in enumerate(filtered.iterrows())]
        selected_idx = st.selectbox(
            "Select play",
            options=list(range(len(filtered))),
            format_func=lambda i: labels[i],
        )
        play = filtered.iloc[selected_idx]

    play_id = _safe_text(play, "id", "playId", default=f"row-{selected_idx}")
    game_id = _safe_text(play, "gameId", "game_id", default="unknown-game")
    saved = _saved_observation(game_id, play_id)
    widget_prefix = f"{game_id}_{play_id}".replace(" ", "_").replace("/", "-")

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

    saved_start = saved.get("video_start_seconds")
    saved_end = saved.get("video_end_seconds")
    start_default = (
        format_timecode(float(saved_start))
        if saved_start is not None and not pd.isna(saved_start)
        else ""
    )
    end_default = (
        format_timecode(float(saved_end))
        if saved_end is not None and not pd.isna(saved_end)
        else ""
    )

    t1, t2 = st.columns(2)
    start_text = t1.text_input(
        "Clip start",
        value=start_default,
        key=f"clip_start_{widget_prefix}",
        help="Enter SS, MM:SS, or HH:MM:SS from the downloaded broadcast.",
    )
    end_text = t2.text_input(
        "Clip end",
        value=end_default,
        key=f"clip_end_{widget_prefix}",
        help="Usually 10-25 seconds is enough for a snap.",
    )

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

    if saved:
        st.success("This play has already been charted. Saved values are loaded for editing.")
    else:
        st.caption("New play: charting fields start clean.")

    personnel_options, personnel_index = _options_with_saved(
        ["", "10", "11", "12", "13", "20", "21", "22", "Other"],
        saved.get("personnel"),
    )
    family_options, family_index = _options_with_saved(
        ["", "Gun", "Pistol", "Under Center", "Goalline", "Other"],
        saved.get("formation_family"),
    )
    initial_structure_options, initial_structure_index = _options_with_saved(
        ["", "2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other"],
        saved.get("initial_formation"),
    )
    final_structure_options, final_structure_index = _options_with_saved(
        ["", "2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other"],
        saved.get("final_formation", saved.get("formation")),
    )
    initial_backfield_options, initial_backfield_index = _options_with_saved(
        ["", "Split backs", "RB left", "RB right", "Pistol dot", "Empty", "Other"],
        saved.get("initial_backfield"),
    )
    final_backfield_options, final_backfield_index = _options_with_saved(
        ["", "Split backs", "RB left", "RB right", "Pistol dot", "Empty", "Other"],
        saved.get("final_backfield"),
    )
    strength_options, strength_index = _options_with_saved(
        ["", "Left", "Right", "Balanced", "Boundary", "Field", "Unknown"],
        saved.get("formation_strength"),
    )
    motion_type_options, motion_type_index = _options_with_saved(
        ["", "Across", "Jet", "Orbit", "Return", "Short", "Out to slot/wide", "Into backfield", "Trade", "Other"],
        saved.get("motion_type", saved.get("motion")),
    )
    run_direction_options, run_direction_index = _options_with_saved(
        ["", "Left", "Right", "Middle", "Boundary", "Field", "Unknown"],
        saved.get("run_direction"),
    )
    shell_options, shell_index = _options_with_saved(
        ["", "1-High", "2-High", "0-High", "Unknown"],
        saved.get("shell"),
    )
    trigger_options, trigger_index = _options_with_saved(
        ["", "None", "Motion", "Shift", "Defensive stem", "Cadence/check", "Other", "Unknown"],
        saved.get("adjustment_trigger"),
    )
    response_options, response_index = _options_with_saved(
        ["", "None", "Bump", "Travel", "Safety rotation", "Front shift", "Box insert", "Box remove", "Other", "Unknown"],
        saved.get("adjustment_type", saved.get("motion_response_type")),
    )
    pressure_family_options, pressure_family_index = _options_with_saved(
        ["", "Standard rush", "Blitz", "Sim pressure", "Creeper", "Zero pressure", "Unknown"],
        saved.get("pressure_family"),
    )

    with st.form(key=f"chart_form_{widget_prefix}"):
        left, right = st.columns(2)

        with left:
            st.markdown("#### Offense")
            personnel = st.selectbox(
                "Personnel",
                personnel_options,
                index=personnel_index,
                help=(
                    "Personnel describes who is on the field, not where they align. "
                    "If a RB motions out to WR, the personnel grouping does not change."
                ),
            )
            formation_family = st.selectbox(
                "Formation family",
                family_options,
                index=family_index,
            )

            st.markdown("##### Formation evolution")
            initial_formation = st.selectbox(
                "Initial receiver structure",
                initial_structure_options,
                index=initial_structure_index,
                help="Receiver distribution before motion or shift.",
            )
            initial_formation_detail = st.text_input(
                "Initial formation detail",
                value=_saved_text(saved, "initial_formation_detail"),
                placeholder="Optional detail: nub, TE attached, condensed, etc.",
            )
            initial_backfield = st.selectbox(
                "Initial backfield alignment",
                initial_backfield_options,
                index=initial_backfield_index,
            )
            final_formation = st.selectbox(
                "Receiver structure at snap",
                final_structure_options,
                index=final_structure_index,
            )
            final_formation_detail = st.text_input(
                "Formation detail at snap",
                value=_saved_text(saved, "final_formation_detail"),
                placeholder="Optional detail: trips boundary, nub TE, condensed, etc.",
            )
            final_backfield = st.selectbox(
                "Backfield at snap",
                final_backfield_options,
                index=final_backfield_index,
            )
            formation_strength = st.selectbox(
                "Formation strength at snap",
                strength_options,
                index=strength_index,
            )

            st.markdown("##### Motion / shift")
            motion_present = st.selectbox(
                "Motion?",
                ["No", "Yes", "Unknown"],
                index=["No", "Yes", "Unknown"].index(_saved_yes_no(saved, "motion_present", "No")),
            )
            motion_player = st.text_input(
                "Motion player",
                value=_saved_text(saved, "motion_player"),
                placeholder="Example: RB #5 / Y / slot WR",
            )
            motion_type = st.selectbox(
                "Motion type",
                motion_type_options,
                index=motion_type_index,
            )
            motion_direction = st.text_input(
                "Motion direction",
                value=_saved_text(saved, "motion_direction"),
                placeholder="Example: left-to-right / field-to-boundary",
            )
            motion_start_alignment = st.text_input(
                "Motion start alignment",
                value=_saved_text(saved, "motion_start_alignment"),
                placeholder="Example: RB in backfield",
            )
            motion_end_alignment = st.text_input(
                "Motion end alignment",
                value=_saved_text(saved, "motion_end_alignment"),
                placeholder="Example: No. 3 receiver in trips",
            )

            shift_present = st.selectbox(
                "Shift?",
                ["No", "Yes", "Unknown"],
                index=["No", "Yes", "Unknown"].index(_saved_yes_no(saved, "shift_present", "No")),
            )
            shift_description = st.text_input(
                "Shift description",
                value=_saved_text(saved, "shift_description", _saved_text(saved, "shift")),
                placeholder="Example: 2x2 to 3x1, multiple players reset",
            )

            play_type_options, play_type_index = _options_with_saved(
                ["", "Run", "Pass", "RPO", "Scramble", "Sack", "Other"],
                saved.get("film_play_type"),
            )
            play_type = st.selectbox(
                "Film play type",
                play_type_options,
                index=play_type_index,
            )
            run_concept = st.text_input(
                "Run concept",
                value=_saved_text(saved, "run_concept"),
            )
            run_direction = st.selectbox(
                "Run direction",
                run_direction_options,
                index=run_direction_index,
            )
            pass_concept = st.text_input(
                "Pass concept",
                value=_saved_text(saved, "pass_concept"),
            )
            rpo = st.selectbox(
                "RPO?",
                ["Unknown", "No", "Yes"],
                index=["Unknown", "No", "Yes"].index(_saved_yes_no(saved, "rpo")),
            )
            play_action = st.selectbox(
                "Play action?",
                ["Unknown", "No", "Yes"],
                index=["Unknown", "No", "Yes"].index(_saved_yes_no(saved, "play_action")),
            )

        with right:
            st.markdown("#### Defense")
            defensive_personnel = st.text_input(
                "Defensive personnel",
                value=_saved_text(saved, "defensive_personnel"),
                placeholder="4-2-5",
            )
            front = st.text_input(
                "Front",
                value=_saved_text(saved, "front"),
                placeholder="Even / Odd / Mint / Bear",
            )
            initial_box_count = st.number_input(
                "Initial box count",
                min_value=0,
                max_value=11,
                value=_saved_int(saved, "pre_motion_box_count", 6),
                step=1,
                help="Count before any offensive motion or defensive adjustment.",
            )
            snap_box_count = st.number_input(
                "Box count at snap",
                min_value=0,
                max_value=11,
                value=_saved_int(saved, "post_motion_box_count", _saved_int(saved, "box_count", 6)),
                step=1,
            )
            shell = st.selectbox(
                "Shell",
                shell_options,
                index=shell_index,
            )
            coverage = st.text_input(
                "Coverage",
                value=_saved_text(saved, "coverage"),
                placeholder="Cover 1 / 3 / 4 / 6 / Match / Unknown",
            )
            rushers = st.number_input(
                "Rushers",
                min_value=0,
                max_value=11,
                value=_saved_int(saved, "rushers", 4),
                step=1,
            )
            blitz = st.selectbox(
                "Blitz?",
                ["Unknown", "No", "Yes"],
                index=["Unknown", "No", "Yes"].index(_saved_yes_no(saved, "blitz")),
            )
            pressure_family = st.selectbox(
                "Pressure family",
                pressure_family_options,
                index=pressure_family_index,
                help=(
                    "Standard rush = normal rush structure; Blitz = 5+ rushers; "
                    "Sim pressure/Creeper = four-man pressure with a non-traditional rusher."
                ),
            )
            pressure_source = st.text_input(
                "Pressure source",
                value=_saved_text(saved, "pressure_source"),
                placeholder="Example: Will / Nickel / Boundary CB / Safety",
            )
            pressure_type = st.text_input(
                "Pressure detail",
                value=_saved_text(saved, "pressure_type"),
                placeholder="Optional detail: boundary CB replace, cross-dog, etc.",
            )

            st.markdown("##### Defensive pre-snap adjustment")
            adjustment_trigger = st.selectbox(
                "Adjustment trigger",
                trigger_options,
                index=trigger_index,
                help="What caused or describes the pre-snap defensive movement.",
            )
            motion_response_type = st.selectbox(
                "Adjustment type",
                response_options,
                index=response_index,
            )
            motion_response_player = st.text_input(
                "Defender adjusting",
                value=_saved_text(saved, "adjustment_player", _saved_text(saved, "motion_response_player")),
                placeholder="Example: Will LB / nickel / safety",
            )
            motion_response = st.text_input(
                "Adjustment detail",
                value=_saved_text(saved, "adjustment_detail", _saved_text(saved, "motion_response")),
                placeholder="Example: Will bumps outside; safety rolls down; Mike inserts into box",
            )

            playbook_match = st.text_input(
                "CFB 27 playbook match",
                value=_saved_text(saved, "playbook_match"),
            )
            saved_confidence = saved.get("match_confidence")
            try:
                saved_confidence_pct = int(round(float(saved_confidence) * 100))
            except (TypeError, ValueError):
                saved_confidence_pct = 0
            match_confidence = st.slider(
                "Playbook match confidence",
                0,
                100,
                saved_confidence_pct,
                5,
            )
            notes = st.text_area(
                "Notes",
                value=_saved_text(saved, "notes"),
            )

        reviewed_default = _saved_yes_no(saved, "reviewed", "Yes") == "Yes"
        reviewed = st.checkbox("Reviewed / validated", value=reviewed_default)

        qc_warnings = []
        motion_details = [
            motion_player,
            motion_type,
            motion_direction,
            motion_start_alignment,
            motion_end_alignment,
        ]
        if motion_present == "No" and any(str(value).strip() for value in motion_details):
            qc_warnings.append("Motion is marked No, but motion detail fields are populated.")
        if shift_present == "No" and shift_description.strip():
            qc_warnings.append("Shift is marked No, but a shift description is populated.")
        if play_type == "Run" and pass_concept.strip():
            qc_warnings.append("Play type is Run, but Pass concept is populated.")
        if play_type == "Pass" and (run_concept.strip() or run_direction):
            qc_warnings.append("Play type is Pass, but Run concept/direction is populated.")
        if adjustment_trigger == "Motion" and motion_present == "No":
            qc_warnings.append(
                "Defensive adjustment trigger is Motion, but Motion is marked No."
            )
        if adjustment_trigger == "Shift" and shift_present == "No":
            qc_warnings.append(
                "Defensive adjustment trigger is Shift, but Shift is marked No."
            )
        if adjustment_trigger in {"", "None"} and (
            motion_response_type not in {"", "None", "Unknown"}
            or motion_response_player.strip()
            or motion_response.strip()
        ):
            qc_warnings.append(
                "A defensive adjustment is populated, but Adjustment trigger is blank/None."
            )
        if pressure_family == "Blitz" and int(rushers) < 5:
            qc_warnings.append(
                "Pressure family is Blitz, but fewer than 5 rushers are charted."
            )
        if pressure_family in {"Sim pressure", "Creeper"} and int(rushers) != 4:
            qc_warnings.append(
                f"Pressure family is {pressure_family}, but rushers is not 4."
            )

        if qc_warnings:
            st.warning("QC check:\n\n- " + "\n- ".join(qc_warnings))

        save_clicked = st.form_submit_button("Save film chart", type="primary")

    if save_clicked:
        try:
            start_seconds = parse_timecode(start_text) if start_text.strip() else None
            end_seconds = parse_timecode(end_text) if end_text.strip() else None
        except Exception:
            start_seconds = None
            end_seconds = None

        observation = {
            "game_id": game_id,
            "play_id": play_id,
            "team": team.strip(),
            "chart_side": "offense" if str(side).endswith("offense") else "defense",
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
            "initial_formation": initial_formation or None,
            "initial_formation_detail": initial_formation_detail or None,
            "initial_backfield": initial_backfield or None,
            "formation": final_formation or None,
            "final_formation": final_formation or None,
            "final_formation_detail": final_formation_detail or None,
            "final_backfield": final_backfield or None,
            "formation_strength": formation_strength or None,
            "motion_present": None if motion_present == "Unknown" else motion_present == "Yes",
            "motion_player": motion_player or None,
            "motion_type": motion_type or None,
            "motion_direction": motion_direction or None,
            "motion_start_alignment": motion_start_alignment or None,
            "motion_end_alignment": motion_end_alignment or None,
            "motion": motion_type or None,
            "shift_present": None if shift_present == "Unknown" else shift_present == "Yes",
            "shift_description": shift_description or None,
            "shift": shift_description or None,
            "film_play_type": play_type or None,
            "run_concept": run_concept or None,
            "run_direction": run_direction or None,
            "pass_concept": pass_concept or None,
            "rpo": None if rpo == "Unknown" else rpo == "Yes",
            "play_action": None if play_action == "Unknown" else play_action == "Yes",
            "defensive_personnel": defensive_personnel or None,
            "front": front or None,
            "pre_motion_box_count": int(initial_box_count),
            "post_motion_box_count": int(snap_box_count),
            "box_count": int(snap_box_count),
            "shell": shell or None,
            "coverage": coverage or None,
            "rushers": int(rushers),
            "blitz": None if blitz == "Unknown" else blitz == "Yes",
            "pressure_family": pressure_family or None,
            "pressure_source": pressure_source or None,
            "pressure_type": pressure_type or None,
            "adjustment_trigger": adjustment_trigger or None,
            "adjustment_type": motion_response_type or None,
            "adjustment_player": motion_response_player or None,
            "adjustment_detail": motion_response or None,
            "motion_response_type": motion_response_type or None,
            "motion_response_player": motion_response_player or None,
            "motion_response": motion_response or None,
            "playbook_match": playbook_match or None,
            "match_confidence": match_confidence / 100 if match_confidence else None,
            "reviewed": reviewed,
            "qc_warning_count": len(qc_warnings),
            "notes": notes or None,
        }
        saved_path = save_film_observation(observation)
        if qc_warnings:
            st.warning(
                f"Saved with {len(qc_warnings)} QC warning(s) to {saved_path}. "
                "Review this row before using it as AI ground truth."
            )
        else:
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

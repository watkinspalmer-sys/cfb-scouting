from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from analytics.benchmark import V5_CORE_FIELDS, category_summary, compare_prediction, field_summary, score_prediction, v5_core_comparison
from data.ai_store import load_ai_predictions, save_ai_prediction
from data.cfbd import fetch_roster, fetch_week_plays
from data.film_store import load_film_chart
from video.analyze import analyze_clip_gemini, analyze_clip_gemini_v2, analyze_clip_gemini_v21, analyze_clip_gemini_v3, analyze_clip_gemini_v31, analyze_clip_gemini_v32_diagnostic, analyze_clip_gemini_v33, analyze_clip_gemini_v34, analyze_clip_gemini_v4, analyze_clip_gemini_v41_repeatability, analyze_clip_gemini_v42_personnel_repeatability, analyze_clip_gemini_v43_repeatability_with_personnel_pass, analyze_clip_gemini_v44_roster_personnel, analyze_clip_gemini_v45_hybrid_personnel, analyze_clip_gemini_v46_personnel_consensus, analyze_clip_gemini_v46_personnel_only, analyze_clip_gemini_v5_core


ANALYZERS = {
    "V5 Core Scouting Analyzer (recommended)": ("v5-core", analyze_clip_gemini_v5_core),
    "V4.6 personnel-only batch validation": ("v4.6-personnel-only", analyze_clip_gemini_v46_personnel_only),
    "V4.6 deterministic personnel consensus": ("v4.6-personnel-consensus", analyze_clip_gemini_v46_personnel_consensus),
    "V4.5 hybrid personnel evidence": ("v4.5-hybrid-personnel", analyze_clip_gemini_v45_hybrid_personnel),
    "V4.4 roster-assisted personnel": ("v4.4-roster-assisted-personnel", analyze_clip_gemini_v44_roster_personnel),
    "V4.3 repeatability + separate personnel pass": ("v4.3-repeatability-separated-personnel", analyze_clip_gemini_v43_repeatability_with_personnel_pass),
    "V4.2 repeatability + mechanical personnel": ("v4.2-mechanical-personnel-repeatability", analyze_clip_gemini_v42_personnel_repeatability),
    "V4.1 repeatability diagnostic — 5x V1": ("v4.1-repeatability", analyze_clip_gemini_v41_repeatability),
    "V4 native video + corrective pass": ("v4-native-corrective", analyze_clip_gemini_v4),
    "V3.4 snap-centered temporal": ("v3.4-snap-centered", analyze_clip_gemini_v34),
    "V3.3 two-pass temporal": ("v3.3-two-pass-temporal", analyze_clip_gemini_v33),
    "V3.2 temporal diagnostic 1-pass": ("v3.2-temporal-diagnostic", analyze_clip_gemini_v32_diagnostic),
    "V3.1 temporal frames 1-pass": ("v3.1-temporal-frames", analyze_clip_gemini_v31),
    "V3 temporal evidence 1-pass": ("v3-temporal-evidence", analyze_clip_gemini_v3),
    "V2.1 mechanical 3-pass": ("v2.1-mechanical", analyze_clip_gemini_v21),
    "V2 specialized 3-pass": ("v2-specialized", analyze_clip_gemini_v2),
    "V1 single-pass baseline": ("v1-single-pass", analyze_clip_gemini),
}


def _secret(name: str, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return default


def _clean(value, default=""):
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    return value


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _id_text(value) -> str:
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value)


def _play_label(row: pd.Series, idx: int) -> str:
    quarter = _clean(row.get("period"), "?")
    clock = _clean(row.get("clock"), "")
    down = _clean(row.get("down"), "?")
    distance = _clean(row.get("distance"), "?")
    text = str(_clean(row.get("play_text"), ""))
    if len(text) > 78:
        text = text[:75] + "..."
    return f"{idx + 1}. Q{quarter} {clock} | {down} & {distance} | {text}"


def _clip_path(row: pd.Series) -> Path:
    game_id = _id_text(_clean(row.get("game_id"), "unknown-game"))
    play_id = _id_text(_clean(row.get("play_id"), "unknown-play"))
    name = f"{game_id}_{play_id}".replace("/", "-").replace(" ", "_") + ".mp4"
    return Path("clips") / name


def _build_side_lookup(reviewed: pd.DataFrame, cfbd_key: str | None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not cfbd_key:
        return lookup
    group_columns = ["team", "year", "week"]
    if not set(group_columns).issubset(reviewed.columns):
        return lookup

    for (team, year, week), _ in reviewed.groupby(group_columns, dropna=False):
        try:
            plays = fetch_week_plays(str(team), int(year), int(week), cfbd_key)
        except Exception:
            continue
        if plays.empty or "id" not in plays.columns:
            continue
        for _, play in plays.iterrows():
            play_id = _id_text(play.get("id"))
            offense = str(_clean(play.get("offense"), ""))
            defense = str(_clean(play.get("defense"), ""))
            if offense == str(team):
                lookup[play_id] = "offense"
            elif defense == str(team):
                lookup[play_id] = "defense"
    return lookup


def _row_side(row: pd.Series, lookup: dict[str, str]) -> str:
    saved = str(_clean(row.get("chart_side"), "")).strip().lower()
    if saved in {"offense", "defense"}:
        return saved
    return lookup.get(_id_text(row.get("play_id")), "")


def _play_context(row: pd.Series, side: str, use_play_text: bool) -> dict:
    return {
        "team": _clean(row.get("team"), ""),
        "chart_side": side,
        "period": _clean(row.get("period"), None),
        "clock": _clean(row.get("clock"), None),
        "down": _clean(row.get("down"), None),
        "distance": _clean(row.get("distance"), None),
        "play_text": _clean(row.get("play_text"), None) if use_play_text else None,
    }




def _position_text(value) -> str:
    if isinstance(value, dict):
        return str(
            value.get("abbreviation")
            or value.get("position")
            or value.get("name")
            or value.get("displayName")
            or ""
        ).strip()
    return str(_clean(value, "")).strip()


def _skill_role_from_roster_position(position) -> str | None:
    text = _position_text(position).upper().replace("-", " ").replace("_", " ")
    compact = " ".join(text.split())

    if compact in {"RB", "HB", "TB", "FB"} or "RUNNING BACK" in compact or "FULLBACK" in compact:
        return "RB/FB"
    if compact in {"TE", "H BACK", "HBACK"} or "TIGHT END" in compact:
        return "TE/H-back"
    if compact == "WR" or "WIDE RECEIVER" in compact:
        return "WR"
    return None


def _roster_skill_entries(roster: pd.DataFrame) -> list[dict]:
    entries = []
    if roster.empty:
        return entries

    for _, row in roster.iterrows():
        role = _skill_role_from_roster_position(row.get("position"))
        if role is None:
            continue

        jersey = _clean(row.get("jersey"), None)
        if jersey is None:
            continue

        try:
            jersey_text = str(int(float(jersey)))
        except (TypeError, ValueError):
            jersey_text = str(jersey).strip()

        name = str(_clean(row.get("name"), "")).strip()
        if not name:
            first = str(_clean(row.get("firstName"), "")).strip()
            last = str(_clean(row.get("lastName"), "")).strip()
            name = " ".join(part for part in (first, last) if part)

        entries.append(
            {
                "jersey": jersey_text,
                "name": name,
                "position": _position_text(row.get("position")),
                "role": role,
            }
        )

    entries.sort(key=lambda item: (int(item["jersey"]) if item["jersey"].isdigit() else 999, item["name"]))
    return entries


def _offense_team_for_row(row: pd.Series, cfbd_key: str | None) -> str | None:
    if not cfbd_key:
        return None
    required = ("team", "year", "week", "play_id")
    if any(_clean(row.get(field), None) is None for field in required):
        return None

    try:
        plays = fetch_week_plays(
            str(row.get("team")),
            int(row.get("year")),
            int(row.get("week")),
            cfbd_key,
        )
    except Exception:
        return None

    if plays.empty or "id" not in plays.columns:
        return None

    target = _id_text(row.get("play_id"))
    matches = plays[plays["id"].map(_id_text).eq(target)]
    if matches.empty:
        return None
    return str(_clean(matches.iloc[0].get("offense"), "")).strip() or None


def _v44_roster_context(row: pd.Series, cfbd_key: str | None) -> dict:
    offense_team = _offense_team_for_row(row, cfbd_key)
    if not offense_team:
        raise ValueError("Could not determine the offensive team from CFBD for this play.")

    year = int(row.get("year"))
    roster = fetch_roster(offense_team, year, cfbd_key)
    skill_roster = _roster_skill_entries(roster)
    if not skill_roster:
        raise ValueError(
            f"No RB/FB, TE/H-back, or WR roster entries were found for {offense_team} {year}."
        )

    return {
        "offense_team": offense_team,
        "offense_skill_roster": skill_roster,
    }




def _comparison_for_prediction(human: dict, prediction: dict) -> pd.DataFrame:
    """Use the analyzer's intended benchmark scope."""
    if str(prediction.get("_analyzer_version", "")) == "v5-core":
        return v5_core_comparison(human, prediction)
    return compare_prediction(human, prediction)


def _render_comparison(human: pd.Series, prediction: dict):
    comparison = _comparison_for_prediction(human.to_dict(), prediction)
    matches, total, accuracy = score_prediction(comparison)

    m1, m2, m3 = st.columns(3)
    m1.metric("Exact benchmark matches", f"{matches}/{total}" if total else "—")
    m2.metric("Exact-match accuracy", f"{accuracy:.0%}" if accuracy is not None else "—")
    confidence = prediction.get("overall_confidence")
    m3.metric("AI overall confidence", f"{float(confidence):.0%}" if confidence is not None else "—")

    display = comparison.copy()
    display["Result"] = display.apply(
        lambda row: "✅" if row["Scored"] and row["Match"] is True
        else "❌" if row["Scored"] and row["Match"] is False
        else "Review",
        axis=1,
    )
    st.dataframe(
        display[["Category", "Field", "Human", "AI", "Result"]],
        hide_index=True,
        use_container_width=True,
    )

    uncertain = prediction.get("uncertain_fields") or []
    if uncertain:
        st.write("**AI says a human should review:**", ", ".join(map(str, uncertain)))
    notes = prediction.get("analysis_notes")
    if notes:
        st.write("**AI notes:**", notes)
    return comparison, matches, total, accuracy


def _run_analyzer(
    analyzer_fn,
    clip_path: Path,
    human: pd.Series,
    side: str,
    use_play_text: bool,
    api_key: str,
    model: str,
    video_fps: float,
    extra_context: dict | None = None,
):
    context = _play_context(human, side, use_play_text)
    if extra_context:
        context.update(extra_context)
    return analyzer_fn(
        clip_path=clip_path,
        play_context=context,
        api_key=api_key,
        model=model.strip(),
        fps=float(video_fps),
    )


def _render_single(
    reviewed: pd.DataFrame,
    side_lookup: dict[str, str],
    cfbd_key: str | None,
    api_key: str | None,
    model: str,
    video_fps: float,
    use_play_text: bool,
    analyzer_label: str,
    analyzer_version: str,
    analyzer_fn,
):
    labels = [_play_label(row, idx) for idx, (_, row) in enumerate(reviewed.iterrows())]
    selected_idx = st.selectbox(
        "Benchmark play",
        options=list(range(len(reviewed))),
        format_func=lambda i: labels[i],
        key="single_benchmark_play",
    )
    human = reviewed.iloc[selected_idx]
    clip_path = _clip_path(human)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quarter", _clean(human.get("period"), "?"))
    c2.metric("Clock", _clean(human.get("clock"), "?"))
    c3.metric("Down", _clean(human.get("down"), "?"))
    c4.metric("Distance", _clean(human.get("distance"), "?"))

    st.write("**Play-by-play:**", _clean(human.get("play_text"), "No play text"))
    if clip_path.exists():
        st.video(str(clip_path))
    else:
        st.error(f"Extracted clip not found at {clip_path}.")

    inferred_side = _row_side(human, side_lookup)
    default_role = 1 if inferred_side == "defense" else 0
    role = st.radio(
        f"{_clean(human.get('team'), 'Chart team')} is:",
        ["Offense", "Defense"],
        index=default_role,
        horizontal=True,
        key="single_role",
    )

    analyze_clicked = st.button(
        f"Analyze snap with {analyzer_label}",
        type="primary",
        disabled=(not bool(api_key) or not clip_path.exists()),
        key="single_analyze",
    )

    result_key = (
        f"ai_result_{_id_text(human.get('game_id'))}_{_id_text(human.get('play_id'))}_"
        f"{model}_{video_fps}_{use_play_text}_{analyzer_version}"
    )

    if analyze_clicked:
        try:
            # Keep analyzer descriptions inside the try block so every branch is valid Python.
            if analyzer_version == "v5-core":
                passes = "one six-field native-video core pass"
            elif analyzer_version == "v4.6-personnel-only":
                passes = "three personnel-only visual-role observations"
            elif analyzer_version == "v4.6-personnel-consensus":
                passes = "five full-chart V1 observations plus three visual-role personnel observations"
            elif analyzer_version == "v4.5-hybrid-personnel":
                passes = "five full-chart V1 observations plus three hybrid personnel-evidence observations"
            elif analyzer_version == "v4.4-roster-assisted-personnel":
                passes = "five full-chart V1 observations plus three roster-assisted jersey-ID observations"
            elif analyzer_version == "v4.3-repeatability-separated-personnel":
                passes = "five full-chart V1 observations plus three personnel-only observations"
            elif analyzer_version == "v4.2-mechanical-personnel-repeatability":
                passes = "five native-video observations with mechanical personnel accounting"
            elif analyzer_version == "v4.1-repeatability":
                passes = "five independent V1 native-video observations"
            elif analyzer_version == "v4-native-corrective":
                passes = "two native-video passes"
            elif analyzer_version == "v3.4-snap-centered":
                passes = "three snap-centered temporal passes"
            elif analyzer_version == "v3.3-two-pass-temporal":
                passes = "two focused temporal passes"
            elif analyzer_version == "v3.2-temporal-diagnostic":
                passes = "one narrow temporal diagnostic pass"
            elif analyzer_version == "v3.1-temporal-frames":
                passes = "one high-resolution temporal-frame pass"
            elif analyzer_version == "v3-temporal-evidence":
                passes = "one temporal-evidence pass"
            elif analyzer_version in {"v2-specialized", "v2.1-mechanical"}:
                passes = "three specialized passes"
            else:
                passes = "one native-video pass"
            with st.spinner(f"Analyzing with {passes} using {model} at {video_fps:.1f} FPS..."):
                extra_context = None
                if analyzer_version in {"v4.6-personnel-consensus", "v4.5-hybrid-personnel", "v4.4-roster-assisted-personnel"}:
                    extra_context = _v44_roster_context(human, cfbd_key)
                prediction = _run_analyzer(
                    analyzer_fn, clip_path, human, role.lower(), use_play_text,
                    api_key, model, video_fps, extra_context,
                )
            st.session_state[result_key] = prediction
        except Exception as exc:
            st.error(f"AI analysis failed: {exc}")

    prediction = st.session_state.get(result_key)
    if not prediction:
        st.info("Run the analyzer on one validated play to see a field-by-field comparison.")
        return

    st.divider()
    st.subheader("AI vs. validated chart")
    _, matches, total, _ = _render_comparison(human, prediction)

    saved_path = save_ai_prediction(
        game_id=_id_text(human.get("game_id")),
        play_id=_id_text(human.get("play_id")),
        model=model.strip(),
        analyzer_version=analyzer_version,
        chart_side=role.lower(),
        prediction=prediction,
        score_matches=matches,
        score_total=total,
        video_fps=float(video_fps),
        use_play_text=bool(use_play_text),
    )
    st.caption(f"Prediction saved locally to {saved_path}.")

    if analyzer_version in {"v4.6-personnel-consensus", "v4.5-hybrid-personnel", "v4.4-roster-assisted-personnel", "v4.3-repeatability-separated-personnel", "v4.2-mechanical-personnel-repeatability", "v4.1-repeatability", "v4-native-corrective", "v3.4-snap-centered", "v3.3-two-pass-temporal", "v3.2-temporal-diagnostic", "v3.1-temporal-frames", "v3-temporal-evidence", "v2-specialized", "v2.1-mechanical"}:
        derived = prediction.get("_derived") or {}
        pass_conf = prediction.get("_pass_confidence") or {}
        d1, d2, d3, d4 = st.columns(4)
        observed_backs = derived.get("back_count", derived.get("rb_count", "—"))
        d1.metric("Observed backs", observed_backs)
        d2.metric("Observed TEs", derived.get("te_count", "—"))
        d3.metric("Derived personnel", prediction.get("personnel", "—"))
        d4.metric("Derived blitz", str(prediction.get("blitz", "—")))
        if analyzer_version == "v4.6-personnel-consensus":
            st.caption(
                "V4.6 preserves five unchanged V1 full-chart calls, then runs three personnel-only "
                "visual-role observations. Each specialist run derives personnel from the five visual "
                "roles only; Python takes a true 2-of-3 vote. Jersey/roster evidence is diagnostic and "
                "can no longer override the football-role count."
            )
            repeatability = prediction.get("_repeatability") or {}
            personnel = prediction.get("_personnel_accounting") or {}
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Full-chart runs", repeatability.get("run_count", 5))
            q2.metric("Personnel runs", personnel.get("run_count", 3))
            q3.metric("Valid personnel runs", f"{personnel.get('valid_runs', 0)}/3")
            q4.metric("Consensus personnel", prediction.get("personnel", "—"))

            derived_runs = personnel.get("personnel_runs") or []
            audits = personnel.get("runs") or []
            if audits:
                rows = []
                for run_index, audit in enumerate(audits, start=1):
                    rows.append(
                        {
                            "Run": run_index,
                            "Personnel": derived_runs[run_index - 1] if run_index - 1 < len(derived_runs) else None,
                            "OL": audit.get("offensive_lineman_count"),
                            "QB": audit.get("quarterback_count"),
                            "RB/FB": audit.get("rb_fb_count"),
                            "TE/H": audit.get("te_hback_count"),
                            "WR": audit.get("wr_count"),
                            "Unknown": audit.get("unknown_eligible_count"),
                            "Valid": audit.get("valid_accounting"),
                        }
                    )
                st.write("**Three visual personnel runs**")
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                with st.expander("View player-role observations"):
                    for run_index, audit in enumerate(audits, start=1):
                        st.write(f"**Personnel run {run_index}**")
                        st.dataframe(
                            pd.DataFrame(audit.get("eligible_players") or []),
                            hide_index=True,
                            use_container_width=True,
                        )

            votes = personnel.get("votes") or []
            if votes:
                st.write(
                    "**Personnel votes:**",
                    " | ".join(f"{item.get('value')}: {item.get('count')}/3" for item in votes),
                )

            jersey = personnel.get("jersey_corroboration") or []
            if jersey:
                with st.expander("Jersey corroboration diagnostics (does not affect personnel)"):
                    jersey_rows = []
                    for item in jersey:
                        jersey_rows.append(
                            {
                                "Initial slot": item.get("initial_slot"),
                                "Corroborated jersey": item.get("corroborated_jersey"),
                                "Clear jersey votes": item.get("clear_jersey_votes"),
                            }
                        )
                    st.dataframe(pd.DataFrame(jersey_rows), hide_index=True, use_container_width=True)
        elif analyzer_version == "v4.5-hybrid-personnel":
            st.caption(
                "V4.5 keeps five unchanged V1 full-chart calls, then runs three isolated personnel "
                "passes that collect both jersey evidence and whole-sequence visual-role evidence. "
                "Clear jersey+roster identity is used only when it does not conflict with the visual role; "
                "partial/unreadable jerseys fall back to visual identity; conflicts are flagged for review."
            )
            repeatability = prediction.get("_repeatability") or {}
            personnel = prediction.get("_personnel_accounting") or {}
            z1, z2, z3, z4 = st.columns(4)
            z1.metric("Full-chart runs", repeatability.get("run_count", 5))
            z2.metric("Hybrid personnel runs", personnel.get("run_count", 3))
            z3.metric("Valid personnel runs", f"{personnel.get('valid_runs', 0)}/3")
            z4.metric("Consensus personnel", prediction.get("personnel", "—"))

            st.write(
                "**Evidence quality:**",
                {
                    "runs with conflicts": personnel.get("conflict_runs", 0),
                    "method": personnel.get("method"),
                },
            )

            personnel_runs = personnel.get("runs") or []
            derived_runs = personnel.get("personnel_runs") or []
            if personnel_runs:
                audit_rows = []
                for run_index, audit in enumerate(personnel_runs, start=1):
                    audit_rows.append(
                        {
                            "Personnel run": run_index,
                            "Derived personnel": derived_runs[run_index - 1] if run_index - 1 < len(derived_runs) else None,
                            "OL": audit.get("offensive_lineman_count"),
                            "QB": audit.get("quarterback_count"),
                            "RB/FB": audit.get("rb_fb_count"),
                            "TE/H": audit.get("te_hback_count"),
                            "WR": audit.get("wr_count"),
                            "Unknown": audit.get("unknown_eligible_count"),
                            "Conflicts": audit.get("conflict_count"),
                            "Valid": audit.get("valid_accounting"),
                        }
                    )
                st.write("**Hybrid personnel runs**")
                st.dataframe(pd.DataFrame(audit_rows), hide_index=True, use_container_width=True)

                with st.expander("View hybrid eligible-player evidence"):
                    for run_index, audit in enumerate(personnel_runs, start=1):
                        st.write(f"**Personnel run {run_index}**")
                        st.dataframe(
                            pd.DataFrame(audit.get("eligible_players") or []),
                            hide_index=True,
                            use_container_width=True,
                        )

            votes = personnel.get("votes") or []
            if votes:
                st.write(
                    "**Personnel votes:**",
                    " | ".join(f"{item.get('value')}: {item.get('count')}/3" for item in votes),
                )
        elif analyzer_version == "v4.4-roster-assisted-personnel":
            st.caption(
                "V4.4 keeps five unchanged V1 full-chart calls, then runs three isolated jersey-ID "
                "passes. Gemini identifies jersey numbers only; Python maps those numbers to the "
                "offensive roster and derives personnel from roster positions."
            )
            repeatability = prediction.get("_repeatability") or {}
            personnel = prediction.get("_personnel_accounting") or {}
            y1, y2, y3, y4 = st.columns(4)
            y1.metric("Full-chart runs", repeatability.get("run_count", 5))
            y2.metric("Roster personnel runs", personnel.get("run_count", 3))
            y3.metric("Valid roster matches", f"{personnel.get('valid_runs', 0)}/3")
            y4.metric("Consensus personnel", prediction.get("personnel", "—"))

            personnel_runs = personnel.get("runs") or []
            derived_runs = personnel.get("personnel_runs") or []
            if personnel_runs:
                audit_rows = []
                for run_index, audit in enumerate(personnel_runs, start=1):
                    audit_rows.append(
                        {
                            "Personnel run": run_index,
                            "Derived personnel": derived_runs[run_index - 1] if run_index - 1 < len(derived_runs) else None,
                            "OL": audit.get("offensive_lineman_count"),
                            "QB": audit.get("quarterback_count"),
                            "RB/FB": audit.get("rb_fb_count"),
                            "TE/H": audit.get("te_hback_count"),
                            "WR": audit.get("wr_count"),
                            "Unknown": audit.get("unknown_eligible_count"),
                            "Valid": audit.get("valid_accounting"),
                        }
                    )
                st.write("**Roster-assisted personnel runs**")
                st.dataframe(pd.DataFrame(audit_rows), hide_index=True, use_container_width=True)

                with st.expander("View jersey-to-roster matches"):
                    for run_index, audit in enumerate(personnel_runs, start=1):
                        st.write(f"**Personnel run {run_index}**")
                        st.dataframe(
                            pd.DataFrame(audit.get("eligible_players") or []),
                            hide_index=True,
                            use_container_width=True,
                        )

            votes = personnel.get("votes") or []
            if votes:
                st.write(
                    "**Personnel votes:**",
                    " | ".join(f"{item.get('value')}: {item.get('count')}/3" for item in votes),
                )
        elif analyzer_version == "v4.3-repeatability-separated-personnel":
            st.caption(
                "V4.3 separates the jobs completely: five unchanged V1 calls chart the whole snap, "
                "while three personnel-only calls perform 5 OL + 1 QB + 5 eligible-player accounting. "
                "Only the personnel field is replaced by the personnel pass."
            )
            repeatability = prediction.get("_repeatability") or {}
            personnel = prediction.get("_personnel_accounting") or {}
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Full-chart runs", repeatability.get("run_count", 5))
            x2.metric("Personnel-only runs", personnel.get("run_count", 3))
            x3.metric("Valid personnel runs", f"{personnel.get('valid_runs', 0)}/3")
            x4.metric("Consensus personnel", prediction.get("personnel", "—"))

            full_runs = repeatability.get("runs") or []
            if full_runs:
                full_rows = []
                for run_index, run in enumerate(full_runs, start=1):
                    run_comp = compare_prediction(human.to_dict(), run)
                    run_matches, run_total, run_accuracy = score_prediction(run_comp)
                    full_rows.append(
                        {
                            "Run": run_index,
                            "Matches": run_matches,
                            "Scored fields": run_total,
                            "Accuracy": run_accuracy,
                            "Model confidence": run.get("overall_confidence"),
                        }
                    )
                full_df = pd.DataFrame(full_rows)
                full_display = full_df.copy()
                full_display["Accuracy"] = full_display["Accuracy"].map(
                    lambda value: f"{value:.0%}" if pd.notna(value) else ""
                )
                full_display["Model confidence"] = full_display["Model confidence"].map(
                    lambda value: f"{value:.0%}" if pd.notna(value) else ""
                )
                st.write("**Five unchanged V1 full-chart runs**")
                st.dataframe(full_display, hide_index=True, use_container_width=True)

            personnel_runs = personnel.get("runs") or []
            if personnel_runs:
                audit_rows = []
                for run_index, audit in enumerate(personnel_runs, start=1):
                    audit_rows.append(
                        {
                            "Personnel run": run_index,
                            "Derived personnel": (personnel.get("personnel_runs") or [None, None, None])[run_index - 1],
                            "OL": audit.get("offensive_lineman_count"),
                            "QB": audit.get("quarterback_count"),
                            "RB/FB": audit.get("rb_fb_count"),
                            "TE/H": audit.get("te_hback_count"),
                            "WR": audit.get("wr_count"),
                            "Unknown": audit.get("unknown_eligible_count"),
                            "Valid": audit.get("valid_accounting"),
                            "Confidence": audit.get("confidence"),
                        }
                    )
                st.write("**Separate mechanical personnel runs**")
                st.dataframe(pd.DataFrame(audit_rows), hide_index=True, use_container_width=True)

                with st.expander("View eligible-player classifications from personnel pass"):
                    for run_index, audit in enumerate(personnel_runs, start=1):
                        st.write(f"**Personnel run {run_index}**")
                        st.dataframe(
                            pd.DataFrame(audit.get("eligible_players") or []),
                            hide_index=True,
                            use_container_width=True,
                        )

            votes = personnel.get("votes") or []
            if votes:
                st.write(
                    "**Personnel votes:**",
                    " | ".join(f"{item.get('value')}: {item.get('count')}/3" for item in votes),
                )
        elif analyzer_version == "v4.2-mechanical-personnel-repeatability":
            st.caption(
                "V4.2 keeps the five-run repeatability test but replaces direct personnel guessing "
                "with player accounting: 5 OL + 1 QB + exactly 5 eligible skill players. Each eligible "
                "player is classified as RB/FB, TE/H-back, WR, or Unknown, and Python derives personnel."
            )
            repeatability = prediction.get("_repeatability") or {}
            personnel = prediction.get("_personnel_accounting") or {}
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Independent runs", repeatability.get("run_count", 5))
            p2.metric("Valid personnel accounting", f"{personnel.get('valid_runs', 0)}/5")
            p3.metric("Consensus personnel", prediction.get("personnel", "—"))
            unresolved = repeatability.get("no_majority_fields") or []
            p4.metric("No-majority fields", len(unresolved))

            audits = personnel.get("runs") or []
            if audits:
                audit_rows = []
                for run_index, audit in enumerate(audits, start=1):
                    audit_rows.append(
                        {
                            "Run": run_index,
                            "OL": audit.get("offensive_lineman_count"),
                            "QB": audit.get("quarterback_count"),
                            "RB/FB": audit.get("rb_fb_count"),
                            "TE/H": audit.get("te_hback_count"),
                            "WR": audit.get("wr_count"),
                            "Unknown": audit.get("unknown_eligible_count"),
                            "Valid": audit.get("valid_accounting"),
                            "Notes": audit.get("notes"),
                        }
                    )
                st.write("**Mechanical personnel accounting by run**")
                st.dataframe(pd.DataFrame(audit_rows), hide_index=True, use_container_width=True)

                with st.expander("View the five eligible-player classifications"):
                    for run_index, audit in enumerate(audits, start=1):
                        st.write(f"**Run {run_index}**")
                        players = audit.get("eligible_players") or []
                        st.dataframe(pd.DataFrame(players), hide_index=True, use_container_width=True)

            runs = repeatability.get("runs") or []
            if runs:
                run_rows = []
                for run_index, run in enumerate(runs, start=1):
                    run_comp = compare_prediction(human.to_dict(), run)
                    run_matches, run_total, run_accuracy = score_prediction(run_comp)
                    run_rows.append(
                        {
                            "Run": run_index,
                            "Personnel": run.get("personnel"),
                            "Matches": run_matches,
                            "Scored fields": run_total,
                            "Accuracy": run_accuracy,
                            "Model confidence": run.get("overall_confidence"),
                        }
                    )
                run_df = pd.DataFrame(run_rows)
                run_display = run_df.copy()
                run_display["Accuracy"] = run_display["Accuracy"].map(
                    lambda value: f"{value:.0%}" if pd.notna(value) else ""
                )
                run_display["Model confidence"] = run_display["Model confidence"].map(
                    lambda value: f"{value:.0%}" if pd.notna(value) else ""
                )
                st.write("**Individual run performance**")
                st.dataframe(run_display, hide_index=True, use_container_width=True)

            stability = repeatability.get("stability") or {}
            if stability:
                personnel_stability = stability.get("personnel") or {}
                votes = personnel_stability.get("vote_counts") or []
                if votes:
                    st.write(
                        "**Personnel consensus votes:**",
                        " | ".join(f"{item.get('value')}: {item.get('count')}/5" for item in votes),
                    )
        elif analyzer_version == "v4.1-repeatability":
            st.caption(
                "V4.1 is a repeatability diagnostic. It uploads the clip once, then asks the same "
                "V1 native-video charting question five independent times. The displayed prediction "
                "is a 3-of-5 majority consensus; fields without a true majority are left unresolved."
            )
            repeatability = prediction.get("_repeatability") or {}
            runs = repeatability.get("runs") or []
            stability = repeatability.get("stability") or {}

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Independent V1 runs", repeatability.get("run_count", len(runs)))
            unanimous = repeatability.get("unanimous_fields") or []
            strong = repeatability.get("strong_majority_fields") or []
            unresolved = repeatability.get("no_majority_fields") or []
            r2.metric("Unanimous fields", len(unanimous))
            r3.metric("4/5 majority fields", len(strong))
            r4.metric("No majority", len(unresolved))

            if runs:
                run_rows = []
                for run_index, run in enumerate(runs, start=1):
                    run_comp = compare_prediction(human.to_dict(), run)
                    run_matches, run_total, run_accuracy = score_prediction(run_comp)
                    run_rows.append(
                        {
                            "Run": run_index,
                            "Matches": run_matches,
                            "Scored fields": run_total,
                            "Accuracy": run_accuracy,
                            "Model confidence": run.get("overall_confidence"),
                        }
                    )
                run_df = pd.DataFrame(run_rows)
                run_display = run_df.copy()
                run_display["Accuracy"] = run_display["Accuracy"].map(
                    lambda value: f"{value:.0%}" if pd.notna(value) else ""
                )
                run_display["Model confidence"] = run_display["Model confidence"].map(
                    lambda value: f"{value:.0%}" if pd.notna(value) else ""
                )
                st.write("**Individual V1 run performance**")
                st.dataframe(run_display, hide_index=True, use_container_width=True)

            if stability:
                comparison_for_labels = compare_prediction(human.to_dict(), prediction)
                category_by_field = dict(
                    zip(comparison_for_labels["Field"], comparison_for_labels["Category"])
                )
                ai_key_by_label = {
                    "Personnel": "personnel",
                    "Formation family": "formation_family",
                    "Initial receiver structure": "initial_formation",
                    "Initial backfield": "initial_backfield",
                    "Receiver structure at snap": "final_formation",
                    "Backfield at snap": "final_backfield",
                    "Formation strength": "formation_strength",
                    "Motion present": "motion_present",
                    "Motion type": "motion_type",
                    "Shift present": "shift_present",
                    "Film play type": "film_play_type",
                    "Run direction": "run_direction",
                    "RPO": "rpo",
                    "Play action": "play_action",
                    "Defensive personnel": "defensive_personnel",
                    "Front": "front",
                    "Initial box count": "initial_box_count",
                    "Box count at snap": "snap_box_count",
                    "Shell": "shell",
                    "Rushers": "rushers",
                    "Blitz": "blitz",
                    "Pressure family": "pressure_family",
                    "Adjustment trigger": "adjustment_trigger",
                    "Adjustment type": "adjustment_type",
                }
                stability_rows = []
                for label, ai_key in ai_key_by_label.items():
                    info = stability.get(ai_key)
                    if not info:
                        continue
                    votes = info.get("vote_counts") or []
                    vote_text = " | ".join(
                        f"{item.get('value')}: {item.get('count')}/5" for item in votes
                    )
                    stability_rows.append(
                        {
                            "Category": category_by_field.get(label, ""),
                            "Field": label,
                            "Consensus": prediction.get(ai_key),
                            "Agreement": info.get("agreement"),
                            "Majority": info.get("has_majority"),
                            "Votes": vote_text,
                        }
                    )
                stability_df = pd.DataFrame(stability_rows)
                if not stability_df.empty:
                    stability_display = stability_df.copy()
                    stability_display["Agreement"] = stability_display["Agreement"].map(
                        lambda value: f"{value:.0%}" if pd.notna(value) else ""
                    )
                    st.write("**Field stability across five identical V1 calls**")
                    st.dataframe(
                        stability_display,
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download V4.1 repeatability field stability CSV",
                        data=stability_df.to_csv(index=False).encode("utf-8"),
                        file_name="v4.1_repeatability_field_stability.csv",
                        mime="text/csv",
                        key="download_v41_stability",
                    )

            if unresolved:
                st.warning(
                    "No 3-of-5 majority: " + ", ".join(unresolved)
                )
        elif analyzer_version == "v4-native-corrective":
            st.caption(
                "V4 keeps the V1 native-video chart as the foundation, then runs one narrow "
                "native-video correction pass. Only personnel, initial/final receiver structure, "
                "initial/final backfield, motion type, and run direction can be overwritten."
            )
            field_sources = prediction.get("_field_sources") or {}
            correction_mechanics = prediction.get("_correction_mechanics") or {}
            corrected = [
                field for field, source in field_sources.items()
                if source == "v4-correction"
            ]
            st.write("**Fields replaced by corrective pass:**", corrected if corrected else "None")
            st.write(
                "**Personnel correction counts:**",
                {
                    "backs": correction_mechanics.get("personnel_back_count"),
                    "TEs": correction_mechanics.get("personnel_te_count"),
                    "direct personnel": correction_mechanics.get("personnel_direct"),
                },
            )
        elif analyzer_version == "v3.4-snap-centered":
            st.caption(
                "V3.4 first locates the snap from coarse frames, then re-extracts a dense timeline "
                "from -5.0s to +4.0s around that estimated snap. Pre-snap and post-snap analysis "
                "then run only on those football-relative frames."
            )
            evidence = prediction.get("_temporal_evidence") or {}
            derived_v34 = prediction.get("_derived") or {}
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Coarse snap frame", evidence.get("locator_snap_frame_index", "—"))
            snap_time = evidence.get("locator_snap_timestamp_seconds")
            e2.metric("Estimated snap", f"{snap_time:.2f}s" if isinstance(snap_time, (int, float)) else "—")
            e3.metric("RB start", derived_v34.get("rb_initial_side", "—"))
            e4.metric("RB final", derived_v34.get("rb_final_side", "—"))
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Initial box", derived_v34.get("initial_box_count", "—"))
            b2.metric("Snap box", derived_v34.get("snap_box_count", "—"))
            rusher_desc = derived_v34.get("rusher_descriptions") or []
            b3.metric("Unique rushers", len(rusher_desc))
            b4.metric("Derived blitz", str(prediction.get("blitz", "—")))
            st.write("**Enumerated rushers:**", rusher_desc if rusher_desc else "—")
            bluff_desc = derived_v34.get("bluff_or_drop_descriptions") or []
            if bluff_desc:
                st.write("**Bluff/drop defenders:**", bluff_desc)
            centered_frames = evidence.get("centered_frames") or []
            with st.expander("View V3.4 snap-centered frames"):
                for item in centered_frames:
                    path = item.get("path")
                    if path and Path(path).exists():
                        rel = item.get("relative_to_snap_seconds")
                        rel_text = f"{float(rel):+.2f}s" if isinstance(rel, (int, float)) else ""
                        st.image(
                            str(path),
                            caption=(
                                f"Frame {int(item.get('frame_index', 0)):02d} — "
                                f"{float(item.get('timestamp_seconds', 0)):.2f}s "
                                f"({rel_text} from snap)"
                            ),
                            use_container_width=True,
                        )
        elif analyzer_version == "v3.3-two-pass-temporal":
            st.caption(
                "V3.3 reuses one 12-frame timeline for two narrow calls: a pre-snap structure/"
                "movement pass and a post-snap play/pressure/coverage pass. Rusher count is derived "
                "from the number of unique rusher descriptions instead of asking for a count directly."
            )
            evidence = prediction.get("_temporal_evidence") or {}
            derived_v33 = prediction.get("_derived") or {}
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Snap frame", evidence.get("snap_frame_index", "—"))
            snap_time = evidence.get("snap_timestamp_seconds")
            e2.metric("Snap time", f"{snap_time:.2f}s" if isinstance(snap_time, (int, float)) else "—")
            e3.metric("RB start", derived_v33.get("rb_initial_side", "—"))
            e4.metric("RB final", derived_v33.get("rb_final_side", "—"))
            rusher_desc = derived_v33.get("rusher_descriptions") or []
            st.write("**Enumerated rushers:**", rusher_desc if rusher_desc else "—")
            bluff_desc = derived_v33.get("bluff_or_drop_descriptions") or []
            if bluff_desc:
                st.write("**Bluff/drop defenders:**", bluff_desc)
            frames = evidence.get("frames") or []
            with st.expander("View V3.3 evidence frames"):
                for item in frames:
                    path = item.get("path")
                    if path and Path(path).exists():
                        st.image(
                            str(path),
                            caption=(
                                f"Frame {int(item.get('frame_index', 0)):02d} — "
                                f"{float(item.get('timestamp_seconds', 0)):.2f}s"
                            ),
                            use_container_width=True,
                        )
        elif analyzer_version == "v3.2-temporal-diagnostic":
            st.caption(
                "V3.2 is a diagnostic only. It uses the same 12 high-resolution temporal frames "
                "as V3.1 but asks Gemini for only snap timing, formation family, RB start/end side, "
                "box counts, and actual rusher count."
            )
            diagnostic = prediction.get("_diagnostic") or {}
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Snap frame", diagnostic.get("snap_frame_index", "—"))
            snap_time = diagnostic.get("snap_timestamp_seconds")
            e2.metric("Snap time", f"{snap_time:.2f}s" if isinstance(snap_time, (int, float)) else "—")
            e3.metric("RB start", diagnostic.get("rb_initial_side", "—"))
            e4.metric("RB final", diagnostic.get("rb_final_side", "—"))
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("RB changed", str(diagnostic.get("rb_changed_sides", "—")))
            b2.metric("Initial box", diagnostic.get("initial_box_count", "—"))
            b3.metric("Snap box", diagnostic.get("snap_box_count", "—"))
            b4.metric("Rushers", diagnostic.get("actual_rusher_count", "—"))
            evidence = prediction.get("_temporal_evidence") or {}
            frames = evidence.get("frames") or []
            with st.expander("View V3.2 diagnostic frames"):
                for item in frames:
                    path = item.get("path")
                    if path and Path(path).exists():
                        st.image(
                            str(path),
                            caption=(
                                f"Frame {int(item.get('frame_index', 0)):02d} — "
                                f"{float(item.get('timestamp_seconds', 0)):.2f}s"
                            ),
                            use_container_width=True,
                        )
        elif analyzer_version == "v3.1-temporal-frames":
            st.caption(
                "V3.1 extracts 12 individual high-resolution frames locally and sends them "
                "to Gemini in chronological order in one request. Python still derives "
                "personnel, backfield labels, box totals, and blitz."
            )
            evidence = prediction.get("_temporal_evidence") or {}
            e1, e2, e3 = st.columns(3)
            e1.metric("Evidence frames", evidence.get("frame_count", "—"))
            e2.metric("Snap frame", evidence.get("snap_frame_index", "—"))
            snap_time = evidence.get("snap_timestamp_seconds")
            e3.metric("Snap time", f"{snap_time:.2f}s" if isinstance(snap_time, (int, float)) else "—")
            frames = evidence.get("frames") or []
            with st.expander("View V3.1 evidence frames"):
                for item in frames:
                    path = item.get("path")
                    if path and Path(path).exists():
                        st.image(
                            str(path),
                            caption=(
                                f"Frame {int(item.get('frame_index', 0)):02d} — "
                                f"{float(item.get('timestamp_seconds', 0)):.2f}s"
                            ),
                            use_container_width=True,
                        )
        elif analyzer_version == "v3-temporal-evidence":
            st.caption(
                "V3 extracts 24 frames locally, builds three timestamped timeline sheets, "
                "and sends those images to Gemini in one request. Python still derives "
                "personnel, backfield labels, box totals, and blitz."
            )
            evidence = prediction.get("_temporal_evidence") or {}
            e1, e2, e3 = st.columns(3)
            e1.metric("Evidence frames", evidence.get("frame_count", "—"))
            e2.metric("Snap frame", evidence.get("snap_frame_index", "—"))
            snap_time = evidence.get("snap_timestamp_seconds")
            e3.metric("Snap time", f"{snap_time:.2f}s" if isinstance(snap_time, (int, float)) else "—")
            sheet_paths = evidence.get("sheet_paths") or []
            with st.expander("View V3 temporal evidence sheets"):
                for sheet_path in sheet_paths:
                    if Path(sheet_path).exists():
                        st.image(str(sheet_path), caption=Path(sheet_path).name, use_container_width=True)
        elif analyzer_version == "v2.1-mechanical":
            st.caption(
                "V2.1 derives personnel from five eligible-player buckets, derives backfield "
                "alignment from explicit RB-side tracking, adds two defensive box layers, and "
                "derives blitz from immediate + delayed actual rushers."
            )
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("RB start side", derived.get("rb_initial_side", "—"))
            x2.metric("RB final side", derived.get("rb_final_side", "—"))
            x3.metric("Immediate rushers", derived.get("immediate_rushers_count", "—"))
            x4.metric("Delayed rushers", derived.get("delayed_rushers_count", "—"))
            st.write(
                "**Eligible-player count:**",
                {
                    "backs": derived.get("back_count"),
                    "attached TE": derived.get("attached_te_count"),
                    "wing/H": derived.get("wing_hback_count"),
                    "flexed TE": derived.get("flexed_te_count"),
                    "WR": derived.get("split_wr_count"),
                    "total": derived.get("eligible_total"),
                },
            )
        else:
            st.caption(
                "V2 derives personnel from RB/TE counts and blitz from actual rusher count "
                "instead of asking the model to guess the shorthand."
            )
        if pass_conf:
            st.write("**Pass confidence:**", pass_conf)

    with st.expander("Raw AI chart"):
        st.json(prediction)


def _stored_v1_accuracy(
    reviewed: pd.DataFrame,
    model: str,
    video_fps: float,
    use_play_text: bool,
    core_only: bool = False,
):
    stored = load_ai_predictions()
    if stored.empty or "prediction_json" not in stored.columns:
        return None

    version = stored.get("analyzer_version")
    if version is None:
        stored = stored.assign(analyzer_version="v1-single-pass")
    else:
        stored["analyzer_version"] = stored["analyzer_version"].fillna("v1-single-pass")

    candidates = stored[
        stored["model"].astype(str).eq(model)
        & stored["analyzer_version"].astype(str).eq("v1-single-pass")
    ].copy()

    if "video_fps" in candidates.columns:
        candidates = candidates[pd.to_numeric(candidates["video_fps"], errors="coerce").eq(video_fps)]
    if "use_play_text" in candidates.columns:
        candidates = candidates[
            candidates["use_play_text"].astype(str).str.lower().eq(str(use_play_text).lower())
        ]

    if candidates.empty:
        return None

    human_by_play = {_id_text(row.get("play_id")): row for _, row in reviewed.iterrows()}
    matches = total = plays = 0
    for _, stored_row in candidates.iterrows():
        play_id = _id_text(stored_row.get("play_id"))
        human = human_by_play.get(play_id)
        if human is None:
            continue
        try:
            pred = json.loads(stored_row["prediction_json"])
        except Exception:
            continue
        comp = compare_prediction(human.to_dict(), pred)
        if core_only:
            comp = comp[comp["Field"].isin(V5_CORE_FIELDS)].copy()
        m, t, _ = score_prediction(comp)
        matches += m
        total += t
        plays += 1

    if not total:
        return None
    return {"plays": plays, "matches": matches, "total": total, "accuracy": matches / total}




def _stored_version_detail(
    reviewed: pd.DataFrame,
    analyzer_version: str,
    model: str,
    video_fps: float,
    use_play_text: bool,
) -> pd.DataFrame:
    """Rebuild per-field benchmark detail from predictions already saved locally."""
    stored = load_ai_predictions()
    if stored.empty or "prediction_json" not in stored.columns:
        return pd.DataFrame()

    if "analyzer_version" not in stored.columns:
        stored["analyzer_version"] = "v1-single-pass"
    else:
        stored["analyzer_version"] = stored["analyzer_version"].fillna("v1-single-pass")

    candidates = stored[
        stored["model"].astype(str).eq(str(model))
        & stored["analyzer_version"].astype(str).eq(str(analyzer_version))
    ].copy()

    if "video_fps" in candidates.columns:
        candidates = candidates[
            pd.to_numeric(candidates["video_fps"], errors="coerce").eq(float(video_fps))
        ]
    if "use_play_text" in candidates.columns:
        candidates = candidates[
            candidates["use_play_text"].astype(str).str.lower().eq(str(use_play_text).lower())
        ]

    if candidates.empty:
        return pd.DataFrame()

    human_by_key = {}
    for _, row in reviewed.iterrows():
        game_id = _id_text(row.get("game_id"))
        play_id = _id_text(row.get("play_id"))
        human_by_key[(game_id, play_id)] = row

    comparisons = []
    for _, stored_row in candidates.iterrows():
        key = (_id_text(stored_row.get("game_id")), _id_text(stored_row.get("play_id")))
        human = human_by_key.get(key)
        if human is None:
            continue
        try:
            prediction = json.loads(stored_row["prediction_json"])
        except Exception:
            continue

        comparison = compare_prediction(human.to_dict(), prediction)
        comparison["game_id"] = key[0]
        comparison["play_id"] = key[1]
        comparison["chart_side"] = str(_clean(stored_row.get("chart_side"), ""))
        comparison["analyzer_version"] = analyzer_version
        comparisons.append(comparison)

    return pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()


def _shared_play_field_comparison(
    reviewed: pd.DataFrame,
    current_version: str,
    model: str,
    video_fps: float,
    use_play_text: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare V1 and the selected analyzer only on plays that exist in both stored result sets.
    Returns current_detail, v1_detail, merged field summary.
    """
    current = _stored_version_detail(
        reviewed, current_version, model, video_fps, use_play_text
    )
    v1 = _stored_version_detail(
        reviewed, "v1-single-pass", model, video_fps, use_play_text
    )
    if current_version == "v5-core":
        current = current[current["Field"].isin(V5_CORE_FIELDS)].copy()
        v1 = v1[v1["Field"].isin(V5_CORE_FIELDS)].copy()
    if current.empty or v1.empty:
        return current, v1, pd.DataFrame()

    current_keys = set(
        zip(current["game_id"].astype(str), current["play_id"].astype(str))
    )
    v1_keys = set(zip(v1["game_id"].astype(str), v1["play_id"].astype(str)))
    shared = current_keys & v1_keys
    if not shared:
        return current.iloc[0:0].copy(), v1.iloc[0:0].copy(), pd.DataFrame()

    current_shared = current[
        current.apply(
            lambda row: (str(row["game_id"]), str(row["play_id"])) in shared,
            axis=1,
        )
    ].copy()
    v1_shared = v1[
        v1.apply(
            lambda row: (str(row["game_id"]), str(row["play_id"])) in shared,
            axis=1,
        )
    ].copy()

    current_fields = field_summary(current_shared).rename(
        columns={
            "Matches": "Current matches",
            "Scored plays": "Current scored plays",
            "Accuracy": "Current accuracy",
        }
    )
    v1_fields = field_summary(v1_shared).rename(
        columns={
            "Matches": "V1 matches",
            "Scored plays": "V1 scored plays",
            "Accuracy": "V1 accuracy",
        }
    )

    merged = v1_fields.merge(
        current_fields,
        on=["Field", "Category"],
        how="outer",
    )
    if not merged.empty:
        merged["Change"] = merged["Current accuracy"] - merged["V1 accuracy"]
        merged = merged.sort_values(
            ["Category", "Change", "Field"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    return current_shared, v1_shared, merged


def _render_stored_field_analysis(
    reviewed: pd.DataFrame,
    analyzer_version: str,
    model: str,
    video_fps: float,
    use_play_text: bool,
):
    if analyzer_version == "v1-single-pass":
        return

    current_detail, v1_detail, comparison = _shared_play_field_comparison(
        reviewed,
        analyzer_version,
        model,
        video_fps,
        use_play_text,
    )
    if comparison.empty:
        return

    current_keys = set(
        zip(current_detail["game_id"].astype(str), current_detail["play_id"].astype(str))
    )
    st.subheader("Stored field-level V1 comparison")
    st.caption(
        f"Built from locally saved predictions only — no new Gemini calls. "
        f"Comparison is restricted to {len(current_keys)} play(s) present in both V1 and "
        f"{analyzer_version}."
    )

    display = comparison.copy()
    for column in ("V1 accuracy", "Current accuracy", "Change"):
        display[column] = display[column].map(
            lambda value: f"{value:+.0%}" if column == "Change" and pd.notna(value)
            else f"{value:.0%}" if pd.notna(value)
            else ""
        )
    st.dataframe(display, hide_index=True, use_container_width=True)

    export = comparison.copy()
    st.download_button(
        "Download V1 vs current field comparison CSV",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name=f"field_comparison_v1_vs_{analyzer_version}.csv",
        mime="text/csv",
        key=f"stored_field_compare_{analyzer_version}",
    )

    current_fields = field_summary(current_detail)
    if not current_fields.empty:
        st.download_button(
            f"Download {analyzer_version} field accuracy CSV",
            data=current_fields.to_csv(index=False).encode("utf-8"),
            file_name=f"field_accuracy_{analyzer_version}.csv",
            mime="text/csv",
            key=f"stored_field_current_{analyzer_version}",
        )




def _render_personnel_batch_validation(
    eligible,
    cfbd_key: str | None,
    api_key: str,
    model: str,
    video_fps: float,
    use_play_text: bool,
    analyzer_fn,
):
    """Run the V4.6 personnel specialist across all validated plays and score Personnel only."""
    st.info(
        "Personnel-only validation: 3 Gemini calls per play. No full-chart V1 calls are made. "
        "Only the validated Personnel field is scored."
    )

    run_batch = st.button(
        f"Run personnel validation on {len(eligible)} plays ({len(eligible) * 3} Gemini calls)",
        type="primary",
        key="run_batch_v46_personnel_only",
    )

    batch_key = f"batch_personnel_only_{model}_{video_fps}_{use_play_text}"
    if run_batch:
        summaries = []
        detail_rows = []
        errors = []
        progress = st.progress(0.0, text="Starting personnel-only validation...")
        status = st.empty()

        for position, (idx, human, clip_path, side) in enumerate(eligible, start=1):
            label = _play_label(human, idx)
            status.write(
                f"Personnel {position}/{len(eligible)} — {label} "
                f"({side}, 3 specialist calls)"
            )

            try:
                try:
                    extra_context = _v44_roster_context(human, cfbd_key)
                except Exception:
                    # Roster data is diagnostic only in V4.6; personnel must still be testable
                    # from whole-sequence visual roles when roster retrieval is unavailable.
                    extra_context = {"offense_skill_roster": []}

                prediction = _run_analyzer(
                    analyzer_fn,
                    clip_path,
                    human,
                    side,
                    use_play_text,
                    api_key,
                    model,
                    video_fps,
                    extra_context,
                )

                comp = compare_prediction(human.to_dict(), prediction)
                personnel_comp = comp[comp["Field"].eq("Personnel")].copy()
                if personnel_comp.empty or not bool(personnel_comp.iloc[0]["Scored"]):
                    human_personnel = _clean(human.get("personnel"), None)
                    raise ValueError(
                        f"No validated human personnel label is available for play {human.get('play_id')} "
                        f"(value={human_personnel!r})."
                    )

                personnel_row = personnel_comp.iloc[0]
                is_match = bool(personnel_row["Match"])
                accounting = prediction.get("_personnel_accounting") or {}
                runs = list(accounting.get("personnel_runs") or [])
                votes = list(accounting.get("votes") or [])
                top_votes = int(votes[0].get("count", 0)) if votes else 0
                consensus = prediction.get("personnel")

                summaries.append(
                    {
                        "Play": position,
                        "Play ID": _id_text(human.get("play_id")),
                        "Quarter": _clean(human.get("period"), ""),
                        "Clock": _clean(human.get("clock"), ""),
                        "Side": side.title(),
                        "Human personnel": personnel_row["Human"],
                        "Run 1": runs[0] if len(runs) > 0 else None,
                        "Run 2": runs[1] if len(runs) > 1 else None,
                        "Run 3": runs[2] if len(runs) > 2 else None,
                        "Consensus": consensus,
                        "Consensus votes": f"{top_votes}/3" if top_votes else "",
                        "Valid accounting runs": accounting.get("valid_runs"),
                        "Correct": is_match,
                        "AI confidence": prediction.get("overall_confidence"),
                        "Status": "Success",
                    }
                )

                for run_number, audit in enumerate(accounting.get("runs") or [], start=1):
                    for player in audit.get("eligible_players") or []:
                        detail_rows.append(
                            {
                                "Play": position,
                                "Play ID": _id_text(human.get("play_id")),
                                "Side": side.title(),
                                "Human personnel": personnel_row["Human"],
                                "Consensus": consensus,
                                "Correct": is_match,
                                "Personnel run": run_number,
                                "Run personnel": runs[run_number - 1] if run_number - 1 < len(runs) else None,
                                "OL count": audit.get("offensive_lineman_count"),
                                "QB count": audit.get("quarterback_count"),
                                "Run RB/FB count": audit.get("rb_fb_count"),
                                "Run TE/H count": audit.get("te_hback_count"),
                                "Run WR count": audit.get("wr_count"),
                                "Run valid": audit.get("valid_accounting"),
                                "Player label": player.get("player_label"),
                                "Initial slot": player.get("initial_slot"),
                                "Visual role": player.get("visual_role"),
                                "Jersey": player.get("jersey"),
                                "Jersey readability": player.get("jersey_readability"),
                                "Roster role": player.get("roster_role"),
                                "Jersey/visual agree": player.get("jersey_visual_agree"),
                                "Initial alignment": player.get("initial_alignment"),
                                "Final alignment": player.get("final_alignment"),
                                "Identity evidence": player.get("identity_evidence"),
                            }
                        )

                save_ai_prediction(
                    game_id=_id_text(human.get("game_id")),
                    play_id=_id_text(human.get("play_id")),
                    model=model.strip(),
                    analyzer_version="v4.6-personnel-only",
                    chart_side=side,
                    prediction=prediction,
                    score_matches=1 if is_match else 0,
                    score_total=1,
                    video_fps=float(video_fps),
                    use_play_text=bool(use_play_text),
                )

            except Exception as exc:
                message = str(exc)
                errors.append({"Play": label, "Error": message})
                summaries.append(
                    {
                        "Play": position,
                        "Play ID": _id_text(human.get("play_id")),
                        "Quarter": _clean(human.get("period"), ""),
                        "Clock": _clean(human.get("clock"), ""),
                        "Side": side.title(),
                        "Human personnel": _clean(human.get("personnel"), None),
                        "Run 1": None,
                        "Run 2": None,
                        "Run 3": None,
                        "Consensus": None,
                        "Consensus votes": "",
                        "Valid accounting runs": None,
                        "Correct": None,
                        "AI confidence": None,
                        "Status": f"Error: {message[:140]}",
                    }
                )

            progress.progress(
                position / len(eligible),
                text=f"Completed {position}/{len(eligible)} personnel plays",
            )
            if position < len(eligible):
                time.sleep(7)

        st.session_state[batch_key] = {
            "summary": pd.DataFrame(summaries),
            "detail": pd.DataFrame(detail_rows),
            "errors": pd.DataFrame(errors),
        }
        status.success("Personnel-only validation finished.")

    result = st.session_state.get(batch_key)
    if not result:
        st.info("Run the personnel validation to test the method across the full benchmark.")
        return

    summary = result["summary"]
    detail = result["detail"]
    errors = result["errors"]

    st.divider()
    st.subheader("V4.6 personnel validation results")

    successful = summary[summary["Status"].eq("Success")].copy()
    correct = int(successful["Correct"].eq(True).sum()) if not successful.empty else 0
    scored = int(successful["Correct"].notna().sum()) if not successful.empty else 0
    accuracy = correct / scored if scored else None
    consensus_reached = int(successful["Consensus"].notna().sum()) if not successful.empty else 0
    unanimous = int(successful["Consensus votes"].eq("3/3").sum()) if not successful.empty else 0

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Successful plays", f"{len(successful)}/{len(summary)}")
    v2.metric("Personnel accuracy", f"{correct}/{scored} ({accuracy:.0%})" if accuracy is not None else "—")
    v3.metric("Consensus reached", f"{consensus_reached}/{len(successful)}" if len(successful) else "—")
    v4.metric("Unanimous consensus", f"{unanimous}/{len(successful)}" if len(successful) else "—")

    display = summary.copy()
    display["Correct"] = display["Correct"].map(
        lambda value: "✅" if value is True else "❌" if value is False else ""
    )
    display["AI confidence"] = display["AI confidence"].map(
        lambda value: f"{value:.0%}" if pd.notna(value) else ""
    )
    st.dataframe(display, hide_index=True, use_container_width=True)

    if not errors.empty:
        with st.expander(f"Personnel validation errors ({len(errors)})"):
            st.dataframe(errors, hide_index=True, use_container_width=True)

    st.download_button(
        "Download personnel validation summary CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="v4.6_personnel_batch_validation.csv",
        mime="text/csv",
        key="download_v46_personnel_summary",
    )

    if not detail.empty:
        st.download_button(
            "Download personnel player-role detail CSV",
            data=detail.to_csv(index=False).encode("utf-8"),
            file_name="v4.6_personnel_batch_player_detail.csv",
            mime="text/csv",
            key="download_v46_personnel_detail",
        )


def _render_batch(
    reviewed: pd.DataFrame,
    side_lookup: dict[str, str],
    cfbd_key: str | None,
    api_key: str | None,
    model: str,
    video_fps: float,
    use_play_text: bool,
    analyzer_label: str,
    analyzer_version: str,
    analyzer_fn,
):
    st.write(
        "Run every reviewed benchmark clip sequentially. Each successful prediction "
        "is saved immediately; individual failures do not erase completed plays."
    )

    eligible, missing_clips, missing_sides = [], [], []
    for idx, row in reviewed.iterrows():
        clip = _clip_path(row)
        side = _row_side(row, side_lookup)
        if not clip.exists():
            missing_clips.append(_play_label(row, idx))
            continue
        if side not in {"offense", "defense"}:
            missing_sides.append(_play_label(row, idx))
            continue
        eligible.append((idx, row, clip, side))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reviewed plays", len(reviewed))
    m2.metric("Ready for batch", len(eligible))
    m3.metric("Analyzer", analyzer_version)
    m4.metric("Model", model)

    if analyzer_version == "v4.6-personnel-consensus":
        st.info(
            "V4.6 is a SINGLE-SNAP diagnostic. It keeps five unchanged V1 full-chart calls and uses "
            "three separate personnel passes. Final personnel is a deterministic 2-of-3 vote on visual "
            "RB/TE/WR counts; jersey reads are kept only for diagnostics."
        )
    elif analyzer_version == "v4.5-hybrid-personnel":
        st.info(
            "V4.5 is a SINGLE-SNAP diagnostic. It preserves the five-run full chart and uses three "
            "separate hybrid personnel passes. Jersey numbers are trusted only when genuinely clear; "
            "otherwise the model falls back to whole-sequence role evidence, and conflicts trigger review."
        )
    elif analyzer_version == "v4.4-roster-assisted-personnel":
        st.info(
            "V4.4 is a SINGLE-SNAP diagnostic. It uses five unchanged V1 full-chart calls plus three "
            "separate jersey-ID calls, then maps visible jerseys to the cached CFBD offensive roster. "
            "Batch mode is disabled while we validate the roster-assisted personnel method."
        )
    elif analyzer_version == "v4.3-repeatability-separated-personnel":
        st.info(
            "V4.3 is a SINGLE-SNAP diagnostic. It uses 5 normal V1 full-chart calls plus 3 separate "
            "personnel-only calls against one uploaded video. The personnel prompt cannot affect any "
            "formation, movement, play, defense, or pressure field."
        )
    elif analyzer_version == "v4.2-mechanical-personnel-repeatability":
        st.info(
            "V4.2 is a SINGLE-SNAP diagnostic. It makes five native-video calls, but personnel is "
            "never accepted directly from Gemini. Each run must account for 5 OL + 1 QB + 5 eligible "
            "players, and Python derives personnel only from complete valid accounting."
        )
    elif analyzer_version == "v4.1-repeatability":
        st.info(
            "V4.1 is a SINGLE-SNAP diagnostic. It makes five independent V1 chart calls against "
            "one uploaded clip and scores a 3-of-5 consensus. Do not use Batch benchmark for this "
            "version; that would multiply the experiment across every reviewed snap."
        )
    elif analyzer_version == "v4-native-corrective":
        st.info(
            "V4 uploads the clip once and makes 2 native-video Gemini calls: the original broad "
            "V1 chart plus a narrow corrective pass for only the historically weak offensive "
            "structure fields. Strong V1 fields such as front, shell, play type, RPO, play action, "
            "rushers/blitz, and coverage are protected from the second pass."
        )
    elif analyzer_version == "v3.4-snap-centered":
        st.info(
            "V3.4 uses 3 Gemini requests per snap: a tiny coarse snap locator, then focused "
            "pre-snap and post-snap passes over newly extracted snap-centered frames. The expensive "
            "football analysis is now relative to the snap instead of percentages of clip length."
        )
    elif analyzer_version == "v3.3-two-pass-temporal":
        st.info(
            "V3.3 makes 2 Gemini requests per snap over one locally extracted 12-frame timeline: "
            "pre-snap structure/movement, then post-snap play/pressure/coverage. This is still fewer "
            "requests than V2.1 and is the first version to enumerate each actual rusher."
        )
    elif analyzer_version == "v3.2-temporal-diagnostic":
        st.info(
            "V3.2 is intentionally narrow: one Gemini request over 12 individual high-resolution "
            "frames, returning only snap timing, formation family, RB start/end side, box counts, "
            "and actual rusher count. Use this before any full 9-play batch."
        )
    elif analyzer_version == "v3.1-temporal-frames":
        st.info(
            "V3.1 performs FFmpeg frame extraction locally, sends 12 individual high-resolution "
            "frames in chronological order, and makes only 1 Gemini image-analysis request per snap. "
            "The FPS slider is ignored for V3.1."
        )
    elif analyzer_version == "v3-temporal-evidence":
        st.info(
            "V3 performs FFmpeg frame extraction locally, builds 3 timestamped evidence sheets, "
            "then makes only 1 Gemini image-analysis request per snap. The FPS slider is ignored "
            "for V3 because it uses a fixed 24-frame timeline."
        )
    elif analyzer_version in {"v2-specialized", "v2.1-mechanical"}:
        version_name = "V2.1" if analyzer_version == "v2.1-mechanical" else "V2"
        st.info(
            f"{version_name} uploads each clip once, then makes 3 Gemini analysis calls against "
            "that upload: structure, movement, and post-snap. This uses more free-tier requests "
            "than V1, but stays on the same Flash-Lite model."
        )

    if missing_clips:
        st.warning(f"{len(missing_clips)} reviewed play(s) do not have an extracted clip.")
    if missing_sides:
        st.warning(f"{len(missing_sides)} play(s) could not be identified as offense or defense.")
    if not api_key:
        st.warning("GEMINI_API_KEY is not configured.")
        return
    if not eligible:
        st.info("No benchmark plays are ready for batch analysis.")
        return

    if analyzer_version == "v4.6-personnel-only":
        _render_personnel_batch_validation(
            eligible,
            cfbd_key,
            api_key,
            model,
            video_fps,
            use_play_text,
            analyzer_fn,
        )
        return

    _render_stored_field_analysis(
        reviewed,
        analyzer_version,
        model.strip(),
        float(video_fps),
        bool(use_play_text),
    )

    if analyzer_version in {"v4.6-personnel-consensus", "v4.5-hybrid-personnel", "v4.4-roster-assisted-personnel", "v4.3-repeatability-separated-personnel", "v4.2-mechanical-personnel-repeatability", "v4.1-repeatability"}:
        st.warning(
            "Repeatability diagnostics are intentionally disabled in Batch benchmark. Run this "
            "version from the Single snap tab on Play 1 first; five observations of one snap are "
            "enough to evaluate the personnel method and repeatability."
        )
        return

    run_batch = st.button(
        f"Run all {len(eligible)} plays with {analyzer_label}",
        type="primary",
        key=f"run_batch_{analyzer_version}",
    )

    batch_key = f"batch_{model}_{video_fps}_{use_play_text}_{analyzer_version}"
    if run_batch:
        summaries, comparisons, errors = [], [], []
        progress = st.progress(0.0, text="Starting batch benchmark...")
        status = st.empty()

        for position, (idx, human, clip_path, side) in enumerate(eligible, start=1):
            label = _play_label(human, idx)
            status.write(
                f"Analyzing {position}/{len(eligible)} — {label} "
                f"({side}, {analyzer_version}, {video_fps:.1f} FPS)"
            )
            try:
                prediction = _run_analyzer(
                    analyzer_fn, clip_path, human, side, use_play_text,
                    api_key, model, video_fps,
                )
                comparison = _comparison_for_prediction(human.to_dict(), prediction)
                matches, total, accuracy = score_prediction(comparison)
                comparison["play_id"] = _id_text(human.get("play_id"))
                comparison["clock"] = _clean(human.get("clock"), "")
                comparison["chart_side"] = side
                comparisons.append(comparison)

                save_ai_prediction(
                    game_id=_id_text(human.get("game_id")),
                    play_id=_id_text(human.get("play_id")),
                    model=model.strip(),
                    analyzer_version=analyzer_version,
                    chart_side=side,
                    prediction=prediction,
                    score_matches=matches,
                    score_total=total,
                    video_fps=float(video_fps),
                    use_play_text=bool(use_play_text),
                )
                summaries.append({
                    "Play": position,
                    "Play ID": _id_text(human.get("play_id")),
                    "Quarter": _clean(human.get("period"), ""),
                    "Clock": _clean(human.get("clock"), ""),
                    "Side": side.title(),
                    "Matches": matches,
                    "Scored fields": total,
                    "Accuracy": accuracy,
                    "AI confidence": prediction.get("overall_confidence"),
                    "Status": "Success",
                })
            except Exception as exc:
                message = str(exc)
                errors.append({"Play": label, "Error": message})
                summaries.append({
                    "Play": position,
                    "Play ID": _id_text(human.get("play_id")),
                    "Quarter": _clean(human.get("period"), ""),
                    "Clock": _clean(human.get("clock"), ""),
                    "Side": side.title(),
                    "Matches": None,
                    "Scored fields": None,
                    "Accuracy": None,
                    "AI confidence": None,
                    "Status": f"Error: {message[:120]}",
                })

            progress.progress(position / len(eligible), text=f"Completed {position}/{len(eligible)}")
            if position < len(eligible):
                if analyzer_version in {"v2-specialized", "v2.1-mechanical"}:
                    time.sleep(8)
                elif analyzer_version == "v4-native-corrective":
                    time.sleep(7)
                elif analyzer_version == "v3.4-snap-centered":
                    time.sleep(8)
                elif analyzer_version == "v3.3-two-pass-temporal":
                    time.sleep(7)
                elif analyzer_version in {"v3.2-temporal-diagnostic", "v3.1-temporal-frames", "v3-temporal-evidence"}:
                    time.sleep(6)
                else:
                    time.sleep(6)

        detail = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
        st.session_state[batch_key] = {
            "summary": pd.DataFrame(summaries),
            "detail": detail,
            "errors": pd.DataFrame(errors),
        }
        status.success("Batch benchmark finished.")

    result = st.session_state.get(batch_key)
    if not result:
        st.info("Run the batch to produce the new benchmark.")
        return

    summary, detail, errors = result["summary"], result["detail"], result["errors"]
    st.divider()
    st.subheader("Batch benchmark results")

    successful = summary[summary["Status"].eq("Success")].copy()
    total_matches = int(successful["Matches"].fillna(0).sum()) if not successful.empty else 0
    total_scored = int(successful["Scored fields"].fillna(0).sum()) if not successful.empty else 0
    overall_accuracy = total_matches / total_scored if total_scored else None

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Successful plays", f"{len(successful)}/{len(summary)}")
    b2.metric("Exact matches", f"{total_matches}/{total_scored}" if total_scored else "—")
    b3.metric("Overall accuracy", f"{overall_accuracy:.0%}" if overall_accuracy is not None else "—")
    avg_conf = successful["AI confidence"].dropna().mean() if not successful.empty else None
    b4.metric("Avg. AI confidence", f"{avg_conf:.0%}" if pd.notna(avg_conf) else "—")

    baseline = _stored_v1_accuracy(
        reviewed,
        model.strip(),
        float(video_fps),
        bool(use_play_text),
        core_only=(analyzer_version == "v5-core"),
    )
    if analyzer_version in {"v5-core", "v4.3-repeatability-separated-personnel", "v4.2-mechanical-personnel-repeatability", "v4.1-repeatability", "v4-native-corrective", "v3.4-snap-centered", "v3.3-two-pass-temporal", "v3.2-temporal-diagnostic", "v3.1-temporal-frames", "v3-temporal-evidence", "v2-specialized", "v2.1-mechanical"} and baseline is not None and overall_accuracy is not None:
        st.subheader("V1 vs current analyzer")
        if analyzer_version == "v5-core":
            st.caption("Both scores below use only the same six V5 core fields.")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Stored V1 baseline",
            f"{baseline['accuracy']:.0%}",
            help=f"{baseline['matches']}/{baseline['total']} across {baseline['plays']} stored plays",
        )
        c2.metric(analyzer_version, f"{overall_accuracy:.0%}")
        c3.metric("Change", f"{overall_accuracy - baseline['accuracy']:+.1%}")

    table = summary.copy()
    table["Accuracy"] = table["Accuracy"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "")
    table["AI confidence"] = table["AI confidence"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "")
    st.dataframe(table, hide_index=True, use_container_width=True)

    categories = category_summary(detail)
    if not categories.empty:
        st.subheader("Accuracy by scouting category")
        category_display = categories.copy()
        category_display["Accuracy"] = category_display["Accuracy"].map(lambda v: f"{v:.0%}")
        st.dataframe(category_display, hide_index=True, use_container_width=True)

    fields = field_summary(detail)
    if not fields.empty:
        st.subheader("Accuracy by benchmark field")
        field_display = fields.copy()
        field_display["Accuracy"] = field_display["Accuracy"].map(
            lambda value: f"{value:.0%}" if pd.notna(value) else ""
        )
        st.dataframe(field_display, hide_index=True, use_container_width=True)

    if not errors.empty:
        with st.expander(f"Batch errors ({len(errors)})"):
            st.dataframe(errors, hide_index=True, use_container_width=True)

    if not detail.empty:
        st.download_button(
            "Download field-level benchmark detail CSV",
            data=detail.to_csv(index=False).encode("utf-8"),
            file_name=f"ai_benchmark_detail_{analyzer_version}.csv",
            mime="text/csv",
            key=f"download_detail_{analyzer_version}",
        )

    if not fields.empty:
        st.download_button(
            "Download field accuracy CSV",
            data=fields.to_csv(index=False).encode("utf-8"),
            file_name=f"ai_benchmark_fields_{analyzer_version}.csv",
            mime="text/csv",
            key=f"download_fields_{analyzer_version}",
        )

    st.download_button(
        "Download batch summary CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name=f"ai_benchmark_{analyzer_version}.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="AI Analyzer", layout="wide")
    st.title("AI Analyzer")
    st.caption(
        "Benchmark Gemini video analysis against validated Film Lab data. "
        "V5 focuses on the six fields our testing has shown the model can recognize most reliably: "
        "formation family, run/pass, RPO, play action, defensive front family, and safety shell."
    )

    chart = load_film_chart()
    if chart.empty:
        st.warning("No reviewed film chart exists yet. Chart benchmark plays in Film Lab first.")
        return

    reviewed = chart[chart["reviewed"].map(_truthy)].copy() if "reviewed" in chart.columns else chart.copy()
    if reviewed.empty:
        st.warning("No plays are marked Reviewed / validated.")
        return
    reviewed = reviewed.reset_index(drop=True)

    api_key = _secret("GEMINI_API_KEY")
    cfbd_key = _secret("CFBD_API_KEY")
    model_default = _secret("GEMINI_MODEL", "gemini-3.1-flash-lite")

    st.markdown("### Analysis settings")
    analyzer_label = st.selectbox(
        "Analyzer version",
        options=list(ANALYZERS.keys()),
        index=0,
        help="Use V5 as the production-candidate benchmark: one native-video call per snap and only six core scouting fields.",
    )
    analyzer_version, analyzer_fn = ANALYZERS[analyzer_label]

    model = st.text_input(
        "Gemini model",
        value=str(model_default),
        help="Keep gemini-3.1-flash-lite for the cheapest path.",
    )
    video_fps = st.slider(
        "Video sampling FPS",
        min_value=1.0,
        max_value=5.0,
        value=5.0,
        step=0.5,
        help="Used by V1/V2/V4 native-video analysis. V3-family still-frame analyzers ignore this where noted.",
    )
    use_play_text = st.checkbox(
        "Give the model CFBD play-by-play context",
        value=True,
        help="PBP is outcome context only; prompts explicitly forbid it from deciding film labels.",
    )

    if not api_key:
        st.warning("GEMINI_API_KEY is not configured in .streamlit/secrets.toml.")

    side_lookup = _build_side_lookup(reviewed, cfbd_key)
    single_tab, batch_tab = st.tabs(["Single snap", "Batch benchmark"])

    with single_tab:
        _render_single(
            reviewed, side_lookup, cfbd_key, api_key, model, video_fps, use_play_text,
            analyzer_label, analyzer_version, analyzer_fn,
        )

    with batch_tab:
        _render_batch(
            reviewed, side_lookup, cfbd_key, api_key, model, video_fps, use_play_text,
            analyzer_label, analyzer_version, analyzer_fn,
        )


if __name__ == "__main__":
    main()

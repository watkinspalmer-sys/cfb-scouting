from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from analytics.benchmark import category_summary, compare_prediction, score_prediction
from data.ai_store import save_ai_prediction
from data.cfbd import fetch_week_plays
from data.film_store import load_film_chart
from video.analyze import analyze_clip_gemini


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
    """Infer chart-team side for older rows that predate the chart_side column."""
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


def _render_comparison(human: pd.Series, prediction: dict):
    comparison = compare_prediction(human.to_dict(), prediction)
    matches, total, accuracy = score_prediction(comparison)

    m1, m2, m3 = st.columns(3)
    m1.metric("Exact benchmark matches", f"{matches}/{total}" if total else "—")
    m2.metric("Exact-match accuracy", f"{accuracy:.0%}" if accuracy is not None else "—")
    confidence = prediction.get("overall_confidence")
    m3.metric(
        "AI overall confidence",
        f"{float(confidence):.0%}" if confidence is not None else "—",
    )

    display = comparison.copy()
    display["Result"] = display.apply(
        lambda row: (
            "✅"
            if row["Scored"] and row["Match"] is True
            else "❌"
            if row["Scored"] and row["Match"] is False
            else "Review"
        ),
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


def _render_single(
    reviewed: pd.DataFrame,
    side_lookup: dict[str, str],
    api_key: str | None,
    model: str,
    video_fps: float,
    use_play_text: bool,
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
        st.error(
            f"Extracted clip not found at {clip_path}. Reopen this play in Film Lab "
            "and click Extract clip first."
        )

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
        "Analyze snap with AI",
        type="primary",
        disabled=(not bool(api_key) or not clip_path.exists()),
        key="single_analyze",
    )

    result_key = (
        f"ai_result_{_id_text(human.get('game_id'))}_"
        f"{_id_text(human.get('play_id'))}_{model}_{video_fps}_{use_play_text}"
    )
    if analyze_clicked:
        try:
            with st.spinner(
                f"Uploading the snap and asking {model} to analyze at {video_fps:.1f} FPS..."
            ):
                prediction = analyze_clip_gemini(
                    clip_path=clip_path,
                    play_context=_play_context(human, role.lower(), use_play_text),
                    api_key=api_key,
                    model=model.strip(),
                    fps=float(video_fps),
                )
            st.session_state[result_key] = prediction
        except Exception as exc:
            st.error(f"AI analysis failed: {exc}")

    prediction = st.session_state.get(result_key)
    if not prediction:
        st.info(
            "Run the analyzer on one validated play to see a field-by-field comparison."
        )
        return

    st.divider()
    st.subheader("AI vs. validated chart")
    comparison, matches, total, _ = _render_comparison(human, prediction)

    saved_path = save_ai_prediction(
        game_id=_id_text(human.get("game_id")),
        play_id=_id_text(human.get("play_id")),
        model=model.strip(),
        chart_side=role.lower(),
        prediction=prediction,
        score_matches=matches,
        score_total=total,
        video_fps=float(video_fps),
        use_play_text=bool(use_play_text),
    )
    st.caption(f"Prediction saved locally to {saved_path}.")

    with st.expander("Raw AI chart"):
        st.json(prediction)


def _render_batch(
    reviewed: pd.DataFrame,
    side_lookup: dict[str, str],
    api_key: str | None,
    model: str,
    video_fps: float,
    use_play_text: bool,
):
    st.write(
        "Run every reviewed benchmark clip sequentially. Each successful prediction "
        "is saved immediately, so one failed play will not erase the rest of the run."
    )

    eligible = []
    missing_clips = []
    missing_sides = []

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

    m1, m2, m3 = st.columns(3)
    m1.metric("Reviewed plays", len(reviewed))
    m2.metric("Ready for batch", len(eligible))
    m3.metric("Model", model)

    if missing_clips:
        st.warning(f"{len(missing_clips)} reviewed play(s) do not have an extracted clip.")
    if missing_sides:
        st.warning(
            f"{len(missing_sides)} play(s) could not be identified as offense or defense. "
            "Re-save those plays once in Film Lab if needed."
        )

    if not api_key:
        st.warning("GEMINI_API_KEY is not configured.")
        return
    if not eligible:
        st.info("No benchmark plays are ready for batch analysis.")
        return

    run_batch = st.button(
        f"Run all {len(eligible)} benchmark plays",
        type="primary",
        key="run_batch",
    )

    batch_key = f"batch_{model}_{video_fps}_{use_play_text}"
    if run_batch:
        summaries = []
        comparisons = []
        errors = []

        progress = st.progress(0.0, text="Starting batch benchmark...")
        status = st.empty()

        for position, (idx, human, clip_path, side) in enumerate(eligible, start=1):
            label = _play_label(human, idx)
            status.write(
                f"Analyzing {position}/{len(eligible)} — {label} "
                f"({side}, {video_fps:.1f} FPS)"
            )

            try:
                prediction = analyze_clip_gemini(
                    clip_path=clip_path,
                    play_context=_play_context(human, side, use_play_text),
                    api_key=api_key,
                    model=model.strip(),
                    fps=float(video_fps),
                )
                comparison = compare_prediction(human.to_dict(), prediction)
                matches, total, accuracy = score_prediction(comparison)

                comparison["play_id"] = _id_text(human.get("play_id"))
                comparison["clock"] = _clean(human.get("clock"), "")
                comparison["chart_side"] = side
                comparisons.append(comparison)

                save_ai_prediction(
                    game_id=_id_text(human.get("game_id")),
                    play_id=_id_text(human.get("play_id")),
                    model=model.strip(),
                    chart_side=side,
                    prediction=prediction,
                    score_matches=matches,
                    score_total=total,
                    video_fps=float(video_fps),
                    use_play_text=bool(use_play_text),
                )

                summaries.append(
                    {
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
                    }
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
                        "Matches": None,
                        "Scored fields": None,
                        "Accuracy": None,
                        "AI confidence": None,
                        "Status": f"Error: {message[:120]}",
                    }
                )

            progress.progress(
                position / len(eligible),
                text=f"Completed {position}/{len(eligible)} benchmark plays",
            )

            # Stay conservative with free-tier request limits.
            if position < len(eligible):
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
        st.info(
            "The batch will use the same model, 5 FPS setting, and CFBD-context choice "
            "shown above."
        )
        return

    summary = result["summary"]
    detail = result["detail"]
    errors = result["errors"]

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

    table = summary.copy()
    table["Accuracy"] = table["Accuracy"].map(
        lambda value: f"{value:.0%}" if pd.notna(value) else ""
    )
    table["AI confidence"] = table["AI confidence"].map(
        lambda value: f"{value:.0%}" if pd.notna(value) else ""
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

    categories = category_summary(detail)
    if not categories.empty:
        st.subheader("Accuracy by scouting category")
        category_display = categories.copy()
        category_display["Accuracy"] = category_display["Accuracy"].map(lambda v: f"{v:.0%}")
        st.dataframe(category_display, hide_index=True, use_container_width=True)

    if not errors.empty:
        with st.expander(f"Batch errors ({len(errors)})"):
            st.dataframe(errors, hide_index=True, use_container_width=True)

    csv_bytes = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download batch summary CSV",
        data=csv_bytes,
        file_name="ai_benchmark_summary.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="AI Analyzer", layout="wide")
    st.title("AI Analyzer")
    st.caption(
        "Use Gemini native video analysis to chart football snaps and benchmark "
        "its structured output against your validated Film Lab data."
    )

    chart = load_film_chart()
    if chart.empty:
        st.warning("No reviewed film chart exists yet. Chart benchmark plays in Film Lab first.")
        return

    if "reviewed" in chart.columns:
        reviewed = chart[chart["reviewed"].map(_truthy)].copy()
    else:
        reviewed = chart.copy()

    if reviewed.empty:
        st.warning("No plays are marked Reviewed / validated.")
        return

    reviewed = reviewed.reset_index(drop=True)

    api_key = _secret("GEMINI_API_KEY")
    cfbd_key = _secret("CFBD_API_KEY")
    model_default = _secret("GEMINI_MODEL", "gemini-3.1-flash-lite")

    st.markdown("### Analysis settings")
    model = st.text_input(
        "Gemini model",
        value=str(model_default),
        help="Keep gemini-3.1-flash-lite for the cheapest benchmark path.",
    )
    video_fps = st.slider(
        "Video sampling FPS",
        min_value=1.0,
        max_value=5.0,
        value=5.0,
        step=0.5,
        help="5 FPS is our current benchmark baseline.",
    )
    use_play_text = st.checkbox(
        "Give the model CFBD play-by-play context",
        value=True,
        help=(
            "Production scouting combines structured PBP with film. Turn this off "
            "only for a harder vision-only benchmark."
        ),
    )

    if not api_key:
        st.warning(
            "GEMINI_API_KEY is not configured. Add it to .streamlit/secrets.toml "
            "before running the analyzer."
        )

    side_lookup = _build_side_lookup(reviewed, cfbd_key)

    single_tab, batch_tab = st.tabs(["Single snap", "Batch benchmark"])

    with single_tab:
        _render_single(
            reviewed=reviewed,
            side_lookup=side_lookup,
            api_key=api_key,
            model=model,
            video_fps=video_fps,
            use_play_text=use_play_text,
        )

    with batch_tab:
        _render_batch(
            reviewed=reviewed,
            side_lookup=side_lookup,
            api_key=api_key,
            model=model,
            video_fps=video_fps,
            use_play_text=use_play_text,
        )


if __name__ == "__main__":
    main()

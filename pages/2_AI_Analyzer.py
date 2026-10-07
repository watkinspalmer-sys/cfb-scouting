from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from analytics.benchmark import compare_prediction, score_prediction
from data.ai_store import save_ai_prediction
from data.film_store import load_film_chart
from video.analyze import analyze_clip_openai


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
    game_id = str(_clean(row.get("game_id"), "unknown-game"))
    play_id = str(_clean(row.get("play_id"), "unknown-play"))
    name = f"{game_id}_{play_id}".replace("/", "-").replace(" ", "_") + ".mp4"
    return Path("clips") / name


def main():
    st.set_page_config(page_title="AI Analyzer", layout="wide")
    st.title("AI Analyzer v1")
    st.caption(
        "Have a multimodal model chart one extracted snap, then compare its "
        "structured answer against your validated Film Lab chart."
    )

    chart = load_film_chart()
    if chart.empty:
        st.warning("No reviewed film chart exists yet. Chart benchmark plays in Film Lab first.")
        st.stop()

    if "reviewed" in chart.columns:
        reviewed = chart[chart["reviewed"].map(_truthy)].copy()
    else:
        reviewed = chart.copy()

    if reviewed.empty:
        st.warning("No plays are marked Reviewed / validated.")
        st.stop()

    reviewed = reviewed.reset_index(drop=True)
    labels = [_play_label(row, idx) for idx, (_, row) in enumerate(reviewed.iterrows())]
    selected_idx = st.selectbox(
        "Benchmark play",
        options=list(range(len(reviewed))),
        format_func=lambda i: labels[i],
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

    saved_side = str(_clean(human.get("chart_side"), "")).strip().lower()
    default_role = 0 if saved_side != "defense" else 1
    role = st.radio(
        f"{_clean(human.get('team'), 'Chart team')} is:",
        ["Offense", "Defense"],
        index=default_role,
        horizontal=True,
    )

    api_key = _secret("OPENAI_API_KEY")
    model_default = _secret("OPENAI_MODEL", "gpt-6-luna")
    model = st.text_input(
        "AI model",
        value=str(model_default),
        help="Stored locally in Streamlit secrets if you set OPENAI_MODEL.",
    )
    frame_count = st.slider(
        "Frames sampled from the snap",
        min_value=6,
        max_value=12,
        value=8,
        step=1,
        help="More frames give more temporal information but use more image input.",
    )
    use_play_text = st.checkbox(
        "Give the model CFBD play-by-play context",
        value=True,
        help=(
            "Production scouting will combine structured PBP with film. Turn this "
            "off if you want a harder vision-only benchmark."
        ),
    )

    if not api_key:
        st.warning(
            "OPENAI_API_KEY is not configured. Add it to .streamlit/secrets.toml "
            "before running the analyzer. Do not paste the key into chat."
        )

    analyze_clicked = st.button(
        "Analyze snap with AI",
        type="primary",
        disabled=(not bool(api_key) or not clip_path.exists()),
    )

    result_key = f"ai_result_{human.get('game_id')}_{human.get('play_id')}_{model}"
    if analyze_clicked:
        play_context = {
            "team": _clean(human.get("team"), ""),
            "chart_side": role.lower(),
            "period": _clean(human.get("period"), None),
            "clock": _clean(human.get("clock"), None),
            "down": _clean(human.get("down"), None),
            "distance": _clean(human.get("distance"), None),
            "play_text": _clean(human.get("play_text"), None) if use_play_text else None,
        }
        try:
            with st.spinner(
                f"Sampling {frame_count} frames and asking {model} to chart the snap..."
            ):
                prediction = analyze_clip_openai(
                    clip_path=clip_path,
                    play_context=play_context,
                    api_key=api_key,
                    model=model.strip(),
                    frame_count=int(frame_count),
                )
            st.session_state[result_key] = prediction
        except Exception as exc:
            st.error(f"AI analysis failed: {exc}")

    prediction = st.session_state.get(result_key)
    if not prediction:
        st.info(
            "Run the analyzer on one of the validated plays. The result will be "
            "compared field-by-field against your human chart."
        )
        st.stop()

    st.divider()
    st.subheader("AI vs. validated chart")

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
        display[["Field", "Human", "AI", "Result"]],
        hide_index=True,
        use_container_width=True,
    )

    uncertain = prediction.get("uncertain_fields") or []
    if uncertain:
        st.write("**AI says a human should review:**", ", ".join(map(str, uncertain)))

    notes = prediction.get("analysis_notes")
    if notes:
        st.write("**AI notes:**", notes)

    saved_path = save_ai_prediction(
        game_id=str(human.get("game_id")),
        play_id=str(human.get("play_id")),
        model=model.strip(),
        chart_side=role.lower(),
        prediction=prediction,
        score_matches=matches,
        score_total=total,
    )
    st.caption(f"Prediction saved locally to {saved_path}.")

    with st.expander("Raw AI chart"):
        st.json(prediction)


if __name__ == "__main__":
    main()

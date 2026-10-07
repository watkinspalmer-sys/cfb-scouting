from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from analytics.benchmark import category_summary, compare_prediction, score_prediction
from data.ai_store import load_ai_predictions, save_ai_prediction
from data.cfbd import fetch_week_plays
from data.film_store import load_film_chart
from video.analyze import analyze_clip_gemini, analyze_clip_gemini_v2, analyze_clip_gemini_v21, analyze_clip_gemini_v3, analyze_clip_gemini_v31


ANALYZERS = {
    "V3.1 temporal frames 1-pass (recommended)": ("v3.1-temporal-frames", analyze_clip_gemini_v31),
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


def _render_comparison(human: pd.Series, prediction: dict):
    comparison = compare_prediction(human.to_dict(), prediction)
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
):
    return analyzer_fn(
        clip_path=clip_path,
        play_context=_play_context(human, side, use_play_text),
        api_key=api_key,
        model=model.strip(),
        fps=float(video_fps),
    )


def _render_single(
    reviewed: pd.DataFrame,
    side_lookup: dict[str, str],
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
            if analyzer_version == "v3.1-temporal-frames":
                passes = "one high-resolution temporal-frame pass"
            elif analyzer_version == "v3-temporal-evidence":
                passes = "one temporal-evidence pass"
            elif analyzer_version in {"v2-specialized", "v2.1-mechanical"}:
                passes = "three specialized passes"
            else:
                passes = "one native-video pass"
            with st.spinner(f"Analyzing with {passes} using {model} at {video_fps:.1f} FPS..."):
                prediction = _run_analyzer(
                    analyzer_fn, clip_path, human, role.lower(), use_play_text,
                    api_key, model, video_fps,
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

    if analyzer_version in {"v3.1-temporal-frames", "v3-temporal-evidence", "v2-specialized", "v2.1-mechanical"}:
        derived = prediction.get("_derived") or {}
        pass_conf = prediction.get("_pass_confidence") or {}
        d1, d2, d3, d4 = st.columns(4)
        observed_backs = derived.get("back_count", derived.get("rb_count", "—"))
        d1.metric("Observed backs", observed_backs)
        d2.metric("Observed TEs", derived.get("te_count", "—"))
        d3.metric("Derived personnel", prediction.get("personnel", "—"))
        d4.metric("Derived blitz", str(prediction.get("blitz", "—")))
        if analyzer_version == "v3.1-temporal-frames":
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


def _stored_v1_accuracy(reviewed: pd.DataFrame, model: str, video_fps: float, use_play_text: bool):
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
        m, t, _ = score_prediction(comp)
        matches += m
        total += t
        plays += 1

    if not total:
        return None
    return {"plays": plays, "matches": matches, "total": total, "accuracy": matches / total}


def _render_batch(
    reviewed: pd.DataFrame,
    side_lookup: dict[str, str],
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

    if analyzer_version == "v3.1-temporal-frames":
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
                elif analyzer_version in {"v3.1-temporal-frames", "v3-temporal-evidence"}:
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

    baseline = _stored_v1_accuracy(reviewed, model.strip(), float(video_fps), bool(use_play_text))
    if analyzer_version in {"v3.1-temporal-frames", "v3-temporal-evidence", "v2-specialized", "v2.1-mechanical"} and baseline is not None and overall_accuracy is not None:
        st.subheader("V1 vs current analyzer")
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

    if not errors.empty:
        with st.expander(f"Batch errors ({len(errors)})"):
            st.dataframe(errors, hide_index=True, use_container_width=True)

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
        "V3.1 turns each clip into 12 high-resolution chronological frames locally, sends one "
        "Gemini image-analysis request, and derives football labels with deterministic rules."
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
        help="Use V3.1 for the high-resolution temporal-frame benchmark. Older analyzers remain available as baselines.",
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
        help="Used by V1/V2 native-video analysis. V3/V3.1 ignore this and use local temporal frames.",
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
            reviewed, side_lookup, api_key, model, video_fps, use_play_text,
            analyzer_label, analyzer_version, analyzer_fn,
        )

    with batch_tab:
        _render_batch(
            reviewed, side_lookup, api_key, model, video_fps, use_play_text,
            analyzer_label, analyzer_version, analyzer_fn,
        )


if __name__ == "__main__":
    main()

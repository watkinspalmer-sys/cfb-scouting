import streamlit as st

from analytics.tendencies import build_tables
from data.cfbd import REQUIRED_COLUMNS, fetch_plays


def get_key():
    try:
        return st.secrets["CFBD_API_KEY"]
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def cached_fetch_plays(team, year, last_week, _key):
    """Cached wrapper around the CFBD data-access layer."""
    return fetch_plays(team, year, last_week, _key)


def main():
    st.set_page_config(page_title="CFB Scouting Tool", layout="wide")
    st.title("College Football Scouting Tool")
    st.caption("Pick a team and a season to see its offensive and defensive tendencies.")

    key = get_key()
    if not key:
        st.error("No API key found. Add CFBD_API_KEY to your app's secrets (see setup steps).")
        st.stop()

    col1, col2, col3 = st.columns(3)
    team = col1.text_input(
        "Team (as spelled on CollegeFootballData.com)",
        value="Tulsa",
    )
    year = col2.number_input(
        "Season",
        min_value=2004,
        max_value=2030,
        value=2026,
        step=1,
    )
    last_week = col3.number_input(
        "Through week",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
        help="Each week uses one of your monthly requests, so don't go higher than the current week.",
    )
    hide_small = st.checkbox("Hide rows with fewer than 15 plays", value=False)

    if not st.button("Generate", type="primary"):
        st.info("Choose a team, season and week, then click Generate.")
        return

    team = team.strip()

    with st.spinner(f"Downloading {team} {int(year)} plays..."):
        try:
            plays = cached_fetch_plays(team, int(year), int(last_week), key)
        except Exception as exc:
            st.error(f"Could not download data: {exc}")
            return

    if plays.empty:
        st.warning(
            "No plays came back. Check the team spelling "
            "(for example 'Texas A&M', 'Ohio State') and the season."
        )
        return

    missing = [column for column in REQUIRED_COLUMNS if column not in plays.columns]
    if missing:
        st.error(f"The data is missing columns this app expects: {missing}")
        st.write("Columns that came back:", list(plays.columns))
        return

    weeks_found = sorted(plays["week"].unique()) if "week" in plays.columns else []
    st.success(
        f"Got data for weeks: {', '.join(str(int(week)) for week in weeks_found)}"
        if weeks_found
        else "Got data."
    )

    offense, defense, tables = build_tables(plays, team)

    if offense.empty and defense.empty:
        st.warning("Plays came back, but none matched this team name. Check the spelling.")
        return

    m1, m2, m3, m4 = st.columns(4)

    if len(offense):
        m1.metric("Offense success %", f"{offense['success'].mean() * 100:.0f}")
        m2.metric("Offense yards/play", f"{offense['yardsGained'].mean():.1f}")

    if len(defense):
        m3.metric(
            "Defense success % allowed",
            f"{defense['success'].mean() * 100:.0f}",
        )
        m4.metric(
            "Defense yards/play allowed",
            f"{defense['yardsGained'].mean():.1f}",
        )

    report_text = (
        f"{team} {int(year)} season tendencies "
        f"(through week {int(last_week)})\n"
    )

    for title, table in tables.items():
        shown = table[table["plays"] >= 15] if hide_small else table
        st.subheader(title)
        st.dataframe(shown, hide_index=True)
        report_text += f"\n=== {title} ===\n{shown.to_string(index=False)}\n"

    with st.expander(
        "Copy-ready text (paste into Claude to get a written scouting report)"
    ):
        st.code(report_text, language=None)


if __name__ == "__main__":
    main()

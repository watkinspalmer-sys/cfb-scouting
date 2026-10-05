import requests
import pandas as pd
import streamlit as st

BASE = "https://api.collegefootballdata.com"

REQUIRED_COLUMNS = ["offense", "defense", "playType", "down", "distance",
                    "yardsToGoal", "yardsGained", "offenseScore", "defenseScore"]


# ---------------------------------------------------------------
# Getting the data
# ---------------------------------------------------------------
def get_key():
    try:
        return st.secrets["CFBD_API_KEY"]
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_plays(team, year, last_week, _key):
    """Downloads one team's plays, week by week. Saved for 24 hours so
    repeat lookups don't use up your monthly request allowance."""
    frames = []
    for week in range(1, last_week + 1):
        r = requests.get(
            f"{BASE}/plays",
            headers={"Authorization": f"Bearer {_key}"},
            params={"year": year, "week": week, "team": team},
            timeout=60,
        )
        if r.status_code == 401:
            raise ValueError("The API key was rejected (401). Check the key in your app secrets.")
        if r.status_code == 429:
            raise ValueError("You have used up your monthly request allowance (429).")
        r.raise_for_status()
        data = r.json()
        if data:
            frames.append(pd.DataFrame(data))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------
# Computing the tendencies
# ---------------------------------------------------------------
def prep(data):
    """Keeps run/pass plays and adds the columns we need."""
    d = data.copy()
    for col in ["down", "distance", "yardsToGoal", "yardsGained", "offenseScore", "defenseScore"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d["is_pass"] = d["playType"].astype(str).str.contains("Pass|Sack", case=False, na=False)
    d["is_run"] = d["playType"].astype(str).str.contains("Rush", case=False, na=False)
    d = d[(d["is_pass"] | d["is_run"]) & (d["down"] >= 1) & (d["down"] <= 4)].copy()
    d["down"] = d["down"].astype(int)

    needed = d["down"].map({1: 0.5, 2: 0.7, 3: 1.0, 4: 1.0}) * d["distance"]
    d["success"] = d["yardsGained"] >= needed
    d["explosive"] = (d["is_run"] & (d["yardsGained"] >= 10)) | \
                     (d["is_pass"] & (d["yardsGained"] >= 20))

    d["distance_group"] = pd.cut(
        d["distance"], bins=[0, 3, 7, 100],
        labels=["Short (1-3)", "Medium (4-7)", "Long (8+)"])
    d["field_zone"] = pd.cut(
        d["yardsToGoal"], bins=[0, 20, 50, 80, 100],
        labels=["Red zone", "Opp. territory", "Own side of field", "Backed up"])
    margin = d["offenseScore"] - d["defenseScore"]
    d["game_situation"] = pd.cut(
        margin, bins=[-100, -9, 8, 100],
        labels=["Trailing by 9+", "Close game", "Leading by 9+"])
    return d


def make_table(data, by):
    """Builds one tendency table, grouped however we ask."""
    g = data.groupby(by, observed=True)
    table = pd.DataFrame({
        "plays": g.size(),
        "pass_%": (g["is_pass"].mean() * 100).round(),
        "success_%": (g["success"].mean() * 100).round(),
        "explosive_%": (g["explosive"].mean() * 100).round(),
        "avg_yards": g["yardsGained"].mean().round(1),
    })
    return table.reset_index()


def build_tables(plays, team):
    offense = prep(plays[plays["offense"] == team])
    defense = prep(plays[plays["defense"] == team])
    tables = {
        "OFFENSE: overall": make_table(offense.assign(Situation="All plays"), "Situation"),
        "OFFENSE: by down": make_table(offense, "down"),
        "OFFENSE: by down and distance": make_table(offense, ["down", "distance_group"]),
        "OFFENSE: by field position": make_table(offense, "field_zone"),
        "OFFENSE: by game situation": make_table(offense, "game_situation"),
        "DEFENSE (what opponents do against them): overall": make_table(defense.assign(Situation="All plays"), "Situation"),
        "DEFENSE: by down": make_table(defense, "down"),
        "DEFENSE: by down and distance": make_table(defense, ["down", "distance_group"]),
        "DEFENSE: by field position": make_table(defense, "field_zone"),
    }
    return offense, defense, tables


# ---------------------------------------------------------------
# The app screen
# ---------------------------------------------------------------
def main():
    st.set_page_config(page_title="CFB Scouting Tool", layout="wide")
    st.title("College Football Scouting Tool")
    st.caption("Pick a team and a season to see its offensive and defensive tendencies.")

    key = get_key()
    if not key:
        st.error("No API key found. Add CFBD_API_KEY to your app's secrets (see setup steps).")
        st.stop()

    col1, col2, col3 = st.columns(3)
    team = col1.text_input("Team (as spelled on CollegeFootballData.com)", value="Tulsa")
    year = col2.number_input("Season", min_value=2004, max_value=2030, value=2026, step=1)
    last_week = col3.number_input("Through week", min_value=1, max_value=15, value=5, step=1,
                                  help="Each week uses one of your monthly requests, so don't go higher than the current week.")
    hide_small = st.checkbox("Hide rows with fewer than 15 plays", value=False)

    if not st.button("Generate", type="primary"):
        st.info("Choose a team, season and week, then click Generate.")
        return

    team = team.strip()
    with st.spinner(f"Downloading {team} {int(year)} plays..."):
        try:
            plays = fetch_plays(team, int(year), int(last_week), key)
        except Exception as e:
            st.error(f"Could not download data: {e}")
            return

    if plays.empty:
        st.warning("No plays came back. Check the team spelling (for example 'Texas A&M', 'Ohio State') and the season.")
        return

    missing = [c for c in REQUIRED_COLUMNS if c not in plays.columns]
    if missing:
        st.error(f"The data is missing columns this app expects: {missing}")
        st.write("Columns that came back:", list(plays.columns))
        return

    weeks_found = sorted(plays["week"].unique()) if "week" in plays.columns else []
    st.success(f"Got data for weeks: {', '.join(str(int(w)) for w in weeks_found)}"
               if weeks_found else "Got data.")

    offense, defense, tables = build_tables(plays, team)
    if offense.empty and defense.empty:
        st.warning("Plays came back, but none matched this team name. Check the spelling.")
        return

    # Top-line numbers
    m1, m2, m3, m4 = st.columns(4)
    if len(offense):
        m1.metric("Offense success %", f"{offense['success'].mean() * 100:.0f}")
        m2.metric("Offense yards/play", f"{offense['yardsGained'].mean():.1f}")
    if len(defense):
        m3.metric("Defense success % allowed", f"{defense['success'].mean() * 100:.0f}")
        m4.metric("Defense yards/play allowed", f"{defense['yardsGained'].mean():.1f}")

    report_text = f"{team} {int(year)} season tendencies (through week {int(last_week)})\n"
    for title, table in tables.items():
        shown = table[table["plays"] >= 15] if hide_small else table
        st.subheader(title)
        st.dataframe(shown, hide_index=True)
        report_text += f"\n=== {title} ===\n{shown.to_string(index=False)}\n"

    with st.expander("Copy-ready text (paste into Claude to get a written scouting report)"):
        st.code(report_text, language=None)


main()

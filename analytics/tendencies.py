import pandas as pd


def prep(data):
    """Keep run/pass plays and derive the fields used by tendency tables."""
    d = data.copy()

    for col in ["down", "distance", "yardsToGoal", "yardsGained", "offenseScore", "defenseScore"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d["is_pass"] = d["playType"].astype(str).str.contains("Pass|Sack", case=False, na=False)
    d["is_run"] = d["playType"].astype(str).str.contains("Rush", case=False, na=False)

    d = d[
        (d["is_pass"] | d["is_run"])
        & (d["down"] >= 1)
        & (d["down"] <= 4)
    ].copy()
    d["down"] = d["down"].astype(int)

    needed = d["down"].map({1: 0.5, 2: 0.7, 3: 1.0, 4: 1.0}) * d["distance"]
    d["success"] = d["yardsGained"] >= needed
    d["explosive"] = (
        (d["is_run"] & (d["yardsGained"] >= 10))
        | (d["is_pass"] & (d["yardsGained"] >= 20))
    )

    d["distance_group"] = pd.cut(
        d["distance"],
        bins=[0, 3, 7, 100],
        labels=["Short (1-3)", "Medium (4-7)", "Long (8+)"],
    )
    d["field_zone"] = pd.cut(
        d["yardsToGoal"],
        bins=[0, 20, 50, 80, 100],
        labels=["Red zone", "Opp. territory", "Own side of field", "Backed up"],
    )

    margin = d["offenseScore"] - d["defenseScore"]
    d["game_situation"] = pd.cut(
        margin,
        bins=[-100, -9, 8, 100],
        labels=["Trailing by 9+", "Close game", "Leading by 9+"],
    )

    return d


def make_table(data, by):
    """Build a tendency table grouped by the requested columns."""
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

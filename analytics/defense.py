import pandas as pd


def blitz_rate(film_plays: pd.DataFrame) -> float:
    """Return blitz rate when a reviewed film dataset contains a blitz column."""
    if film_plays.empty or "blitz" not in film_plays.columns:
        return float("nan")

    valid = film_plays["blitz"].dropna()
    if valid.empty:
        return float("nan")

    return float(valid.astype(bool).mean())

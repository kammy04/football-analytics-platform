import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from football import db, features, matches  # noqa: E402


def toy_matches():
    rows = [
        ("2024-08-10", "2024-25", "A", "B", 2, 0, "H"),
        ("2024-08-17", "2024-25", "B", "C", 1, 1, "D"),
        ("2024-08-24", "2024-25", "C", "A", 0, 3, "A"),
        ("2024-08-31", "2024-25", "A", "B", 1, 0, "H"),
    ]
    m = pd.DataFrame(rows, columns=["date", "season", "home_team", "away_team", "home_goals", "away_goals", "result"])
    m["date"] = pd.to_datetime(m["date"])
    m["home_sot"], m["away_sot"] = 5, 3
    return m


def test_first_match_has_no_form():
    f, _, _ = features.add_features(toy_matches())
    assert np.isnan(f.loc[0, "home_form_pts"])
    assert f.loc[0, "elo_diff"] == 0


def test_features_use_only_past_matches():
    """Changing a result must not change the features of that match or any earlier one."""
    m = toy_matches()
    f1, _, _ = features.add_features(m)
    m2 = m.copy()
    m2.loc[2, ["home_goals", "away_goals", "result"]] = [5, 0, "H"]
    f2, _, _ = features.add_features(m2)
    cols = ["elo_diff", "home_form_pts", "away_form_pts"]
    pd.testing.assert_frame_equal(f1.loc[:2, cols], f2.loc[:2, cols])
    assert not f1.loc[3, cols].equals(f2.loc[3, cols])   # ...but later matches do see it


def test_elo_is_zero_sum_and_winner_gains():
    f, elo, _ = features.add_features(toy_matches())
    assert sum(elo.values()) == pytest.approx(1500 * 3)
    assert elo["A"] > 1500 > elo["C"]


def test_form_points():
    f, _, _ = features.add_features(toy_matches())
    # before match 4, A has played two (W, W) -> 3 pts/game; B has played two (L, D) -> 0.5
    assert f.loc[3, "home_form_pts"] == 3
    assert f.loc[3, "away_form_pts"] == 0.5


def test_bookmaker_probs_sum_to_one():
    df = pd.DataFrame({"odds_home": [2.0, 1.5], "odds_draw": [3.4, 4.0], "odds_away": [3.8, 6.5]})
    p = matches.bookmaker_probs(df)
    assert np.allclose(p.sum(axis=1), 1)


def test_rps_perfect_and_worst():
    assert matches.rps(np.array([[0, 0, 1.0]]), np.array(["H"])) == 0
    assert matches.rps(np.array([[1.0, 0, 0]]), np.array(["H"])) == 1


def test_team_query_is_parameterised(tmp_path):
    path = tmp_path / "t.db"
    with sqlite3.connect(path) as con:
        pd.DataFrame({"home_team": ["A"], "away_team": ["B"], "date": ["2024-01-01"]}).to_sql("matches", con, index=False)
    ro = db.connect_readonly(str(path))
    assert len(db.team_matches(ro, "A")) == 1
    assert len(db.team_matches(ro, "A' OR '1'='1")) == 0      # injection attempt matches nothing
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("DELETE FROM matches")                       # read-only connection

"""
features.py - pre-match features built ONLY from matches played before kick-off.

Using any information from the match itself (or later matches) would leak the answer
into the model; tests/test_features.py checks that this never happens.
"""

from collections import defaultdict, deque

import numpy as np
import pandas as pd

ELO_START = 1500
ELO_NEW_TEAM = 1430          # promoted sides start below average
ELO_K = 20
ELO_HOME_ADV = 60
SEASON_REGRESSION = 0.2      # pull ratings 20% back to the mean between seasons
FORM_WINDOW = 5


def expected_home(home_elo, away_elo, home_adv=ELO_HOME_ADV):
    return 1 / (1 + 10 ** ((away_elo - home_elo - home_adv) / 400))


def add_features(matches):
    """Return matches with Elo + rolling-form features, computed in date order."""
    m = matches.sort_values("date").reset_index(drop=True)
    elo = {}
    history = defaultdict(lambda: deque(maxlen=FORM_WINDOW))   # team -> recent match stats
    last_played = {}
    season = None
    first_season = m["season"].min()
    rows = []

    for r in m.itertuples(index=False):
        if r.season != season:   # new season: regress everyone toward the mean
            season = r.season
            elo = {t: ELO_START + (e - ELO_START) * (1 - SEASON_REGRESSION) for t, e in elo.items()}

        h, a = r.home_team, r.away_team
        # in the first season every team is new to the model, so all start level
        new = ELO_START if r.season == first_season else ELO_NEW_TEAM
        eh, ea = elo.get(h, new), elo.get(a, new)

        def form(team):
            hist = history[team]
            if not hist:
                return dict(pts=np.nan, gf=np.nan, ga=np.nan, sot=np.nan, sot_against=np.nan)
            arr = np.array(hist, dtype=float)
            return dict(pts=arr[:, 0].mean(), gf=arr[:, 1].mean(), ga=arr[:, 2].mean(),
                        sot=arr[:, 3].mean(), sot_against=arr[:, 4].mean())

        fh, fa = form(h), form(a)
        rest_h = (r.date - last_played[h]).days if h in last_played else np.nan
        rest_a = (r.date - last_played[a]).days if a in last_played else np.nan

        rows.append({
            "home_elo": eh, "away_elo": ea, "elo_diff": eh - ea,
            "elo_home_win_prob": expected_home(eh, ea),
            **{f"home_form_{k}": v for k, v in fh.items()},
            **{f"away_form_{k}": v for k, v in fa.items()},
            "home_rest_days": rest_h, "away_rest_days": rest_a,
        })

        # ---- update state AFTER recording the pre-match features ----
        score_h = 1.0 if r.result == "H" else 0.5 if r.result == "D" else 0.0
        delta = ELO_K * np.log1p(abs(r.home_goals - r.away_goals) or 1) * (score_h - expected_home(eh, ea))
        elo[h], elo[a] = eh + delta, ea - delta
        pts_h = 3 if r.result == "H" else 1 if r.result == "D" else 0
        pts_a = 3 if r.result == "A" else 1 if r.result == "D" else 0
        history[h].append((pts_h, r.home_goals, r.away_goals, r.home_sot, r.away_sot))
        history[a].append((pts_a, r.away_goals, r.home_goals, r.away_sot, r.home_sot))
        last_played[h] = last_played[a] = r.date

    f = pd.concat([m, pd.DataFrame(rows)], axis=1)
    f["form_pts_diff"] = f["home_form_pts"] - f["away_form_pts"]
    f["form_gd_diff"] = (f["home_form_gf"] - f["home_form_ga"]) - (f["away_form_gf"] - f["away_form_ga"])
    f["form_sot_diff"] = (f["home_form_sot"] - f["home_form_sot_against"]) - (f["away_form_sot"] - f["away_form_sot_against"])
    f["rest_diff"] = (f["home_rest_days"].clip(upper=14) - f["away_rest_days"].clip(upper=14))
    return f, elo, history


MODEL_FEATURES = ["elo_diff", "form_pts_diff", "form_gd_diff", "form_sot_diff", "rest_diff"]


def current_state(matches):
    """Latest Elo and form for every team, for predicting a fixture that hasn't been played yet."""
    _, elo, history = add_features(matches)
    return elo, history


def fixture_features(elo, history, home, away):
    def form(team):
        arr = np.array(history[team], dtype=float)
        return arr.mean(axis=0) if len(arr) else np.full(5, np.nan)
    fh, fa = form(home), form(away)
    return pd.DataFrame([{
        "elo_diff": elo.get(home, ELO_NEW_TEAM) - elo.get(away, ELO_NEW_TEAM),
        "form_pts_diff": fh[0] - fa[0],
        "form_gd_diff": (fh[1] - fh[2]) - (fa[1] - fa[2]),
        "form_sot_diff": (fh[3] - fh[4]) - (fa[3] - fa[4]),
        "rest_diff": 0.0,
    }])

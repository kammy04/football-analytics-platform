"""
matches.py - predict Premier League results (home win / draw / away win).

Train on 2019-20 to 2023-24, test on the whole 2024-25 season, which the model never sees.
Every model is compared against three baselines:
  * always pick a home win
  * Elo only (a single rating per team)
  * the bookmakers (average closing odds, margin removed). This is the bar to beat.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .features import MODEL_FEATURES

CLASSES = ["A", "D", "H"]    # sklearn sorts labels alphabetically
TEST_SEASON = "2024-25"


def split(f, test_season=TEST_SEASON):
    f = f.dropna(subset=MODEL_FEATURES)
    return f[f["season"] < test_season], f[f["season"] == test_season]


def bookmaker_probs(df):
    inv = 1 / df[["odds_away", "odds_draw", "odds_home"]].to_numpy()
    return inv / inv.sum(axis=1, keepdims=True)     # strip the bookmaker margin


def rps(probs, y):
    """Ranked probability score: the standard metric for ordered football outcomes (lower = better)."""
    order = {"A": 0, "D": 1, "H": 2}
    # reorder columns to away, draw, home (already in that order) and build cumulative distributions
    obs = np.zeros_like(probs)
    obs[np.arange(len(y)), [order[v] for v in y]] = 1
    return float(np.mean(np.sum((np.cumsum(probs, 1) - np.cumsum(obs, 1)) ** 2, axis=1) / 2))


def brier(probs, y):
    obs = np.zeros_like(probs)
    obs[np.arange(len(y)), [CLASSES.index(v) for v in y]] = 1
    return float(np.mean(np.sum((probs - obs) ** 2, axis=1)))


def score(name, probs, y):
    pred = np.array(CLASSES)[probs.argmax(1)]
    return {"model": name, "accuracy": accuracy_score(y, pred), "log_loss": log_loss(y, probs, labels=CLASSES),
            "rps": rps(probs, y), "brier": brier(probs, y)}


def models():
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)),
        "Decision tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=40, random_state=0),
    }


def evaluate(f):
    train, test = split(f)
    y = test["result"].to_numpy()
    rows = []

    home_rate = (train["result"].value_counts(normalize=True)).reindex(CLASSES).to_numpy()
    rows.append(score("Baseline: always home win", np.tile(home_rate, (len(test), 1)), y))

    elo_only = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    elo_only.fit(train[["elo_diff"]], train["result"])
    rows.append(score("Baseline: Elo only", elo_only.predict_proba(test[["elo_diff"]]), y))

    fitted = {}
    for name, mdl in models().items():
        mdl.fit(train[MODEL_FEATURES], train["result"])
        fitted[name] = mdl
        rows.append(score(name, mdl.predict_proba(test[MODEL_FEATURES]), y))

    rows.append(score("Bookmakers (closing odds)", bookmaker_probs(test), y))
    results = pd.DataFrame(rows).set_index("model").round(4)
    return results, fitted, train, test


def calibration_table(probs_home, y, bins=6):
    d = pd.DataFrame({"p": probs_home, "home_win": (y == "H").astype(int)})
    d["bin"] = pd.qcut(d["p"], bins, duplicates="drop")
    return d.groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("home_win", "mean"), n=("p", "size"))


def betting_backtest(model, test, edge=0.05):
    """Bet 1 unit whenever the model's probability beats the bookmaker's implied probability by `edge`."""
    p_model = model.predict_proba(test[MODEL_FEATURES])
    p_book = bookmaker_probs(test)
    odds = test[["odds_away", "odds_draw", "odds_home"]].to_numpy()
    bets, profit = 0, 0.0
    for i, res in enumerate(test["result"].to_numpy()):
        for j, outcome in enumerate(CLASSES):
            if p_model[i, j] - p_book[i, j] > edge:
                bets += 1
                profit += (odds[i, j] - 1) if res == outcome else -1
    return {"bets": bets, "profit_units": round(profit, 2), "roi": round(profit / bets, 3) if bets else None}

"""
Streamlit front end:  streamlit run app/streamlit_app.py
(run `python scripts/run_pipeline.py` first to build the database and models)
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from football import db, features, players  # noqa: E402

st.set_page_config(page_title="Football Analytics", page_icon="⚽", layout="wide")


@st.cache_resource
def load():
    con = db.connect_readonly()                 # read-only: the UI can never write to the DB
    bundle = joblib.load(os.path.join(ROOT, "models", "match_model.joblib"))
    p = db.read_players(con)
    clustered, centers, X = players.cluster(p, k=6)
    return con, bundle, clustered, X


con, bundle, clustered, X = load()
elo = bundle["elo"]
history = {k: v for k, v in bundle["history"].items()}
teams = sorted(t for t in elo if len(history.get(t, [])) >= 5)

st.title("⚽ Football Analytics")
tab1, tab2, tab3 = st.tabs(["Match predictor", "Player archetypes", "Team history"])

with tab1:
    st.caption("Probabilities from a logistic regression on Elo rating and last-5-match form. "
               "On the 2024-25 test season it scored 53% vs the bookmakers' 54%.")
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
    away = c2.selectbox("Away team", [t for t in teams if t != home])
    # selectbox values come from the whitelist above, but validate anyway before using them
    if home in teams and away in teams and home != away:
        x = features.fixture_features(elo, {k: v for k, v in history.items()}, home, away)
        probs = bundle["model"].predict_proba(x[features.MODEL_FEATURES])[0]
        labels = {"H": f"{home} win", "D": "Draw", "A": f"{away} win"}
        cols = st.columns(3)
        for col, cls in zip(cols, ["H", "D", "A"]):
            col.metric(labels[cls], f"{probs[list(bundle['model'].classes_).index(cls)]:.0%}")
        st.write(f"Elo: **{home}** {elo[home]:.0f} vs **{away}** {elo[away]:.0f}")

with tab2:
    st.caption("k-means (k=6) on 29 in-game attributes from EA FC 26, standardised per player so "
               "clusters describe playing style rather than overall quality.")
    names = clustered.sort_values("overall", ascending=False)["short_name"].tolist()
    name = st.selectbox("Player", names)
    if name in set(names):
        row = clustered[clustered["short_name"] == name].iloc[0]
        st.subheader(f"{row['short_name']} · {row['club_name']} · {row['overall']} OVR")
        st.write(f"Archetype: **{row['archetype']}**")
        st.write("Most similar players by style:")
        st.dataframe(players.similar_players(clustered, X, name), hide_index=True, width="stretch")
    st.write("Archetype sizes")
    st.bar_chart(clustered["archetype"].str.split(" \\(").str[0].value_counts())

with tab3:
    team = st.selectbox("Team", teams, key="history_team")
    if team in teams:
        m = db.team_matches(con, team)      # parameterised query
        m["venue"] = np.where(m["home_team"] == team, "Home", "Away")
        m["opponent"] = np.where(m["venue"] == "Home", m["away_team"], m["home_team"])
        m["goals_for"] = np.where(m["venue"] == "Home", m["home_goals"], m["away_goals"])
        m["goals_against"] = np.where(m["venue"] == "Home", m["away_goals"], m["home_goals"])
        m["points"] = np.select([m["goals_for"] > m["goals_against"], m["goals_for"] == m["goals_against"]], [3, 1], 0)
        st.dataframe(m.groupby("season").agg(played=("points", "size"), points=("points", "sum"),
                                             goals_for=("goals_for", "sum"), goals_against=("goals_against", "sum")),
                     width="stretch")
        st.dataframe(m[["date", "season", "venue", "opponent", "goals_for", "goals_against", "points"]]
                     .sort_values("date", ascending=False).head(20), hide_index=True, width="stretch")

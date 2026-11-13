"""
db.py - load raw data into SQLite and read it back safely.

Security notes (see SECURITY.md):
  * every query that takes user input uses ? placeholders, never string formatting
  * the app opens the database read-only (mode=ro), so a bug in the UI can't modify data
  * team and player names coming from the UI are checked against the values in the database
"""

import glob
import os
import sqlite3
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw")
DB_PATH = os.path.join(ROOT, "data", "football.db")

EPL_URL = "https://football-data.co.uk/mmz4281/{season}/E0.csv"
EPL_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425"]
PLAYERS_URL = "https://raw.githubusercontent.com/ismailoksuz/EAFC26-DataHub/main/data/players.csv"

MATCH_COLS = {
    "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team", "FTHG": "home_goals",
    "FTAG": "away_goals", "FTR": "result", "HS": "home_shots", "AS": "away_shots",
    "HST": "home_sot", "AST": "away_sot", "HC": "home_corners", "AC": "away_corners",
    "AvgH": "odds_home", "AvgD": "odds_draw", "AvgA": "odds_away",
}

ATTRIBUTES = [
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy", "attacking_short_passing",
    "attacking_volleys", "skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing",
    "skill_ball_control", "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance", "power_shot_power", "power_jumping", "power_stamina",
    "power_strength", "power_long_shots", "mentality_aggression", "mentality_interceptions",
    "mentality_positioning", "mentality_vision", "mentality_penalties", "mentality_composure",
    "defending_marking_awareness", "defending_standing_tackle", "defending_sliding_tackle",
]
PLAYER_COLS = ["player_id", "short_name", "long_name", "player_positions", "overall", "potential", "age",
               "club_name", "league_name", "nationality_name", "value_eur"] + ATTRIBUTES


def download():
    os.makedirs(RAW, exist_ok=True)
    for s in EPL_SEASONS:
        path = os.path.join(RAW, f"E0_{s}.csv")
        if not os.path.exists(path):
            urllib.request.urlretrieve(EPL_URL.format(season=s), path)
    path = os.path.join(RAW, "fc26_players.csv")
    if not os.path.exists(path):
        urllib.request.urlretrieve(PLAYERS_URL, path)


def load_matches_raw():
    frames = []
    for path in sorted(glob.glob(os.path.join(RAW, "E0_*.csv"))):
        season = os.path.basename(path)[3:7]
        df = pd.read_csv(path, encoding="utf-8-sig")
        # odds columns were renamed over the years; fall back to Bet365 if the average is missing
        for a, b in [("AvgH", "B365H"), ("AvgD", "B365D"), ("AvgA", "B365A")]:
            if a not in df.columns:
                df[a] = df[b]
        df = df[list(MATCH_COLS)].rename(columns=MATCH_COLS)
        df["season"] = f"20{season[:2]}-{season[2:]}"
        frames.append(df)
    m = pd.concat(frames, ignore_index=True).dropna(subset=["home_team", "result"])
    m["date"] = pd.to_datetime(m["date"], dayfirst=True)
    return m.sort_values("date").reset_index(drop=True)


def build(db_path=DB_PATH):
    download()
    matches = load_matches_raw()
    players = pd.read_csv(os.path.join(RAW, "fc26_players.csv"), usecols=PLAYER_COLS, low_memory=False)
    players = players[~players["player_positions"].str.startswith("GK")]   # outfield players only
    if os.path.exists(db_path):
        os.remove(db_path)
    with sqlite3.connect(db_path) as con:
        matches.assign(date=matches["date"].dt.strftime("%Y-%m-%d")).to_sql("matches", con, index=False)
        players.to_sql("players", con, index=False)
        con.execute("CREATE INDEX idx_matches_teams ON matches(home_team, away_team)")
        con.execute("CREATE INDEX idx_players_name ON players(short_name)")
    return len(matches), len(players)


def connect_readonly(db_path=DB_PATH):
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)


def read_matches(con):
    m = pd.read_sql_query("SELECT * FROM matches ORDER BY date", con, parse_dates=["date"])
    return m


def read_players(con):
    return pd.read_sql_query("SELECT * FROM players", con)


def team_matches(con, team):
    """Parameterised query: the team name is bound, never pasted into the SQL string."""
    return pd.read_sql_query(
        "SELECT * FROM matches WHERE home_team = ? OR away_team = ? ORDER BY date",
        con, params=(team, team), parse_dates=["date"])

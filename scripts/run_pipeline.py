"""
Run everything end to end:
    python scripts/run_pipeline.py

1. download data and build data/football.db (SQLite)
2. engineer pre-match features, train + evaluate match models on the 2024-25 season
3. cluster players into archetypes, train the high-potential classifier
4. write reports/, figures/ and models/ (used by the Streamlit app)
"""

import json
import os
import sys

import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from football import charts, db, features, matches, players  # noqa: E402

ROOT = db.ROOT
for d in ["reports", "figures", "models"]:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

print("1/4 building database ...")
n_matches, n_players = db.build()
con = db.connect_readonly()

print("2/4 match models ...")
m = db.read_matches(con)
f, elo, history = features.add_features(m)
results, fitted, train, test = matches.evaluate(f)
best = fitted["Logistic regression"]
backtest = matches.betting_backtest(best, test)
cal = matches.calibration_table(best.predict_proba(test[features.MODEL_FEATURES])[:, 2], test["result"].to_numpy())
results.to_csv(os.path.join(ROOT, "reports", "match_model_results.csv"))
charts.model_comparison(results, os.path.join(ROOT, "figures", "model_comparison.png"))
charts.calibration(cal, os.path.join(ROOT, "figures", "calibration.png"))
charts.elo_history(f, ["Man City", "Liverpool", "Arsenal", "Man United"], os.path.join(ROOT, "figures", "elo_history.png"))
# the app refits on all seasons so its predictions use the most recent data
final = matches.models()["Logistic regression"].fit(f.dropna(subset=features.MODEL_FEATURES)[features.MODEL_FEATURES],
                                                    f.dropna(subset=features.MODEL_FEATURES)["result"])
joblib.dump({"model": final, "elo": elo, "history": {k: list(v) for k, v in history.items()}},
            os.path.join(ROOT, "models", "match_model.joblib"))

print("3/4 player archetypes ...")
p = db.read_players(con)
k_table = players.choose_k(players.style_matrix(p))
clustered, centers, X = players.cluster(p, k=6)
charts.archetype_heatmap(centers, players.SHORT, os.path.join(ROOT, "figures", "archetypes.png"))
pot_metrics, pot_coefs = players.potential_classifier(p)
clustered[["player_id", "short_name", "archetype", "pc1", "pc2"]].to_csv(
    os.path.join(ROOT, "reports", "player_archetypes.csv"), index=False)

print("4/4 writing report ...")
summary = {
    "matches": n_matches, "outfield_players": n_players, "test_season": matches.TEST_SEASON,
    "match_models": results.reset_index().to_dict(orient="records"),
    "value_bet_backtest": backtest,
    "k_selection": k_table.round(4).to_dict(orient="records"),
    "archetype_sizes": clustered["archetype"].value_counts().to_dict(),
    "potential_classifier": pot_metrics,
    "potential_top_coefficients": pot_coefs.tail(5).round(3).to_dict(),
}
with open(os.path.join(ROOT, "reports", "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2, default=str)
print(results)
print("backtest:", backtest)
print("potential classifier:", pot_metrics)
print(clustered["archetype"].value_counts())

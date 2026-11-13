"""
players.py - player archetypes (k-means) and a high-potential classifier.

Clustering uses the 29 detailed in-game attributes (finishing, vision, sliding tackle...)
of every outfield player in EA FC 26. Attributes are standardised *within each player*
first, so clusters describe a player's *style* (what they're relatively good at) rather
than just how good they are overall.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .db import ATTRIBUTES

SHORT = {a: a.split("_", 1)[1].replace("_", " ") for a in ATTRIBUTES}


def style_matrix(players):
    X = players[ATTRIBUTES].to_numpy(dtype=float)
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)   # per-player profile
    return StandardScaler().fit_transform(X)


def choose_k(X, ks=range(4, 13), seed=0, sample=6000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    out = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        out.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X[idx], km.labels_[idx])})
    return pd.DataFrame(out)


def cluster(players, k=8, seed=0):
    X = style_matrix(players)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
    p = players.copy()
    p["cluster"] = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=ATTRIBUTES)
    names = name_clusters(p, centers)
    p["archetype"] = p["cluster"].map(names)
    coords = PCA(n_components=2, random_state=seed).fit_transform(X)
    p["pc1"], p["pc2"] = coords[:, 0], coords[:, 1]
    return p, centers.rename(index=names), X


POSITION_ROLE = {
    "CB": "Centre-back", "LB": "Full-back", "RB": "Full-back", "LWB": "Full-back", "RWB": "Full-back",
    "ST": "Striker", "CF": "Striker", "CAM": "Creative attacker", "LW": "Creative attacker",
    "RW": "Creative attacker", "LM": "Creative attacker", "RM": "Creative attacker",
    "CM": "Midfielder", "CDM": "Midfielder",
}


def name_clusters(p, centers):
    """
    Human-readable names: the cluster's most common position, plus its signature attributes.
    When two clusters share a role (e.g. two kinds of midfielder), the one that leans more
    defensive is called 'deep-lying' and the other 'box-to-box'.
    """
    names, roles = {}, {}
    for c in centers.index:
        pos = p.loc[p["cluster"] == c, "player_positions"].str.split(",").str[0].value_counts().index[0]
        roles[c] = POSITION_ROLE.get(pos, pos)
    for role in set(roles.values()):
        cs = [c for c in centers.index if roles[c] == role]
        if len(cs) == 1:
            names[cs[0]] = role
            continue
        defensive = centers.loc[cs, ["mentality_interceptions", "defending_standing_tackle"]].mean(axis=1)
        order = defensive.sort_values(ascending=False).index
        labels = ["Deep-lying " + role.lower(), "Box-to-box " + role.lower()] + \
                 [f"{role} {i}" for i in range(3, len(cs) + 1)]
        names.update(dict(zip(order, labels)))
    for c in centers.index:
        top = centers.loc[c].nlargest(3).index
        names[c] = f"{names[c]} ({', '.join(SHORT[t] for t in top)})"
    return names


def similar_players(p, X, name, n=8):
    idx = p.index[p["short_name"] == name]
    if len(idx) == 0:
        raise KeyError(name)
    i = p.index.get_loc(idx[0])
    nn = NearestNeighbors(n_neighbors=n + 1).fit(X)
    _, ind = nn.kneighbors(X[i:i + 1])
    return p.iloc[ind[0][1:]][["short_name", "age", "player_positions", "club_name", "overall", "archetype"]]


def potential_classifier(players, threshold=80, max_age=21, seed=0):
    """Can current attributes + age predict which young players EA rates as future stars?"""
    young = players[players["age"] <= max_age].copy()
    y = (young["potential"] >= threshold).astype(int)
    X = young[ATTRIBUTES + ["age"]]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    coefs = pd.Series(clf[-1].coef_[0], index=ATTRIBUTES + ["age"]).sort_values()
    return {"auc": round(roc_auc_score(yte, p), 3), "positives": int(y.sum()), "young_players": len(young)}, coefs

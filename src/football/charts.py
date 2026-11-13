"""charts.py - static figures for the README."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a3a29c"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f6c3a8", "#f3f2ee", "#a9cbf3", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def model_comparison(results, path):
    r = results.sort_values("rps", ascending=False)
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    colors = [ORANGE if "Bookmakers" in i else GREY if "Baseline" in i else BLUE for i in r.index]
    ax.barh(r.index, r["rps"], color=colors, height=0.6)
    for y, (v, acc) in enumerate(zip(r["rps"], r["accuracy"])):
        ax.text(v + 0.0008, y, f"RPS {v:.4f}  ·  {acc:.1%} correct", va="center", fontsize=9)
    ax.set_xlim(0.19, max(r["rps"]) + 0.022)
    ax.set_xlabel("Ranked probability score (lower is better)")
    ax.set_title("Predicting the unseen 2024-25 Premier League season")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def calibration(table, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=INK_2, linestyle="--", linewidth=1)
    ax.plot(table["predicted"], table["actual"], color=BLUE, linewidth=2, marker="o", markersize=7)
    ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.9)
    ax.set_xlabel("Predicted home-win probability")
    ax.set_ylabel("Actual home-win rate")
    ax.set_title("Calibration (2024-25)")
    ax.text(0.12, 0.2, "dashed: perfect calibration", color=INK_2, fontsize=8.5)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def elo_history(f, teams, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    for team, color in zip(teams, [BLUE, ORANGE, AQUA, YELLOW]):
        h = f[f["home_team"] == team][["date", "home_elo"]].rename(columns={"home_elo": "elo"})
        a = f[f["away_team"] == team][["date", "away_elo"]].rename(columns={"away_elo": "elo"})
        s = np.concatenate([h.values, a.values])
        s = s[np.argsort(s[:, 0])]
        ax.plot(s[:, 0], s[:, 1].astype(float), color=color, linewidth=2, label=team)
    ax.set_ylabel("Elo rating (before each match)")
    ax.set_title("Team strength over six seasons (Elo)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def archetype_heatmap(centers, short, path):
    c = centers.copy()
    c.columns = [short[x] for x in c.columns]
    keep = c.abs().max().sort_values(ascending=False).index[:18]
    c = c[keep]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(c.values, cmap=DIVERGING, vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(c.shape[1]), c.columns, rotation=40, ha="right")
    ax.set_yticks(range(c.shape[0]), [i.split(" (")[0] for i in c.index])
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="relative strength (z)")
    ax.set_title("Six player archetypes found by k-means: what each is relatively good at")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

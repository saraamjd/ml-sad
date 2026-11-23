"""Figures for the ML-SAD analysis (matplotlib, no seaborn dependency)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # safe for headless / script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .statistics import STRATEGY_COLUMNS

SAD_COLOR = "#F8AD7D"
NOSAD_COLOR = "mediumseagreen"


def _save(fig, path: str | Path | None):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def age_distribution(df: pd.DataFrame, path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(df["age"], bins=18, color=NOSAD_COLOR, edgecolor="darkgreen", alpha=0.8)
    mean_age = df["age"].mean()
    ax.axvline(mean_age, color="navy", linestyle="dashed", linewidth=1,
               label=f"Mean: {mean_age:.1f}")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    ax.set_title("Age Distribution")
    ax.legend()
    return _save(fig, path)


def strategy_bars_by_group(df: pd.DataFrame, group_col: str = "sad",
                           labels=("SAD", "No SAD"), path=None):
    sub = df.dropna(subset=[group_col])
    g1 = [sub[sub[group_col] == 1][c].mean() for c in STRATEGY_COLUMNS]
    g0 = [sub[sub[group_col] == 0][c].mean() for c in STRATEGY_COLUMNS]
    x = np.arange(len(STRATEGY_COLUMNS))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, g1, width, label=labels[0], color=SAD_COLOR)
    ax.bar(x + width / 2, g0, width, label=labels[1], color=NOSAD_COLOR)
    ax.set_ylabel("Score")
    ax.set_title("Emotion-Regulation Strategies")
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in STRATEGY_COLUMNS])
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def scree_plot(eigenvalues, path=None):
    fig, ax = plt.subplots()
    ax.plot(range(len(eigenvalues)), eigenvalues, marker="o")
    ax.set_title("Scree Plot")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Eigenvalue")
    ax.grid(True)
    return _save(fig, path)


def severity_vs_strategies(spin_df: pd.DataFrame, path=None):
    order = ["minimal", "mild", "moderate", "severe", "very_severe"]
    present = [s for s in order if s in set(spin_df["spin_severity"])]
    grouped = spin_df.groupby("spin_severity")[list(STRATEGY_COLUMNS)].mean().reindex(present)
    fig, ax = plt.subplots(figsize=(9, 5))
    for col, mk in zip(STRATEGY_COLUMNS, ["o", "s", "^"]):
        ax.plot(range(len(present)), grouped[col], marker=mk, label=col.title())
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=30, ha="right")
    ax.set_xlabel("SPIN severity")
    ax.set_ylabel("Score")
    ax.set_title("Emotion Regulation vs. SPIN Severity")
    ax.legend(title="Strategy")
    fig.tight_layout()
    return _save(fig, path)

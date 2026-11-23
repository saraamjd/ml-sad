"""Group comparisons and effect sizes for emotion-regulation strategies."""
from __future__ import annotations

import pandas as pd
from scipy import stats

STRATEGY_COLUMNS = ("suppression", "avoidance", "reappraisal")


def manova(df: pd.DataFrame):
    """MANOVA of the three strategies on demographic / clinical predictors.

    Reproduces ``supression + avoidance + reappraisal ~ stress_level + female
    + diagnosed + age``. Returns the fitted MANOVA result object.
    """
    from statsmodels.multivariate.manova import MANOVA

    data = df.rename(columns={
        "suppression": "supression", "stress_level": "st",
        "female": "f", "diagnosed": "d", "age": "ag",
    })
    formula = "supression + avoidance + reappraisal ~ st + f + d + ag"
    return MANOVA.from_formula(formula, data=data).mv_test()


def rank_biserial(u: float, n1: int, n2: int) -> float:
    """Rank-biserial effect size for a Mann-Whitney U statistic."""
    return (2 * u) / (n1 * n2) - 1


def mann_whitney_by_group(df: pd.DataFrame, group_col: str = "sad") -> pd.DataFrame:
    """Mann-Whitney U comparing each strategy between the two groups.

    Also reports mean maladaptive score and the rank-biserial effect size.
    """
    labelled = df.dropna(subset=[group_col])
    g1 = labelled[labelled[group_col] == 1]
    g0 = labelled[labelled[group_col] == 0]
    rows = []
    for col in (*STRATEGY_COLUMNS, "mean_maladaptive"):
        u, p = stats.mannwhitneyu(g1[col], g0[col])
        rows.append({
            "variable": col,
            "U": u,
            "p": p,
            "effect_rank_biserial": rank_biserial(u, len(g1), len(g0)),
            "mean_group1": g1[col].mean(),
            "mean_group0": g0[col].mean(),
        })
    return pd.DataFrame(rows)


def strategy_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlations among the three strategy scores."""
    rows = []
    pairs = [("suppression", "reappraisal"), ("avoidance", "reappraisal"),
             ("avoidance", "suppression")]
    for a, b in pairs:
        r, p = stats.pearsonr(df[a], df[b])
        rows.append({"pair": f"{a}~{b}", "r": r, "p": p})
    return pd.DataFrame(rows)


def group_means(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Mean strategy scores by a grouping column (e.g. ``sad`` or ``female``)."""
    return (df.dropna(subset=[group_col])
              .groupby(group_col)[list(STRATEGY_COLUMNS)]
              .mean())

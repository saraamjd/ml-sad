"""Reliability and factor-analytic validation of the emotion-regulation scale."""
from __future__ import annotations

import pandas as pd
from scipy import stats

from .preprocessing import ITEM_COLUMNS


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Cronbach's alpha for a set of item columns (respondents x items)."""
    k = items.shape[1]
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def subscale_reliability(df: pd.DataFrame) -> dict[str, float]:
    """Cronbach's alpha for each subscale and the full scale."""
    s = df[[f"{i}S" for i in range(1, 11)]]
    a = df[[f"{i}A" for i in range(1, 11)]]
    r = df[[f"{i}R" for i in range(1, 11)]]
    return {
        "Suppression": cronbach_alpha(s),
        "Avoidance": cronbach_alpha(a),
        "Reappraisal": cronbach_alpha(r),
        "Total": cronbach_alpha(df[ITEM_COLUMNS]),
    }


def item_total_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of each item with its own subscale total."""
    rows = []
    for i in range(1, 11):
        for strat, total in (("S", "sum_S"), ("A", "sum_A"), ("R", "sum_R")):
            col = f"{i}{strat}"
            rho, p = stats.spearmanr(df[col], df[total])
            rows.append({"item": col, "subscale": strat, "rho": rho, "p": p})
    return pd.DataFrame(rows)


def kmo_and_bartlett(df: pd.DataFrame):
    """KMO sampling adequacy and Bartlett's test of sphericity on the 30 items."""
    from factor_analyzer.factor_analyzer import (
        calculate_bartlett_sphericity, calculate_kmo,
    )
    from sklearn.preprocessing import StandardScaler

    scaled = pd.DataFrame(
        StandardScaler().fit_transform(df[ITEM_COLUMNS]), columns=ITEM_COLUMNS
    )
    chi2, p = calculate_bartlett_sphericity(scaled)
    _, kmo_model = calculate_kmo(scaled)
    return {"bartlett_chi2": chi2, "bartlett_p": p, "kmo": kmo_model}


def exploratory_factor_analysis(df: pd.DataFrame, n_factors: int = 2,
                                rotation: str = "varimax"):
    """Fit an EFA on the 30 items; return (loadings, eigenvalues)."""
    from factor_analyzer import FactorAnalyzer
    from sklearn.preprocessing import StandardScaler

    scaled = pd.DataFrame(
        StandardScaler().fit_transform(df[ITEM_COLUMNS]), columns=ITEM_COLUMNS
    )
    eig = FactorAnalyzer(len(ITEM_COLUMNS), rotation=None)
    eig.fit(scaled)
    eigenvalues = eig.get_eigenvalues()[0]

    fa = FactorAnalyzer(n_factors, rotation=rotation, method="minres")
    fa.fit(scaled)
    loadings = pd.DataFrame(
        fa.loadings_,
        index=ITEM_COLUMNS,
        columns=[f"Factor{i + 1}" for i in range(n_factors)],
    )
    return loadings, eigenvalues


def normality(df: pd.DataFrame, column: str, group_col: str = "sad") -> pd.DataFrame:
    """Shapiro-Wilk normality test of ``column`` within each group."""
    rows = []
    for g, sub in df.dropna(subset=[group_col]).groupby(group_col):
        w, p = stats.shapiro(sub[column])
        rows.append({"group": int(g), "variable": column, "W": w, "p": p})
    return pd.DataFrame(rows)

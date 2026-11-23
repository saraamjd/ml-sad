#!/usr/bin/env python3
"""Reliability and factor analysis of the emotion-regulation scale.

Usage: python scripts/01_psychometrics.py
Writes a summary to results/psychometrics.txt and a scree plot to figures/.
"""
from __future__ import annotations

from pathlib import Path

from ml_sad import preprocessing as pp
from ml_sad import psychometrics as ps
from ml_sad import visualization as viz

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    df = pp.load_scores()
    lines = []

    lines.append("Cronbach's alpha")
    for scale, alpha in ps.subscale_reliability(df).items():
        lines.append(f"  {scale:12s}: {alpha:.3f}")

    lines.append("\nKMO & Bartlett")
    try:
        kb = ps.kmo_and_bartlett(df)
        lines.append(f"  KMO = {kb['kmo']:.3f}")
        lines.append(f"  Bartlett chi2 = {kb['bartlett_chi2']:.1f}, p = {kb['bartlett_p']:.3g}")

        loadings, eig = ps.exploratory_factor_analysis(df, n_factors=2)
        viz.scree_plot(eig, REPO_ROOT / "figures" / "scree_plot.png")
        lines.append("\nEFA (2 factors, varimax) loadings:")
        lines.append(loadings.round(3).to_string())
    except Exception as exc:  # factor_analyzer optional at runtime
        lines.append(f"  [skipped EFA: {exc}]")

    lines.append("\nItem-total Spearman correlations:")
    lines.append(ps.item_total_correlations(df).round(3).to_string(index=False))

    report = "\n".join(lines)
    print(report)
    out = REPO_ROOT / "results" / "psychometrics.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Correlate emotion-regulation strategies with SPIN social-anxiety severity.

Usage: python scripts/03_spin_correlation.py
Writes results/spin_correlation.txt and figures/severity_vs_strategies.png.
"""
from __future__ import annotations

from pathlib import Path

from ml_sad import preprocessing as pp
from ml_sad import visualization as viz

REPO_ROOT = Path(__file__).resolve().parents[1]
COLS = ["suppression", "avoidance", "reappraisal", "spin_total"]


def main() -> None:
    spin = pp.load_spin_scores()
    lines = ["SPIN severity distribution:",
             spin["spin_severity"].value_counts().to_string(),
             "\nCorrelation matrix (Pearson):",
             spin[COLS].corr().round(3).to_string(),
             "\nSpearman (strategy vs spin_total):"]
    for c in ["suppression", "avoidance", "reappraisal"]:
        rho = spin[c].corr(spin["spin_total"], method="spearman")
        lines.append(f"  {c:12s}: {rho:.3f}")

    viz.severity_vs_strategies(
        spin, path=REPO_ROOT / "figures" / "severity_vs_strategies.png")

    report = "\n".join(lines)
    print(report)
    out = REPO_ROOT / "results" / "spin_correlation.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)


if __name__ == "__main__":
    main()

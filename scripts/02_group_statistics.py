#!/usr/bin/env python3
"""Group comparisons of emotion-regulation strategies (SAD vs. no-SAD, sex).

Usage: python scripts/02_group_statistics.py
Writes results/group_statistics.txt and figures/strategies_by_group.png.
"""
from __future__ import annotations

from pathlib import Path

from ml_sad import preprocessing as pp
from ml_sad import statistics as st
from ml_sad import visualization as viz

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    df = pp.load_scores()
    lines = []

    lines.append("Mann-Whitney U (SAD vs no-SAD):")
    lines.append(st.mann_whitney_by_group(df, "sad").round(4).to_string(index=False))

    lines.append("\nMann-Whitney U (female vs male):")
    lines.append(st.mann_whitney_by_group(df, "female").round(4).to_string(index=False))

    lines.append("\nPearson correlations among strategies:")
    lines.append(st.strategy_correlations(df).round(4).to_string(index=False))

    lines.append("\nGroup means by SAD status:")
    lines.append(st.group_means(df, "sad").round(4).to_string())

    try:
        lines.append("\nMANOVA (strategies ~ stress + sex + diagnosed + age):")
        lines.append(str(st.manova(df)))
    except Exception as exc:
        lines.append(f"\n[MANOVA skipped: {exc}]")

    viz.strategy_bars_by_group(df, "sad",
                               path=REPO_ROOT / "figures" / "strategies_by_group.png")

    report = "\n".join(lines)
    print(report)
    out = REPO_ROOT / "results" / "group_statistics.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)


if __name__ == "__main__":
    main()

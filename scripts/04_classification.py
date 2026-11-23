#!/usr/bin/env python3
"""Train and evaluate the SAD screening classifiers.

Usage: python scripts/04_classification.py [--seed 49]
Writes:
  results/cross_validated_scores.csv
  results/confusion_matrices_by_classifier.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ml_sad import classification as clf
from ml_sad import preprocessing as pp

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=clf.DEFAULT_SEED)
    args = parser.parse_args()

    df = pp.load_scores()
    X, y = pp.screening_dataset(df)
    print(f"Labelled respondents: n={len(y)}  SAD+={int(y.sum())}  SAD-={int((y == 0).sum())}\n")

    cv = clf.round_report(clf.cross_validated_scores(X, y, seed=args.seed))
    print("Cross-validated (repeated stratified 5-fold) scores:")
    print(cv.to_string(index=False))

    cm = clf.round_report(clf.confusion_matrices(X, y, seed=args.seed))
    print("\nHeld-out confusion matrices and DTA metrics:")
    print(cm.to_string(index=False))

    results = REPO_ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    cv.to_csv(results / "cross_validated_scores.csv", index=False)
    cm.to_csv(results / "confusion_matrices_by_classifier.csv", index=False)


if __name__ == "__main__":
    main()

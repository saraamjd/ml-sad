#!/usr/bin/env python3
"""Regenerate the de-identified tables in ``data/`` from the raw survey exports.

The raw JSON exports are private (they contain emails, IPs, geolocation, names
and phone numbers) and are NOT part of this repository. Point this script at a
local copy to reproduce the published, de-identified CSVs.

Usage:
    python scripts/00_deidentify.py --raw-dir /path/to/raw [--salt SECRET]

Expects, in ``--raw-dir``:
    fileV206_edited.json       (emotion-regulation questionnaire)
    socialanxietyusers1.json   (SPIN questionnaire, ipgeolocation already removed)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ml_sad import preprocessing as pp

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True,
                        help="Directory containing the private raw JSON exports.")
    parser.add_argument("--salt", default="ml-sad-public",
                        help="Salt for pseudonymous ids (keep private for stability).")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data"))
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scores = pp.load_emotion_regulation(raw / "fileV206_edited.json", salt=args.salt)
    pp.assert_no_pii(scores)
    scores.to_csv(out / "emotion_regulation.csv", index=False)
    print(f"emotion_regulation.csv: {len(scores)} respondents, {scores.shape[1]} columns")

    spin = pp.load_spin(raw / "socialanxietyusers1.json", salt=args.salt)
    pp.assert_no_pii(spin)
    spin.to_csv(out / "spin_severity.csv", index=False)
    print(f"spin_severity.csv: {len(spin)} respondents, {spin.shape[1]} columns")


if __name__ == "__main__":
    main()

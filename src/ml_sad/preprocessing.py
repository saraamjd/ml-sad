"""Load, de-identify and score the ML-SAD survey exports.

The study used two raw JSON-Lines exports that are **not** distributed with this
repository because they contain directly identifying information (email
addresses, IP-based geolocation, names and phone numbers):

- ``fileV206_edited.json``   – the emotion-regulation strategy questionnaire.
- ``socialanxietyusers.json`` – the SPIN social-anxiety severity questionnaire.

This module reconstructs the exact de-identified tables shipped in ``data/`` from
those raw files. All direct identifiers are dropped and each respondent is given
a stable pseudonymous id (``P0001`` …) derived from a salted hash, so the public
tables can never be re-linked to the originals.

Emotion-regulation instrument
------------------------------
Ten scenarios, each offering three responses mapped to one coping strategy:
Suppression (S), Avoidance (A) or Reappraisal (R). For every scenario the
respondent allocates the score to exactly one strategy, so each scenario's three
item columns (e.g. ``1S``/``1A``/``1R``) sum to the same per-scenario maximum.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

# Raw export column S<scenario>D<dimension> -> analysis code "<scenario><strategy>".
# The dimension->strategy assignment is scenario-specific and comes from the
# questionnaire design.
ITEM_RENAME = {
    "S1D1": "1S", "S1D2": "1R", "S1D3": "1A",
    "S2D1": "2R", "S2D2": "2S", "S2D3": "2A",
    "S3D1": "3S", "S3D2": "3A", "S3D3": "3R",
    "S4D1": "4A", "S4D2": "4R", "S4D3": "4S",
    "S5D1": "5A", "S5D2": "5S", "S5D3": "5R",
    "S6D1": "6S", "S6D2": "6R", "S6D3": "6A",
    "S7D1": "7A", "S7D2": "7S", "S7D3": "7R",
    "S8D1": "8A", "S8D2": "8S", "S8D3": "8R",
    "S9D1": "9R", "S9D2": "9S", "S9D3": "9A",
    "S10D1": "10A", "S10D2": "10R", "S10D3": "10S",
}
ITEM_COLUMNS = [f"{i}{s}" for i in range(1, 11) for s in ("S", "A", "R")]

# Normalisation constant used throughout the original analysis (see README /
# data/README.md for the rationale). Kept as-is for exact reproducibility.
NORM_M = 80

AGE_MIN, AGE_MAX = 18, 35

# Columns that directly or indirectly identify a respondent and must never be
# published.
_PII_FIELDS = ("Email=", "id", "description", "DateTime", "demographic",
               "ipgeolocation", "submittime")

_DEFAULT_SALT = "ml-sad-public"


def _pseudonym(value: str, salt: str) -> str:
    """Deterministic, non-reversible short id for a raw identifier."""
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return digest[:12]


def _read_jsonl(path: str | os.PathLike) -> list[dict]:
    """Read a JSON-Lines file, tolerating ``None`` tokens and blank lines."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip().replace(": None", ": null")
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _to_int(value) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# Emotion-regulation questionnaire
# --------------------------------------------------------------------------- #
def load_emotion_regulation(raw_path: str | os.PathLike,
                            salt: str = _DEFAULT_SALT) -> pd.DataFrame:
    """Build the de-identified, scored emotion-regulation table from the raw export.

    Returns one row per respondent (after the 18–35 age filter) with:
    pseudonymous ``participant_id``; demographics ``age``, ``female``,
    ``diagnosed``, ``stress_level``; the 30 scored items; subscale sums and
    normalised scores; and the derived binary ``sad`` / ``ml_label`` targets.
    """
    records = _read_jsonl(raw_path)
    df = pd.DataFrame(records)

    # Age filter used in the study.
    df = df[df["Age="].apply(_to_int).between(AGE_MIN, AGE_MAX)].copy()

    # Pseudonymise before touching anything else, then drop raw identifiers.
    df["participant_id"] = df["Email="].astype(str).map(lambda v: _pseudonym(v, salt))

    out = pd.DataFrame()
    out["participant_id"] = df["participant_id"].values
    out["age"] = df["Age="].apply(_to_int).values
    out["female"] = df["Female="].apply(_to_int).values
    out["diagnosed"] = df["Diagnised="].apply(_to_int).values
    out["stress_level"] = df["Stress Level="].apply(_to_int).values

    df = df.rename(columns=ITEM_RENAME)
    for col in ITEM_COLUMNS:
        out[col] = pd.to_numeric(df[col], errors="coerce").values

    out = add_subscale_scores(out)
    out = add_targets(out)

    # Stable, reproducible ordering and anonymous sequential ids.
    out = out.sort_values("participant_id").reset_index(drop=True)
    out["participant_id"] = [f"P{i + 1:04d}" for i in range(len(out))]
    return out


def add_subscale_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add Suppression/Avoidance/Reappraisal sums and normalised scores."""
    df = df.copy()
    df["sum_S"] = df[[f"{i}S" for i in range(1, 11)]].sum(axis=1)
    df["sum_A"] = df[[f"{i}A" for i in range(1, 11)]].sum(axis=1)
    df["sum_R"] = df[[f"{i}R" for i in range(1, 11)]].sum(axis=1)
    df["suppression"] = df["sum_S"] / NORM_M
    df["avoidance"] = df["sum_A"] / NORM_M
    df["reappraisal"] = df["sum_R"] / NORM_M
    df["mean_maladaptive"] = (df["suppression"] + df["avoidance"]) / 2 * NORM_M
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary screening targets used for group stats and classification.

    ``sad`` (and the equivalent ML target) is defined on the *extreme groups*
    only, following the study design:

    - SAD positive  : clinically diagnosed AND self-reported high stress (4–5)
    - SAD negative  : not diagnosed AND self-reported low stress (1–2)

    Respondents in the ambiguous middle are left as ``NA`` and excluded from the
    supervised analyses.
    """
    df = df.copy()
    pos = (df["diagnosed"] == 1) & (df["stress_level"].isin([4, 5]))
    neg = (df["diagnosed"] == 0) & (df["stress_level"].isin([1, 2]))
    df["sad"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[pos, "sad"] = 1
    df.loc[neg, "sad"] = 0
    # ``ml_label`` kept for traceability with the original notebooks (0=SAD).
    df["ml_label"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[pos, "ml_label"] = 0
    df.loc[neg, "ml_label"] = 1
    return df


def screening_dataset(df: pd.DataFrame):
    """Return ``(X, y)`` for supervised screening: 30 items, SAD=1 positive."""
    labelled = df.dropna(subset=["sad"]).copy()
    X = labelled[ITEM_COLUMNS].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = labelled["sad"].astype(int)
    return X, y


# --------------------------------------------------------------------------- #
# SPIN severity questionnaire
# --------------------------------------------------------------------------- #
SPIN_SEVERITY_BINS = [
    (0, 20, "minimal"),
    (21, 30, "mild"),
    (31, 40, "moderate"),
    (41, 50, "severe"),
    (51, 68, "very_severe"),
]


def _spin_total(spin: dict) -> int | None:
    if not spin:
        return None
    questions = spin.get("questions")
    if not isinstance(questions, dict):
        questions = {f"question{i}": spin.get(f"question{i}", 0) for i in range(1, 18)}
    try:
        return sum(int(v) for v in questions.values())
    except (TypeError, ValueError):
        return None


def spin_severity_label(total: int | None) -> str | None:
    if total is None:
        return None
    for low, high, name in SPIN_SEVERITY_BINS:
        if low <= total <= high:
            return name
    return None


def load_spin(raw_path: str | os.PathLike, salt: str = _DEFAULT_SALT) -> pd.DataFrame:
    """Build the de-identified SPIN table (severity + emotion-regulation sums)."""
    records = _read_jsonl(raw_path)
    rows = []
    for rec in records:
        total = _spin_total(rec.get("spinquestionnaire"))
        if total is None:
            continue
        rows.append({
            "participant_id": _pseudonym(str(rec.get("id")), salt),
            "spin_total": total,
            "spin_severity": spin_severity_label(total),
            "suppression": rec.get("supression_sum"),
            "avoidance": rec.get("avoidance_sum"),
            "reappraisal": rec.get("reappraisal_sum"),
            "sad_label": rec.get("label"),
        })
    out = pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)
    out["participant_id"] = [f"S{i + 1:04d}" for i in range(len(out))]
    return out


# --------------------------------------------------------------------------- #
# Loading the shipped, already de-identified tables
# --------------------------------------------------------------------------- #
def data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def load_scores(path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Load the de-identified emotion-regulation table shipped in ``data/``."""
    path = path or data_dir() / "emotion_regulation.csv"
    df = pd.read_csv(path)
    df["sad"] = df["sad"].astype("Int64")
    df["ml_label"] = df["ml_label"].astype("Int64")
    return df


def load_spin_scores(path: str | os.PathLike | None = None) -> pd.DataFrame:
    path = path or data_dir() / "spin_severity.csv"
    return pd.read_csv(path)


def assert_no_pii(df: pd.DataFrame, fields: Iterable[str] = _PII_FIELDS) -> None:
    """Raise if any known identifying column survived into ``df``."""
    leaked = [c for c in df.columns if c in set(fields)]
    if leaked:
        raise AssertionError(f"Identifying columns present: {leaked}")

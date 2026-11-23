# Data dictionary & provenance

Both tables are **de-identified**. They were produced from private raw survey
exports by [`scripts/00_deidentify.py`](../scripts/00_deidentify.py), which:

- removes all direct identifiers (email, IP/geolocation, name, phone, free-text
  description, exact submission timestamp);
- replaces each respondent key with a salted SHA-256 pseudonym, then a stable
  sequential id (`P0001…`, `S0001…`);
- applies the study's age filter (18–35) to the emotion-regulation table.

The raw files are **not** distributed and cannot be reconstructed from these
tables.

## `emotion_regulation.csv` — 488 respondents

| Column | Type | Description |
|---|---|---|
| `participant_id` | str | Anonymous id (`P0001…`). Not linkable to the raw export. |
| `age` | int | Age in years (18–35). |
| `female` | 0/1 | 1 = female, 0 = male (as recorded). |
| `diagnosed` | 0/1 | 1 = prior clinical SAD diagnosis (self-reported). |
| `stress_level` | int | Self-reported stress level, 1 (low) – 5 (high). |
| `1S … 10R` | int | 30 item scores, coded `<scenario><strategy>`; strategy ∈ S=Suppression, A=Avoidance, R=Reappraisal. Each scenario's three items sum to that scenario's allocation. |
| `sum_S`, `sum_A`, `sum_R` | int | Subscale sums across the 10 scenarios. |
| `suppression`, `avoidance`, `reappraisal` | float | Subscale sums normalised by a constant (80). |
| `mean_maladaptive` | float | Combined suppression+avoidance index, `(suppression+avoidance)/2 × 80`. |
| `sad` | 0/1/NA | Screening target (extreme groups). 1 = diagnosed & stress 4–5; 0 = not diagnosed & stress 1–2; NA = ambiguous middle (excluded from supervised analyses). |
| `ml_label` | 0/1/NA | Legacy label used in the original notebooks (**0 = SAD**, 1 = non-SAD); inverse coding of `sad`. Kept for traceability. |

> The normalisation constant (80) and the `mean_maladaptive` formula are kept
> exactly as in the original analysis so published statistics reproduce. They
> are monotonic rescalings and do not affect correlations, rank tests, or the
> classifiers (which use the raw 30 items).

## `spin_severity.csv` — 128 respondents

Derived from the SPIN (Social Phobia Inventory) questionnaire. The raw SPIN
export is a separate instrument from the emotion-regulation survey; ids here
(`S0001…`) are **not** aligned with `emotion_regulation.csv`.

| Column | Type | Description |
|---|---|---|
| `participant_id` | str | Anonymous id (`S0001…`). |
| `spin_total` | int | SPIN total score (sum of 17 items, 0–68). |
| `spin_severity` | str | Severity band: minimal (0–20), mild (21–30), moderate (31–40), severe (41–50), very_severe (51–68). |
| `suppression`, `avoidance`, `reappraisal` | float | Emotion-regulation subscale scores recorded with the SPIN submission. |
| `sad_label` | 0/1 | SAD label recorded in the source (1 = SAD). |

## Important limitations

- Self-report measures; **not** a diagnostic instrument for individuals.
- Convenience sample, ages 18–35; results may not generalise.
- `sad` uses extreme-group labelling by design, which inflates separability
  relative to an unselected population — interpret accuracy accordingly.

License: **CC BY 4.0** (see [`LICENSE`](LICENSE)).

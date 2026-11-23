"""ML-SAD: Machine-learning screening of Social Anxiety Disorder from emotion-regulation strategies.

Public, reproducible analysis code accompanying the ML-SAD study. The package is
organised into focused modules:

- :mod:`ml_sad.preprocessing`  – load raw survey exports, de-identify, score subscales.
- :mod:`ml_sad.psychometrics`  – reliability (Cronbach's alpha), EFA, item-total correlations.
- :mod:`ml_sad.statistics`     – group comparisons (MANOVA, Mann-Whitney U, effect sizes).
- :mod:`ml_sad.classification` – ML screening models and diagnostic-accuracy metrics.
- :mod:`ml_sad.visualization`  – figures used in the paper.
"""

__version__ = "1.0.0"

STRATEGIES = ("S", "A", "R")  # Suppression, Avoidance, Reappraisal
STRATEGY_NAMES = {"S": "Suppression", "A": "Avoidance", "R": "Reappraisal"}
N_SCENARIOS = 10

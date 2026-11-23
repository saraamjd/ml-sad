"""Machine-learning screening models and diagnostic-accuracy metrics.

Four classifiers are compared on the 30 emotion-regulation items to screen for
Social Anxiety Disorder (SAD): k-NN, SVM (RBF), Perceptron and Random Forest.
Two evaluation views are produced:

1. Cross-validated performance (repeated stratified 5-fold) – the honest
   estimate of generalisation.
2. A single held-out confusion matrix per model plus the full battery of
   diagnostic-test-accuracy (DTA) metrics (sensitivity, specificity, PPV, NPV,
   likelihood ratios, diagnostic odds ratio, Youden's J), for direct comparison
   with the SPIN self-report screener.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import (
    RepeatedStratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

DEFAULT_SEED = 49


def build_models(seed: int = DEFAULT_SEED) -> dict:
    return {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": SVC(kernel="rbf", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced_subsample", random_state=seed),
        "Perceptron": Perceptron(random_state=seed),
    }


def cross_validated_scores(X, y, seed: int = DEFAULT_SEED,
                           n_splits: int = 5, n_repeats: int = 3) -> pd.DataFrame:
    """Repeated stratified k-fold scores for every model."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=1)
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for name, model in build_models(seed).items():
        row = {"Classifier": name}
        for m in metrics:
            scoring = m if m != "roc_auc" else "roc_auc"
            try:
                row[m] = float(np.mean(
                    cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)))
            except Exception:
                row[m] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _dta_metrics(tp, fp, fn, tn) -> dict:
    """Diagnostic-test-accuracy metrics from a 2x2 count table (SAD = positive)."""
    def safe(n, d):
        return n / d if d else np.nan

    sens = safe(tp, tp + fn)
    spec = safe(tn, tn + fp)
    ppv = safe(tp, tp + fp)
    npv = safe(tn, tn + fn)
    acc = safe(tp + tn, tp + fp + fn + tn)
    f1 = safe(2 * tp, 2 * tp + fp + fn)
    lr_pos = safe(sens, 1 - spec) if spec not in (None, np.nan) else np.nan
    lr_neg = safe(1 - sens, spec) if spec else np.nan
    dor = safe(tp * tn, fp * fn)
    youden = (sens + spec - 1) if not (np.isnan(sens) or np.isnan(spec)) else np.nan
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Sensitivity": sens, "Specificity": spec, "PPV": ppv, "NPV": npv,
        "Accuracy": acc, "F1": f1, "LR+": lr_pos, "LR-": lr_neg,
        "DOR": dor, "Youden_J": youden,
    }


def confusion_matrices(X, y, seed: int = DEFAULT_SEED,
                       test_size: float = 0.30) -> pd.DataFrame:
    """Held-out 2x2 confusion matrix and DTA metrics for each classifier.

    All models share the same stratified train/test split so their confusion
    matrices are directly comparable.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    rows = []
    for name, model in build_models(seed).items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
        rows.append({"Classifier": name, **_dta_metrics(tp, fp, fn, tn)})
    return pd.DataFrame(rows)


def round_report(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    num = df.select_dtypes(include="number").columns
    out = df.copy()
    out[num] = out[num].round(digits)
    return out

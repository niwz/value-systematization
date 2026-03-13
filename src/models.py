"""Statistical models for pilot analysis."""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import DELTA_FEATURE_NAMES


def prepare_Xy(
    df: pd.DataFrame,
    log_scale: bool = False,
    interactions: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature matrix X and binary target y from results dataframe.

    Target: 1 if original_choice == 'A', 0 if 'B'.

    Args:
        log_scale: Apply np.sign(x) * np.log1p(np.abs(x)) to magnitude delta columns.
        interactions: Append pairwise interaction terms for all moral features.
    """
    valid = df[df["original_choice"].isin(["A", "B"])].copy()
    feature_cols = [c for c in DELTA_FEATURE_NAMES if c in valid.columns]
    X = valid[feature_cols].values.astype(float)
    names = list(feature_cols)

    if log_scale:
        magnitude_cols = ["delta_benefit_magnitude", "delta_harm_magnitude"]
        for col_name in magnitude_cols:
            if col_name in feature_cols:
                idx = feature_cols.index(col_name)
                X[:, idx] = np.sign(X[:, idx]) * np.log1p(np.abs(X[:, idx]))

    if interactions:
        interaction_cols = []
        interaction_names = []
        for i, j in combinations(range(len(feature_cols)), 2):
            interaction_cols.append(X[:, i] * X[:, j])
            interaction_names.append(f"{feature_cols[i]}*{feature_cols[j]}")
        if interaction_cols:
            X = np.column_stack([X] + interaction_cols)
            names.extend(interaction_names)

    y = (valid["original_choice"] == "A").astype(int).values
    return X, y, names


def fit_logistic(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], cv_folds: int = 5
) -> dict:
    """Fit L1-regularized logistic regression with cross-validation.

    Uses a Pipeline to avoid scaling leakage across CV folds.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegressionCV(
            penalty="l1",
            solver="saga",
            cv=cv_folds,
            scoring="neg_log_loss",
            max_iter=5000,
            random_state=42,
        )),
    ])
    pipe.fit(X, y)

    # Cross-validated metrics (scaling happens inside each fold)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    ll_scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_log_loss")

    lr_model = pipe.named_steps["lr"]
    coef_table = pd.DataFrame({
        "feature": feature_names,
        "coefficient": lr_model.coef_[0],
        "nonzero": lr_model.coef_[0] != 0,
    }).sort_values("coefficient", key=abs, ascending=False)

    return {
        "model": pipe,
        "cv_accuracy_mean": acc_scores.mean(),
        "cv_accuracy_std": acc_scores.std(),
        "cv_logloss_mean": -ll_scores.mean(),
        "cv_logloss_std": ll_scores.std(),
        "n_nonzero": (lr_model.coef_[0] != 0).sum(),
        "coefficients": coef_table,
        "n_samples": len(y),
        "base_rate": y.mean(),
    }


def fit_decision_tree(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], cv_folds: int = 5
) -> dict:
    """Fit a shallow decision tree baseline."""
    model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    ll_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss")

    model.fit(X, y)

    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model": model,
        "cv_accuracy_mean": acc_scores.mean(),
        "cv_accuracy_std": acc_scores.std(),
        "cv_logloss_mean": -ll_scores.mean(),
        "cv_logloss_std": ll_scores.std(),
        "tree_depth": model.get_depth(),
        "n_leaves": model.get_n_leaves(),
        "feature_importances": importances,
        "n_samples": len(y),
    }


def fit_random_forest(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], cv_folds: int = 5
) -> dict:
    """Fit a random forest for nonlinear prediction with overfitting resistance."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    ll_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss")

    model.fit(X, y)

    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model": model,
        "cv_accuracy_mean": acc_scores.mean(),
        "cv_accuracy_std": acc_scores.std(),
        "cv_logloss_mean": -ll_scores.mean(),
        "cv_logloss_std": ll_scores.std(),
        "feature_importances": importances,
        "n_samples": len(y),
    }

"""Statistical models for pilot analysis."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .features import DELTA_FEATURE_NAMES


def prepare_Xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature matrix X and binary target y from results dataframe.

    Target: 1 if original_choice == 'A', 0 if 'B'.
    """
    valid = df[df["original_choice"].isin(["A", "B"])].copy()
    feature_cols = [c for c in DELTA_FEATURE_NAMES if c in valid.columns]
    X = valid[feature_cols].values.astype(float)
    y = (valid["original_choice"] == "A").astype(int).values
    return X, y, feature_cols


def fit_logistic(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], cv_folds: int = 5
) -> dict:
    """Fit L1-regularized logistic regression with cross-validation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        cv=cv_folds,
        scoring="neg_log_loss",
        max_iter=5000,
        random_state=42,
    )
    model.fit(X_scaled, y)

    # Cross-validated metrics
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    acc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
    ll_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="neg_log_loss")

    coef_table = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_[0],
        "nonzero": model.coef_[0] != 0,
    }).sort_values("coefficient", key=abs, ascending=False)

    return {
        "model": model,
        "scaler": scaler,
        "cv_accuracy_mean": acc_scores.mean(),
        "cv_accuracy_std": acc_scores.std(),
        "cv_logloss_mean": -ll_scores.mean(),
        "cv_logloss_std": ll_scores.std(),
        "n_nonzero": (model.coef_[0] != 0).sum(),
        "coefficients": coef_table,
        "n_samples": len(y),
        "base_rate": y.mean(),
    }


def fit_decision_tree(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], cv_folds: int = 5
) -> dict:
    """Fit a shallow decision tree baseline."""
    model = DecisionTreeClassifier(max_depth=3, random_state=42)

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

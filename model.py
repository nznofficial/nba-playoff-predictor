"""
model.py — Train and evaluate logistic regression for NBA playoff game prediction.

Uses temporal train/test split (not random) to prevent future data leakage.
TimeSeriesSplit for cross-validation. Saves fitted pipeline with joblib.
"""

import logging
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report
)

log = logging.getLogger(__name__)

MODEL_FEATURES = [
    "net_rtg_diff",
    "off_rtg_diff",
    "def_rtg_diff",
    "pace_diff",
    "rest_days_diff",
    "home_court",
    "win_pct_diff",
    "series_game_num",
    "series_lead",
    "three_pt_rate_diff",
    "playoff_exp_diff",
    "is_bubble",
    "series_pts_diff",
    "prior_playoff_pts_diff",
]

# Seasons used for temporal split
TRAIN_SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22",
]
TEST_SEASONS = ["2022-23", "2023-24"]


def _prepare_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Fill missing feature values with 0 and return only feature columns."""
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        log.warning(f"Features not in DataFrame (filling with 0): {missing}")
    result = df[available].copy()
    for f in missing:
        result[f] = 0
    return result[features]


def train_model(
    training_df: pd.DataFrame,
    features: list = None,
    target: str = "won",
    model_path: str = "models/logistic_model.joblib",
) -> tuple:
    """
    Train logistic regression on training_df using a temporal train/test split.
    Returns (fitted_pipeline, metrics_dict).
    Saves pipeline to model_path.
    """
    if features is None:
        features = MODEL_FEATURES

    Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)

    # Temporal split: train on earlier seasons, test on recent
    train_df = training_df[training_df["season"].isin(TRAIN_SEASONS)]
    test_df = training_df[training_df["season"].isin(TEST_SEASONS)]

    log.info(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    if len(train_df) == 0:
        raise RuntimeError("No training data found. Check season labels match TRAIN_SEASONS.")
    if len(test_df) == 0:
        log.warning("No test data — evaluating on training data only.")
        test_df = train_df

    X_train = _prepare_features(train_df, features)
    y_train = train_df[target].astype(int)
    X_test = _prepare_features(test_df, features)
    y_test = test_df[target].astype(int)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logisticregression", LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, solver="lbfgs"
        )),
    ])
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    test_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_auc": float(roc_auc_score(y_test, test_prob)),
        "classification_report": classification_report(y_test, test_pred),
        "y_test": y_test.values,
        "y_prob": test_prob,
        "feature_names": features,
    }

    # Extract coefficients
    coef = pipeline.named_steps["logisticregression"].coef_[0]
    metrics["feature_importances"] = dict(zip(features, coef.tolist()))

    joblib.dump(pipeline, model_path)
    log.info(f"Model saved to {model_path}")
    log.info(f"Train accuracy: {metrics['train_accuracy']:.3f}")
    log.info(f"Test accuracy:  {metrics['test_accuracy']:.3f}")
    log.info(f"Test AUC:       {metrics['test_auc']:.3f}")

    return pipeline, metrics


def cross_validate_model(
    training_df: pd.DataFrame,
    features: list = None,
    target: str = "won",
    n_splits: int = 5,
) -> dict:
    """
    Perform TimeSeriesSplit cross-validation.
    Data must be sorted by season/game_date before passing in.
    Returns mean/std accuracy and AUC across folds.
    """
    if features is None:
        features = MODEL_FEATURES

    df_sorted = training_df.sort_values(["season", "game_date"]).reset_index(drop=True)
    X = _prepare_features(df_sorted, features)
    y = df_sorted[target].astype(int)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_accuracies = []
    cv_aucs = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("logisticregression", LogisticRegression(
                C=1.0, max_iter=1000, random_state=42, solver="lbfgs"
            )),
        ])
        pipe.fit(X_tr, y_tr)

        preds = pipe.predict(X_val)
        probs = pipe.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, preds)
        try:
            auc = roc_auc_score(y_val, probs)
        except ValueError:
            auc = 0.5  # only one class in fold

        cv_accuracies.append(float(acc))
        cv_aucs.append(float(auc))
        log.info(f"Fold {fold+1}: accuracy={acc:.3f}, AUC={auc:.3f}")

    result = {
        "cv_accuracies": cv_accuracies,
        "mean_accuracy": float(np.mean(cv_accuracies)),
        "std_accuracy": float(np.std(cv_accuracies)),
        "cv_auc_scores": cv_aucs,
        "mean_auc": float(np.mean(cv_aucs)),
        "std_auc": float(np.std(cv_aucs)),
    }
    log.info(f"CV mean accuracy: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}")
    log.info(f"CV mean AUC:      {result['mean_auc']:.3f} ± {result['std_auc']:.3f}")
    return result


def load_model(model_path: str = "models/logistic_model.joblib") -> Pipeline:
    """Load fitted sklearn Pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run main.py to train first."
        )
    return joblib.load(model_path)


def print_model_report(model: Pipeline, metrics: dict, feature_names: list = None) -> None:
    """Pretty-print model performance and logistic regression coefficients."""
    if feature_names is None:
        feature_names = metrics.get("feature_names", MODEL_FEATURES)

    print("\n" + "=" * 60)
    print("NBA PLAYOFF PREDICTOR — MODEL REPORT")
    print("=" * 60)
    print(f"Train Accuracy: {metrics['train_accuracy']:.3f}")
    print(f"Test Accuracy:  {metrics['test_accuracy']:.3f}")
    print(f"Test AUC:       {metrics['test_auc']:.3f}")
    print()
    print("Classification Report (Test Set):")
    print(metrics["classification_report"])

    print("Feature Coefficients (positive = favors win):")
    print(f"{'Feature':<25} {'Coefficient':>12}")
    print("-" * 38)
    coef_items = sorted(
        metrics["feature_importances"].items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    for feat, coef in coef_items:
        sign = "+" if coef >= 0 else ""
        print(f"  {feat:<23} {sign}{coef:>10.4f}")
    print("=" * 60 + "\n")

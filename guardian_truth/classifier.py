from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NON_FEATURE_COLUMNS = {
    "sample_id",
    "prompt",
    "response",
    "label",
    "error",
    "response_token_ids",
    "response_tokens",
}


@dataclass
class TrainArtifacts:
    pipeline: Pipeline
    feature_columns: List[str]
    metrics: Dict[str, float]


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    work = df.copy()

    if "error" in work.columns:
        work = work[work["error"].isna() | (work["error"] == "")]

    work = work[work["label"].notna()].copy()
    work["label"] = work["label"].astype(int)

    feature_columns = select_feature_columns(work)
    X = work[feature_columns].copy()
    y = work["label"].to_numpy(dtype=int)

    return X, y, feature_columns


def train_logreg(
    df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.2,
    max_iter: int = 2000,
) -> TrainArtifacts:
    X, y, feature_columns = prepare_xy(df)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=max_iter, random_state=random_state)),
        ]
    )

    pipeline.fit(X_train, y_train)

    valid_scores = pipeline.predict_proba(X_valid)[:, 1]
    valid_pred = (valid_scores >= 0.5).astype(int)

    pr_auc = average_precision_score(y_valid, valid_scores)

    precision, recall, thresholds = precision_recall_curve(y_valid, valid_scores)
    best_f1 = 0.0
    best_threshold = 0.5

    for i, thr in enumerate(thresholds):
        p = precision[i]
        r = recall[i]
        if (p + r) > 0:
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(thr)

    report = classification_report(y_valid, valid_pred, output_dict=True, zero_division=0)

    metrics = {
        "pr_auc": float(pr_auc),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "valid_accuracy": float(report["accuracy"]),
        "valid_precision_1": float(report["1"]["precision"]),
        "valid_recall_1": float(report["1"]["recall"]),
        "valid_f1_1": float(report["1"]["f1-score"]),
    }

    return TrainArtifacts(
        pipeline=pipeline,
        feature_columns=feature_columns,
        metrics=metrics,
    )


def save_artifacts(artifacts: TrainArtifacts, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": artifacts.pipeline,
            "feature_columns": artifacts.feature_columns,
            "metrics": artifacts.metrics,
        },
        path,
    )


def load_artifacts(path: str | Path) -> dict:
    return joblib.load(path)
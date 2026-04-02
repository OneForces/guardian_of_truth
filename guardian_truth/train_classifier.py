from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from guardian_truth.classifier import save_artifacts, train_logreg
from guardian_truth.config import AppConfig, ensure_project_dirs


def main() -> None:
    ensure_project_dirs()
    cfg = AppConfig()

    features_path = cfg.paths.train_features_csv
    model_path = cfg.paths.model_artifact
    metrics_path = Path(model_path).with_suffix(".metrics.json")

    print("Loading features from:", features_path)
    df = pd.read_csv(features_path)

    artifacts = train_logreg(
        df=df,
        random_state=cfg.train.random_state,
        test_size=cfg.train.test_size,
        max_iter=cfg.train.max_iter,
    )

    save_artifacts(artifacts, model_path)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(artifacts.metrics, f, ensure_ascii=False, indent=2)

    print("Model saved to:", model_path)
    print("Metrics saved to:", metrics_path)
    print("Feature columns:", len(artifacts.feature_columns))
    print("Metrics:")
    for k, v in artifacts.metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import pandas as pd

from guardian_truth.classifier import load_artifacts
from guardian_truth.config import AppConfig


def main() -> None:
    cfg = AppConfig()

    artifacts = load_artifacts(cfg.paths.model_artifact)
    pipeline = artifacts["pipeline"]
    feature_columns = artifacts["feature_columns"]

    clf = pipeline.named_steps["clf"]
    coefs = clf.coef_[0]

    df = pd.DataFrame(
        {
            "feature": feature_columns,
            "coef": coefs,
            "abs_coef": abs(coefs),
        }
    ).sort_values("abs_coef", ascending=False)

    print("TOP 30 features:")
    print(df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
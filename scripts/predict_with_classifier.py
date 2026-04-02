from __future__ import annotations

import pandas as pd

from guardian_truth.classifier import load_artifacts
from guardian_truth.config import AppConfig
from guardian_truth.features import FeatureExtractor
from guardian_truth.modeling import ModelWrapper


def main() -> None:
    cfg = AppConfig()

    artifacts = load_artifacts(cfg.paths.model_artifact)
    pipeline = artifacts["pipeline"]
    feature_columns = artifacts["feature_columns"]
    metrics = artifacts["metrics"]

    print("Loaded metrics:", metrics)

    wrapper = ModelWrapper(cfg)
    wrapper.load()

    extractor = FeatureExtractor(wrapper)

    prompt = "Вопрос: Кто написал роман 'Война и мир'?\nОтвет:"
    response = " Роман написал Александр Пушкин."

    result = extractor.extract(prompt=prompt, response=response)

    row = {col: 0.0 for col in feature_columns}
    for k, v in result.features.items():
        if k in row:
            row[k] = v

    X = pd.DataFrame([row], columns=feature_columns)

    proba = float(pipeline.predict_proba(X)[:, 1][0])
    threshold = float(metrics.get("best_threshold", 0.5))
    pred = int(proba >= threshold)

    print("hallucination_probability:", proba)
    print("threshold:", threshold)
    print("predicted_label:", pred)
    print("predicted_text:", "hallucination" if pred == 1 else "not_hallucination")


if __name__ == "__main__":
    main()
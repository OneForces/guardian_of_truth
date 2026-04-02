from __future__ import annotations

from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from guardian_truth.config import AppConfig, ensure_project_dirs
from guardian_truth.dataset import load_samples_from_csv
from guardian_truth.features import FeatureExtractor
from guardian_truth.modeling import ModelWrapper


def main() -> None:
    ensure_project_dirs()
    cfg = AppConfig()

    input_path = cfg.paths.public_bench_csv
    output_path = cfg.paths.train_features_csv

    print("Loading samples from:", input_path)
    samples = load_samples_from_csv(input_path)
    print("Num samples:", len(samples))

    wrapper = ModelWrapper(cfg)
    wrapper.load()

    extractor = FeatureExtractor(wrapper)

    rows: List[Dict[str, object]] = []

    for sample in tqdm(samples, desc="Extracting features"):
        try:
            result = extractor.extract(prompt=sample.prompt, response=sample.response)

            row: Dict[str, object] = {
                "sample_id": sample.sample_id,
                "prompt": sample.prompt,
                "response": sample.response,
                "label": sample.label,
                "response_token_ids": " ".join(map(str, result.response_token_ids)),
                "response_tokens": " | ".join(result.response_tokens),
            }
            row.update(result.features)
            rows.append(row)

        except Exception as e:
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "prompt": sample.prompt,
                    "response": sample.response,
                    "label": sample.label,
                    "error": str(e),
                }
            )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print("Saved features to:", output_path)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))


if __name__ == "__main__":
    main()
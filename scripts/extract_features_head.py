from __future__ import annotations

from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from guardian_truth.config import AppConfig
from guardian_truth.dataset import load_samples_from_csv
from guardian_truth.features import FeatureExtractor
from guardian_truth.modeling import ModelWrapper


def main() -> None:
    cfg = AppConfig()
    samples = load_samples_from_csv(cfg.paths.public_bench_csv)[:5]

    print("Num samples:", len(samples))

    wrapper = ModelWrapper(cfg)
    wrapper.load()

    extractor = FeatureExtractor(wrapper)

    rows: List[Dict[str, object]] = []

    for sample in tqdm(samples, desc="Extracting HEAD features"):
        result = extractor.extract(prompt=sample.prompt, response=sample.response)
        row: Dict[str, object] = {
            "sample_id": sample.sample_id,
            "label": sample.label,
        }
        row.update(result.features)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.head())
    print("columns:", df.columns.tolist())


if __name__ == "__main__":
    main()
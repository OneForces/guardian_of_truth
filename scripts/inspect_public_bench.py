from __future__ import annotations

from guardian_truth.config import AppConfig
from guardian_truth.dataset import inspect_csv_dataset


def main() -> None:
    cfg = AppConfig()
    info = inspect_csv_dataset(cfg.paths.public_bench_csv)

    print("dataset_path:", info.path)
    print("num_rows:", info.num_rows)
    print("columns:", info.columns)


if __name__ == "__main__":
    main()
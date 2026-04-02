from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from guardian_truth.classifier import load_artifacts
from guardian_truth.config import AppConfig, ensure_project_dirs
from guardian_truth.features import FeatureExtractor
from guardian_truth.modeling import ModelWrapper


def detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final hallucination prediction pipeline")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Путь к входному CSV",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/submission.csv",
        help="Путь к выходному submission CSV",
    )

    parser.add_argument(
        "--id-col",
        type=str,
        default="",
        help="Имя колонки с id, если нужно указать вручную",
    )

    parser.add_argument(
        "--prompt-col",
        type=str,
        default="",
        help="Имя колонки с prompt, если нужно указать вручную",
    )

    parser.add_argument(
        "--response-col",
        type=str,
        default="",
        help="Имя колонки с ответом модели, если нужно указать вручную",
    )

    # 🔥 НОВОЕ: быстрый режим
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Ограничить количество строк (для быстрого теста)",
    )

    return parser


def main() -> None:
    ensure_project_dirs()
    cfg = AppConfig()

    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {input_path}")

    print("Loading artifacts...")
    artifacts = load_artifacts(cfg.paths.model_artifact)
    pipeline = artifacts["pipeline"]
    feature_columns = artifacts["feature_columns"]

    print("Loading model wrapper...")
    wrapper = ModelWrapper(cfg)
    wrapper.load()
    extractor = FeatureExtractor(wrapper)

    print("Reading input CSV:", input_path)
    df = pd.read_csv(input_path)

    # 🔥 FAST MODE
    if args.limit > 0:
        df = df.head(args.limit)
        print(f"[FAST MODE] Using only first {len(df)} rows")

    id_col = args.id_col.strip() or detect_column(df, ["id", "sample_id", "uid"])
    prompt_col = args.prompt_col.strip() or detect_column(df, ["prompt", "question", "input", "query"])
    response_col = args.response_col.strip() or detect_column(
        df,
        ["response", "answer", "output", "generation", "model_answer"],
    )

    if prompt_col is None:
        raise ValueError(
            f"Не найдена колонка prompt/question/input/query. Колонки: {list(df.columns)}"
        )

    if response_col is None:
        raise ValueError(
            f"Не найдена колонка response/answer/output/generation/model_answer. "
            f"Колонки: {list(df.columns)}"
        )

    if id_col is None:
        print("Колонка id не найдена, будет использован индекс строки.")
    else:
        print("Using id column:", id_col)

    print("Using prompt column:", prompt_col)
    print("Using response column:", response_col)

    submission_rows: List[Dict[str, object]] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        sample_id = row[id_col] if id_col is not None else idx
        prompt = "" if pd.isna(row[prompt_col]) else str(row[prompt_col])
        response = "" if pd.isna(row[response_col]) else str(row[response_col])

        try:
            result = extractor.extract(prompt=prompt, response=response)

            feature_row = {col: 0.0 for col in feature_columns}
            for k, v in result.features.items():
                if k in feature_row:
                    feature_row[k] = v

            X = pd.DataFrame([feature_row], columns=feature_columns)
            score = float(pipeline.predict_proba(X)[:, 1][0])

        except Exception as e:
            print(f"[WARN] sample_id={sample_id} failed: {e}")
            score = 0.5  # fallback

        submission_rows.append(
            {
                "id": sample_id,
                "score": score,
            }
        )

    submission_df = pd.DataFrame(submission_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False, encoding="utf-8")

    print("Saved submission to:", output_path)
    print(submission_df.head())


if __name__ == "__main__":
    main()
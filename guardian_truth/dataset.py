from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from guardian_truth.schemas import Sample


@dataclass
class DatasetInfo:
    path: Path
    num_rows: int
    columns: List[str]


def _detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_samples_from_csv(path: str | Path) -> List[Sample]:
    path = Path(path)
    df = pd.read_csv(path)

    prompt_col = _detect_column(
        df,
        ["prompt", "question", "input", "query"],
    )
    response_col = _detect_column(
        df,
        ["response", "answer", "output", "generation", "model_answer"],
    )
    label_col = _detect_column(
        df,
        ["label", "target", "y", "is_hallucination", "hallucination"],
    )
    id_col = _detect_column(
        df,
        ["id", "sample_id", "uid"],
    )

    if prompt_col is None:
        raise ValueError(
            f"Не найдена колонка prompt/question/input/query в {path}. "
            f"Колонки: {list(df.columns)}"
        )

    if response_col is None:
        raise ValueError(
            f"Не найдена колонка response/answer/output/generation/model_answer в {path}. "
            f"Колонки: {list(df.columns)}"
        )

    samples: List[Sample] = []

    for i, row in df.iterrows():
        prompt = "" if pd.isna(row[prompt_col]) else str(row[prompt_col])
        response = "" if pd.isna(row[response_col]) else str(row[response_col])

        label = None
        if label_col is not None and not pd.isna(row[label_col]):
            try:
                label = int(row[label_col])
            except Exception:
                val = str(row[label_col]).strip().lower()
                if val in {"true", "yes", "1"}:
                    label = 1
                elif val in {"false", "no", "0"}:
                    label = 0
                else:
                    label = None

        sample_id = None
        if id_col is not None and not pd.isna(row[id_col]):
            sample_id = str(row[id_col])
        else:
            sample_id = str(i)

        samples.append(
            Sample(
                prompt=prompt,
                response=response,
                label=label,
                sample_id=sample_id,
            )
        )

    return samples


def inspect_csv_dataset(path: str | Path) -> DatasetInfo:
    path = Path(path)
    df = pd.read_csv(path)
    return DatasetInfo(
        path=path,
        num_rows=len(df),
        columns=list(df.columns),
    )
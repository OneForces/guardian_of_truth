from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass
class ModelConfig:
    model_name_or_path: str = str(Path.home() / "models" / "GigaChat3-10B-A1.8B-bf16")
    tokenizer_name_or_path: Optional[str] = None
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = False
    max_length: int = 2048
    output_hidden_states: bool = True


@dataclass
class FeatureConfig:
    probe_layers: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 20, 24])
    top_k_logits: int = 5
    low_conf_threshold: float = 0.2
    answer_min_tokens: int = 1


@dataclass
class TrainConfig:
    random_state: int = 42
    test_size: float = 0.2
    classifier_type: str = "logreg"
    max_iter: int = 2000


@dataclass
class PathsConfig:
    train_csv: Path = RAW_DATA_DIR / "train.csv"
    valid_csv: Path = RAW_DATA_DIR / "valid.csv"
    public_bench_csv: Path = RAW_DATA_DIR / "knowledge_bench_public.csv"
    features_parquet: Path = INTERIM_DATA_DIR / "features.parquet"
    train_features_csv: Path = INTERIM_DATA_DIR / "train_features.csv"
    model_artifact: Path = MODELS_DIR / "hallucination_detector.joblib"


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def ensure_project_dirs() -> None:
    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
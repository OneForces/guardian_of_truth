from dataclasses import dataclass
from typing import Optional

@dataclass
class Sample:
    prompt: str
    response: str
    label: Optional[int] = None
    sample_id: Optional[str] = None

@dataclass
class PredictionResult:
    hallucination_probability: float
    predicted_label: int
    latency_ms: float

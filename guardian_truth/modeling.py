from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from guardian_truth.config import AppConfig


@dataclass
class ForwardBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_len: int
    response_len: int
    full_text: str


@dataclass
class ForwardOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    logits: torch.Tensor
    hidden_states: List[torch.Tensor]
    prompt_len: int
    response_len: int
    response_token_mask: torch.Tensor


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_name.lower().strip()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[key]


class ModelWrapper:
    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.model_cfg = self.config.model
        self.tokenizer = None
        self.model = None
        self.device = self.model_cfg.device
        self.dtype = _resolve_torch_dtype(self.model_cfg.torch_dtype)

    def _validate_local_path(self, path: str, label: str) -> None:
        if not path or not os.path.isdir(path):
            raise FileNotFoundError(
                f"{label} not found: {path}. "
                "Модель не скачалась в образ или путь указан неверно."
            )

    def load(self) -> None:
        logging.getLogger("transformers").setLevel(logging.ERROR)

        model_name = self.model_cfg.model_name_or_path
        tokenizer_name = self.model_cfg.tokenizer_name_or_path or model_name

        self._validate_local_path(model_name, "Model path")
        self._validate_local_path(tokenizer_name, "Tokenizer path")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.model_cfg.trust_remote_code,
            local_files_only=True,
        )

        target_device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"

        if target_device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=self.model_cfg.trust_remote_code,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            self.model = self.model.to("cuda")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=self.model_cfg.trust_remote_code,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            self.model = self.model.to("cpu")

        self.model.eval()

    def build_batch(self, prompt: str, response: str) -> ForwardBatch:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded. Call load() first.")

        prompt = prompt if prompt is not None else ""
        response = response if response is not None else ""

        full_text = prompt + response

        prompt_enc = self.tokenizer(prompt, return_tensors="pt")
        full_enc = self.tokenizer(full_text, return_tensors="pt")

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        prompt_len = int(prompt_enc["input_ids"].shape[1])
        full_len = int(input_ids.shape[1])
        response_len = max(full_len - prompt_len, 0)

        return ForwardBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_len=prompt_len,
            response_len=response_len,
            full_text=full_text,
        )

    @torch.no_grad()
    def forward(self, prompt: str, response: str) -> ForwardOutput:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        batch = self.build_batch(prompt=prompt, response=response)

        device = next(self.model.parameters()).device
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        seq_len = int(input_ids.shape[1])
        response_token_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

        start_idx = min(batch.prompt_len, seq_len)
        end_idx = min(batch.prompt_len + batch.response_len, seq_len)

        if end_idx > start_idx:
            response_token_mask[start_idx:end_idx] = True

        return ForwardOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits=outputs.logits,
            hidden_states=list(outputs.hidden_states),
            prompt_len=batch.prompt_len,
            response_len=batch.response_len,
            response_token_mask=response_token_mask,
        )

    def decode_tokens(self, input_ids: torch.Tensor) -> List[str]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded. Call load() first.")

        ids = input_ids.detach().cpu().tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]

        return [self.tokenizer.decode([token_id]) for token_id in ids]

    def get_vocab_size(self) -> int:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
        return int(self.model.config.vocab_size)
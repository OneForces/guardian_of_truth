from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from guardian_truth.modeling import ForwardOutput, ModelWrapper


@dataclass
class FeatureExtractionResult:
    features: Dict[str, float]
    response_token_ids: List[int]
    response_tokens: List[str]


def _safe_float(x: torch.Tensor | float | int) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    if x.numel() == 0:
        return 0.0
    return float(x.detach().float().cpu().item())


def _tensor_stats(prefix: str, values: torch.Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }

    values = values.detach().float()
    std = values.std(unbiased=False) if values.numel() > 1 else torch.tensor(0.0, device=values.device)

    return {
        f"{prefix}_mean": _safe_float(values.mean()),
        f"{prefix}_std": _safe_float(std),
        f"{prefix}_min": _safe_float(values.min()),
        f"{prefix}_max": _safe_float(values.max()),
    }


class FeatureExtractor:
    def __init__(self, model_wrapper: ModelWrapper) -> None:
        self.model_wrapper = model_wrapper

    def extract_from_forward_output(self, out: ForwardOutput) -> FeatureExtractionResult:
        """
        Извлекаем признаки только по токенам ответа.
        Для causal LM логиты на позиции t предсказывают токен t+1,
        поэтому для ответа используем позиции [start-1, end-1].
        """
        input_ids = out.input_ids[0]              # [seq_len]
        logits = out.logits[0]                    # [seq_len, vocab]
        hidden_states = out.hidden_states         # list[[1, seq_len, hidden]]
        mask = out.response_token_mask            # [seq_len]

        response_positions = torch.where(mask)[0]
        features: Dict[str, float] = {}

        if response_positions.numel() == 0:
            features["response_num_tokens"] = 0.0
            features["prompt_len"] = float(out.prompt_len)
            features["response_len"] = float(out.response_len)
            features["vocab_size"] = float(logits.shape[-1])
            return FeatureExtractionResult(
                features=features,
                response_token_ids=[],
                response_tokens=[],
            )

        # Позиции токенов ответа
        start = int(response_positions[0].item())
        end = int(response_positions[-1].item()) + 1

        # Для токена ответа на позиции p нужен логит с позиции p-1
        valid_response_positions = response_positions[response_positions > 0]

        if valid_response_positions.numel() == 0:
            features["response_num_tokens"] = 0.0
            features["prompt_len"] = float(out.prompt_len)
            features["response_len"] = float(out.response_len)
            features["vocab_size"] = float(logits.shape[-1])
            return FeatureExtractionResult(
                features=features,
                response_token_ids=[],
                response_tokens=[],
            )

        logit_positions = valid_response_positions - 1
        target_token_ids = input_ids[valid_response_positions]

        selected_logits = logits[logit_positions]  # [n_resp, vocab]
        log_probs = F.log_softmax(selected_logits, dim=-1)
        probs = torch.softmax(selected_logits, dim=-1)

        token_log_probs = log_probs.gather(
            dim=-1,
            index=target_token_ids.unsqueeze(-1),
        ).squeeze(-1)

        token_probs = token_log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)

        top1_probs, top1_ids = probs.max(dim=-1)
        top2_probs = probs.topk(k=2, dim=-1).values[:, 1]
        top1_gap = top1_probs - top2_probs

        is_top1_correct = (top1_ids == target_token_ids).float()

        # Базовые признаки
        features["prompt_len"] = float(out.prompt_len)
        features["response_len"] = float(out.response_len)
        features["response_num_tokens"] = float(valid_response_positions.numel())
        features["vocab_size"] = float(logits.shape[-1])

        # Uncertainty признаки
        features.update(_tensor_stats("token_logprob", token_log_probs))
        features.update(_tensor_stats("token_prob", token_probs))
        features.update(_tensor_stats("entropy", entropy))
        features.update(_tensor_stats("top1_prob", top1_probs))
        features.update(_tensor_stats("top1_gap", top1_gap))
        features.update(_tensor_stats("top1_correct", is_top1_correct))

        features["low_conf_ratio_p_lt_0_1"] = _safe_float((token_probs < 0.1).float().mean())
        features["low_conf_ratio_p_lt_0_2"] = _safe_float((token_probs < 0.2).float().mean())
        features["low_conf_ratio_p_lt_0_5"] = _safe_float((token_probs < 0.5).float().mean())
        features["high_entropy_ratio"] = _safe_float((entropy > entropy.mean()).float().mean())

        # Динамика по ответу
        if token_probs.numel() > 1:
            prob_diff = token_probs[1:] - token_probs[:-1]
            entropy_diff = entropy[1:] - entropy[:-1]
            features.update(_tensor_stats("token_prob_diff", prob_diff))
            features.update(_tensor_stats("entropy_diff", entropy_diff))
        else:
            features.update(_tensor_stats("token_prob_diff", torch.tensor([], device=token_probs.device)))
            features.update(_tensor_stats("entropy_diff", torch.tensor([], device=token_probs.device)))

        # Hidden-state признаки по последнему слою
        last_hidden = hidden_states[-1][0]  # [seq_len, hidden]
        response_hidden = last_hidden[valid_response_positions]  # [n_resp, hidden]

        hidden_norms = torch.norm(response_hidden.float(), dim=-1)
        features.update(_tensor_stats("hidden_norm_last", hidden_norms))

        # Hidden-state признаки по нескольким слоям
        candidate_layers = [0, len(hidden_states) // 2, len(hidden_states) - 1]
        candidate_layers = sorted(set(candidate_layers))

        layer_means = []
        for layer_idx in candidate_layers:
            h = hidden_states[layer_idx][0][valid_response_positions].float()   # [n_resp, hidden]
            h_norm = torch.norm(h, dim=-1)
            features.update(_tensor_stats(f"hidden_norm_layer_{layer_idx}", h_norm))
            layer_means.append(h.mean(dim=0))

        # Косинусная близость между усреднёнными представлениями слоёв
        if len(layer_means) >= 2:
            for i in range(len(layer_means) - 1):
                a = layer_means[i].unsqueeze(0)
                b = layer_means[i + 1].unsqueeze(0)
                cos = F.cosine_similarity(a, b).squeeze(0)
                features[f"layer_cosine_{i}_{i+1}"] = _safe_float(cos)

        response_token_ids = target_token_ids.detach().cpu().tolist()
        response_tokens = self.model_wrapper.decode_tokens(target_token_ids)

        return FeatureExtractionResult(
            features=features,
            response_token_ids=response_token_ids,
            response_tokens=response_tokens,
        )

    def extract(self, prompt: str, response: str) -> FeatureExtractionResult:
        out = self.model_wrapper.forward(prompt=prompt, response=response)
        return self.extract_from_forward_output(out)
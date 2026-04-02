from __future__ import annotations

import torch

from guardian_truth.modeling import ModelWrapper


def main() -> None:
    wrapper = ModelWrapper()
    wrapper.load()

    prompt = "Вопрос: Кто написал роман 'Война и мир'?\nОтвет:"
    response = " Лев Толстой."

    out = wrapper.forward(prompt=prompt, response=response)

    print("model_device:", next(wrapper.model.parameters()).device)
    print("input_ids shape:", tuple(out.input_ids.shape))
    print("logits shape:", tuple(out.logits.shape))
    print("hidden_states:", len(out.hidden_states))
    print("prompt_len:", out.prompt_len)
    print("response_len:", out.response_len)
    print("response_mask_sum:", int(out.response_token_mask.sum().item()))

    probs = torch.softmax(out.logits[:, :-1, :], dim=-1)
    print("probs shape:", tuple(probs.shape))


if __name__ == "__main__":
    main()
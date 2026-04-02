from __future__ import annotations

from guardian_truth.modeling import ModelWrapper
from guardian_truth.features import FeatureExtractor


def main() -> None:
    wrapper = ModelWrapper()
    wrapper.load()

    extractor = FeatureExtractor(wrapper)

    prompt = "Вопрос: Кто написал роман 'Война и мир'?\nОтвет:"
    response = " Лев Толстой."

    result = extractor.extract(prompt=prompt, response=response)

    print("response_token_ids:", result.response_token_ids)
    print("response_tokens:", result.response_tokens)
    print("num_features:", len(result.features))

    for k in sorted(result.features.keys())[:30]:
        print(f"{k}: {result.features[k]}")


if __name__ == "__main__":
    main()
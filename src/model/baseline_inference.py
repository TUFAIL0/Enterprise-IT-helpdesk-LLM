
import truststore

# Use the macOS certificate store for Hugging Face requests.
truststore.inject_into_ssl()

from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


MODEL_NAME = "mlx-community/Qwen2.5-1.5B-4bit"
MAX_TOKENS = 120
TEMPERATURE = 0.0

RESULTS_PATH = Path("experiments/baseline_results.md")

PROMPTS = [
    "A Linux server becomes unresponsive after a restart. What should an administrator check first?",
    "A Linux machine cannot reach an internal service. What diagnostic steps should be performed?",
    "A user receives 'Permission denied' when accessing a file on Linux. How should this be investigated?",
]


def main():
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load(MODEL_NAME)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as results_file:
        results_file.write("# Baseline Inference Results\n\n")
        results_file.write(f"- Model: `{MODEL_NAME}`\n")
        results_file.write("- Framework: MLX-LM\n")
        results_file.write("- Quantization: 4-bit\n")
        results_file.write(f"- Maximum generated tokens: {MAX_TOKENS}\n")
        results_file.write(f"- Sampling temperature: {TEMPERATURE}\n")

        for index, prompt in enumerate(PROMPTS, start=1):
            print(f"\n{'=' * 70}")
            print(f"Baseline prompt {index}: {prompt}")
            print("-" * 70)

            sampler = make_sampler(temp=TEMPERATURE)

            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                sampler=sampler,
            )

            print(response)

            results_file.write(f"\n## Prompt {index}\n\n")
            results_file.write(f"**Question:** {prompt}\n\n")
            results_file.write("**Model response:**\n\n")
            results_file.write(response.strip() + "\n\n")
            results_file.flush()

    print(f"\nBaseline results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

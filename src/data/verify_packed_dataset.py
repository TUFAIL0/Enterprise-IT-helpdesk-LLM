import truststore
truststore.inject_into_ssl()
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DATASET_PATH = Path("data/packed/train.parquet")
SEQUENCE_LENGTH = 1024


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = load_dataset(
        "parquet",
        data_files=str(DATASET_PATH),
        split="train",
    )

    lengths = [len(row) for row in dataset["input_ids"]]
    max_token_id = max(
        max(row) for row in dataset["input_ids"]
    )

    print("\n=== Packed Dataset Verification ===")
    print(f"Dataset file: {DATASET_PATH}")
    print(f"Number of sequences: {len(dataset)}")
    print(f"Expected sequence length: {SEQUENCE_LENGTH}")
    print(f"All sequences have expected length: {all(n == SEQUENCE_LENGTH for n in lengths)}")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"Maximum token ID in dataset: {max_token_id}")
    print(f"All token IDs fit tokenizer vocabulary: {max_token_id < len(tokenizer)}")

    print("\n=== Sample Decoded Text ===")
    sample_ids = dataset[0]["input_ids"][:80]
    print(tokenizer.decode(sample_ids))

    if not all(n == SEQUENCE_LENGTH for n in lengths):
        raise ValueError("Some sequences have an unexpected length.")

    if max_token_id >= len(tokenizer):
        raise ValueError("A token ID is outside the tokenizer vocabulary.")

    print("\nDataset verification passed.")


if __name__ == "__main__":
    main()

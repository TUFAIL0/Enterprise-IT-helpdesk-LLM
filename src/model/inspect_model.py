from transformers import AutoTokenizer, AutoConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B"


from transformers import AutoConfig, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B"


def inspect_model():
    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\n=== Model Architecture ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Architecture: {config.architectures}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Transformer layers: {config.num_hidden_layers}")
    print(f"Attention heads: {config.num_attention_heads}")
    print(
        "Key-value heads: "
        f"{getattr(config, 'num_key_value_heads', 'Not specified')}"
    )
    print(
        "Head dimension: "
        f"{config.hidden_size // config.num_attention_heads}"
    )
    print(f"Config vocabulary size: {config.vocab_size}")
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"Tokenizer length: {len(tokenizer)}")
    print(
        "Maximum position embeddings: "
        f"{getattr(config, 'max_position_embeddings', 'Not specified')}"
    )


if __name__ == "__main__":
    inspect_model()

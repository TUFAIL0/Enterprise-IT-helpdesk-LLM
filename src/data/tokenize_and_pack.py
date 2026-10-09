from pathlib import Path
import truststore

truststore.inject_into_ssl()
from transformers import AutoTokenizer
from datasets import Dataset


""" Model name and Dataset configuration"""
MODEL_NAME = "Qwen/Qwen2.5-1.5B"

CLEANED_DIR = Path("data/cleaned")
PACKED_DIR = Path("data/packed")

SEQUENCE_LENGTH = 1024

def load_tokenizer():
    """Local the tokenizer with selected model"""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Tokenizer loaded: {MODEL_NAME}")
    print(f"Vocablary size: {tokenizer.vocab_size}")
    print(f"BOS token: {tokenizer.bos_token}")
    print(f"EOS token: {tokenizer.eos_token}")

    return tokenizer


def load_documents():
    """Load all cleaned text documents."""
    documents = []

    for path in sorted(CLEANED_DIR.rglob("*.txt")):
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).strip()

        if text:
            documents.append({
                "source": str(path),
                "text": text,
            })

    if not documents:
        raise ValueError(
            f"No non-empty .txt documents found in {CLEANED_DIR}"
        )

    print(f"Documents loaded: {len(documents)}")
    for document in documents:
        print(
            f"  {document['source']}: "
            f"{len(document['text'].split())} words"
        )

    return documents


def tokenize_and_pack(documents, tokenizer):
   """ Tokenize the document and pack them into fix length sequence"""
   all_token_ids = []
   document_token_count = []
   packed_sequence = []

   for document in documents:
      token_id = tokenizer.encode(
         document["text"],
         add_special_tokens = False
      )

      if tokenizer.eos_token_id is not None:
         token_id.append(tokenizer.eos_token_id)

      document_token_count.append(len(token_id)) 
      all_token_ids.extend(token_id)

   total_tokens = len(all_token_ids)
   print(f"Total tokens before packing: {total_tokens}")
   print(f"Required tokens for one sequence: {SEQUENCE_LENGTH}")
   for i in range(0,total_tokens, SEQUENCE_LENGTH):
      sequence = all_token_ids[i: i + SEQUENCE_LENGTH]
      packed_sequence.append(sequence)

   """Keep only full length Sequence"""
   packed_sequences = [
      sequence 
      for sequence in packed_sequence
      if len(sequence) == SEQUENCE_LENGTH
   ]

   if not packed_sequences:
      raise ValueError(
          "No full sequences were created. "
          "Check the corpus size or sequence length."
        )

   print(f"Total token before packing: {total_tokens}")
   print("Average token per document:" f"{sum(document_token_count)/len(document_token_count)}:.2f)")
   print(f"Sequence length: {SEQUENCE_LENGTH}")
   print(f"Packed sequences: {len(packed_sequences)}")
   print(
      f"Tokens discarded from final partial chunk: "
      f"{total_tokens % SEQUENCE_LENGTH}"
      )

   return packed_sequences



def save_packed_dataset(packed_sequences):
    """Save fixed-length token sequences as a Parquet dataset."""
    PACKED_DIR.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_dict({
        "input_ids": packed_sequences,
    })

    output_path = PACKED_DIR / "train.parquet"
    dataset.to_parquet(str(output_path))

    print(f"Dataset saved to: {output_path}")
    print(f"Saved rows: {len(dataset)}")


if __name__ == "__main__":
   tokenizer = load_tokenizer()
   documents = load_documents()
   packed_sequence = tokenize_and_pack(documents, tokenizer)
   save_packed_dataset(packed_sequences=packed_sequence)
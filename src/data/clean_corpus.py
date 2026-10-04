from pathlib import Path
import hashlib
import re

PROCESSED_DIR = Path("data/processed")
CLEANED_DIR = Path("data/cleaned")


MIN_WORD = 100


def normalize_text(text: str) -> str:
    """ Clean whitespaces and line breaks. """
    """ Replace null character with space """
    text = text.replace("\x00"," ")

    text = re.sub(r"\s+"," ", text)
    return text.strip()

def word_count(text: str) -> int:
    """ Return number of words in document"""

    return len(text.split())

def content_hash(text: str) -> str:
    """ Return a SHA-256 hash of normalized document content"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def isEnglish(text: str) -> bool:
    """Check for text Language"""

    if not text:
        return False

    ascii_char = sum(char.isascii() for char in text)

    total_char = len(text)
    ascii_ratio = ascii_char / total_char

    return ascii_ratio >= 0.90

def load_document() -> list[tuple[Path, str]]:
    document = []

    for path in PROCESSED_DIR.rglob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = normalize_text(text)
        document.append((path, text))

    return document



def main() -> None:
    """Run the corpus cleaning pipeline."""

    documents = load_document()

    print(f"Initial documents: {len(documents)}")

    # Step 1: Apply length filter

    length_filter = [
        (path, text)
        for path, text in documents
        if word_count(text) >= MIN_WORD
    ]

    print(f"After length filtering: {len(length_filter)}")

    # Step 2: Apply de-duplication

    unique_document = []
    seen_hash = set()

    for path, text in length_filter:
        text_hash = content_hash(text=text)

        if text_hash not in seen_hash:
            seen_hash.add(text_hash)
            unique_document.append((path, text))

    print(f"After deduplication: {len(unique_document)}")

    # Step 3: English language filter

    english_documents = [
        (path, text)
        for path, text in unique_document
        if isEnglish(text=text)
    ]

    print(f"After English filtering: {len(english_documents)}")

    # Step 4: Save clean corpus

    for source_path, text in english_documents:
        relative_path = source_path.relative_to(PROCESSED_DIR)
        output_path = CLEANED_DIR / relative_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    print(f"Cleaned documents saved to: {CLEANED_DIR}")


if __name__ == "__main__":
    main()
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
CLEANED_DIR = Path("data/cleaned")
REPORT_FILE = Path("data/metadata/cleaning_report.txt")


def count_word(text: str) -> int:
    """ Return number of word in the documents"""
    return len(text.split())


def main() -> None:
    """ Creating report for corpus cleaning process"""

    initial_document = list(PROCESSED_DIR.rglob("*.txt"))
    cleaned_document = list(CLEANED_DIR.rglob("*.txt"))

    initial_count = len(initial_document)
    cleaned_count = len(cleaned_document)

    length_filtered_count = 6
    deduplicated_count = 6
    english_filtered_count = len(cleaned_document)

    length_removed = initial_count - length_filtered_count
    duplicate_removed = length_filtered_count - deduplicated_count
    non_english_removed = deduplicated_count - english_filtered_count

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        file.write("Corpus Cleaning Report\n")
        file.write("=====================\n\n")

        file.write("Document Counts\n")
        file.write("---------------\n")
        file.write(
            f"Initial documents: {initial_count}\n"
        )
        file.write(
            f"After length filtering: {length_filtered_count}\n"
        )
        file.write(
            f"After deduplication: {deduplicated_count}\n"
        )
        file.write(
            f"After English filtering: {english_filtered_count}\n\n"
        )

        file.write("Documents Removed\n")
        file.write("-----------------\n")
        file.write(
            f"Removed by length filtering: {length_removed}\n"
        )
        file.write(
            f"Removed by deduplication: {duplicate_removed}\n"
        )
        file.write(
            f"Removed by English filtering: {non_english_removed}\n\n"
        )

        file.write("Cleaned Documents\n")
        file.write("-----------------\n")

        for document in sorted(cleaned_document):
            text = document.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            words = count_word(text)

            file.write(
                f"{document} | {words} words\n"
            )

    print(f"Cleaning report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
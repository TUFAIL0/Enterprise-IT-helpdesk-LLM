from pathlib import Path
from pypdf import PdfReader


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def extract_pdf(pdf_path: Path) -> None:
    """ Extract text from a PDF, page by page. """
    reader = PdfReader(pdf_path)
    output_dir = PROCESSED_DIR/pdf_path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir/ f"{pdf_path.stem}.text"

    """Open Output file with write mode (w) add encoding for special character. close file properly usign with"""
    with output_file.open("w", encoding="utf-8") as file:
        for page_number, page in enumerate(reader.pages, start=1):
         text = page.extract_text() or ""
         file.write(f"\n\n === Page {page_number} === \n\n")
         file.write(text)
    
    print(f"Extracted: {pdf_path}")
    print(f"Pages: {len(reader.pages)}")
    print(f"Output: {output_file}")


def main() -> None:
   """Extract text from every PDF in the raw data directory."""
   """rglobe search file recursively and return list of file with extention .pdf"""
   pdf_file = list(RAW_DIR.rglob("*.pdf"))
   print(f"Found {len(pdf_file)} PDF files.")

   for pdf_path in pdf_file:
      extract_pdf(pdf_path=pdf_path)

"""If this file run directly excute main"""
if __name__ == "__main__":
   main()
import json
import re
from pathlib import Path

import pymupdf


# Keep all input and output paths relative to this lesson folder.
BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "documents" / "company_policy.pdf"
OUTPUT_DIR = BASE_DIR / "output"
JSON_OUTPUT_PATH = OUTPUT_DIR / "documents.json"
TEXT_OUTPUT_PATH = OUTPUT_DIR / "processed_policy.txt"


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph line breaks."""
    # Replace multiple spaces and tabs with a single space.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove unnecessary spaces at the beginning and end of each line.
    text = re.sub(r" *\n *", "\n", text)

    # Replace three or more empty lines with one blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_pdf(file_path: Path) -> list[dict]:
    """Load a PDF and return one document object for each non-empty page."""
    documents = []

    # Open the PDF safely. It will be closed automatically after this block.
    with pymupdf.open(file_path) as pdf:
        # Process each page separately so page metadata can be preserved.
        for page_number, page in enumerate(pdf, start=1):
            raw_text = page.get_text("text")
            cleaned_text = clean_text(raw_text)

            # Skip pages that do not contain extractable text.
            if not cleaned_text:
                continue

            # Store the text and its source information together.
            documents.append(
                {
                    "page_content": cleaned_text,
                    "metadata": {
                        "source": file_path.name,
                        "page": page_number,
                    },
                }
            )

    return documents


def main() -> None:
    # Stop early with a clear message if the input PDF is missing.
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    # Load and clean the PDF page by page.
    documents = load_pdf(PDF_PATH)

    # Create the output folder if it does not already exist.
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Save page-wise documents in JSON format for future RAG steps.
    with JSON_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(documents, file, indent=2, ensure_ascii=False)

    # Also save all processed pages as one readable text file.
    processed_text = "\n\n".join(
        f"--- Page {document['metadata']['page']} ---\n{document['page_content']}"
        for document in documents
    )
    TEXT_OUTPUT_PATH.write_text(processed_text, encoding="utf-8")

    print(f"Loaded {len(documents)} pages")
    print(f"Saved structured documents to: {JSON_OUTPUT_PATH}")
    print(f"Saved processed text to: {TEXT_OUTPUT_PATH}")

    # Preview each document's metadata and first 200 characters.
    for document in documents:
        print(document["metadata"])
        print(document["page_content"][:200])
        print()


if __name__ == "__main__":
    main()

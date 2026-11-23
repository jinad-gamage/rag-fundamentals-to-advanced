
import hashlib
import re

import pymupdf

from .config import MAX_FILE_SIZE, MAX_PAGES


def clean_text(text: str) -> str:
    """Normalize whitespace without removing meaningful text."""

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove repeated spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def calculate_document_hash(pdf_bytes: bytes) -> str:
    """Create a stable identifier for PDF contents."""

    return hashlib.sha256(pdf_bytes).hexdigest()


def load_pdf(
    pdf_bytes: bytes,
    filename: str,
) -> tuple[str, list[dict]]:
    """Extract text and metadata from an uploaded PDF."""

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise ValueError(
            "PDF exceeds the 10 MB upload limit."
        )

    # Identify the document using its content hash.
    document_id = calculate_document_hash(pdf_bytes)

    documents = []

    # Open the uploaded PDF directly from memory.
    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    ) as pdf:

        if pdf.page_count > MAX_PAGES:
            raise ValueError(
                f"PDF exceeds the {MAX_PAGES}-page limit."
            )

        # Extract text from every page.
        for page_number, page in enumerate(
            pdf,
            start=1,
        ):

            raw_text = page.get_text("text")

            cleaned_text = clean_text(raw_text)

            # Skip pages without extractable text.
            if not cleaned_text:
                continue

            documents.append(
                {
                    "page_content": cleaned_text,
                    "metadata": {
                        "source": filename,
                        "page": page_number,
                        "document_id": document_id,
                    },
                }
            )

    return document_id, documents

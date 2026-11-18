
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Keep all paths relative to the lesson folders.
BASE_DIR = Path(__file__).parent

# Read the processed documents from Lesson 02.
INPUT_PATH = (
    BASE_DIR.parent
    / "Lesson-02-Document-Loading-Text-Processing"
    / "output"
    / "documents.json"
)

# Output files for Lesson 03.
OUTPUT_DIR = BASE_DIR / "output"
JSON_OUTPUT_PATH = OUTPUT_DIR / "chunks.json"
TEXT_OUTPUT_PATH = OUTPUT_DIR / "chunks.txt"

# Chunking configuration (measured in characters).
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def load_documents(file_path: Path) -> list[dict]:
    """Load processed documents generated in Lesson 02."""

    # Check whether the input JSON file exists.
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input JSON file not found: {file_path}"
        )

    # Read the JSON file.
    with file_path.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    # Validate the expected JSON structure.
    if not isinstance(documents, list):
        raise ValueError("Expected a list of documents.")

    return documents

def create_chunks(documents: list[dict]) -> list[dict]:
    """Split documents into chunks and preserve metadata."""

    chunks = []

    # Configure recursive character splitting.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Process each document page separately.
    for document in documents:

        page_content = document["page_content"]
        metadata = document["metadata"]

        # Skip empty documents.
        if not page_content.strip():
            continue

        # Split page content into smaller chunks.
        text_chunks = text_splitter.split_text(page_content)

        # Store each generated chunk with metadata.
        for chunk_number, chunk_text in enumerate(
            text_chunks, start=1
        ):

            # Create an identifier for the chunk.
            chunk_id = (
                f"{Path(metadata['source']).stem}"
                f"_page_{metadata['page']}"
                f"_chunk_{chunk_number}"
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page_content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_number": chunk_number,
                    },
                }
            )

    return chunks

def save_chunks(chunks: list[dict]) -> None:
    """Save chunks in JSON and readable text formats."""

    # Create output directory if it does not exist.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save structured chunks as JSON.
    with JSON_OUTPUT_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Create a readable text representation.
    processed_text = "\n\n".join(
        f"--- {chunk['chunk_id']} ---\n"
        f"{chunk['page_content']}"
        for chunk in chunks
    )

    # Save readable chunks as TXT.
    TEXT_OUTPUT_PATH.write_text(
        processed_text,
        encoding="utf-8"
    )

def main() -> None:

    # Load processed documents from Lesson 02.
    documents = load_documents(INPUT_PATH)

    # Split documents into chunks.
    chunks = create_chunks(documents)

    # Save generated chunks.
    save_chunks(chunks)

    # Display summary.
    print(f"Loaded {len(documents)} documents")
    print(f"Generated {len(chunks)} chunks")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")

    print(f"Saved JSON to: {JSON_OUTPUT_PATH}")
    print(f"Saved text to: {TEXT_OUTPUT_PATH}")

    # Preview each generated chunk.
    for chunk in chunks:
        print()
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Length: {len(chunk['page_content'])} characters")
        print(chunk["page_content"][:200])

if __name__ == "__main__":
    main()
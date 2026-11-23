
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from .config import CHUNK_SIZE, CHUNK_OVERLAP


def create_chunks(documents: list[dict]) -> list[dict]:
    """Split extracted PDF pages into smaller chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for document in documents:

        page_content = document["page_content"]
        metadata = document["metadata"]

        # Split each page independently.
        text_chunks = splitter.split_text(
            page_content
        )

        for chunk_number, chunk_text in enumerate(
            text_chunks,
            start=1,
        ):

            # Create a stable unique chunk ID.
            chunk_id = (
                f"{metadata['document_id']}"
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

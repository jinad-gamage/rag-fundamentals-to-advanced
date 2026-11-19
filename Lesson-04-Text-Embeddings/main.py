
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# Keep paths relative to the lesson folders.
BASE_DIR = Path(__file__).parent

# Read the chunks generated in Lesson 03.
INPUT_PATH = (
    BASE_DIR.parent
    / "Lesson-03-Text-Chunking"
    / "output"
    / "chunks.json"
)

# Output configuration.
OUTPUT_DIR = BASE_DIR / "output"
EMBEDDINGS_OUTPUT_PATH = OUTPUT_DIR / "embeddings.json"

# Embedding model configuration.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Number of relevant chunks to retrieve.
TOP_K = 3


def load_chunks(file_path: Path) -> list[dict]:
    """Load the text chunks generated in Lesson 03."""

    # Check whether the input JSON file exists.
    if not file_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {file_path}"
        )

    # Read the JSON file.
    with file_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    # Validate the JSON structure.
    if not isinstance(chunks, list):
        raise ValueError("Expected a list of chunks.")

    if not chunks:
        raise ValueError("No chunks found in the input file.")

    return chunks


def load_embedding_model() -> SentenceTransformer:
    """Load the pre-trained sentence embedding model."""

    print(f"Loading embedding model: {MODEL_NAME}")

    # The model is downloaded automatically if not cached.
    model = SentenceTransformer(MODEL_NAME)

    return model


def generate_embeddings(
    chunks: list[dict],
    model: SentenceTransformer,
) -> tuple[list[dict], np.ndarray]:
    """Generate embeddings for all document chunks."""

    # Extract text from each chunk.
    texts = [chunk["page_content"] for chunk in chunks]

    # Convert all chunk texts into numerical vectors.
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Combine embeddings with their original metadata.
    embedded_chunks = []

    for chunk, vector in zip(chunks, vectors):
        embedded_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page_content": chunk["page_content"],
                "metadata": chunk["metadata"],
                "embedding": vector.tolist(),
            }
        )

    return embedded_chunks, vectors


def save_embeddings(embedded_chunks: list[dict]) -> None:
    """Save generated embeddings in JSON format."""

    # Create the output directory.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the embeddings and metadata.
    with EMBEDDINGS_OUTPUT_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "model": MODEL_NAME,
                "dimension": len(
                    embedded_chunks[0]["embedding"]
                ),
                "chunks": embedded_chunks,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


def cosine_similarity(
    query_vector: np.ndarray,
    document_vectors: np.ndarray,
) -> np.ndarray:
    """Calculate cosine similarity for normalized vectors."""

    # Both query and document vectors are normalized.
    # Their dot product is therefore cosine similarity.
    return document_vectors @ query_vector


def semantic_search(
    query: str,
    model: SentenceTransformer,
    chunks: list[dict],
    document_vectors: np.ndarray,
    top_k: int = TOP_K,
) -> list[dict]:
    """Retrieve the most semantically similar chunks."""

    # Convert the user question into an embedding.
    query_vector = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Calculate similarity with every document chunk.
    similarities = cosine_similarity(
        query_vector,
        document_vectors,
    )

    # Get indices of the highest similarity scores.
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    # Collect the most relevant chunks.
    for index in top_indices:
        chunk = chunks[int(index)]

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page_content": chunk["page_content"],
                "metadata": chunk["metadata"],
                "similarity_score": float(similarities[index]),
            }
        )

    return results


def main() -> None:

    # Load document chunks from Lesson 03.
    chunks = load_chunks(INPUT_PATH)

    print(f"Loaded {len(chunks)} chunks")

    # Load the embedding model.
    model = load_embedding_model()

    # Generate document embeddings.
    embedded_chunks, document_vectors = generate_embeddings(
        chunks,
        model,
    )

    # Save embeddings for later lessons.
    save_embeddings(embedded_chunks)

    print(f"Generated {len(embedded_chunks)} embeddings")
    print(f"Embedding dimensions: {document_vectors.shape[1]}")
    print(f"Saved embeddings to: {EMBEDDINGS_OUTPUT_PATH}")

    # Ask a question about the company policy.
    query = "How many annual leave days do employees receive?"

    print(f"\nUser Question: {query}")

    # Find the most relevant document chunks.
    results = semantic_search(
        query,
        model,
        chunks,
        document_vectors,
    )

    # Display retrieved results.
    print("\nTop Relevant Chunks:")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Similarity: {result['similarity_score']:.4f}")
        print(f"Metadata: {result['metadata']}")
        print(f"Content:\n{result['page_content']}")


if __name__ == "__main__":
    main()

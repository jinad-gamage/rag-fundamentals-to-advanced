
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Keep all paths relative to the lesson folders.
BASE_DIR = Path(__file__).resolve().parent

# Read embeddings generated in Lesson 04.
INPUT_PATH = (
    BASE_DIR.parent
    / "Lesson-04-Text-Embeddings"
    / "output"
    / "embeddings.json"
)

# Persistent vector database directory.
DB_PATH = BASE_DIR / "chroma_db"

# Database configuration.
COLLECTION_NAME = "company_policies"

# Use the same model from Lesson 04.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Number of results to retrieve.
TOP_K = 3


def load_embeddings(file_path: Path) -> dict:
    """Load the embeddings generated in Lesson 04."""

    # Check whether the input file exists.
    if not file_path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {file_path}"
        )

    # Load the JSON data.
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Validate the expected structure.
    if not isinstance(data, dict):
        raise ValueError("Expected an embeddings JSON object.")

    if not data.get("chunks"):
        raise ValueError("No embeddings found in the input file.")

    # Ensure the embedding model matches Lesson 04.
    if data.get("model") != MODEL_NAME:
        raise ValueError(
            "Embedding model mismatch. "
            "Use the same model as Lesson 04."
        )

    return data


def initialize_database():
    """Initialize a persistent ChromaDB collection."""

    # Create a persistent ChromaDB client.
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # Create the collection if it does not exist.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
    )

    return collection


def store_embeddings(collection, embedded_chunks: list[dict]) -> None:
    """Store text chunks and pre-generated embeddings in ChromaDB."""

    # Extract the required fields.
    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in embedded_chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["page_content"])
        embeddings.append(chunk["embedding"])
        metadatas.append(chunk["metadata"])

    # Insert or update records using their IDs.
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Stored {len(ids)} chunks in ChromaDB")


def semantic_search(
    collection,
    model: SentenceTransformer,
    query: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """Search the vector database using a query embedding."""

    # Convert the user question into an embedding.
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Ensure the requested result count is valid.
    result_count = min(top_k, collection.count())

    if result_count == 0:
        return []

    # Search ChromaDB using the query embedding.
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=result_count,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    # Convert ChromaDB results into readable records.
    retrieved_chunks = []

    for index, chunk_id in enumerate(results["ids"][0]):
        retrieved_chunks.append(
            {
                "chunk_id": chunk_id,
                "page_content": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": results["distances"][0][index],
            }
        )

    return retrieved_chunks


def main() -> None:

    # Load pre-generated embeddings from Lesson 04.
    data = load_embeddings(INPUT_PATH)

    embedded_chunks = data["chunks"]

    print(f"Loaded {len(embedded_chunks)} embeddings")

    # Initialize the persistent vector database.
    collection = initialize_database()

    # Store the document embeddings.
    store_embeddings(collection, embedded_chunks)

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total stored records: {collection.count()}")
    print(f"Database location: {DB_PATH}")

    # Load the same embedding model used in Lesson 04.
    model = SentenceTransformer(MODEL_NAME)

    # Ask a question about the company policy.
    query = "How many vacation days do employees receive?"

    print(f"\nUser Question: {query}")

    # Retrieve relevant chunks.
    results = semantic_search(
        collection,
        model,
        query,
    )

    # Display search results.
    print("\nTop Relevant Chunks:")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Metadata: {result['metadata']}")
        print(f"Content:\n{result['page_content']}")


if __name__ == "__main__":
    main()

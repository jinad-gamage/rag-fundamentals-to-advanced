from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

# Keep all paths relative to the lesson folders.
BASE_DIR = Path(__file__).resolve().parent

# Reuse the persistent ChromaDB database from Lesson 05.
DB_PATH = (
    BASE_DIR.parent
    / "Lesson-05-Vector-Databases"
    / "chroma_db"
)

# Output configuration.
OUTPUT_DIR = BASE_DIR / "output"
CONTEXT_OUTPUT_PATH = OUTPUT_DIR / "retrieved_context.txt"

# Database and embedding model configuration.
COLLECTION_NAME = "company_policies"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval configuration.
CANDIDATE_K = 5
TOP_K = 3
MIN_SIMILARITY = 0.40

def load_vector_database() -> Collection:
    """Connect to the persistent ChromaDB collection."""

    # Prevent accidentally creating a new empty database.
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {DB_PATH}\n"
            "Run the Lesson 05 application first."
        )

    # Connect to the existing persistent database.
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # Retrieve the collection created in Lesson 05.
    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # Check whether the collection contains records.
    if collection.count() == 0:
        raise ValueError(
            "The collection is empty. "
            "Run the Lesson 05 application first."
        )

    return collection

def load_embedding_model() -> SentenceTransformer:
    """Load the same embedding model used in previous lessons."""

    print(f"Loading embedding model: {MODEL_NAME}")

    return SentenceTransformer(MODEL_NAME)

def retrieve_candidates(
    query: str,
    collection: Collection,
    model: SentenceTransformer,
    candidate_k: int = CANDIDATE_K,
    source_filter: str | None = None,
) -> list[dict]:
    """Retrieve candidate chunks from ChromaDB."""

    # Convert the user question into a normalized embedding.
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Limit the requested number of candidates.
    result_count = min(candidate_k, collection.count())

    # Optional metadata filtering.
    where_filter = None

    if source_filter:
        where_filter = {
            "source": source_filter
        }

    # Search the vector database.
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=result_count,
        where=where_filter,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    candidates = []

    # Convert ChromaDB results into structured records.
    for index, chunk_id in enumerate(results["ids"][0]):

        distance = float(results["distances"][0][index])

        # Lesson 05 uses cosine distance.
        similarity = 1.0 - distance

        candidates.append(
            {
                "chunk_id": chunk_id,
                "page_content": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": distance,
                "similarity": similarity,
            }
        )

    return candidates

def select_relevant_chunks(
    candidates: list[dict],
    min_similarity: float = MIN_SIMILARITY,
    top_k: int = TOP_K,
) -> list[dict]:
    """Filter and select the most relevant chunks."""

    # Remove candidates below the similarity threshold.
    relevant_chunks = [
        chunk
        for chunk in candidates
        if chunk["similarity"] >= min_similarity
    ]

    # Sort candidates by similarity, highest first.
    relevant_chunks.sort(
        key=lambda chunk: chunk["similarity"],
        reverse=True,
    )

    # Select the final Top-K chunks.
    return relevant_chunks[:top_k]

def build_context(chunks: list[dict]) -> str:
    """Build an LLM-ready context from selected chunks."""

    if not chunks:
        return "No sufficiently relevant information was found."

    context_parts = []

    # Add source information before each chunk.
    for index, chunk in enumerate(chunks, start=1):

        metadata = chunk["metadata"]

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")

        context_part = (
            f"[Context {index}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Chunk ID: {chunk['chunk_id']}\n\n"
            f"{chunk['page_content']}"
        )

        context_parts.append(context_part)

    # Separate individual context sections.
    return "\n\n---\n\n".join(context_parts)

def save_context(context: str) -> None:
    """Save the selected context in a readable text file."""

    # Create the output directory if necessary.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the context for the next lesson.
    CONTEXT_OUTPUT_PATH.write_text(
        context,
        encoding="utf-8",
    )

def main() -> None:

    # Connect to the database created in Lesson 05.
    collection = load_vector_database()

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored chunks: {collection.count()}")

    # Load the embedding model.
    model = load_embedding_model()

    print("\nRAG Retrieval System")
    print("Type 'exit' to stop the application.")

    # Interactive question loop.
    while True:

        query = input("\nEnter your question: ").strip()

        if query.lower() == "exit":
            print("Exiting retrieval system.")
            break

        if not query:
            print("Please enter a valid question.")
            continue

        # Optional source filtering.
        source_filter = input(
            "Source filter (press Enter to skip): "
        ).strip()

        if not source_filter:
            source_filter = None

        # Step 1: Retrieve candidate chunks.
        candidates = retrieve_candidates(
            query=query,
            collection=collection,
            model=model,
            candidate_k=CANDIDATE_K,
            source_filter=source_filter,
        )

        print(f"\nRetrieved {len(candidates)} candidates")

        # Step 2: Apply threshold and Top-K selection.
        selected_chunks = select_relevant_chunks(
            candidates=candidates,
            min_similarity=MIN_SIMILARITY,
            top_k=TOP_K,
        )

        print(f"Selected {len(selected_chunks)} relevant chunks")

        # Step 3: Build context.
        context = build_context(selected_chunks)

        # Step 4: Save the context.
        save_context(context)

        # Display selected chunks.
        for rank, chunk in enumerate(selected_chunks, start=1):

            print(f"\n--- Result {rank} ---")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Similarity: {chunk['similarity']:.4f}")
            print(f"Source: {chunk['metadata'].get('source')}")
            print(f"Page: {chunk['metadata'].get('page')}")
            print(f"Content:\n{chunk['page_content']}")

        # Display the final context.
        print("\n--- Final Retrieved Context ---")
        print(context)

        print(f"\nContext saved to: {CONTEXT_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
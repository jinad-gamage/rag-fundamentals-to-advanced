
import chromadb

from .config import (
    DB_PATH,
    COLLECTION_NAME,
    CANDIDATE_K,
    TOP_K,
    MIN_SIMILARITY,
)

from .embeddings import generate_embeddings


def get_collection():
    """Connect to the persistent ChromaDB collection."""

    DB_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
    )

    return collection


def document_exists(
    collection,
    document_id: str,
) -> bool:
    """Check whether a document is already indexed."""

    results = collection.get(
        where={
            "document_id": document_id
        },
        limit=1,
        include=["metadatas"],
    )

    return len(results["ids"]) > 0


def store_chunks(
    chunks: list[dict],
) -> int:
    """Generate and store embeddings for document chunks."""

    if not chunks:
        return 0

    collection = get_collection()

    # Extract the required fields.
    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    texts = [
        chunk["page_content"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    # Generate embeddings.
    embeddings = generate_embeddings(texts)

    # Store or update records in ChromaDB.
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(ids)


def retrieve_chunks(
    question: str,
) -> list[dict]:
    """Retrieve relevant chunks using cosine similarity."""

    collection = get_collection()

    if collection.count() == 0:
        return []

    # Generate the query embedding.
    query_embedding = generate_embeddings(
        [question]
    )[0]

    candidate_count = min(
        CANDIDATE_K,
        collection.count(),
    )

    # Search the vector database.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    candidates = []

    for index, chunk_id in enumerate(
        results["ids"][0]
    ):

        distance = float(
            results["distances"][0][index]
        )

        # Convert cosine distance into similarity.
        similarity = 1.0 - distance

        # Apply similarity threshold filtering.
        if similarity < MIN_SIMILARITY:
            continue

        candidates.append(
            {
                "chunk_id": chunk_id,
                "page_content": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "similarity": similarity,
            }
        )

    # Sort by similarity.
    candidates.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    # Return the final Top-K chunks.
    return candidates[:TOP_K]


def clear_database() -> None:
    """Delete the collection and recreate it empty."""

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
    except chromadb.errors.NotFoundError:
        pass

    # Recreate the collection with cosine distance.
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
    )

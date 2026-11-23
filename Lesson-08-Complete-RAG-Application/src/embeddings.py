
import streamlit as st

from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model."""

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


def generate_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """Convert text into normalized embedding vectors."""

    model = load_embedding_model()

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return vectors.tolist()

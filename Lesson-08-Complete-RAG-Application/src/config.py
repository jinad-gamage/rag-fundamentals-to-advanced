
import os
from pathlib import Path

from dotenv import load_dotenv


# Project root directory.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables.
load_dotenv(BASE_DIR / ".env")

# Persistent ChromaDB directory.
DB_PATH = BASE_DIR / "data" / "chroma_db"

# Collection name.
COLLECTION_NAME = "rag_documents"

# Embedding model.
EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Gemini configuration.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)

# Chunking configuration.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval configuration.
CANDIDATE_K = 5
TOP_K = 3
MIN_SIMILARITY = 0.40

# Maximum PDF upload size: 10 MB.
MAX_FILE_SIZE = 10 * 1024 * 1024

# Maximum number of PDF pages per upload.
MAX_PAGES = 100

# bank_policy/config.py

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory for raw and processed documents
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Vector store directory
VECTORSTORE_DIR = BASE_DIR / "models" / "vectorstore"

# Hugging Face model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "google/flan-t5-small"

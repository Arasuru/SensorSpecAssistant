import os
from pathlib import Path

#base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "inputs"
VECTORS_DIR = DATA_DIR / "vector_store"

#file paths
PDF_FILENAME = "bst-bme280-ds002.pdf"
PDF_PATH = INPUT_DIR / PDF_FILENAME

#Model settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_KWARGS = {'device': 'cpu'}
LLM_MODEL_NAME = "llama-3.1-8b-instant"

#Text splitting settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

#Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)

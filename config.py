# config.py
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
LOGS_DIR = DATA_DIR / 'logs' / 'exhibition'

# Create directories if they don't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Colors
YOUR_WORDS_COLOR = 'blue'
VISITOR_WORDS_COLOR = 'orange'

# t-SNE Parameters
TSNE_COMPONENTS = 3
TSNE_PERPLEXITY = 30
TSNE_ITERATIONS = 1000

# API Configuration
API_HOST = '0.0.0.0'
API_PORT = 5000

# Embedding Model
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
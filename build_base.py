import os
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

"""
Direct pipeline: Text → Embeddings → t-SNE → JSON → Visualization
Base build only. Run once, then the exhibition runs cheaply.
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

# Configuration
INPUT_FILE = "data/raw/my_writing.txt"
CACHE_DIR = Path("data/cache")

YOUR_COLOR = "#2196f3"     # Blue
VISITOR_COLOR = "#ff9800"  # Orange (used later)

BASE_WORDS_JSON = CACHE_DIR / "base_words.json"
BASE_EMB_NPY = CACHE_DIR / "base_embeddings.npy"
BASE_POS_NPY = CACHE_DIR / "base_positions.npy"

def load_and_process_text(filepath):
    """Load your writing and extract words"""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    words = text.lower().split()

    seen = set()
    unique_words = []
    for word in words:
        word = word.strip('.,!?;:"()[]{}')
        if len(word) >= 2 and word not in seen:
            seen.add(word)
            unique_words.append(word)

    return unique_words

def create_embeddings(words):
    """Generate semantic embeddings (offline after first model download)"""
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Generating embeddings for {len(words)} words...")
    embeddings = model.encode(
        words,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine sim = dot product
        batch_size=64
    )
    return np.asarray(embeddings, dtype=np.float32)

def compute_tsne(embeddings, perplexity=40):
    """Reduce dimensions to 3D using t-SNE (base build only)"""
    print("Computing t-SNE projection...")

    tsne = TSNE(
        n_components=3,
        perplexity=min(perplexity, len(embeddings) - 1),
        random_state=42,
        n_iter=750,
        init="pca",
        learning_rate="auto",
        verbose=1
    )

    positions = tsne.fit_transform(embeddings).astype(np.float32)

    # Normalize to reasonable scale (-5 to 5)
    positions = positions - positions.mean(axis=0, keepdims=True)
    max_range = float(np.abs(positions).max()) or 1.0
    positions = (positions / max_range) * 5.0

    return positions

def save_cache(words, embeddings, positions):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASE_WORDS_JSON, "w", encoding="utf-8") as f:
        json.dump({"words": words}, f, indent=2)
    np.save(BASE_EMB_NPY, embeddings)
    np.save(BASE_POS_NPY, positions)
    print(f"✓ Cached base map to {CACHE_DIR}/")

def main():
    print("=" * 60)
    print("Direct Pipeline: Text → Embeddings → t-SNE → CACHE")
    print("=" * 60 + "\n")

    print("Step 1: Loading text...")
    words = load_and_process_text(INPUT_FILE)
    print(f"  ✓ Extracted {len(words)} unique words\n")

    print("Step 2: Creating embeddings...")
    embeddings = create_embeddings(words)
    print(f"  ✓ Generated {embeddings.shape[1]}-dimensional embeddings\n")

    print("Step 3: Computing t-SNE...")
    positions = compute_tsne(embeddings)
    print("  ✓ Reduced to 3D coordinates\n")

    print("Step 4: Saving cache...")
    save_cache(words, embeddings, positions)

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("Next: run server.py and open web/index.html")
    print("=" * 60)

if __name__ == "__main__":
    main()

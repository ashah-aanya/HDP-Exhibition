import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from openTSNE import TSNE

INPUT_FILE = "data/raw/my_writing.txt"
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_WORDS_JSON = CACHE_DIR / "base_words.json"
BASE_EMB_NPY = CACHE_DIR / "base_embeddings.npy"
FRAMES_NPY = CACHE_DIR / "frames.npy"

YOUR_COLOR = "#2196f3"

def load_words():
    text = Path(INPUT_FILE).read_text(encoding="utf-8")
    words = text.lower().split()
    seen, uniq = set(), []
    for w in words:
        w = w.strip('.,!?;:"()[]{}')
        if len(w) >= 2 and w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq

def normalize_positions(pos, target=5.0):
    pos = pos - pos.mean(axis=0, keepdims=True)
    mx = float(np.abs(pos).max()) or 1.0
    return (pos / mx) * target

def main():
    print("Loading words...")
    words = load_words()
    print(f"✓ {len(words)} words")

    print("Embedding...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(words, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)

    # cache base
    with open(BASE_WORDS_JSON, "w", encoding="utf-8") as f:
        json.dump({"words": words}, f, indent=2)
    np.save(BASE_EMB_NPY, emb)
    print("✓ Saved base words/embeddings")

    # Evolving TSNE frames (real optimization snapshots)
    # Keep these modest for stability/power: you can tune later.
    perplexity = min(30, len(words) - 1)
    total_iter = 1200          # total optimization iterations
    save_every = 20            # snapshot interval
    # => ~61 frames

    print("Computing evolving t-SNE frames (openTSNE)...")
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        random_state=42,
        n_jobs=1,
        verbose=True
    )

    embedding = tsne.fit(emb)  # initial + starts optimization

    frames = []
    # initial state
    frames.append(normalize_positions(np.asarray(embedding), target=5.0).astype(np.float32))

    it = 0
    while it < total_iter:
        embedding = embedding.optimize(n_iter=save_every, momentum=0.8)
        it += save_every
        frames.append(normalize_positions(np.asarray(embedding), target=5.0).astype(np.float32))
        print(f"  ✓ frame {len(frames)-1} @ iter {it}")

    frames = np.stack(frames, axis=0)  # (F, N, 3)
    np.save(FRAMES_NPY, frames)
    print(f"✓ Saved {frames.shape[0]} frames to {FRAMES_NPY}")
    print("Next: python server.py, then open web/index.html")

if __name__ == "__main__":
    main()

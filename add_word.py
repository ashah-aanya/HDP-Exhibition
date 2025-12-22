#!/usr/bin/env python3
"""
Interactive script to add words to the visualization
Adds words to my_writing.txt and regenerates the cluster
Type 'stop' to exit
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from openTSNE import TSNE
import sys
import os

# Disable tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def add_word_to_file(word):
    """Add a word to my_writing.txt"""
    with open('data/raw/my_writing.txt', 'a') as f:
        f.write(f'\n{word}')
    print(f"[1/4] ✓ Added '{word}' to my_writing.txt")

def regenerate_cluster():
    """Regenerate the t-SNE cluster with all words"""
    # Load all words
    print("[2/4] Loading all words from file...")
    with open('data/raw/my_writing.txt', 'r') as f:
        words = [line.strip() for line in f if line.strip()]
    print(f"      ✓ Loaded {len(words)} total words")

    # Load embedding model
    print("[3/4] Generating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(words, show_progress_bar=False)
    print(f"      ✓ Generated {len(embeddings)} embeddings")

    # Run t-SNE with faster settings
    print("[4/4] Running t-SNE clustering (faster mode, ~15-20 seconds)...")
    perplexity = min(30, len(words) - 1)
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        learning_rate=1500,
        random_state=42,
        n_jobs=4,  # Use multiple cores for speed
        verbose=False,
        initialization='pca',  # PCA initialization is faster than random
        early_exaggeration=12.0,
        n_iter=250,  # Reduced from 500 to 250 iterations
        early_exaggeration_iter=100  # Reduced from 250 to 100 iterations
    )

    positions = tsne.fit(embeddings)

    # Normalize positions
    positions_array = np.array(positions)
    positions_array = positions_array - positions_array.mean(axis=0)
    max_range = np.abs(positions_array).max()
    if max_range > 0:
        positions_array = (positions_array / max_range) * 5

    # Create exhibition data format
    # Last word is orange, all others are blue
    word_data = []
    for i, word in enumerate(words):
        color = '#ff9800' if i == len(words) - 1 else "#0E2233"
        word_data.append({'word': word, 'color': color})

    exhibition_data = {
        'words': word_data,
        'frames': [positions_array.tolist()]
    }

    # Save to web directory
    output_path = Path('web/exhibition_data.json')
    with open(output_path, 'w') as f:
        json.dump(exhibition_data, f, indent=2)

    print(f"      ✓ Cluster regenerated! New word '{words[-1]}' is now ORANGE")

def main():
    """Main function - interactive loop"""
    print("="*60)
    print("Interactive Word Addition Tool")
    print("="*60)
    print("Type words and press Enter to add them to the visualization.")
    print("Type 'stop' to exit.\n")

    word_count = 0

    while True:
        try:
            word = input("Enter a word: ").strip()

            if not word:
                print("⚠ Empty input, please enter a word\n")
                continue

            if word.lower() == 'stop':
                print("\n" + "="*60)
                print(f"Stopped. Added {word_count} word(s) total.")
                print("Restart the server to see your new words!")
                print("="*60)
                break

            word_count += 1
            print(f"\n--- Processing word #{word_count}: '{word}' ---")

            # Add word to file
            add_word_to_file(word)

            # Regenerate cluster
            regenerate_cluster()

            print("✓ Done!\n")

        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print(f"Interrupted. Added {word_count} word(s) total.")
            print("Restart the server to see your new words!")
            print("="*60)
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")
            continue

if __name__ == "__main__":
    main()

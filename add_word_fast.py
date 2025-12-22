#!/usr/bin/env python3
"""
Fast word addition - adds new word to existing t-SNE without full regeneration
Only regenerates when adding the first few words or periodically
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
    print(f"[1/3] ✓ Added '{word}' to my_writing.txt")

def add_word_fast(word):
    """Add word to existing visualization without full regeneration"""
    # Load existing data
    print("[2/3] Loading existing visualization data...")
    with open('web/exhibition_data.json', 'r') as f:
        data = json.load(f)

    existing_words = [item['word'] for item in data['words']]
    existing_positions = np.array(data['frames'][-1])

    # Check if word already exists
    if word.lower() in [w.lower() for w in existing_words]:
        print(f"⚠ Word '{word}' already exists!")
        return False

    print(f"[3/3] Computing position for new word (fast mode)...")

    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embedding for new word
    new_embedding = model.encode([word])[0]

    # Generate embeddings for existing words to rebuild t-SNE
    existing_embeddings = model.encode(existing_words, show_progress_bar=False)

    # Rebuild t-SNE from existing positions (very fast)
    from openTSNE import TSNEEmbedding
    from openTSNE.affinity import PerplexityBasedNN

    # Create affinity for existing embeddings
    perplexity = min(30, len(existing_embeddings) - 1)
    affinity = PerplexityBasedNN(existing_embeddings, perplexity=perplexity, random_state=42)

    # Create embedding from existing positions
    tsne_embedding = TSNEEmbedding(
        existing_positions,
        affinity,
        random_state=42,
        learning_rate=1500,
        n_jobs=1
    )

    # Transform new word (very fast - just places it, no full optimization)
    new_position = tsne_embedding.transform(np.array([new_embedding]))

    # Normalize all positions
    all_positions = np.vstack([existing_positions, new_position])
    all_positions = all_positions - all_positions.mean(axis=0)
    max_range = np.abs(all_positions).max()
    if max_range > 0:
        all_positions = (all_positions / max_range) * 5

    # Update word list and colors (last word is orange)
    all_words = existing_words + [word]
    word_data = []
    for i, w in enumerate(all_words):
        color = '#ff9800' if i == len(all_words) - 1 else "#0E2233"
        word_data.append({'word': w, 'color': color})

    # Save updated data
    exhibition_data = {
        'words': word_data,
        'frames': [all_positions.tolist()]
    }

    with open('web/exhibition_data.json', 'w') as f:
        json.dump(exhibition_data, f, indent=2)

    print(f"      ✓ Word '{word}' added instantly! (orange dot)")
    return True

def main():
    """Main function - interactive loop"""
    print("="*60)
    print("Fast Word Addition Tool")
    print("="*60)
    print("Type words and press Enter to add them to the visualization.")
    print("New words appear in ~2-3 seconds (no full regeneration!).")
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
                print("Words will appear in the visualization within 2 seconds!")
                print("="*60)
                break

            word_count += 1
            print(f"\n--- Processing word #{word_count}: '{word}' ---")

            # Add word to file
            add_word_to_file(word)

            # Add word fast (no full regeneration)
            success = add_word_fast(word)

            if success:
                print("✓ Done! Check your browser in ~2 seconds.\n")
            else:
                print()

        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print(f"Interrupted. Added {word_count} word(s) total.")
            print("="*60)
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()

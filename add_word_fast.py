#!/usr/bin/env python3
"""
Fast word addition - adds new word to existing t-SNE without full regeneration
Sends WebSocket message to server for instant updates
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from openTSNE import TSNE
import sys
import os
import asyncio
import websockets

# Disable tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def add_word_to_file(word):
    """Add a word to my_writing.txt"""
    with open('data/raw/my_writing.txt', 'a') as f:
        f.write(f'\n{word}')
    print(f"[1/4] ✓ Added '{word}' to my_writing.txt")

async def notify_server(word, position):
    """Send WebSocket message to server with new word"""
    try:
        async with websockets.connect('ws://localhost:8080/ws') as ws:
            # Send add_word command with position
            await ws.send(json.dumps({
                'command': 'add_word_with_position',
                'word': word,
                'position': position.tolist()
            }))

            # Wait for response
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(response)

            if data.get('success'):
                print(f"[4/4] ✓ Server updated! Word visible in browser now.")
                return True
            else:
                print(f"[4/4] ⚠ Server response: {data.get('message', 'Unknown error')}")
                return False
    except Exception as e:
        print(f"[4/4] ⚠ Could not notify server (visualization will update in ~2s): {e}")
        return False

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

    print(f"[2/4] Computing position for new word (ultra-fast mode)...")

    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embedding ONLY for new word (this is all we need!)
    new_embedding = model.encode([word])[0]

    # Simple nearest-neighbor approach: find similar words and place nearby
    # This is MUCH faster than rebuilding t-SNE affinity

    # Load cached embeddings if available, otherwise compute them
    cache_file = Path('web/embeddings_cache.npy')
    if cache_file.exists():
        existing_embeddings = np.load(cache_file)
    else:
        # First time: compute and cache embeddings
        print("      (First run: caching embeddings for future speed...)")
        existing_embeddings = model.encode(existing_words, show_progress_bar=False)
        np.save(cache_file, existing_embeddings)

    # Find k nearest neighbors in embedding space
    k = min(5, len(existing_embeddings))
    similarities = np.dot(existing_embeddings, new_embedding)
    nearest_indices = np.argsort(similarities)[-k:]

    # Place new word at centroid of nearest neighbors in t-SNE space
    nearest_positions = existing_positions[nearest_indices]
    new_position = nearest_positions.mean(axis=0, keepdims=True)
    

    # Add small random offset to avoid exact overlap
    new_position += np.random.randn(1, 3) * 0.1

    # Just append the new position (no need to renormalize everything)
    all_positions = np.vstack([existing_positions, new_position])

    # Update colors efficiently: change previous orange to blue, new word is orange
    word_data = [{'word': item['word'], 'color': '#133A5B'} for item in data['words']]
    word_data.append({'word': word, 'color': '#ff9800'})

    # Save updated data
    exhibition_data = {
        'words': word_data,
        'frames': [all_positions.tolist()]
    }

    with open('web/exhibition_data.json', 'w') as f:
        json.dump(exhibition_data, f, indent=2)

    # Update embedding cache with new word
    cache_file = Path('web/embeddings_cache.npy')
    updated_embeddings = np.vstack([existing_embeddings, new_embedding])
    np.save(cache_file, updated_embeddings)

    print(f"[3/4] ✓ Data saved! Notifying server...")

    # Return the new word and position for server notification
    return (word, new_position[0])

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
            result = add_word_fast(word)

            if result:
                word, position = result
                # Notify server via WebSocket
                asyncio.run(notify_server(word, position))
                print("✓ Done!\n")
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

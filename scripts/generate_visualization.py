"""
Direct pipeline: Text → Embeddings → t-SNE → JSON → Visualization
No TensorBoard required!
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

# Configuration
INPUT_FILE = 'data/raw/my_writing.txt'
OUTPUT_FILE = 'web/data.json'
YOUR_COLOR = '#2196f3'  # Blue
VISITOR_COLOR = '#ff9800'  # Orange

def load_and_process_text(filepath):
    """Load your writing and extract words"""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Extract words
    words = text.lower().split()
    
    # Clean and deduplicate
    seen = set()
    unique_words = []
    for word in words:
        word = word.strip('.,!?;:"()[]{}')
        if len(word) >= 2 and word not in seen:
            seen.add(word)
            unique_words.append(word)
    
    return unique_words

def create_embeddings(words):
    """Generate semantic embeddings"""
    print(f"Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Generating embeddings for {len(words)} words...")
    embeddings = model.encode(words, show_progress_bar=True)
    
    return embeddings

def compute_tsne(embeddings, perplexity=30):
    """Reduce dimensions to 3D using t-SNE"""
    print(f"Computing t-SNE projection...")
    
    tsne = TSNE(
        n_components=3,
        perplexity=min(perplexity, len(embeddings) - 1),
        random_state=42,
        n_iter=1000,
        verbose=1
    )
    
    positions = tsne.fit_transform(embeddings)
    
    # Normalize to reasonable scale (-5 to 5)
    positions = positions - positions.mean(axis=0)
    max_range = np.abs(positions).max()
    positions = (positions / max_range) * 5
    
    return positions

def export_to_json(words, positions, output_file):
    """Export to JSON for Three.js visualization"""
    data = []
    
    for i, word in enumerate(words):
        data.append({
            'word': word,
            'position': positions[i].tolist(),
            'source': 'your_writing',
            'color': YOUR_COLOR
        })
    
    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Exported {len(data)} words to {output_file}")

def main():
    print("="*60)
    print("Direct Pipeline: Text → Embeddings → t-SNE → Visualization")
    print("="*60 + "\n")
    
    # Step 1: Load text
    print("Step 1: Loading text...")
    words = load_and_process_text(INPUT_FILE)
    print(f"  ✓ Extracted {len(words)} unique words\n")
    
    # Step 2: Create embeddings
    print("Step 2: Creating embeddings...")
    embeddings = create_embeddings(words)
    print(f"  ✓ Generated {embeddings.shape[1]}-dimensional embeddings\n")
    
    # Step 3: Compute t-SNE
    print("Step 3: Computing t-SNE...")
    positions = compute_tsne(embeddings)
    print(f"  ✓ Reduced to 3D coordinates\n")
    
    # Step 4: Export
    print("Step 4: Exporting to JSON...")
    export_to_json(words, positions, OUTPUT_FILE)
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print(f"Next: Open web/visualization.html in your browser")
    print("="*60)

if __name__ == '__main__':
    main()

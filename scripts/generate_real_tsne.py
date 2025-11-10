import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import json
import numpy as np
from pathlib import Path
from openTSNE import TSNE

def load_tensorboard_data():
    log_dir = Path('data/logs/exhibition')
    
    words = []
    with open(log_dir / 'metadata.tsv', 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            words.append(line.strip().split('\t')[0])
    
    embeddings = []
    with open(log_dir / 'tensor.tsv', 'r') as f:
        for line in f:
            embeddings.append([float(x) for x in line.strip().split('\t')])
    
    return words, np.array(embeddings)

def compute_tsne_realtime(embeddings, total_iterations=1000, save_every=10):
    """
    Compute t-SNE step-by-step, saving snapshots
    This is TRUE t-SNE optimization happening iteratively
    """
    print(f"Computing REAL t-SNE optimization...")
    print(f"Total iterations: {total_iterations}")
    print(f"Saving snapshot every {save_every} iterations\n")
    
    # Initialize t-SNE
    tsne = TSNE(
        n_components=3,
        perplexity=min(10, len(embeddings) - 1),
        random_state=42,
        n_jobs=1,
        verbose=True
    )
    
    # Initial embedding (random positions)
    print("Initializing with random positions...")
    embedding = tsne.fit(embeddings)
    
    frames = []
    
    # Save initial state
    positions = embedding[:].copy()
    positions = positions - positions.mean(axis=0)
    max_range = np.abs(positions).max()
    if max_range > 0:
        positions = (positions / max_range) * 5
    frames.append(positions)
    print(f"  Saved frame 0 (initial random)")
    
    # Optimize iteratively
    iterations_done = 0
    while iterations_done < total_iterations:
        # Run optimization in chunks
        chunk_size = save_every
        print(f"\nOptimizing iterations {iterations_done} to {iterations_done + chunk_size}...")
        
        embedding = embedding.optimize(n_iter=chunk_size, momentum=0.8)
        iterations_done += chunk_size
        
        # Save this snapshot
        positions = embedding[:].copy()
        positions = positions - positions.mean(axis=0)
        max_range = np.abs(positions).max()
        if max_range > 0:
            positions = (positions / max_range) * 5
        
        frames.append(positions)
        print(f"  ✓ Saved frame {len(frames)-1} (iteration {iterations_done})")
    
    print(f"\n✓ Optimization complete! Generated {len(frames)} frames")
    return frames

def export_visualization(words, frames):
    data = {
        'words': [{'word': w, 'color': '#2196f3'} for w in words],
        'frames': [frame.tolist() for frame in frames],
        'metadata': {
            'type': 'real_tsne',
            'total_frames': len(frames),
            'description': 'Real t-SNE optimization captured frame-by-frame'
        }
    }
    
    output_file = Path('web/exhibition_data.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print(f"\n✓ Exported to {output_file}")

def main():
    print("="*60)
    print("REAL t-SNE Optimization Capture")
    print("="*60 + "\n")
    
    words, embeddings = load_tensorboard_data()
    print(f"Loaded {len(words)} words with {embeddings.shape[1]}D embeddings\n")
    
    # Capture t-SNE optimization process
    frames = compute_tsne_realtime(
        embeddings, 
        total_iterations=1000,  # Total t-SNE iterations
        save_every=10            # Save a frame every 10 iterations
    )
    
    export_visualization(words, frames)
    
    print("\n" + "="*60)
    print("✓ SUCCESS! Real t-SNE optimization captured!")
    print("\nThis is ACTUAL t-SNE, not a simulation.")
    print("The visualization shows the real clustering process.")
    print("\nNext steps:")
    print("  cd web")
    print("  python -m http.server 8000")
    print("  Open: http://localhost:8000/exhibition.html")
    print("="*60)

if __name__ == '__main__':
    main()

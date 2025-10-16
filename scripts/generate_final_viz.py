import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def load_tensorboard_data():
    """Load embeddings from TensorBoard files"""
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

def create_smooth_evolution(embeddings, n_frames=100):
    """Create smooth rotating projection through embedding space"""
    print(f"Creating {n_frames} evolution frames...")
    
    # Get principal components
    pca = PCA(n_components=min(20, embeddings.shape[1]))
    reduced = pca.fit_transform(embeddings)
    
    frames = []
    
    for i in range(n_frames):
        progress = i / n_frames
        angle = progress * 4 * np.pi  # 2 full rotations
        
        # Create rotation matrix through high-dimensional space
        # This simulates the optimization process of t-SNE
        weights = np.zeros((reduced.shape[1], 3))
        
        # First 3 components - primary structure
        weights[0] = [np.cos(angle), np.sin(angle), np.cos(angle * 1.3)]
        weights[1] = [np.sin(angle * 1.1), np.cos(angle * 0.9), np.sin(angle * 1.4)]
        weights[2] = [np.cos(angle * 0.7), np.sin(angle * 1.2), np.cos(angle * 0.8)]
        
        # Additional components - add variation
        for j in range(3, min(10, reduced.shape[1])):
            phase = angle * (0.3 + j * 0.2)
            weights[j] = [
                np.cos(phase) * 0.5,
                np.sin(phase) * 0.5,
                np.cos(phase * 1.3) * 0.5
            ]
        
        # Project to 3D
        positions = reduced @ weights
        
        # Normalize
        positions = positions - positions.mean(axis=0)
        max_range = np.abs(positions).max()
        if max_range > 0:
            positions = (positions / max_range) * 5
        
        frames.append(positions)
        
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_frames} frames complete")
    
    return frames

def export_visualization(words, frames):
    """Export to JSON for web viewer"""
    data = {
        'words': [{'word': w, 'color': '#2196f3'} for w in words],
        'frames': [frame.tolist() for frame in frames]
    }
    
    output_file = Path('web/exhibition_data.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print(f"\n✓ Exported to {output_file}")
    return output_file

def main():
    print("="*60)
    print("Generating Clean Exhibition Visualization")
    print("="*60 + "\n")
    
    words, embeddings = load_tensorboard_data()
    print(f"Loaded {len(words)} words with {embeddings.shape[1]}D embeddings\n")
    
    frames = create_smooth_evolution(embeddings, n_frames=100)
    
    output_file = export_visualization(words, frames)
    
    print("\n" + "="*60)
    print("✓ SUCCESS!")
    print("\nNext steps:")
    print("  cd web")
    print("  python -m http.server 8000")
    print("  Open: http://localhost:8000/exhibition.html")
    print("="*60)

if __name__ == '__main__':
    main()

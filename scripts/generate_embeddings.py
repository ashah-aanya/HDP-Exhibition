"""
Generate embeddings from your writing and export to TensorBoard format
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from sentence_transformers import SentenceTransformer
from tensorboard.plugins import projector
import config

def load_text(filepath):
    """Load your writing from a text file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def extract_words(text, min_length=2):
    """Extract words from text"""
    words = text.lower().split()
    
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in words:
        word = word.strip('.,!?;:"()[]{}')
        if len(word) >= min_length and word not in seen:
            seen.add(word)
            unique_words.append(word)
    
    return unique_words

def create_embeddings(words):
    """Generate embeddings using sentence transformers"""
    print(f"Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Generating embeddings for {len(words)} words...")
    embeddings = model.encode(words, show_progress_bar=True)
    
    return embeddings

def export_to_tensorboard(words, embeddings, log_dir):
    """Export embeddings to TensorBoard format"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata.tsv (word labels)
    metadata_path = log_dir / 'metadata.tsv'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write('word\tsource\tcolor\n')
        for word in words:
            f.write(f'{word}\tyour_writing\tblue\n')
    
    # Create tensor.tsv (embeddings)
    tensor_path = log_dir / 'tensor.tsv'
    with open(tensor_path, 'w') as f:
        for embedding in embeddings:
            f.write('\t'.join(str(x) for x in embedding) + '\n')
    
    # Create projector config
    projector_config = projector.ProjectorConfig()
    embedding_config = projector_config.embeddings.add()
    embedding_config.metadata_path = 'metadata.tsv'
    embedding_config.tensor_path = 'tensor.tsv'
    
    projector.visualize_embeddings(log_dir, projector_config)
    
    print(f"✓ TensorBoard files created in {log_dir}")

def main():
    print("Starting embedding generation...")
    
    # Check if input file exists
    input_file = Path('data/raw/my_writing.txt')
    
    if not input_file.exists():
        print(f"ERROR: {input_file} not found!")
        print(f"Please create the file and add your writing to it.")
        return
    
    # Load and process text
    print(f"Loading text from {input_file}")
    text = load_text(input_file)
    print(f"Loaded {len(text)} characters")
    
    words = extract_words(text)
    print(f"Extracted {len(words)} unique words")
    
    if len(words) == 0:
        print("ERROR: No words extracted! Check your input file.")
        return
    
    # Generate embeddings
    embeddings = create_embeddings(words)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Export to TensorBoard
    log_dir = Path('data/logs/exhibition')
    export_to_tensorboard(words, embeddings, log_dir)
    
    print("\n" + "="*60)
    print("SUCCESS! Next steps:")
    print(f"  1. Run: tensorboard --logdir={log_dir}")
    print("  2. Open: http://localhost:6006/#projector")
    print("  3. In TensorBoard, click 'Projector' tab")
    print("  4. Select 't-SNE' for dimensionality reduction")
    print("="*60)

if __name__ == '__main__':
    main()

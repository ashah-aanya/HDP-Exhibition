"""
Real-time WebSocket server for evolving t-SNE visualization
Supports adding new words dynamically and broadcasting updates
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import asyncio
import json
import numpy as np
from pathlib import Path
from aiohttp import web, WSMsgType
import aiohttp_cors
from sentence_transformers import SentenceTransformer
from openTSNE import TSNE
from openTSNE.affinity import PerplexityBasedNN
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
model = None
tsne = None
embedding_obj = None
words = []
word_colors = []  # Store colors for each word
embeddings = []
current_positions = []
websocket_clients = set()
animation_task = None
is_animating = False


async def init_model():
    """Initialize the embedding model"""
    global model
    logger.info("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Model loaded!")


def initialize_tsne(initial_embeddings, perplexity=25):
    """Initialize t-SNE with initial embeddings - increased scatter and fast learning"""
    global tsne, embedding_obj

    # Perplexity must be less than n_samples
    perplexity = min(perplexity, max(1, len(initial_embeddings) - 1))

    logger.info(f"Initializing t-SNE with {len(initial_embeddings)} points (faster smooth mode)...")
    tsne = TSNE(
        n_components=3,
        random_state=42,
        verbose=False,
        n_jobs=1,
        initialization="pca",   # big win vs random
        early_exaggeration=32,  # faster cluster separation (try 24–64)
        learning_rate="auto",   # usually more stable than hardcoding 1500
        negative_gradient_method="bh",  # Barnes-Hut = faster
        n_iter=400,             # shorter initial run; you animate further anyway
        exaggeration=None       # let early_exaggeration handle it
    )

    embedding_obj = tsne.fit(initial_embeddings)

    # Get initial positions
    positions = embedding_obj[:].copy()
    positions = normalize_positions(positions)

    logger.info("t-SNE initialized in high-drama mode!")
    return positions

def normalize_positions(positions):
    """Normalize positions to a consistent scale"""
    positions = positions - positions.mean(axis=0)
    max_range = np.abs(positions).max()
    if max_range > 0:
        positions = (positions / max_range) * 5
    return positions


async def add_word(word, color="#151A1D"):
    """Add a new word to the visualization"""
    global words, embeddings, current_positions, embedding_obj

    if word.lower() in [w.lower() for w in words]:
        logger.warning(f"Word '{word}' already exists")
        return False

    # Generate embedding for new word
    logger.info(f"Adding word: {word}")
    new_embedding = model.encode([word])[0]

    words.append(word)
    embeddings.append(new_embedding.tolist())  # Store as list for consistency

    # If this is the first word or we need to reinitialize
    if len(words) == 1:
        embedding_array = np.array([new_embedding])
        positions = initialize_tsne(embedding_array)
        current_positions = positions.tolist()
    elif len(words) == 2:
        # Need at least 2 points to do t-SNE, so reinitialize
        embedding_array = np.array(embeddings)
        positions = initialize_tsne(embedding_array)
        current_positions = positions.tolist()
    else:
        # Transform new point using existing t-SNE
        try:
            new_position = embedding_obj.transform(np.array([new_embedding]))
            new_position = normalize_positions(new_position)
            current_positions.append(new_position[0].tolist())
        except Exception as e:
            logger.error(f"Error transforming new word: {e}")
            # Fallback: reinitialize t-SNE with all words
            embedding_array = np.array(embeddings)
            positions = initialize_tsne(embedding_array)
            current_positions = positions.tolist()

    # Broadcast update to all clients
    await broadcast_update()

    logger.info(f"Word '{word}' added successfully. Total words: {len(words)}")
    return True


async def optimize_tsne_step(iterations=50):
    """Run optimization with high momentum for perpetual movement"""
    global embedding_obj, current_positions

    if embedding_obj is None or len(words) < 2:
        return

    # Very high momentum = never fully converges, keeps drifting
    embedding_obj = embedding_obj.optimize(n_iter=iterations, momentum=0.99)

    # Get updated positions
    positions = embedding_obj[:].copy()
    positions = normalize_positions(positions)
    current_positions = positions.tolist()

    # Broadcast update
    await broadcast_update()


async def broadcast_update():
    """Send current state to all connected clients"""
    if not websocket_clients:
        return

    # Build word data with colors from word_colors
    word_data = []
    for i, w in enumerate(words):
        color = word_colors[i] if i < len(word_colors) else '#2196f3'
        word_data.append({'word': w, 'color': color})

    data = {
        'type': 'update',
        'words': word_data,
        'positions': current_positions
    }

    message = json.dumps(data)

    # Send to all connected clients
    disconnected = set()
    for ws in websocket_clients:
        try:
            await ws.send_str(message)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            disconnected.add(ws)

    # Remove disconnected clients
    websocket_clients.difference_update(disconnected)

## draw points through line smooth
async def animation_loop():
    """Continuously optimize t-SNE with perpetual movement"""
    global is_animating, embedding_obj, current_positions

    logger.info("Animation loop started - ULTRA-FAST perpetual motion mode")

    while is_animating:
        try:
            if len(words) >= 2 and embedding_obj is not None:
                # Optimize t-SNE with more iterations for dramatic clustering
                embedding_obj = embedding_obj.optimize(n_iter=15, momentum=0.95)

                # Get updated positions
                positions = embedding_obj[:].copy()
                positions = normalize_positions(positions)
                current_positions = positions.tolist()

                # Broadcast to all clients
                await broadcast_update()

            await asyncio.sleep(0.1)  # Update 10 times per second - more stable for large datasets
        except Exception as e:
            logger.error(f"Error in animation loop: {e}")
            await asyncio.sleep(0.05)

    logger.info("Animation loop stopped")


async def websocket_handler(request):
    """Handle WebSocket connections"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    websocket_clients.add(ws)
    logger.info(f"New WebSocket connection. Total clients: {len(websocket_clients)}")

    # Send initial state with colors from word_colors
    word_data = []
    for i, w in enumerate(words):
        color = word_colors[i] if i < len(word_colors) else '#2196f3'
        word_data.append({'word': w, 'color': color})

    initial_data = {
        'type': 'init',
        'words': word_data,
        'positions': current_positions
    }
    await ws.send_str(json.dumps(initial_data))

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    command = data.get('command')

                    if command == 'add_word':
                        word = data.get('word', '').strip()
                        if word:
                            try:
                                success = await add_word(word)
                                await ws.send_str(json.dumps({
                                    'type': 'response',
                                    'success': success,
                                    'message': f"Word '{word}' {'added' if success else 'already exists'}"
                                }))
                            except Exception as e:
                                logger.error(f"Error adding word '{word}': {e}")
                                await ws.send_str(json.dumps({
                                    'type': 'response',
                                    'success': False,
                                    'message': f"Error adding word: {str(e)}"
                                }))

                    elif command == 'start_animation':
                        global is_animating, animation_task
                        if not is_animating:
                            is_animating = True
                            animation_task = asyncio.create_task(animation_loop())
                            await ws.send_str(json.dumps({
                                'type': 'response',
                                'message': 'Animation started'
                            }))

                    elif command == 'stop_animation':
                        is_animating = False
                        if animation_task:
                            await animation_task
                        await ws.send_str(json.dumps({
                            'type': 'response',
                            'message': 'Animation stopped'
                        }))

                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                except Exception as e:
                    logger.error(f"Error processing command: {e}")

            elif msg.type == WSMsgType.ERROR:
                logger.error(f'WebSocket error: {ws.exception()}')

    finally:
        websocket_clients.remove(ws)
        logger.info(f"WebSocket connection closed. Total clients: {len(websocket_clients)}")

    return ws


async def load_existing_data():
    """Load existing words from the data file if available"""
    global words, word_colors, embeddings, current_positions, embedding_obj

    data_file = Path('web/exhibition_data.json')
    if data_file.exists():
        logger.info("Loading existing data...")
        with open(data_file, 'r') as f:
            data = json.load(f)

        if data.get('words') and data.get('frames'):
            # Load words and their colors
            words = [item['word'] for item in data['words']]
            word_colors = [item['color'] for item in data['words']]

            # Generate embeddings for existing words
            logger.info(f"Generating embeddings for {len(words)} words...")
            embeddings_array = model.encode(words)
            embeddings = embeddings_array.tolist()  # Keep as list for consistency

            # Initialize t-SNE with existing data
            # This updates the global embedding_obj
            initialize_tsne(embeddings_array)

            # Use the positions from the saved file
            current_positions = data['frames'][-1]

            logger.info(f"Loaded {len(words)} existing words")
    else:
        logger.info("No existing data found. Starting fresh.")


async def watch_data_file():
    """Watch exhibition_data.json for changes and reload when modified"""
    data_file = Path('web/exhibition_data.json')
    last_modified = data_file.stat().st_mtime if data_file.exists() else 0

    logger.info("File watcher started - monitoring exhibition_data.json for changes")

    while True:
        await asyncio.sleep(2)  # Check every 2 seconds

        if data_file.exists():
            current_modified = data_file.stat().st_mtime

            if current_modified > last_modified:
                logger.info("="*60)
                logger.info("exhibition_data.json has been updated! Reloading...")
                logger.info("="*60)

                await load_existing_data()
                await broadcast_update()  # Send new data to all connected clients

                last_modified = current_modified
                logger.info("✓ Data reloaded and sent to all clients!")


async def start_background_tasks(app):
    """Initialize model and load data on startup"""
    await init_model()
    await load_existing_data()

    # Start file watcher in background
    asyncio.create_task(watch_data_file())


async def cleanup_background_tasks(app):
    """Cleanup on shutdown"""
    global is_animating, animation_task

    is_animating = False
    if animation_task:
        await animation_task


def create_app():
    """Create and configure the web application"""
    app = web.Application()

    # Add routes
    app.router.add_get('/ws', websocket_handler)

    # Configure CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })

    # Apply CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)

    # Setup startup/cleanup
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    return app


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("Real-time t-SNE Visualization Server")
    logger.info("="*60)

    app = create_app()
    web.run_app(app, host='localhost', port=8080)

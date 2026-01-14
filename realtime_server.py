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
base_embedding = None  # Keep reference to base embedding for transform()
words = []
word_colors = []  # Store colors for each word
embeddings = []
current_positions = []
websocket_clients = set()
animation_task = None
is_animating = False
tsne_warmup_ticks = 2   # <-- ADD THIS (e.g., 30 ticks ≈ 3 seconds at 10 FPS)
display_mean = None
display_scale = None



async def init_model():
    """Initialize the embedding model"""
    global model
    logger.info("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Model loaded!")


def initialize_tsne(initial_embeddings, perplexity=25, fast_mode=False):
    """Initialize t-SNE with initial embeddings - increased scatter and fast learning"""
    global tsne, embedding_obj, base_embedding, tsne_warmup_ticks


    # Perplexity must be less than n_samples
    perplexity = min(perplexity, max(1, len(initial_embeddings) - 1))

    if fast_mode:
        # Ultra-fast settings for incremental word addition
        logger.info(f"Initializing t-SNE with {len(initial_embeddings)} points (ULTRA-FAST rebuild mode)...")
        tsne = TSNE(
            n_components=3,
            random_state=42,
            verbose=False,
            n_jobs=1,
            initialization="pca",   # PCA is fastest
            early_exaggeration=12,  # Lower = faster (but less dramatic separation)
            learning_rate="auto",
            negative_gradient_method="bh",  # Barnes-Hut
            n_iter=150,             # Reduced from 400 for speed
            exaggeration=None
        )
    else:
        # Original settings for initial load
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
    base_embedding = embedding_obj  # Keep reference for transform()

    positions = embedding_obj[:].copy()
    global display_mean, display_scale
    display_mean, display_scale = fit_display_transform(positions, target=5.0)

    logger.info("t-SNE initialized in high-drama mode!")
    return positions

def fit_display_transform(positions, target=5.0):
    mean = positions.mean(axis=0)
    shifted = positions - mean
    max_range = np.abs(shifted).max()
    scale = (target / max_range) if max_range > 0 else 1.0
    return mean, scale

def apply_display_transform(positions):
    global display_mean, display_scale
    if display_mean is None or display_scale is None:
        return positions
    return (positions - display_mean) * display_scale

def topk_cosine_neighbors(embeddings_list, new_embedding, k=8):
    """
    embeddings_list: list of embeddings as python lists (same order as words)
    new_embedding: np.array shape (D,)
    returns: list of indices of closest existing points (excluding the new one)
    """
    E = np.asarray(embeddings_list, dtype=np.float32)  # (N, D)
    x = np.asarray(new_embedding, dtype=np.float32)    # (D,)

    # normalize
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    x_norm = x / (np.linalg.norm(x) + 1e-12)

    sims = E_norm @ x_norm  # (N,)
    # The new embedding is appended last in your current flow, so exclude last index
    sims[-1] = -1.0

    k = min(k, max(0, len(sims) - 1))
    if k <= 0:
        return [], []

    idx = np.argpartition(-sims, kth=k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    return idx.tolist(), sims[idx].tolist()

async def add_word(word, color="#151A1D"):
    """Add a new word to the visualization"""
    global words, embeddings, current_positions, embedding_obj

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
            new_position = apply_display_transform(new_position)
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
    positions = apply_display_transform(positions)
    current_positions = positions.tolist()

    # Broadcast update
    await broadcast_update()

async def broadcast_update(neighborsForNewWords=None):
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

    if neighborsForNewWords is not None:
        data['neighborsForNewWords'] = neighborsForNewWords

    message = json.dumps(data)

    # Send to all connected clients (make a copy to avoid iteration issues)
    disconnected = set()
    for ws in list(websocket_clients):
        try:
            await ws.send_str(message)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            disconnected.add(ws)

    # Remove disconnected clients
    websocket_clients.difference_update(disconnected)

def kmeans_simple(X, k=8, n_iter=8, seed=42):
    """
    Tiny k-means (no sklearn) for clustering positions.
    X: (N, D)
    returns: labels (N,), centers (k, D)
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    k = max(1, min(k, N))

    # init centers by sampling points
    centers = X[rng.choice(N, size=k, replace=False)].copy()

    for _ in range(n_iter):
        # assign
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (N,k)
        labels = d2.argmin(axis=1)

        # update
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = X[mask].mean(axis=0)
            else:
                centers[j] = X[rng.integers(0, N)]
    return labels, centers


def cluster_drift_step(current_positions, k=8, cluster_update_every=20,
                       cluster_drift_scale=0.0005, point_wobble_scale=0.005,
                       seed=42, recluster_every=600):
    """
    Move points by cluster-level drift + tiny per-point wobble.
    Uses clusters as a loose guideline but never converges - perpetual gentle drift.
    """
    # keep state on the function
    if not hasattr(cluster_drift_step, "frame"):
        cluster_drift_step.frame = 0
        cluster_drift_step.labels = None
        cluster_drift_step.centers = None
        cluster_drift_step.cluster_velocities = None  # velocities instead of offsets
        cluster_drift_step.point_velocities = None    # individual point velocities

    cluster_drift_step.frame += 1
    positions = np.asarray(current_positions, dtype=np.float32)
    N, D = positions.shape

    # Choose k based on N (keeps it calm and stable)
    k_eff = max(2, min(k, int(np.sqrt(N)) if N > 4 else N))

    # Recompute clusters occasionally (not every frame)
    if (cluster_drift_step.labels is None or
        cluster_drift_step.frame % cluster_update_every == 1 or
        cluster_drift_step.frame % recluster_every == 0 or
        cluster_drift_step.labels.shape[0] != N):

        labels, centers = kmeans_simple(positions, k=k_eff, n_iter=6, seed=seed)
        cluster_drift_step.labels = labels
        cluster_drift_step.centers = centers

        # Initialize or resize velocities
        rng = np.random.default_rng(seed + cluster_drift_step.frame)
        if cluster_drift_step.cluster_velocities is None or cluster_drift_step.cluster_velocities.shape[0] != k_eff:
            cluster_drift_step.cluster_velocities = rng.normal(0.0, cluster_drift_scale * 0.5, size=centers.shape).astype(np.float32)

        if cluster_drift_step.point_velocities is None or cluster_drift_step.point_velocities.shape[0] != N:
            cluster_drift_step.point_velocities = rng.normal(0.0, point_wobble_scale * 0.5, size=(N, D)).astype(np.float32)

    labels = cluster_drift_step.labels
    centers = cluster_drift_step.centers

    # Use random number generator
    rng = np.random.default_rng(seed + cluster_drift_step.frame)

    # Update cluster velocities with visible drift - slow convergence/divergence cycles
    # Add continuous random forces that change direction slowly
    cluster_noise = rng.normal(0.0, cluster_drift_scale * 1.5, size=cluster_drift_step.cluster_velocities.shape).astype(np.float32)
    cluster_drift_step.cluster_velocities = (
        0.96 * cluster_drift_step.cluster_velocities +  # very high momentum - slow, smooth direction changes
        0.04 * cluster_noise                             # small random push
    )

    # Update individual point velocities with visible wobble
    point_noise = rng.normal(0.0, point_wobble_scale * 1.5, size=(N, D)).astype(np.float32)
    cluster_drift_step.point_velocities = (
        0.94 * cluster_drift_step.point_velocities +  # very high momentum - smooth motion
        0.06 * point_noise                            # small random push
    )

    # Apply velocities instead of direct position changes
    cluster_velocity = cluster_drift_step.cluster_velocities[labels]  # (N,D)
    point_velocity = cluster_drift_step.point_velocities              # (N,D)

    # Faster movement - long slow convergence/divergence cycles
    positions = positions + 2.0 * cluster_velocity + 1.2 * point_velocity

    # No damping - we want perpetual motion, not convergence
    # positions *= damping  # REMOVED

    return positions

## draw points through line smooth
async def animation_loop():
    """Continuously optimize t-SNE with perpetual movement"""
    global is_animating, embedding_obj, current_positions, tsne_warmup_ticks

    logger.info("Animation loop started - ULTRA-FAST perpetual motion mode")

    while is_animating:
        try:
            if len(words) >= 2 and embedding_obj is not None:
                # Allow small mismatch during word addition (race condition)
                if abs(len(embedding_obj) - len(words)) <= 1:
                    if tsne_warmup_ticks > 0:
                        # 🔥 Warmup phase: run real t-SNE for the first few ticks
                        embedding_obj = embedding_obj.optimize(n_iter=15, momentum=0.95)
                        tsne_warmup_ticks -= 1

                        positions = embedding_obj[:].copy()
                        positions = apply_display_transform(positions)
                        current_positions = positions.tolist()

                        await broadcast_update()
                    else:
                        # 🟢 After warmup: cluster-based gentle drift (calm, cohesive)
                        positions = cluster_drift_step(
                            current_positions,
                            k=8,
                            cluster_update_every=40,      # recompute clustering every ~4s at 10 FPS
                            cluster_drift_scale=0.008,    # cluster drift
                            point_wobble_scale=0.003      # individual wobble
                        )
                        current_positions = positions.tolist()
                        await broadcast_update()
                else:
                    # This shouldn't happen anymore since we update embedding_obj immediately
                    # But just in case, log it and skip this frame
                    logger.warning(f"Mismatch detected: {len(embedding_obj)} embeddings vs {len(words)} words. Skipping frame...")

            await asyncio.sleep(0.1)  # Update 10 times per second - more stable for large datasets
        except Exception as e:
            logger.error(f"Error in animation loop: {e}")
            await asyncio.sleep(0.05)

    logger.info("Animation loop stopped")


async def websocket_handler(request):
    """Handle WebSocket connections"""
    global words, word_colors, embeddings, current_positions, embedding_obj, is_animating, animation_task

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

                    elif command == 'add_word_with_position':
                        # Fast word addition - rebuild t-SNE with cached embeddings + new word
                        try:
                            word = data.get('word', '').strip()
                            position = data.get('position', [0, 0, 0])

                            logger.info(f"Received add_word_with_position: word='{word}', position={position}")

                            if word:
                                # Add word
                                words.append(word)
                                word_colors.append('#ff9800')  # Orange for new word

                                # Change previous orange words to blue
                                for i in range(len(word_colors) - 1):
                                    if word_colors[i] == '#ff9800':
                                        word_colors[i] = "#475E9D"

                                # Generate embedding for the new word (ONLY new word, not all)
                                new_embedding = model.encode([word])[0]
                                embeddings.append(new_embedding.tolist())

                                # Compute nearest neighbors in embedding space (for labeling)
                                neighbors_payload = None
                                if len(words) >= 3:
                                    neighbor_idxs, neighbor_scores = topk_cosine_neighbors(embeddings, new_embedding, k=8)
                                    new_idx = len(words) - 1
                                    neighbors_payload = [{
                                        "newWordIndex": new_idx,
                                        "neighbors": neighbor_idxs,
                                        "scores": neighbor_scores
                                    }]

                                # Use t-SNE transform to add new word instantly
                                logger.info(f"Transforming new word '{word}' using existing t-SNE...")

                                # Use base_embedding for transform (it has the method)
                                new_pos = base_embedding.transform(np.array([new_embedding]))
                                new_pos = apply_display_transform(new_pos)
                                current_positions.append(new_pos[0].tolist())

                                # Update embedding_obj to include the new word
                                all_embeddings = np.array(embeddings)
                                embedding_obj = base_embedding.prepare_partial(all_embeddings)

                                # Run a few optimization steps to let the new word settle
                                logger.info(f"Running quick optimization to settle new word...")
                                embedding_obj = embedding_obj.optimize(n_iter=20, momentum=0.8)

                                # Update positions after optimization
                                positions = embedding_obj[:].copy()
                                positions = apply_display_transform(positions)
                                current_positions = positions.tolist()

                                logger.info(f"Word '{word}' added and optimized. Total: {len(words)} words")

                                # Broadcast to all clients immediately (include neighbors payload)
                                await broadcast_update(neighborsForNewWords=neighbors_payload)

                                await ws.send_str(json.dumps({
                                    'type': 'response',
                                    'success': True,
                                    'message': f"Word '{word}' added"
                                }))
                            else:
                                logger.warning(f"Word '{word}' is invalid (empty)")
                                await ws.send_str(json.dumps({
                                    'type': 'response',
                                    'success': False,
                                    'message': f"Word is invalid (empty)"
                                }))
                        except Exception as e:
                            logger.error(f"Error in add_word_with_position: {e}")
                            import traceback
                            traceback.print_exc()
                            await ws.send_str(json.dumps({
                                'type': 'response',
                                'success': False,
                                'message': f"Error: {str(e)}"
                            }))

                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                except Exception as e:
                    logger.error(f"Error processing command: {e}")

            elif msg.type == WSMsgType.ERROR:
                logger.error(f'WebSocket error: {ws.exception()}')

    finally:
        websocket_clients.discard(ws)  # Use discard instead of remove to avoid KeyError
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
            EMB_CACHE = Path("web/embeddings_cache.npy")
            if EMB_CACHE.exists():
                embeddings_array = np.load(EMB_CACHE)
            else:
                embeddings_array = model.encode(words, show_progress_bar=False)
                np.save(EMB_CACHE, embeddings_array)
            embeddings = embeddings_array.tolist()
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

    # File watcher disabled - using WebSocket messaging for updates instead
    # asyncio.create_task(watch_data_file())


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

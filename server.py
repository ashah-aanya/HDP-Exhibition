import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import asyncio
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import websockets

CACHE_DIR = Path("data/cache")
BASE_WORDS_JSON = CACHE_DIR / "base_words.json"
BASE_EMB_NPY = CACHE_DIR / "base_embeddings.npy"
FRAMES_NPY = CACHE_DIR / "frames.npy"

YOUR_COLOR = "#2196f3"
VISITOR_COLOR = "#ff9800"
MODEL_NAME = "all-MiniLM-L6-v2"

# Stream rate: low power + smooth client lerp
SERVER_FPS = 2.0  # 2 updates/sec; client lerp makes it look continuous

def load_cache():
    words = json.loads(BASE_WORDS_JSON.read_text(encoding="utf-8"))["words"]
    base_emb = np.load(BASE_EMB_NPY).astype(np.float32, copy=False)   # (N, D), normalized
    frames = np.load(FRAMES_NPY).astype(np.float32, copy=False)       # (F, N, 3)
    return words, base_emb, frames

def parse_words(raw: str):
    raw = raw.replace(",", " ")
    parts = [p.strip().lower() for p in raw.split()]
    out, seen = [], set()
    for w in parts:
        w = w.strip('.,!?;:"()[]{}')
        if len(w) >= 2 and w not in seen:
            seen.add(w)
            out.append(w)
    return out

def embed_words(model, words):
    emb = model.encode(words, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)

def knn_attach(new_emb, base_emb, k=12):
    """
    Return neighbor indices + softmax weights so the new word can be positioned
    each frame as a weighted average of its neighbors' evolving positions.
    """
    sims = new_emb @ base_emb.T
    k = min(k, base_emb.shape[0])
    idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]

    attaches = []
    for i in range(new_emb.shape[0]):
        nn = idx[i]
        nn_sims = sims[i, nn]
        order = np.argsort(-nn_sims)
        nn = nn[order]
        nn_sims = nn_sims[order]

        # softmax weights
        x = nn_sims - nn_sims.max()
        w = np.exp(x / 0.07)
        w = w / (w.sum() + 1e-9)
        attaches.append((nn.astype(int), w.astype(np.float32)))
    return attaches

def make_words_payload(words, colors):
    return [{"word": w, "color": c} for w, c in zip(words, colors)]

async def ws_handler(websocket):
    base_words, base_emb, frames = load_cache()
    model = SentenceTransformer(MODEL_NAME)

    # State
    words = list(base_words)
    colors = [YOUR_COLOR] * len(words)

    # New words: store attachment info so they move with the frames
    # each entry: {"word": str, "neighbors": np.array, "weights": np.array, "color": orange}
    new_word_attachments = []

    # Client might connect multiple times; we stream continuously after init
    initialized = False
    frame_idx = 0
    F = frames.shape[0]
    N = len(base_words)

    async def send_init():
        nonlocal initialized
        # initial positions = frame 0 with any attached new words appended
        base_pos = frames[frame_idx]
        positions = [base_pos.tolist()]  # placeholder; we'll build properly below

    async def build_positions_for_frame(fi: int):
        base_pos = frames[fi]  # (N, 3)

        # start with base positions
        pos_list = base_pos.tolist()

        # append positions for new words (weighted average of neighbor positions)
        for item in new_word_attachments:
            nn = item["neighbors"]
            w = item["weights"]
            p = (base_pos[nn] * w[:, None]).sum(axis=0)
            pos_list.append([float(p[0]), float(p[1]), float(p[2])])

        return pos_list

    async def stream_loop():
        nonlocal frame_idx
        while True:
            await asyncio.sleep(1.0 / SERVER_FPS)
            frame_idx = (frame_idx + 1) % F

            positions = await build_positions_for_frame(frame_idx)
            payload = {
                "type": "update",
                "words": make_words_payload(words, colors),
                "positions": positions,
            }
            await websocket.send(json.dumps(payload))

    stream_task = None

    async for message in websocket:
        req = json.loads(message)
        cmd = req.get("command")

        if cmd in ("init", "start_animation"):
            # Send init frame once
            positions = await build_positions_for_frame(frame_idx)
            payload = {
                "type": "init",
                "words": make_words_payload(words, colors),
                "positions": positions,
            }
            await websocket.send(json.dumps(payload))
            initialized = True

            # Start streaming frames
            if stream_task is None:
                stream_task = asyncio.create_task(stream_loop())

        elif cmd == "add_words":
            raw = req.get("raw", "")
            new_words = parse_words(raw)
            if not new_words:
                await websocket.send(json.dumps({"type": "notice", "message": "No valid words."}))
                continue

            existing = set(words)
            new_words = [w for w in new_words if w not in existing]
            if not new_words:
                await websocket.send(json.dumps({"type": "notice", "message": "All words already exist."}))
                continue

            new_emb = embed_words(model, new_words)
            attaches = knn_attach(new_emb, base_emb, k=12)

            start_index = len(words)
            for w, (nn, ww) in zip(new_words, attaches):
                words.append(w)
                colors.append(VISITOR_COLOR)
                new_word_attachments.append({
                    "word": w,
                    "neighbors": nn,
                    "weights": ww
                })

            # Send one immediate update + neighbor indices for labeling
            positions = await build_positions_for_frame(frame_idx)

            neighbors_payload = []
            for i, (nn, _) in enumerate(attaches):
                neighbors_payload.append({
                    "newWordIndex": start_index + i,
                    "neighbors": nn[:8].tolist()
                })

            await websocket.send(json.dumps({
                "type": "update",
                "words": make_words_payload(words, colors),
                "positions": positions,
                "neighborsForNewWords": neighbors_payload
            }))

        else:
            await websocket.send(json.dumps({"type": "notice", "message": f"Unknown command: {cmd}"}))

async def main():
    async def router(websocket):
        await ws_handler(websocket)

    async with websockets.serve(router, "localhost", 8080, ping_interval=20):
        print("WS running at ws://localhost:8080")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())

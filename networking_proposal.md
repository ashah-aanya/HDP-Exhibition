# Exhibition Networking Proposal

## Network Requirement (applies to all approaches)

All three approaches require the two computers to **reach each other over a local network**. The transport (WebSocket, HTTP POST, fetch) is irrelevant if the network blocks device-to-device traffic (client isolation). The network must allow direct communication between the two machines.

---

## Approach 1: Laptop-Generated WiFi Hotspot

**How it works:** Computer 1 enables Internet Sharing (System Settings → General → Sharing → Internet Sharing), creating its own private WiFi network. Computer 2 joins that network and opens `http://<computer1-ip>:8080/add` in a browser.

**Merits:**
- No code changes required (server already binds to `0.0.0.0:8080`)
- Only one-line fix needed in `add_word.html`
- Private network — no client isolation, guaranteed to work
- No dependencies on venue WiFi

**Drawbacks:**
- Computer 1 must have Internet Sharing available and enabled
- Computer 1 loses its own internet access while hosting

**Network requirement:** None — the hotspot *is* the network.

---

## Approach 2: Same-Network fetch() / WebSocket (Venue WiFi)

**How it works:** Both computers connect to the same WiFi. Computer 2 opens `http://<computer1-ip>:8080/add`. The existing WebSocket in `add_word.html` is updated to use Computer 1's IP instead of `localhost`.

**Merits:**
- No code changes to the server
- Only one-line fix in `add_word.html`
- Both computers retain internet access

**Drawbacks:**
- Unreliable at public/venue WiFi — client isolation often blocks device-to-device traffic
- Requires knowing Computer 1's IP in advance (can change on reconnect)

**Network requirement:** Both devices on the same subnet with client isolation disabled.

---

## Approach 3: HTTP POST API Endpoint

**How it works:** Add a simple HTTP POST route to the existing aiohttp server. Computer 2's UI uses `fetch()` to POST `{ "word": "..." }` to `http://<computer1-ip>:8080/add_word`. No need for FastAPI — this can be added to the existing server in ~5 lines.

**Merits:**
- Simple, stateless — no persistent WebSocket connection to maintain
- Easy to test manually with a browser or curl
- `add_word.html` becomes simpler (no WebSocket reconnect logic)

**Drawbacks:**
- No real-time feedback to Computer 2 when the visualization updates (one-way only)
- Still fails if the network has client isolation — same limitation as Approach 2
- Slightly more code change than the one-line WebSocket fix

**Network requirement:** Same as Approach 2 — same subnet, no client isolation.

---

## Recommendation

**Use Approach 1 (hotspot) for reliability, with the one-line code fix from Approach 2.**

The hotspot eliminates all network uncertainty. The code change is minimal: one line in `add_word.html`:

```js
// Change this:
ws = new WebSocket('ws://localhost:8080/ws');

// To this:
ws = new WebSocket(`ws://${window.location.hostname}:8080/ws`);
```

FastAPI/POST adds complexity without solving the underlying network problem. If the network works, the existing WebSocket is already sufficient.

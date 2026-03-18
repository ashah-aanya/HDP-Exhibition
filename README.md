# HDP Exhibition — Word Embedding Visualization

A real-time 3D word embedding visualization. Visitors type words into an input screen; the words appear live in a 3D semantic space projected on the display screen.

---

## Architecture

```
Laptop A (display)                    Laptop B (visitor input)
┌─────────────────────────┐           ┌─────────────────────────┐
│  Python server          │           │  Browser                │
│  (ML model + t-SNE)     │◄─────────►│  add_word.html          │
│                         │  WiFi     │                         │
│  Browser (fullscreen)   │  hotspot  │  Visitor types a word   │
│  realtime_exhibition    │           │  → word appears on      │
│  .html (3D viz)         │           │    Laptop A's screen    │
└─────────────────────────┘           └─────────────────────────┘
```

**Laptop A** creates a WiFi hotspot. **Laptop B** connects to it. The server binds to `0.0.0.0` so it's reachable over the hotspot. Both pages auto-detect the correct WebSocket address from `window.location.host` — no manual IP configuration needed.

---

## One-Time Setup

Run these commands once before the exhibition (on Laptop A, with internet access):

```bash
# Clone / navigate to the project
cd ~/HDP-Exhibition

# Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Three.js is bundled locally (already done — web/vendor/three.min.js)
# If it's missing, re-download it:
# curl -o web/vendor/three.min.js https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js
```

---

## Laptop A — Before the Exhibition

### 1. macOS system settings (do once, before opening day)

- **System Preferences > Battery (or Energy Saver)**
  - "Prevent computer from sleeping automatically when the display is off" → ON
  - "Start up automatically after a power failure" → ON (if available)
- **System Preferences > Security & Privacy > Firewall**
  - Add Python to allowed apps, or temporarily disable the firewall
- **System Preferences > Users & Groups > Login Options**
  - Enable automatic login (so the machine recovers after a power cut without a password)
- **Display settings**
  - Set display sleep to "Never" while the exhibition is running

### 2. Enable the WiFi hotspot

1. Open **System Preferences > Sharing**
2. Select **Internet Sharing** in the left list
3. "Share your connection from:" → choose **Ethernet** (if plugged in) or **iPhone USB**
4. "To computers using:" → check **Wi-Fi**
5. Click **Wi-Fi Options…** and set a network name and password
6. Check the **Internet Sharing** checkbox to turn it on

> Note: If you have no ethernet/USB connection to share from, use a phone hotspot instead — both laptops connect to the phone hotspot, and Laptop B reaches Laptop A via the `.local` hostname.

---

## Laptop A — Starting the Exhibition

```bash
cd ~/HDP-Exhibition
./start_exhibition.sh
```

Wait for output like:
```
Loaded 5600 existing words
```

Then open the display in Chrome:
```
http://localhost:8080/
```

Press `Cmd+Ctrl+F` to go fullscreen. The 3D visualization will start animating automatically.

**The script will print the URL for Laptop B**, e.g.:
```
http://aanyas-macbook-pro.local:8080/add
```

---

## Laptop B — Connecting

### Step 1 — Join the hotspot
Connect to Laptop A's WiFi hotspot network (the name you set in Internet Sharing).

### Step 2 — Open the input page
In any browser, go to:
```
http://<laptop-a-hostname>.local:8080/add
```

The `.local` hostname is printed when you run `./start_exhibition.sh` on Laptop A. You can also find it by running `hostname` in Laptop A's terminal.

**Example:** `http://aanyas-macbook-pro.local:8080/add`

### Step 3 — Verify
The page should show a green **"Connected"** dot. Type a word and submit — it should appear on Laptop A's screen within ~2 seconds.

Bookmark the URL or set it as the browser's homepage for quick recovery.

---

## Day-of Quick Reference

### Laptop A
```bash
# 1. Enable hotspot in System Preferences > Sharing
# 2. Start server:
cd ~/HDP-Exhibition && ./start_exhibition.sh
# 3. Open display in Chrome (fullscreen):
#    http://localhost:8080/
# 4. Verify health:
#    http://localhost:8080/health
```

### Laptop B
```bash
# 1. Connect to Laptop A's WiFi hotspot
# 2. Open in browser:
#    http://<hostname>.local:8080/add
# 3. Confirm green "Connected" dot
# 4. Submit a test word
```

---

## Troubleshooting

### Laptop B can't connect / page won't load
- Confirm Laptop B is on Laptop A's hotspot (not the venue WiFi)
- Try the IP address instead: `http://192.168.2.1:8080/add` (or whatever IP `./start_exhibition.sh` printed)
- Check `http://<hostname>.local:8080/health` from Laptop B — if this works but `/add` doesn't, try a different browser
- Restart Laptop A's hotspot (turn Internet Sharing off and on in System Preferences)

### Green dot shows "Connecting..." but never connects
- The server may still be loading (it takes 30–60 seconds on first startup to load the ML model and 5600+ word embeddings)
- Wait for the terminal on Laptop A to show "Loaded N existing words" before Laptop B tries to connect

### Word submitted but doesn't appear on the display
- Check that `http://localhost:8080/` is open and visible on Laptop A (not minimized)
- The display page auto-reconnects — if it lost connection, refresh it once
- Check `exhibition.log` in the project directory for errors

### Server crashed
```bash
# On Laptop A — restart the server:
cd ~/HDP-Exhibition && ./start_exhibition.sh
```
Words are auto-saved every 5 minutes. At most 5 minutes of visitor contributions could be lost.

### Display visualization is blank / black screen
Three.js is bundled locally (`web/vendor/three.min.js`) so no internet is needed. If the file is missing:
```bash
cd ~/HDP-Exhibition
curl -o web/vendor/three.min.js https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js
```
Then refresh the display page.

### After power cut or reboot
Laptop A should auto-login and you just need to run `./start_exhibition.sh` again. All previously submitted words are preserved in `web/exhibition_data.json`.

---

## Logs

Server logs are written to `exhibition.log` (rotates at 10MB, keeps 5 files):
```bash
# Watch live logs:
tail -f ~/HDP-Exhibition/exhibition.log

# Check for errors:
grep ERROR ~/HDP-Exhibition/exhibition.log
```

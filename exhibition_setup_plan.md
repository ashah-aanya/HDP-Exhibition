# Exhibition Setup Meeting Plan

## Context
- **Computer 1 (your laptop):** runs the server, t-SNE computation, and visualization display. Connected to the control center/projector/screen.
- **Computer 2 (second laptop):** runs only the add-word UI in a browser. Visitors interact with this.
- School venue — network reliability unknown.

---

## Meeting Outline

### 1. Arrive early — test the ethernet first (5 min)
- Plug Computer 1 into the control center ethernet
- Confirm internet works:
  ```bash
  ping google.com
  ```
- If ethernet works → proceed with hotspot setup (most reliable)
- If no ethernet → fall back to school WiFi (risk of client isolation)

---

### 2. Start the server on Computer 1 (2 min)
```bash
cd /Users/aanyashah/HDP-Exhibition
python realtime_server.py
```
- Note the IP printed in the terminal, e.g.:
  ```
  This machine's IP: 192.168.2.1
  ```

---

### 3. Set up the hotspot on Computer 1 (if ethernet available) (3 min)
- System Settings → General → Sharing → Internet Sharing
- Share from: **Ethernet**
- To devices using: **Wi-Fi**
- Turn on Internet Sharing
- Note the hotspot network name (usually your Mac's hostname)

---

### 4. Connect Computer 2 to the hotspot (2 min)
- On Computer 2: join the Wi-Fi network created by Computer 1
- Open browser and go to:
  ```
  http://<computer1-ip>:8080/add
  ```
- Confirm the green "Connected" dot appears in the UI

---

### 5. Open the visualization on Computer 1 (1 min)
```
http://localhost:8080/
```
- Connect to the projector/screen
- Confirm the 3D visualization loads

---

### 6. Test end-to-end (5 min)
- Type a word on Computer 2 → hit Enter
- Confirm it appears on the visualization on Computer 1
- Type 3–5 more words and confirm the t-SNE updates
- Confirm the connection status dot stays green

---

### 7. Fallback: if no ethernet, use school WiFi
- Connect both laptops to the same school WiFi
- Find Computer 1's IP:
  ```bash
  ifconfig | grep "inet 192"
  ```
- On Computer 2, navigate to `http://<computer1-ip>:8080/add`
- Test the same way — if words don't appear, the network has client isolation and you'll need ethernet or iPhone USB-C tethering

---

## Key Commands Cheatsheet

| Task | Command |
|------|---------|
| Start server | `cd /Users/aanyashah/HDP-Exhibition && python realtime_server.py` |
| Find IP | `ifconfig | grep "inet 192"` |
| Test network connectivity | `ping <computer1-ip>` |
| Check server is running | `curl http://localhost:8080/health` |
| Stop server | `Ctrl+C` |

---

## What to confirm with the venue/IT
- Is there an ethernet port at the control center?
- Does the school WiFi allow device-to-device traffic (no client isolation)?
- What is the screen/projector input? (HDMI, USB-C?)

---

## Priorities
1. **Ethernet available** → hotspot setup (guaranteed to work)
2. **No ethernet, school WiFi** → test client isolation first
3. **Neither works** → iPhone USB-C tethering to Computer 1, then hotspot

---

## Plan B: Two-Computer Fallbacks (no ethernet, no phone)

### B1: School WiFi — just try it
Many school networks work fine for device-to-device traffic. Test before assuming it won't work.

1. Connect both laptops to the same school WiFi
2. Start server on Computer 1, note the IP from terminal output
3. On Computer 2, open `http://<computer1-ip>:8080/add`
4. Type a word — if it appears on the visualization, you're done

**Test quickly:** `ping <computer1-ip>` from Computer 2. If it responds, it will work.

---

### B2: Computer 2 as a mobile hotspot (if it's a Windows laptop)
Windows laptops can create a hotspot without needing ethernet.

1. On Computer 2 (Windows): Settings → Network → Mobile Hotspot → turn on
2. Computer 1 joins that hotspot
3. Start the server on Computer 1, note the IP
4. On Computer 2, open `http://<computer1-ip>:8080/add`

This works because Computer 1 (the server) joins Computer 2's network — direction doesn't matter, they just need to be on the same network.

---

### B3: macOS ad-hoc network from Computer 1 (no internet needed)
Creates a direct computer-to-computer WiFi network without needing ethernet or a router.

1. On Computer 1, hold **Option** and click the WiFi icon in the menu bar
2. Select **"Create Network..."**
3. Give it a name and click Create
4. Computer 2 joins that network from its WiFi settings
5. Find Computer 1's IP:
   ```bash
   ifconfig | grep "inet 169"
   ```
   (ad-hoc networks use 169.254.x.x addresses)
6. On Computer 2, open `http://169.254.x.x:8080/add`

**Note:** Neither computer will have internet while on this network, but the exhibition runs fully offline so that's fine.

---

### B4: Last resort — USB-C ethernet adapter
If the venue has ethernet ports, a USB-C to ethernet adapter (~$15–25, available at any electronics store) plugged into Computer 1 enables the primary hotspot plan. Worth buying before the exhibition as insurance.

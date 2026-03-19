#!/bin/bash
# Exhibition startup script for Laptop A (display + server)
# Run this every time you start the exhibition.

cd "$(dirname "$0")"

# Activate Python virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Kill any existing server instances
pkill -f realtime_server.py 2>/dev/null
sleep 1

# Get the stable .local mDNS hostname (works even if IP changes)
MDNS_HOST="$(hostname -s).local"

# Also get the current hotspot/WiFi IP for reference
HOTSPOT_IP=$(ipconfig getifaddr bridge100 2>/dev/null)
if [ -z "$HOTSPOT_IP" ]; then
    HOTSPOT_IP=$(ipconfig getifaddr en0 2>/dev/null)
fi

echo ""
echo "============================================================"
echo "  EXHIBITION SERVER STARTING"
echo "============================================================"
echo ""
echo "  Laptop B (visitor input) — open this URL in a browser:"
echo ""
echo "    http://$MDNS_HOST:8080/add"
echo ""
echo "  (This address works even if the IP changes)"
echo ""
if [ -n "$HOTSPOT_IP" ]; then
echo "  Current IP address (backup):  http://$HOTSPOT_IP:8080/add"
fi
echo "  Health check:                  http://$MDNS_HOST:8080/health"
echo "  Display URL (this laptop):     http://localhost:8080/"
echo ""
echo "============================================================"
echo ""

# caffeinate -s prevents macOS from sleeping while the server is running
caffeinate -s python realtime_server.py

#!/bin/bash

# Sovereign Byte-Level Firewall - Launch Script (macOS)
echo "================================================="
echo "   SOVEREIGN BYTE-LEVEL FIREWALL LAUNCHER        "
echo "================================================="

# 1. Start the Firewall Daemon in the background
# (macOS requires sudo to sniff live packets on en0 via BPF)
echo "[*] Starting the Firewall Daemon..."
echo "[!] Note: You may be asked for your Mac password to grant packet sniffing permissions."

# Use the exact absolute path to the Conda Python executable 
# This bypasses the macOS 'sudo' secure_path wipe
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_EXEC="$CONDA_PREFIX/bin/python"
else
    PYTHON_EXEC=$(which python3)
fi

echo "[*] Using Python at: $PYTHON_EXEC"
# Forward all arguments (like --learning_time) directly to the python script
sudo "$PYTHON_EXEC" firewall_daemon.py --interface en0 "$@" &
DAEMON_PID=$!

# Wait a second to let the WebSocket server boot up
sleep 2

# 3. Open the Dashboard in the default browser (macOS 'open' command)
echo "[*] Launching Web Dashboard..."
open dashboard/index.html

# 4. Handle clean shutdown
echo "================================================="
echo "   FIREWALL IS LIVE AND STREAMING TO DASHBOARD   "
echo "================================================="
echo "Press Ctrl+C to stop the firewall and exit."

# Trap Ctrl+C to kill the background python daemon safely
trap "echo -e '\n[*] Shutting down Firewall...'; sudo kill $DAEMON_PID; exit 0" SIGINT

# Keep script running to maintain the trap
wait $DAEMON_PID

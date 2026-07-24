#!/usr/bin/env python3
"""
Manual test harness for the dashboard's threat-type UI.

Spins up a WebSocket server on localhost:8765 (the same URL dashboard/app.js
connects to) and plays back a fixed script of synthetic alerts covering every
alert type the dashboard understands -- including the new SLOW_DISTRIBUTED
(multi-source botnet) and CRITICAL_BYTE (hard Gold Ceiling breach) types --
so you can eyeball the rendering without needing a live capture, a trained
model, or a real attack.

Usage:
    pip install websockets
    python dashboard/mock_ws_server.py
    # then open dashboard/index.html in a browser

Ctrl+C to stop. The script loops forever so you have time to check each
rendering state at your own pace.
"""
import asyncio
import json
import random
import time

import websockets

CLIENTS = set()


async def handler(ws):
    CLIENTS.add(ws)
    print(f"[+] dashboard connected ({len(CLIENTS)} client(s))")
    try:
        await ws.wait_closed()
    finally:
        CLIENTS.discard(ws)
        print(f"[-] dashboard disconnected ({len(CLIENTS)} client(s))")


async def broadcast(payload):
    if CLIENTS:
        msg = json.dumps(payload)
        websockets.broadcast(CLIENTS, msg)
    print(f"-> sent {payload['type']:<18} score={payload.get('score')}")


def make_heatmap(n=256, hot_at=None):
    surprise = [round(random.uniform(0.2, 2.5), 2) for _ in range(n)]
    for i in (hot_at or []):
        surprise[i] = round(random.uniform(7.0, 8.5), 2)
    return {"bytes": [random.randint(0, 255) for _ in range(n)], "surprise": surprise}


# (delay_before_sending_seconds, alert_dict)
SCRIPT = [
    (1, dict(type="BYTE",
             message="Byte-level anomaly: window surprise 6.10 bits",
             score=6.10,
             enrichment={"heatmap": make_heatmap(hot_at=[40, 41, 42, 43])})),

    (3, dict(type="RATE",
             message="Rate anomaly: SYN flood detected (420 SYNs/s)",
             score=420,
             enrichment={"syns": 420, "top_ports": [443, 80]})),

    (3, dict(type="SLOW",
             message=("Slow/low anomaly: cumulative surprise 18.4 bits from "
                      "10.0.0.5 <-> 10.0.0.9 (persistent sub-threshold activity)"),
             score=18.4,
             enrichment={"cusum_level": 18.4})),

    (3, dict(type="SLOW_DISTRIBUTED",
             message=("Slow/low distributed anomaly: cumulative surprise 28.5 bits "
                      "targeting 10.0.0.9:443 (multi-source attack campaign)"),
             score=28.5,
             enrichment={
                 "cusum_level": 28.5,
                 "top_talkers": [
                     {"pair": "10.0.0.5 -> 10.0.0.9", "bytes": 51200},
                     {"pair": "10.0.0.6 -> 10.0.0.9", "bytes": 48200},
                     {"pair": "10.0.0.7 -> 10.0.0.9", "bytes": 39900},
                 ],
                 "top_ports": [443, 8443],
                 "proto_mix_pct": {"TCP": 96, "UDP": 4},
                 "syns": 812,
             })),

    (4, dict(type="CRITICAL_BYTE",
             message=("CRITICAL: Hard static Gold Baseline ceiling breached! "
                      "Surprise: 9.12 bits > gold_threshold 7.50"),
             score=9.12,
             enrichment={
                 "top_talkers": [{"pair": "203.0.113.4 -> 10.0.0.9", "bytes": 8800}],
                 "top_ports": [4444],
                 "proto_mix_pct": {"TCP": 100},
                 "syns": 3,
             })),

    # Fires again quickly, while the banner from the previous one is still up
    # -- confirms the auto-dismiss timer resets instead of the banner
    # disappearing mid-campaign.
    (2, dict(type="CRITICAL_BYTE",
             message=("CRITICAL: Hard static Gold Baseline ceiling breached! "
                      "Surprise: 9.40 bits > gold_threshold 7.50"),
             score=9.40,
             enrichment={
                 "top_talkers": [{"pair": "203.0.113.4 -> 10.0.0.9", "bytes": 9100}],
                 "top_ports": [4444],
                 "proto_mix_pct": {"TCP": 100},
                 "syns": 5,
             })),
]


async def play_script():
    print("Waiting for dashboard to connect on ws://localhost:8765 ...")
    while not CLIENTS:
        await asyncio.sleep(0.5)
    print("Client connected. Playing test script (loops until Ctrl+C)...\n")
    while True:
        for delay, alert in SCRIPT:
            await asyncio.sleep(delay)
            await broadcast(dict(alert, timestamp=time.time()))
        print("\nScript complete. Restarting in 6s so you can compare states again...\n")
        await asyncio.sleep(6)


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        await play_script()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")

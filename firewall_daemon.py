#!/usr/bin/env python3
"""
firewall_daemon.py
==================
Sovereign Byte-Level Firewall - Live Traffic Daemon.
Listens to a local network interface, extracts packets, and streams them 
through the OR-fused IDS (Byte-level Transformer + SYN Rate Detector).
Now with real-time WebSocket broadcasting for the Web Dashboard!
"""

import os
import sys
import time
import argparse
import logging
import json
import threading
import asyncio
import websockets

import torch
import torch.nn.functional as F
from scapy.all import sniff, TCP, IP

from src.model import NetworkBytePatcher
from src.dataloader import RawPcapIterableDataset

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Global variables for WebSockets
CONNECTED_CLIENTS = set()
loop = None

async def register(websocket):
    CONNECTED_CLIENTS.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        CONNECTED_CLIENTS.remove(websocket)

async def broadcast_alert(alert_data):
    if CONNECTED_CLIENTS:
        message = json.dumps(alert_data)
        websockets.broadcast(CONNECTED_CLIENTS, message)

def trigger_alert_async(alert_type, message, score=None):
    if loop is not None:
        alert_data = {
            "timestamp": time.time(),
            "type": alert_type,
            "message": message,
            "score": score
        }
        asyncio.run_coroutine_threadsafe(broadcast_alert(alert_data), loop)

def parse_args():
    parser = argparse.ArgumentParser(description="Live Fused Firewall Daemon")
    parser.add_argument("--interface", type=str, default="en0", help="Network interface to sniff (e.g. en0, eth0)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest_patcher_ep0_gs750000_mid_epoch.pt", help="Path to best checkpoint")
    parser.add_argument("--byte_threshold", type=float, default=393.1, help="Byte anomaly threshold")
    parser.add_argument("--learning_time", type=int, default=0, help="Run in Learning Mode for X seconds to calculate custom threshold (0 = Disabled)")
    parser.add_argument("--rate_threshold", type=int, default=75, help="SYN rate threshold per 100ms window")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length for byte patcher")
    parser.add_argument("--ws_port", type=int, default=8765, help="WebSocket port for dashboard")
    return parser.parse_args()

def sniff_thread(args, device, model, masker):
    tls_state = {}
    byte_buffer = bytearray()
    current_bucket = int(time.time() * 10)
    syn_count = 0
    
    # --- LEARNING MODE STATE ---
    is_learning = args.learning_time > 0
    learning_start = time.time()
    learning_scores = []
    
    if is_learning:
        logging.info(f"Firewall is in LEARNING MODE on '{args.interface}' for {args.learning_time} seconds...")
        logging.info(f"Please use your browser normally so the AI can learn your traffic baseline.")
    else:
        logging.info(f"Firewall is LIVE. Monitoring traffic on '{args.interface}'...")
    
    def packet_callback(packet):
        nonlocal byte_buffer, current_bucket, syn_count, tls_state, is_learning, learning_scores
        
        # 1. Rate Detector (SYN Flood Check)
        t_bucket = int(time.time() * 10)
        if t_bucket > current_bucket:
            if not is_learning and syn_count > args.rate_threshold:
                msg = f"Volumetric anomaly detected! {syn_count} SYNs in 100ms window."
                logging.warning(f"[RATE ALARM] {msg}")
                trigger_alert_async("RATE", msg, score=syn_count)
            syn_count = 0
            current_bucket = t_bucket
            
        if TCP in packet and packet[TCP].flags & 0x02: # SYN flag
            syn_count += 1
            
        # 2. Byte-level Payload Detector
        raw_bytes = bytes(packet)
        if not raw_bytes:
            return
            
        # Mask the packet
        masked = masker._mask_packet_addresses(raw_bytes, stream_tls_state=tls_state)
        byte_buffer.extend(masked)
        
        while len(byte_buffer) >= args.seq_len:
            window = list(byte_buffer[:args.seq_len])
            del byte_buffer[:args.seq_len]
            
            with torch.no_grad():
                batch = torch.tensor([window], dtype=torch.long, device=device)
                
                # Protect against OOB indexing
                x_in = torch.clamp(batch[:, :-1], min=0)
                
                logits = model(x_in)
                probs = F.softmax(logits, dim=-1)
                entropy_vals = -torch.sum(probs * torch.log2(probs + 1e-9), dim=-1)
                
                # Check for extreme "surprise" threshold
                total_entropy = entropy_vals.sum().item()
                
                if is_learning:
                    learning_scores.append(total_entropy)
                    elapsed = time.time() - learning_start
                    if elapsed > args.learning_time:
                        if len(learning_scores) == 0:
                            logging.error("No packets processed during learning mode. Try again.")
                        else:
                            scores_tensor = torch.tensor(learning_scores)
                            mean_val = scores_tensor.mean().item()
                            std_val = scores_tensor.std().item() if len(learning_scores) > 1 else 0.0
                            calc_threshold = mean_val + (3 * std_val)
                            logging.info(f"=================================================")
                            logging.info(f" CALIBRATION COMPLETE for '{args.interface}' ")
                            logging.info(f" Processed Windows: {len(learning_scores)}")
                            logging.info(f" Mean Entropy:      {mean_val:.2f}")
                            logging.info(f" Std Deviation:     {std_val:.2f}")
                            logging.info(f" -> YOUR MACBOOK THRESHOLD: {calc_threshold:.2f} <- ")
                            logging.info(f" Run again with: --byte_threshold {calc_threshold:.2f}")
                            logging.info(f"=================================================")
                        os._exit(0)  # Kill the daemon cleanly
                else:
                    if total_entropy > args.byte_threshold:
                        msg = f"Payload exploit detected! Entropy score: {total_entropy:.2f} > {args.byte_threshold}"
                        logging.critical(f"[BYTE ALARM] {msg}")
                        trigger_alert_async("BYTE", msg, score=total_entropy)

    try:
        sniff(iface=args.interface, prn=packet_callback, store=False)
    except PermissionError:
        logging.error("Permission denied! Sniffing live traffic requires root privileges. Try running with 'sudo'.")
    except Exception as e:
        logging.error(f"Sniffer crashed: {e}")

async def main_async():
    global loop
    loop = asyncio.get_running_loop()
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    logging.info(f"Hardware Acceleration: {device}")
    
    if not os.path.exists(args.checkpoint):
        logging.error(f"Checkpoint not found at {args.checkpoint}. Exiting.")
        sys.exit(1)
        
    logging.info("Loading NetworkBytePatcher (gs75000 configuration)...")
    model = NetworkBytePatcher(d_model=128, nhead=4, num_layers=2, max_sequence_length=args.seq_len)
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    state_dict = ckpt['model_state']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    masker = RawPcapIterableDataset("dummy.pcap", mask_addresses=True)
    
    # Run the blocking sniffer in a background thread so asyncio can handle WebSockets
    sniffer_thread = threading.Thread(target=sniff_thread, args=(args, device, model, masker), daemon=True)
    sniffer_thread.start()
    
    logging.info(f"Started WebSocket Broadcast Server on ws://localhost:{args.ws_port}")
    async with websockets.serve(register, "localhost", args.ws_port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logging.info("Shutting down Sovereign Firewall.")

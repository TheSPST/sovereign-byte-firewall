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
import math
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

# Scoring is aligned with evaluate_zero_day.py --score_agg topk --topk_frac 0.1
# (the validated recipe). Score = mean of the top 10% per-byte surprise
# (-log2 P(actual next byte)), in bits. 12.0 ~= the Youden threshold found on
# the CIC eval (12.025 bits); per-environment calibration should replace it.
SCORE_METRIC = "surprise_topk"
DEFAULT_BYTE_THRESHOLD = 12.0

def parse_args():
    parser = argparse.ArgumentParser(description="Live Fused Firewall Daemon")
    parser.add_argument("--interface", type=str, default="en0", help="Network interface to sniff (e.g. en0, eth0)")
    parser.add_argument("--checkpoint", type=str, default="ckpt_best/checkpoints/latest_patcher_ep0_gs75000_mid_epoch.pt",
                        help="Path to best checkpoint (default: v2-masking gs75000 - the validated peak)")
    parser.add_argument("--byte_threshold", type=float, default=None,
                        help=f"Byte anomaly threshold in surprise bits. If omitted, uses saved calibration for the interface, else {DEFAULT_BYTE_THRESHOLD}")
    parser.add_argument("--topk_frac", type=float, default=0.1,
                        help="Fraction of bytes (by surprise, highest first) averaged into the window score (default 0.1, matching the validated eval recipe)")
    parser.add_argument("--learning_time", type=int, default=0,
                        help="Calibrate on benign traffic for X seconds, save threshold, then go LIVE automatically (0 = Disabled)")
    parser.add_argument("--rate_threshold", type=int, default=75, help="SYN rate threshold per 100ms window")
    parser.add_argument("--dedup_window", type=float, default=60.0,
                        help="Collapse repeated alerts of the same type within this many seconds into one incident (0 = disable)")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length for byte patcher")
    parser.add_argument("--ws_port", type=int, default=8765, help="WebSocket port for dashboard")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-environment threshold calibration (persisted per interface)
# ---------------------------------------------------------------------------

def calibration_path(interface):
    return f"calibration_{interface}.json"

def load_calibration(interface):
    path = calibration_path(interface)
    if os.path.exists(path):
        try:
            with open(path) as f:
                calib = json.load(f)
            if calib.get("score_metric") != SCORE_METRIC:
                logging.warning(f"Ignoring {path}: calibrated with metric "
                                f"'{calib.get('score_metric', 'entropy_sum (legacy)')}', current metric is "
                                f"'{SCORE_METRIC}'. Re-run with --learning_time to recalibrate.")
                return None
            if "byte_threshold" in calib:
                return calib
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"Could not read {path}: {e}")
    return None

def save_calibration(interface, scores_tensor):
    """Derive a robust threshold from benign scores and persist it.

    Uses the 99.9th percentile (robust to a few outlier windows) and keeps
    mean+3*std as a reference. Returns the calibration dict.
    """
    quantile_thr = torch.quantile(scores_tensor, 0.999).item()
    mean_val = scores_tensor.mean().item()
    std_val = scores_tensor.std().item() if scores_tensor.numel() > 1 else 0.0
    sigma_thr = mean_val + 3 * std_val
    calib = {
        "interface": interface,
        "score_metric": SCORE_METRIC,
        "byte_threshold": max(quantile_thr, sigma_thr),
        "quantile_999": quantile_thr,
        "mean_plus_3sigma": sigma_thr,
        "mean": mean_val,
        "std": std_val,
        "num_windows": int(scores_tensor.numel()),
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = calibration_path(interface)
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)
    logging.info(f"Calibration saved to {path}")
    return calib


# ---------------------------------------------------------------------------
# Incident aggregation (alert deduplication)
# ---------------------------------------------------------------------------

class IncidentAggregator:
    """Collapses repeated alerts of the same type within a time window into
    one incident: the first alert fires immediately, repeats are counted and
    summarized when the incident goes quiet. Turns alert storms (e.g. one
    sustained flood = hundreds of raw alerts) into a handful of incidents."""

    def __init__(self, window_seconds, alert_fn):
        self.window = window_seconds
        self.alert_fn = alert_fn
        self.incidents = {}  # key -> {first_ts, last_ts, count, max_score}

    def report(self, key, message, score=None):
        """Returns True if the alert was emitted, False if suppressed."""
        if self.window <= 0:
            self.alert_fn(key, message, score)
            return True
        now = time.time()
        inc = self.incidents.get(key)
        if inc is not None and (now - inc["last_ts"]) > self.window:
            self._close(key, inc)
            inc = None
        if inc is None:
            self.incidents[key] = {"first_ts": now, "last_ts": now, "count": 1,
                                   "max_score": score if score is not None else 0.0}
            self.alert_fn(key, message, score)
            return True
        inc["count"] += 1
        inc["last_ts"] = now
        if score is not None and score > inc["max_score"]:
            inc["max_score"] = score
        return False

    def flush(self):
        """Close and summarize incidents whose window has expired. Call periodically."""
        now = time.time()
        for key, inc in list(self.incidents.items()):
            if (now - inc["last_ts"]) > self.window:
                self._close(key, inc)

    def _close(self, key, inc):
        if inc["count"] > 1:
            duration = max(1.0, inc["last_ts"] - inc["first_ts"])
            msg = (f"Incident closed: {inc['count']} {key} alerts over {duration:.0f}s "
                   f"(peak score {inc['max_score']:.2f})")
            logging.warning(f"[{key} INCIDENT] {msg}")
            self.alert_fn(key, msg, inc["max_score"])
        self.incidents.pop(key, None)

def sniff_thread(args, device, model, masker):
    tls_state = {}
    byte_buffer = bytearray()
    current_bucket = int(time.time() * 10)
    syn_count = 0

    aggregator = IncidentAggregator(args.dedup_window, trigger_alert_async)

    # --- CALIBRATION MODE STATE ---
    is_learning = args.learning_time > 0
    learning_start = time.time()
    learning_scores = []

    if is_learning:
        logging.info(f"CALIBRATING on '{args.interface}' for {args.learning_time} seconds...")
        logging.info("Use the network normally (benign traffic only) so the AI can learn this environment's baseline.")
        trigger_alert_async("INFO", f"Calibrating baseline for {args.learning_time}s...")
    else:
        logging.info(f"Firewall is LIVE. Monitoring traffic on '{args.interface}' "
                     f"(byte_threshold={args.byte_threshold:.2f}, dedup_window={args.dedup_window:.0f}s)...")

    def finish_calibration():
        nonlocal is_learning
        if len(learning_scores) < 10:
            logging.error(f"Only {len(learning_scores)} windows seen during calibration - not enough. "
                          "Staying in calibration mode; generate more traffic.")
            return
        calib = save_calibration(args.interface, torch.tensor(learning_scores))
        args.byte_threshold = calib["byte_threshold"]
        is_learning = False
        logging.info("=================================================")
        logging.info(f" CALIBRATION COMPLETE for '{args.interface}'")
        logging.info(f" Windows: {calib['num_windows']} | Mean: {calib['mean']:.2f} | Std: {calib['std']:.2f}")
        logging.info(f" q99.9: {calib['quantile_999']:.2f} | mean+3s: {calib['mean_plus_3sigma']:.2f}")
        logging.info(f" -> LIVE with byte_threshold = {args.byte_threshold:.2f} <-")
        logging.info("=================================================")
        trigger_alert_async("INFO", f"Calibration complete. LIVE with threshold {args.byte_threshold:.2f}")

    def packet_callback(packet):
        nonlocal byte_buffer, current_bucket, syn_count, tls_state

        # 1. Rate Detector (SYN Flood Check)
        t_bucket = int(time.time() * 10)
        if t_bucket > current_bucket:
            if not is_learning and syn_count > args.rate_threshold:
                msg = f"Volumetric anomaly detected! {syn_count} SYNs in 100ms window."
                if aggregator.report("RATE", msg, score=syn_count):
                    logging.warning(f"[RATE ALARM] {msg}")
            aggregator.flush()
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
                targets = batch[:, 1:]

                # Surprise scoring - mirrors evaluate_zero_day.py
                # (--score_agg topk --topk_frac 0.1), the recipe validated on
                # CIC: surprise_t = -log2 P(byte_t+1 | byte_<=t), window score
                # = mean of the top 10% most surprising bytes.
                logits = model(x_in)
                log_probs = F.log_softmax(logits, dim=-1)
                gather_idx = torch.clamp(targets, min=0).unsqueeze(-1)
                token_logprob = log_probs.gather(-1, gather_idx).squeeze(-1)
                surprise_bits = -token_logprob / math.log(2)
                k = max(1, min(surprise_bits.shape[1], int(round(args.topk_frac * surprise_bits.shape[1]))))
                topk_vals, _ = torch.topk(surprise_bits, k=k, dim=1)
                window_score = topk_vals.mean().item()

                if is_learning:
                    learning_scores.append(window_score)
                    if (time.time() - learning_start) > args.learning_time:
                        finish_calibration()  # transitions to LIVE in-place, no restart needed
                else:
                    if window_score > args.byte_threshold:
                        msg = f"Payload anomaly detected! Surprise: {window_score:.2f} bits > {args.byte_threshold:.2f}"
                        if aggregator.report("BYTE", msg, score=window_score):
                            logging.critical(f"[BYTE ALARM] {msg}")

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

    # Threshold resolution: explicit CLI flag > saved per-interface calibration > built-in default
    if args.byte_threshold is None:
        calib = load_calibration(args.interface)
        if calib is not None:
            args.byte_threshold = calib["byte_threshold"]
            logging.info(f"Loaded saved calibration for '{args.interface}' "
                         f"(threshold={args.byte_threshold:.2f}, calibrated {calib.get('calibrated_at', '?')}, "
                         f"{calib.get('num_windows', '?')} windows)")
        else:
            args.byte_threshold = DEFAULT_BYTE_THRESHOLD
            if args.learning_time == 0:
                logging.warning(f"No calibration found for '{args.interface}'. Using default threshold "
                                f"{DEFAULT_BYTE_THRESHOLD} - consider running with --learning_time 300 first.")
    
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

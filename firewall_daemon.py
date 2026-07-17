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
import bisect
import argparse
import logging
import json
import threading
import asyncio
import websockets
from collections import deque, Counter

import torch
import torch.nn.functional as F
from scapy.all import sniff, TCP, IP, UDP

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

def trigger_alert_async(alert_type, message, score=None, enrichment=None):
    if loop is not None:
        alert_data = {
            "timestamp": time.time(),
            "type": alert_type,
            "message": message,
            "score": score,
            "enrichment": enrichment or {},
        }
        asyncio.run_coroutine_threadsafe(broadcast_alert(alert_data), loop)


# ---------------------------------------------------------------------------
# Incident enrichment (deterministic facts, no LLM)
# ---------------------------------------------------------------------------

def compute_enrichment(pkt_meta):
    """Summarize the recent packet-metadata ring buffer into human-readable
    facts an analyst can triage from: top talkers, ports, protocol mix, SYNs.
    pkt_meta entries: (ts, src, dst, sport, dport, proto, size, is_syn)."""
    if not pkt_meta:
        return {}
    talkers, ports, protos = Counter(), Counter(), Counter()
    total_bytes = syns = 0
    for (_ts, src, dst, _sp, dport, proto, size, is_syn) in pkt_meta:
        talkers[f"{src} -> {dst}"] += size
        if dport:
            ports[dport] += 1
        protos[proto] += 1
        total_bytes += size
        if is_syn:
            syns += 1
    n = len(pkt_meta)
    return {
        "packets": n,
        "bytes": total_bytes,
        "top_talkers": [{"pair": k, "bytes": v} for k, v in talkers.most_common(3)],
        "top_ports": [p for p, _ in ports.most_common(5)],
        "proto_mix_pct": {p: round(100.0 * c / n) for p, c in protos.most_common()},
        "syns": syns,
    }


def score_percentile(score, calib):
    """Map a live surprise score to its percentile in the benign calibration
    distribution, using the stored quantile array. Returns None if unavailable."""
    if not calib:
        return None
    q = calib.get("score_quantiles")
    if not q:
        return None
    # q[i] is the i-th percentile value (i in 0..100); locate the score.
    return min(100, max(0, bisect.bisect_left(q, score)))

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
    parser.add_argument("--target_incidents_per_hour", type=float, default=1.0,
                        help="Alert budget: pick the calibration threshold that yields at most this many "
                             "byte-alerts/hour on the benign baseline (0 = fall back to the 99.9th-percentile / "
                             "mean+3sigma rule). Actual incidents are <= this after dedup.")
    parser.add_argument("--rate_threshold", type=int, default=75, help="SYN rate threshold per 100ms window")
    parser.add_argument("--dedup_window", type=float, default=60.0,
                        help="Collapse repeated alerts of the same type within this many seconds into one incident (0 = disable)")
    parser.add_argument("--incident_log", type=str, default=None,
                        help="CSV file to append closed incidents to (default: incidents_<interface>.csv; 'none' to disable)")
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

def save_calibration(interface, scores_tensor, elapsed_seconds=0.0, target_iph=0.0):
    """Derive a per-environment threshold from benign scores and persist it.

    If target_iph > 0 (an alert budget in byte-alerts/hour), the threshold is
    the score quantile that yields that many benign exceedances/hour given the
    measured window throughput - i.e. you tell it how many alerts/hour you can
    tolerate and it sets the bar. Raw exceedances upper-bound incidents (dedup
    only reduces them), so real incidents/hour <= target.

    If target_iph <= 0, falls back to max(99.9th percentile, mean+3*std).
    """
    quantile_thr = torch.quantile(scores_tensor, 0.999).item()
    mean_val = scores_tensor.mean().item()
    std_val = scores_tensor.std().item() if scores_tensor.numel() > 1 else 0.0
    sigma_thr = mean_val + 3 * std_val
    n = int(scores_tensor.numel())

    window_rate = (n / elapsed_seconds) if elapsed_seconds > 0 else 0.0  # windows/sec
    budget_thr = None
    budget_q = None
    if target_iph > 0 and window_rate > 0:
        windows_per_hour = window_rate * 3600.0
        frac = target_iph / windows_per_hour           # fraction of windows allowed over the bar
        frac = min(max(frac, 1e-6), 0.5)               # clamp to a sane range
        budget_q = 1.0 - frac
        budget_thr = torch.quantile(scores_tensor, budget_q).item()

    if budget_thr is not None:
        threshold = budget_thr
        method = f"alert-budget ({target_iph:g}/hr -> q{budget_q:.5f})"
    else:
        threshold = max(quantile_thr, sigma_thr)
        method = "max(q99.9, mean+3sigma)"

    # 0..100th percentile values, for mapping a live score to its percentile.
    try:
        qs = torch.tensor([i / 100.0 for i in range(101)])
        score_quantiles = torch.quantile(scores_tensor, qs).tolist()
    except Exception:
        score_quantiles = None

    calib = {
        "interface": interface,
        "score_metric": SCORE_METRIC,
        "byte_threshold": threshold,
        "threshold_method": method,
        "score_quantiles": score_quantiles,
        "target_incidents_per_hour": target_iph,
        "window_rate_per_sec": round(window_rate, 3),
        "budget_quantile": budget_q,
        "budget_threshold": budget_thr,
        "quantile_999": quantile_thr,
        "mean_plus_3sigma": sigma_thr,
        "mean": mean_val,
        "std": std_val,
        "num_windows": n,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = calibration_path(interface)
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)
    logging.info(f"Calibration saved to {path} (method: {method})")
    return calib


# ---------------------------------------------------------------------------
# Incident aggregation (alert deduplication)
# ---------------------------------------------------------------------------

class IncidentAggregator:
    """Collapses repeated alerts of the same type within a time window into
    one incident: the first alert fires immediately, repeats are counted and
    summarized when the incident goes quiet. Turns alert storms (e.g. one
    sustained flood = hundreds of raw alerts) into a handful of incidents."""

    def __init__(self, window_seconds, alert_fn, incident_log=None):
        self.window = window_seconds
        self.alert_fn = alert_fn
        self.incidents = {}  # key -> {first_ts, last_ts, count, max_score}
        self.incident_log = incident_log
        if incident_log and not os.path.exists(incident_log):
            with open(incident_log, "w") as f:
                f.write("opened_at,closed_at,type,raw_alerts,peak_score,context\n")

    def report(self, key, message, score=None, enrichment=None):
        """Returns True if the alert was emitted, False if suppressed.
        Enrichment (computed facts) is captured on incident open and carried
        through to the WS payload and the incident-log context column."""
        if self.window <= 0:
            self.alert_fn(key, message, score, enrichment)
            return True
        now = time.time()
        inc = self.incidents.get(key)
        if inc is not None and (now - inc["last_ts"]) > self.window:
            self._close(key, inc)
            inc = None
        if inc is None:
            self.incidents[key] = {"first_ts": now, "last_ts": now, "count": 1,
                                   "max_score": score if score is not None else 0.0,
                                   "enrichment": enrichment or {}}
            self.alert_fn(key, message, score, enrichment)
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
            self.alert_fn(key, msg, inc["max_score"], inc.get("enrichment"))
        if self.incident_log:
            try:
                # context = compact JSON of the opening enrichment (CSV-safe: quoted).
                # Drop the heatmap array — it's for the live dashboard, not the CSV.
                ctx_data = {k: v for k, v in (inc.get("enrichment") or {}).items() if k != "heatmap"}
                ctx = json.dumps(ctx_data, separators=(",", ":")).replace('"', "'")
                with open(self.incident_log, "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(inc['first_ts']))},"
                            f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(inc['last_ts']))},"
                            f"{key},{inc['count']},{inc['max_score']:.2f},\"{ctx}\"\n")
            except OSError as e:
                logging.warning(f"Could not write incident log: {e}")
        self.incidents.pop(key, None)

def sniff_thread(args, device, model, masker, calib=None):
    tls_state = {}
    byte_buffer = bytearray()
    current_bucket = int(time.time() * 10)
    syn_count = 0
    current_calib = calib  # updated in-place when calibration finishes

    # Rolling packet-metadata ring buffer (~last 10s) for incident enrichment.
    META_WINDOW = 10.0
    pkt_meta = deque()

    incident_log = args.incident_log
    if incident_log is None:
        incident_log = f"incidents_{args.interface}.csv"
    elif incident_log.lower() == "none":
        incident_log = None
    if incident_log:
        logging.info(f"Incident log: {incident_log}")
    aggregator = IncidentAggregator(args.dedup_window, trigger_alert_async, incident_log=incident_log)

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
        nonlocal current_calib
        elapsed = time.time() - learning_start
        calib = save_calibration(args.interface, torch.tensor(learning_scores),
                                 elapsed_seconds=elapsed, target_iph=args.target_incidents_per_hour)
        current_calib = calib
        args.byte_threshold = calib["byte_threshold"]
        is_learning = False
        logging.info("=================================================")
        logging.info(f" CALIBRATION COMPLETE for '{args.interface}'")
        logging.info(f" Windows: {calib['num_windows']} | Mean: {calib['mean']:.2f} | Std: {calib['std']:.2f} "
                     f"| Rate: {calib['window_rate_per_sec']:.1f} win/s")
        logging.info(f" Method: {calib['threshold_method']} | q99.9: {calib['quantile_999']:.2f} "
                     f"| mean+3s: {calib['mean_plus_3sigma']:.2f}")
        logging.info(f" -> LIVE with byte_threshold = {args.byte_threshold:.2f} "
                     f"(budget {args.target_incidents_per_hour:g} alerts/hr) <-")
        logging.info("=================================================")
        trigger_alert_async("INFO", f"Calibration complete. LIVE with threshold {args.byte_threshold:.2f}")

    def packet_callback(packet):
        nonlocal byte_buffer, current_bucket, syn_count, tls_state

        now = time.time()
        is_syn = bool(TCP in packet and packet[TCP].flags & 0x02)

        # 1. Rate Detector (SYN Flood Check)
        t_bucket = int(now * 10)
        if t_bucket > current_bucket:
            if not is_learning and syn_count > args.rate_threshold:
                msg = f"Volumetric anomaly detected! {syn_count} SYNs in 100ms window."
                if aggregator.report("RATE", msg, score=syn_count, enrichment=compute_enrichment(pkt_meta)):
                    logging.warning(f"[RATE ALARM] {msg}")
            aggregator.flush()
            syn_count = 0
            current_bucket = t_bucket

        if is_syn:
            syn_count += 1

        raw_bytes = bytes(packet)
        if not raw_bytes:
            return

        # Capture packet metadata for incident enrichment (cheap, best-effort).
        try:
            if IP in packet:
                ipl = packet[IP]
                if TCP in packet:
                    proto, dport = "TCP", packet[TCP].dport
                elif UDP in packet:
                    proto, dport = "UDP", packet[UDP].dport
                else:
                    proto, dport = "other", 0
                pkt_meta.append((now, ipl.src, ipl.dst, 0, dport, proto, len(raw_bytes), is_syn))
                cutoff = now - META_WINDOW
                while pkt_meta and pkt_meta[0][0] < cutoff:
                    pkt_meta.popleft()
        except Exception:
            pass

        # 2. Byte-level Payload Detector
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
                        pct = score_percentile(window_score, current_calib)
                        pct_str = f" ({pct}th pct of baseline)" if pct is not None else ""
                        msg = (f"Payload anomaly detected! Surprise: {window_score:.2f} bits > "
                               f"{args.byte_threshold:.2f}{pct_str}")
                        enrich = compute_enrichment(pkt_meta)
                        if pct is not None:
                            enrich["score_percentile"] = pct
                        # Per-byte surprise for the flagged window -> dashboard
                        # heatmap ("what the model saw"). surprise_bits[i] scores
                        # the prediction of window[i+1], so pair them 1:1.
                        sb = surprise_bits[0].tolist()
                        enrich["heatmap"] = {
                            "bytes": [int(b) for b in window[1:]],
                            "surprise": [round(float(x), 2) for x in sb],
                        }
                        if aggregator.report("BYTE", msg, score=window_score, enrichment=enrich):
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

    # Load saved calibration (used for the threshold if not overridden, and for
    # score-percentile enrichment regardless).
    loaded_calib = load_calibration(args.interface)

    # Threshold resolution: explicit CLI flag > saved per-interface calibration > built-in default
    if args.byte_threshold is None:
        if loaded_calib is not None:
            args.byte_threshold = loaded_calib["byte_threshold"]
            logging.info(f"Loaded saved calibration for '{args.interface}' "
                         f"(threshold={args.byte_threshold:.2f}, calibrated {loaded_calib.get('calibrated_at', '?')}, "
                         f"{loaded_calib.get('num_windows', '?')} windows)")
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
    sniffer_thread = threading.Thread(target=sniff_thread, args=(args, device, model, masker, loaded_calib), daemon=True)
    sniffer_thread.start()
    
    logging.info(f"Started WebSocket Broadcast Server on ws://localhost:{args.ws_port}")
    async with websockets.serve(register, "localhost", args.ws_port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logging.info("Shutting down Sovereign Firewall.")

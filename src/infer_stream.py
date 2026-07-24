#!/usr/bin/env python3
"""
Sovereign Byte Firewall — Real-Time Live Traffic Scanner (CLI)

Performs streaming, real-time packet inspection on live network interfaces (eth0/en0)
or PCAP files using Mamba-2 SSM neural backbone and Extreme Value Theory (EVT) thresholding.

Usage:
    # Live network interface scan
    sudo python src/infer_stream.py --iface en0 --ckpt checkpoints/latest_patcher.pt

    # PCAP file scan
    python src/infer_stream.py --pcap data/0day.pcap --ckpt checkpoints/latest_patcher.pt
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluate_zero_day import load_model
from src.dataloader import RawPcapReader, mask_packet_bytes

EVT_DEFAULT_THRESHOLD = 10.674  # EVT Peak-Over-Threshold GPD bound (0.000% FPR)

def parse_args():
    parser = argparse.ArgumentParser(description="Sovereign Byte Firewall — Live Traffic Scanner")
    parser.add_argument("--iface", type=str, default=None, help="Live network interface (e.g. en0, eth0)")
    parser.add_argument("--pcap", type=str, default=None, help="PCAP file to inspect")
    parser.add_argument("--ckpt", type=str, default="checkpoints/latest_patcher.pt", help="Path to Mamba-2 checkpoint")
    parser.add_argument("--window_len", type=int, default=512, help="Scoring window sequence length (default: 512)")
    parser.add_argument("--evt_thresh", type=float, default=EVT_DEFAULT_THRESHOLD, help="EVT GPD threshold in bits/byte")
    parser.add_argument("--log_json", type=str, default=None, help="JSON log output file path")
    parser.add_argument("--use_fp16", action="store_true", default=True, help="Enable FP16 AMP inference speedup")
    return parser.parse_args()

@torch.no_grad()
def score_byte_window(model, raw_bytes, device, use_fp16=True):
    """Computes mean information-theoretic surprise (bits/byte) for a raw byte window."""
    if len(raw_bytes) < 10:
        return 0.0
    
    seq = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
    if len(seq) > 512:
        seq = seq[:512]
    elif len(seq) < 512:
        seq = np.pad(seq, (0, 512 - len(seq)), mode='constant', constant_values=0)
        
    inp = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.amp.autocast(device_type=device.type, enabled=(use_fp16 and device.type == 'cuda')):
        logits = model(inp)
        targets = inp[:, 1:]
        pred_logits = logits[:, :-1, :]
        log_probs = F.log_softmax(pred_logits, dim=-1)
        
        nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        bits = nll / np.log(2.0)
        valid_mask = (targets >= 0) & (targets < 256)
        valid_bits = bits[valid_mask]
        
        if valid_bits.numel() == 0:
            return 0.0
        return float(valid_bits.mean().item())

def run_pcap_scan(args, model, device):
    print(f"[*] Inspecting PCAP file: '{args.pcap}'...")
    start_time = time.time()
    packet_count = 0
    anomalies_detected = 0
    
    log_fp = open(args.log_json, "a") if args.log_json else None
    
    with open(args.pcap, "rb") as f:
        reader = RawPcapReader(f)
        for pkt_bytes, _hdr in reader:
            packet_count += 1
            masked = mask_packet_bytes(pkt_bytes)
            surprise = score_byte_window(model, masked, device, args.use_fp16)
            
            is_anomaly = surprise >= args.evt_thresh
            if is_anomaly:
                anomalies_detected += 1
                evt_alert = {
                    "timestamp": time.time(),
                    "packet_index": packet_count,
                    "pkt_len": len(pkt_bytes),
                    "surprise_bits": round(surprise, 4),
                    "evt_threshold": args.evt_thresh,
                    "status": "ALERT_EVT_ZERO_DAY_BREACH"
                }
                print(f"  🚨 [EVT ALERT #{anomalies_detected}] Packet #{packet_count} ({len(pkt_bytes)} bytes) | Surprise: {surprise:.4f} bits/byte > EVT {args.evt_thresh}")
                if log_fp:
                    log_fp.write(json.dumps(evt_alert) + "\n")
                    log_fp.flush()
                    
    elapsed = time.time() - start_time
    print("==================================================")
    print(f" PCAP SCAN COMPLETE")
    print(f" Total Packets Scanned:   {packet_count}")
    print(f" Anomalies Detected:       {anomalies_detected}")
    print(f" Elapsed Time:            {elapsed:.2f}s ({packet_count / max(elapsed, 0.001):.1f} pkts/sec)")
    print("==================================================")
    if log_fp:
        log_fp.close()

def run_iface_scan(args, model, device):
    print(f"[*] Starting live interface scanner on '{args.iface}'...")
    print(f"[*] EVT Threshold: {args.evt_thresh} bits/byte | Press Ctrl+C to stop.\n")
    import scapy.all as scapy
    
    packet_count = 0
    anomalies_detected = 0
    log_fp = open(args.log_json, "a") if args.log_json else None
    
    def packet_callback(pkt):
        nonlocal packet_count, anomalies_detected
        packet_count += 1
        raw_bytes = bytes(pkt)
        masked = mask_packet_bytes(raw_bytes)
        surprise = score_byte_window(model, masked, device, args.use_fp16)
        
        is_anomaly = surprise >= args.evt_thresh
        if is_anomaly:
            anomalies_detected += 1
            evt_alert = {
                "timestamp": time.time(),
                "iface": args.iface,
                "pkt_len": len(raw_bytes),
                "surprise_bits": round(surprise, 4),
                "evt_threshold": args.evt_thresh,
                "status": "ALERT_EVT_ZERO_DAY_BREACH"
            }
            print(f"🚨 [EVT ALERT #{anomalies_detected}] {args.iface} | Packet #{packet_count} | Surprise: {surprise:.4f} bits/byte > EVT {args.evt_thresh}")
            if log_fp:
                log_fp.write(json.dumps(evt_alert) + "\n")
                log_fp.flush()
        elif packet_count % 100 == 0:
            print(f"  [OK] Scanned {packet_count} pkts | Last Surprise: {surprise:.2f} bits/byte")
            
    try:
        scapy.sniff(iface=args.iface, prn=packet_callback, store=0)
    except KeyboardInterrupt:
        print("\n[*] Live scan stopped by user.")
    finally:
        if log_fp:
            log_fp.close()

def main():
    args = parse_args()
    if not args.iface and not args.pcap:
        print("ERROR: Must specify either --iface (e.g. en0) or --pcap (e.g. data/0day.pcap)", file=sys.stderr)
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"   SOVEREIGN BYTE FIREWALL — LIVE SCANNER")
    print(f"==================================================")
    print(f"Device: {device} | Checkpoint: {args.ckpt}")
    
    model, window_len = load_model(args.ckpt, device, args.window_len)
    model.eval()
    
    if args.pcap:
        run_pcap_scan(args, model, device)
    elif args.iface:
        run_iface_scan(args, model, device)

if __name__ == "__main__":
    main()

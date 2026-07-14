#!/usr/bin/env python3
"""
firewall_daemon.py
==================
Sovereign Byte-Level Firewall - Live Traffic Daemon.
Listens to a local network interface, extracts packets, and streams them 
through the OR-fused IDS (Byte-level Transformer + SYN Rate Detector).
"""

import os
import sys
import time
import argparse
import logging
from collections import deque

import torch
import torch.nn.functional as F
from scapy.all import sniff, TCP, IP

from src.model import NetworkBytePatcher
from src.dataloader import RawPcapIterableDataset

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Live Fused Firewall Daemon")
    parser.add_argument("--interface", type=str, default="en0", help="Network interface to sniff (e.g. en0, eth0)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest_patcher_ep0_gs75000_mid_epoch.pt", help="Path to best checkpoint")
    parser.add_argument("--byte_threshold", type=float, default=393.1, help="Byte anomaly threshold (gs75000 Youden)")
    parser.add_argument("--rate_threshold", type=int, default=75, help="SYN rate threshold per 100ms window")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length for byte patcher")
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    logging.info(f"Starting Sovereign Firewall Daemon on Interface: {args.interface}")
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
    
    # Set up our address masker to sanitize raw IPs and MACs
    masker = RawPcapIterableDataset("dummy.pcap", mask_addresses=True)
    tls_state = {}
    
    byte_buffer = bytearray()
    
    # Rate detector state
    current_bucket = int(time.time() * 10) # 100ms buckets
    syn_count = 0
    
    logging.info(f"Firewall is LIVE. Monitoring traffic on '{args.interface}'...")
    
    def packet_callback(packet):
        nonlocal byte_buffer, current_bucket, syn_count, tls_state
        
        # 1. Rate Detector (SYN Flood Check)
        t_bucket = int(time.time() * 10)
        if t_bucket > current_bucket:
            # Time bucket rolled over, check previous bucket
            if syn_count > args.rate_threshold:
                logging.warning(f"[RATE ALARM] Volumetric anomaly detected! {syn_count} SYNs in 100ms window.")
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
                
                # Protect against OOB indexing on -1 sentinel masks if any leaked through
                x_in = torch.clamp(batch[:, :-1], min=0)
                
                logits = model(x_in)
                probs = F.softmax(logits, dim=-1)
                entropy_vals = -torch.sum(probs * torch.log2(probs + 1e-9), dim=-1)
                
                # Check for extreme "surprise" threshold
                total_entropy = entropy_vals.sum().item()
                if total_entropy > args.byte_threshold:
                    logging.critical(f"[BYTE ALARM] Payload exploit detected! Entropy score: {total_entropy:.2f} > {args.byte_threshold}")

    try:
        sniff(iface=args.interface, prn=packet_callback, store=False)
    except PermissionError:
        logging.error("Permission denied! Sniffing live traffic requires root privileges. Try running with 'sudo'.")
    except Exception as e:
        logging.error(f"Sniffer crashed: {e}")

if __name__ == "__main__":
    main()

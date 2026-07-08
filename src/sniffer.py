#!/usr/bin/env python3
"""
Live Packet Sniffer & Next-Byte Anomaly Detection Firewall.
Sniffs live packet sequences on a network interface, applies the dynamic
enterprise masking rules, runs predictions via the compiled ONNX model,
and measures Shannon entropy spikes (surprise metrics) to identify zero-days.

Usage:
  sudo python src/sniffer.py --onnx_path checkpoints/latest_patcher.onnx --interface en0 --threshold 5.5
"""

import os
import sys
import argparse
import math
import numpy as np
import onnxruntime as ort

try:
    from scapy.all import sniff, IP, IPv6, ARP, TCP
except ImportError:
    print("ERROR: Scapy is required to run the sniffer. Install it via: pip install scapy")
    sys.exit(1)

# Helper function to softmax numpy logits
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class LiveAnomalyFirewall:
    def __init__(self, onnx_path, threshold=5.5, mask_addresses=True):
        self.threshold = threshold
        self.mask_addresses = mask_addresses
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found at: {onnx_path}")
            
        print(f"Loading ONNX Inference session from: '{onnx_path}'...")
        # Use CPU provider for lightweight inline routing
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print("ONNX model loaded successfully [OK]")

    def mask_packet(self, packet_data):
        """
        Applies Layer 2, 3, 4, and 7 masking rules in-place to ensure the model
        remains network-agnostic.
        """
        pkt_bytes = bytearray(packet_data)
        n = len(pkt_bytes)
        
        # 1. Layer 2 MAC Erasure
        if n >= 12:
            pkt_bytes[0:12] = b'\x00' * 12
            
        # 2. Layer 3 Offset Calculation (VLAN / QinQ / MPLS / IPv4 / IPv6 / ARP)
        if n < 14:
            return bytes(pkt_bytes)
            
        offset = 12
        ethertype = (pkt_bytes[offset] << 8) | pkt_bytes[offset + 1]
        offset += 2
        
        # Handle single/double VLAN tag loop
        while ethertype in (0x8100, 0x88a8, 0x9100, 0x9200):
            if n < offset + 4:
                return bytes(pkt_bytes)
            ethertype = (pkt_bytes[offset + 2] << 8) | pkt_bytes[offset + 3]
            offset += 4
            
        l3_offset = offset
        resolved_ethertype = ethertype
        
        # Handle MPLS (0x8847 / 0x8848)
        if resolved_ethertype in (0x8847, 0x8848):
            while True:
                if n < l3_offset + 4:
                    return bytes(pkt_bytes)
                bos = pkt_bytes[l3_offset + 2] & 0x01
                l3_offset += 4
                if bos:
                    break
            if n < l3_offset + 1:
                return bytes(pkt_bytes)
            version = (pkt_bytes[l3_offset] >> 4) & 0x0F
            if version == 4:
                resolved_ethertype = 0x0800
            elif version == 6:
                resolved_ethertype = 0x86DD
            else:
                resolved_ethertype = None
                
        # 3. Layer 3 Address Masking
        l4_offset = None
        is_tcp = False
        
        if resolved_ethertype == 0x0800:  # IPv4
            if n >= l3_offset + 20:
                pkt_bytes[l3_offset + 12 : l3_offset + 20] = b'\x00' * 8
                protocol = pkt_bytes[l3_offset + 9]
                if protocol == 6:  # TCP
                    ihl = pkt_bytes[l3_offset] & 0x0F
                    l4_offset = l3_offset + (ihl * 4)
                    is_tcp = True
                    
        elif resolved_ethertype == 0x86DD:  # IPv6
            if n >= l3_offset + 40:
                pkt_bytes[l3_offset + 8 : l3_offset + 40] = b'\x00' * 32
                next_header = pkt_bytes[l3_offset + 6]
                if next_header == 6:  # TCP
                    l4_offset = l3_offset + 40
                    is_tcp = True
                    
        elif resolved_ethertype == 0x0806:  # ARP
            if n >= l3_offset + 28:
                pkt_bytes[l3_offset + 8 : l3_offset + 28] = b'\x00' * 20
                
        # 4. Layer 4 TCP & Layer 7 TLS Masking
        if is_tcp and l4_offset is not None:
            if n >= l4_offset + 20:
                tcp_data_offset = (pkt_bytes[l4_offset + 12] >> 4) & 0x0F
                tcp_header_len = tcp_data_offset * 4
                
                # TCP Option Masking
                if tcp_header_len > 20 and n >= l4_offset + tcp_header_len:
                    pkt_bytes[l4_offset + 20 : l4_offset + tcp_header_len] = b'\x00' * (tcp_header_len - 20)
                    
                # TLS Encrypted Payload Bypass
                sport = (pkt_bytes[l4_offset] << 8) | pkt_bytes[l4_offset + 1]
                dport = (pkt_bytes[l4_offset + 2] << 8) | pkt_bytes[l4_offset + 3]
                if sport == 443 or dport == 443:
                    tls_offset = l4_offset + tcp_header_len
                    if n >= tls_offset + 5:
                        tls_content_type = pkt_bytes[tls_offset]
                        if tls_content_type == 0x17:  # Application Data
                            pkt_bytes[tls_offset + 5 : n] = b'\x00' * (n - (tls_offset + 5))
                            
        return bytes(pkt_bytes)

    def analyze_packet(self, packet):
        """
        Callback triggered on packet arrival. Evaluates next-byte surprises.
        """
        raw_data = bytes(packet)
        if not raw_data:
            return

        # Step 1: Pre-process with masking rules
        processed_data = self.mask_packet(raw_data) if self.mask_addresses else raw_data
        
        # Step 2: Prepare sequence inputs (Shape: 1, Sequence_Length)
        seq_len = len(processed_data)
        if seq_len < 2:
            return
            
        input_tokens = np.array([list(processed_data)], dtype=np.int64)
        
        # Step 3: Run ONNX Inference
        try:
            # inputs shape: (1, seq_len) -> outputs shape: (1, seq_len, 256)
            logits = self.session.run([self.output_name], {self.input_name: input_tokens})[0]
        except Exception as e:
            # Handles sequence length edge-cases or shape compilation limits gracefully
            return
            
        # Step 4: Evaluate Shannon Surprise (Entropy) for the sequence transitions
        # We look at predictions at index i, and surprise compared to true next byte at i+1
        predicted_logits = logits[0, :-1, :]  # Shape: (seq_len - 1, 256)
        next_byte_targets = input_tokens[0, 1:]  # Shape: (seq_len - 1,)
        
        probabilities = softmax(predicted_logits)
        target_probs = probabilities[np.arange(len(next_byte_targets)), next_byte_targets]
        
        # Clip to prevent log(0) underflows
        target_probs = np.clip(target_probs, 1e-9, 1.0)
        surprises = -np.log2(target_probs)
        
        # Step 5: Detect Anomaly
        avg_surprise = np.mean(surprises)
        max_surprise = np.max(surprises)
        
        if avg_surprise > self.threshold:
            print(f"[ALERT] Anomaly Detected! Average Surprise: {avg_surprise:.4f} bits | Max: {max_surprise:.4f} bits | Length: {seq_len} bytes")
            # If integrated with iptables or nfqueue, drop the packet here.
        else:
            print(f"[PASS] Packet ok. Avg Surprise: {avg_surprise:.4f} bits | Length: {seq_len} bytes")

def main():
    parser = argparse.ArgumentParser(description="Live Packet Anomaly Detection Firewall")
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="checkpoints/latest_patcher.onnx",
        help="Path to compiled ONNX model (default: checkpoints/latest_patcher.onnx)"
    )
    parser.add_argument(
        "--interface",
        type=str,
        default=None,
        help="Interface to sniff on (e.g. en0, eth0, or None for default)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.5,
        help="Anomaly threshold in bits of Shannon Surprise (default: 5.5)"
    )
    parser.add_argument(
        "--disable_masking",
        action="store_true",
        default=False,
        help="Disable Dynamic Protocol Address Masking"
    )
    args = parser.parse_args()
    
    try:
        firewall = LiveAnomalyFirewall(
            onnx_path=args.onnx_path,
            threshold=args.threshold,
            mask_addresses=not args.disable_masking
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please download the PyTorch checkpoint from Kaggle and run export_onnx.py first.")
        sys.exit(1)
        
    print(f"\nStarting live packet sniffing on interface: '{args.interface or 'default'}'...")
    print("Press Ctrl+C to terminate.")
    
    try:
        sniff(iface=args.interface, prn=firewall.analyze_packet, store=False)
    except PermissionError:
        print("ERROR: Sniffing requires root privileges. Please run with 'sudo python src/sniffer.py'")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nSniffer firewall terminated by user.")

if __name__ == "__main__":
    main()

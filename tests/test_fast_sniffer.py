"""
tests/test_fast_sniffer.py
===========================
Unit tests and throughput benchmarks for src/fast_sniffer.py.
Compares C-struct zero-Scapy parsing against Scapy packet inspection over 10,000 iterations.
"""
import time
import struct
import socket
import pytest

from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP, UDP

from src.fast_sniffer import parse_packet_fast


def _build_tcp_packet(src_ip="192.168.1.100", dst_ip="10.0.0.1", sport=12345, dport=80, flags="S"):
    pkt = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags=flags) / b"HELLO_WORLD_PAYLOAD"
    return bytes(pkt)


def _build_udp_packet(src_ip="192.168.1.100", dst_ip="10.0.0.1", sport=5353, dport=53):
    pkt = Ether() / IP(src=src_ip, dst=dst_ip) / UDP(sport=sport, dport=dport) / b"DNS_QUERY"
    return bytes(pkt)


def test_parse_packet_fast_tcp_syn():
    raw = _build_tcp_packet(flags="S")
    parsed = parse_packet_fast(raw)
    assert parsed is not None
    src_ip, sport, dst_ip, dport, proto, is_syn = parsed
    assert src_ip == "192.168.1.100"
    assert sport == 12345
    assert dst_ip == "10.0.0.1"
    assert dport == 80
    assert proto == "TCP"
    assert is_syn is True


def test_parse_packet_fast_tcp_ack():
    raw = _build_tcp_packet(flags="A")
    parsed = parse_packet_fast(raw)
    assert parsed is not None
    src_ip, sport, dst_ip, dport, proto, is_syn = parsed
    assert src_ip == "192.168.1.100"
    assert sport == 12345
    assert dst_ip == "10.0.0.1"
    assert dport == 80
    assert proto == "TCP"
    assert is_syn is False


def test_parse_packet_fast_udp():
    raw = _build_udp_packet()
    parsed = parse_packet_fast(raw)
    assert parsed is not None
    src_ip, sport, dst_ip, dport, proto, is_syn = parsed
    assert src_ip == "192.168.1.100"
    assert sport == 5353
    assert dst_ip == "10.0.0.1"
    assert dport == 53
    assert proto == "UDP"
    assert is_syn is False


def test_parse_packet_fast_invalid():
    assert parse_packet_fast(b"too_short") is None
    assert parse_packet_fast(None) is None


def test_fast_sniffer_speed_benchmark():
    """Benchmark parse_packet_fast vs Scapy over 10,000 packet parsing iterations."""
    raw_tcp = _build_tcp_packet()
    iterations = 10000

    # 1. Measure Scapy parsing time
    t0 = time.perf_counter()
    for _ in range(iterations):
        scapy_pkt = Ether(raw_tcp)
        if IP in scapy_pkt:
            src = scapy_pkt[IP].src
            dst = scapy_pkt[IP].dst
            if TCP in scapy_pkt:
                dp = scapy_pkt[TCP].dport
                pr = "TCP"
    scapy_time = time.perf_counter() - t0

    # 2. Measure FastSniffer parsing time
    t0 = time.perf_counter()
    for _ in range(iterations):
        fast_res = parse_packet_fast(raw_tcp)
        if fast_res is not None:
            src, sp, dst, dp, pr, syn = fast_res
    fast_time = time.perf_counter() - t0

    speedup = scapy_time / max(1e-6, fast_time)
    print(f"\n[BENCHMARK] Scapy time ({iterations:,} pkts): {scapy_time:.4f}s ({scapy_time/iterations*1e6:.2f} µs/pkt)")
    print(f"[BENCHMARK] FastSniffer time ({iterations:,} pkts): {fast_time:.4f}s ({fast_time/iterations*1e6:.2f} µs/pkt)")
    print(f"[BENCHMARK] Speedup: {speedup:.1f}x FASTER!")

    # Verify at least 15x speedup
    assert speedup >= 15.0, f"Expected at least 15x speedup, got {speedup:.1f}x"

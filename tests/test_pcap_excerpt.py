"""
tests/test_pcap_excerpt.py
===========================
Unit tests for PCAP incident excerpt generator in firewall_daemon.py.
"""
import os
import shutil
import tempfile
import pytest
from scapy.all import rdpcap, Ether, IP, TCP

from firewall_daemon import save_pcap_excerpt, compute_enrichment


def _make_dummy_raw_packets(count=5):
    packets = []
    for i in range(count):
        pkt = Ether() / IP(src=f"192.168.1.{10+i}", dst="10.0.0.1") / TCP(sport=1000+i, dport=80) / f"PAYLOAD_{i}".encode()
        packets.append(bytes(pkt))
    return packets


def test_save_pcap_excerpt():
    """Verify that save_pcap_excerpt writes valid .pcap files readable by Scapy rdpcap."""
    raw_pkts = _make_dummy_raw_packets(5)
    temp_dir = tempfile.mkdtemp(prefix="test_pcap_")

    try:
        pcap_path = save_pcap_excerpt(raw_pkts, incident_type="CRITICAL_BYTE", output_dir=temp_dir)
        assert pcap_path is not None
        assert os.path.exists(pcap_path)
        assert pcap_path.endswith(".pcap")

        read_pkts = rdpcap(pcap_path)
        assert len(read_pkts) == 5
        assert read_pkts[0][IP].dst == "10.0.0.1"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_compute_enrichment_includes_pcap_excerpt():
    """Verify compute_enrichment attaches pcap_excerpt path when pkt_raw_ring is supplied."""
    raw_pkts = _make_dummy_raw_packets(3)
    pkt_meta = [
        (1700000000.0, "192.168.1.10", "10.0.0.1", 1000, 80, "TCP", 64, True),
        (1700000001.0, "192.168.1.11", "10.0.0.1", 1001, 80, "TCP", 64, False),
    ]

    temp_dir = tempfile.mkdtemp(prefix="test_pcap_enr_")
    try:
        # Patch save_pcap_excerpt output_dir by calling save_pcap_excerpt directly
        pcap_path = save_pcap_excerpt(raw_pkts, "BYTE", output_dir=temp_dir)
        assert pcap_path is not None

        enr = compute_enrichment(pkt_meta, raw_pkts, "BYTE")
        assert "pcap_excerpt" in enr
        assert os.path.exists(enr["pcap_excerpt"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

"""
tests/test_flow_buffer.py
==========================
Unit tests for FlowBufferManager (per-flow session buffering) and Gold Threshold Ceiling guardrail.
"""
import time
import pytest
import torch
from firewall_daemon import FlowBufferManager, save_calibration


def test_flow_buffer_separation():
    """Verify that FlowBufferManager isolates bytes per 4-tuple flow so windows never mix bytes across streams."""
    fbm = FlowBufferManager(idle_ttl=300.0)
    now = time.time()

    flow_a = ("192.168.1.10", 1234, "10.0.0.1", 80)
    flow_b = ("192.168.1.20", 5678, "10.0.0.2", 443)

    # Add 300 bytes of 0xAA to Flow A
    fbm.add_bytes(flow_a, [0xAA] * 300, now=now)
    # Add 300 bytes of 0xBB to Flow B
    fbm.add_bytes(flow_b, [0xBB] * 300, now=now + 1)

    # Neither flow has 512 bytes yet -> pop_windows(512) yields nothing
    ready = list(fbm.pop_windows(512))
    assert len(ready) == 0

    # Add another 250 bytes of 0xAA to Flow A -> total 550 bytes
    fbm.add_bytes(flow_a, [0xAA] * 250, now=now + 2)

    # Flow A has 550 bytes -> pop_windows(512) yields 1 window from Flow A
    ready = list(fbm.pop_windows(512))
    assert len(ready) == 1
    key, window = ready[0]
    assert key == flow_a
    assert len(window) == 512
    assert all(b == 0xAA for b in window), "Window contained mixed bytes from another flow!"

    # Flow B still has only 300 bytes -> remainder is preserved
    assert len(fbm.buffers[flow_b]) == 300


def test_flow_buffer_idle_eviction():
    """Verify that idle flows exceeding idle_ttl are automatically evicted."""
    fbm = FlowBufferManager(idle_ttl=10.0)
    now = time.time()

    flow_idle = ("1.1.1.1", 100, "2.2.2.2", 80)
    flow_active = ("3.3.3.3", 200, "4.4.4.4", 80)

    fbm.add_bytes(flow_idle, [0x01] * 50, now=now - 20.0)    # 20s ago (> 10s TTL)
    fbm.add_bytes(flow_active, [0x02] * 50, now=now)          # active now

    assert flow_idle not in fbm.buffers, "Idle flow was not evicted"
    assert flow_active in fbm.buffers


def test_gold_threshold_calibration():
    """Verify that save_calibration computes and saves gold_threshold = threshold + 3.0."""
    dummy_scores = torch.tensor([5.0 + i * 0.1 for i in range(100)])
    calib = save_calibration("test_iface", dummy_scores, elapsed_seconds=60.0, target_iph=1.0)
    assert "gold_threshold" in calib
    assert abs(calib["gold_threshold"] - (calib["byte_threshold"] + 3.0)) < 1e-3

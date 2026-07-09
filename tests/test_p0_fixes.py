"""
tests/test_p0_fixes.py
======================
Isolated pytest suite for the three P0 fixes applied before the AI Kosh run.

  Fix W4 — SYN Flood mislabeling
      Every bare SYN was unconditionally labelled TCP_SYN_Flood.
      Now uses a per-source-IP 1-second sliding window (threshold = 50 SYNs/s).

  Fix W5 — Padding sentinel + FocalLoss alignment
      Trailing windows were padded with 0x00 (a valid byte value).
      Now padded with -1 (outside the 0-255 range).
      FocalLoss ignore_index changed from -100 (never fired) to -1.

  Fix B_CKPT — Checkpoint resilience
      Interrupt handler saved 'start_epoch' (already-done epoch) instead of
      'epoch' (epoch in flight).  Added global_step tracking and mid-epoch
      step-interval saves every checkpoint_interval_steps steps.

Run with:
    pytest tests/test_p0_fixes.py -v
"""

import os
import time
import struct
from collections import defaultdict, deque
from unittest.mock import patch

import torch
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pcap_bytes(packets):
    """Build a minimal valid pcap file (little-endian, link type 1 = Ethernet)."""
    PCAP_MAGIC    = 0xA1B2C3D4
    VERSION_MAJOR = 2
    VERSION_MINOR = 4
    THISZONE      = 0
    SIGFIGS       = 0
    SNAPLEN       = 65535
    NETWORK       = 1  # Ethernet
    header = struct.pack(
        "<IHHiIII",
        PCAP_MAGIC, VERSION_MAJOR, VERSION_MINOR,
        THISZONE, SIGFIGS, SNAPLEN, NETWORK,
    )
    body = b""
    ts = int(time.time())
    for pkt in packets:
        body += struct.pack("<IIII", ts, 0, len(pkt), len(pkt))
        body += pkt
    return header + body


def _eth_ip_tcp(src_ip, flags, payload=None):
    """Build a minimal Ethernet/IP/TCP frame."""
    if payload is None:
        payload = b"\x00" * 20
    eth = b"\xff\xff\xff\xff\xff\xff" + b"\x00\x11\x22\x33\x44\x55" + b"\x08\x00"
    src_octets = tuple(int(o) for o in src_ip.split("."))
    dst_octets = (192, 168, 0, 1)
    tcp = (
        b"\x00\x50\xc0\xa8"
        + b"\x00\x00\x00\x01"
        + b"\x00\x00\x00\x00"
        + b"\x50"
        + bytes([flags])
        + b"\xff\xff\x00\x00\x00\x00"
        + payload
    )
    total_len = 20 + len(tcp)
    ip = bytes([
        0x45, 0x00,
        (total_len >> 8) & 0xFF,
        total_len & 0xFF,
        0x00, 0x01,
        0x40, 0x00,
        0x40, 0x06,
        0x00, 0x00,
        *src_octets,
        *dst_octets,
    ])
    return eth + ip + tcp


# ---------------------------------------------------------------------------
# W4 — SYN Flood rate-gating tests
# ---------------------------------------------------------------------------

class TestW4SynFloodRateGating:

    @staticmethod
    def _run_tracker(events, threshold=50, window_sec=1.0):
        """Mirror the sliding-window logic from dataloader.__iter__."""
        tracker = defaultdict(deque)
        results = []
        for src_ip in events:
            now = time.monotonic()
            q = tracker[src_ip]
            q.append(now)
            while q and (now - q[0]) > window_sec:
                q.popleft()
            results.append(len(q) > threshold)
        return results

    def test_single_syn_is_not_flood(self):
        """A single SYN must NOT trigger the flood flag."""
        flags = self._run_tracker(["10.0.0.1"])
        assert flags == [False], "Single bare SYN incorrectly flagged as SYN flood"

    def test_syn_ack_guard_logic(self):
        """SYN+ACK (0x12) must fail the (SYN set, ACK not set) guard in the dataloader."""
        syn_only = 0x02
        syn_ack  = 0x12
        assert (syn_only & 0x02) and not (syn_only & 0x10), \
            "Bare SYN should pass the guard"
        assert not ((syn_ack & 0x02) and not (syn_ack & 0x10)), \
            "SYN+ACK should fail the guard and never enter the tracker"

    def test_flood_fires_after_threshold(self):
        """51st SYN/sec from same IP must set the flood flag."""
        ip = "10.0.0.3"
        below = self._run_tracker([ip] * 50)
        assert all(f is False for f in below), "First 50 SYNs should NOT be flagged"
        above = self._run_tracker([ip] * 51)
        assert above[-1] is True, "51st SYN within 1s should be flagged as flood"

    def test_flood_resets_after_window_expires(self):
        """Old SYNs must be evicted so a single fresh SYN is not a flood."""
        ip = "10.0.0.4"
        tracker = defaultdict(deque)
        old_time = time.monotonic() - 2.0
        for _ in range(60):
            tracker[ip].append(old_time)

        now = time.monotonic()
        q = tracker[ip]
        q.append(now)
        while q and (now - q[0]) > 1.0:
            q.popleft()

        assert len(q) == 1, f"Sliding window should have evicted old SYNs; got {len(q)}"

    def test_different_ips_do_not_cross_contaminate(self):
        """SYNs from one IP must not count against a different IP."""
        ip_a = "10.1.1.1"
        ip_b = "10.1.1.2"
        events = [ip_a] * 60 + [ip_b]   # 60 from A, 1 from B
        results = self._run_tracker(events)
        assert results[-1] is False, \
            "ip_b should NOT be flagged due to ip_a flooding"


# ---------------------------------------------------------------------------
# W5 — Padding sentinel + FocalLoss alignment tests
# ---------------------------------------------------------------------------

class TestW5PaddingSentinel:

    def test_padding_sentinel_is_minus_one(self, tmp_path):
        """Trailing window must contain -1 sentinel, not 0."""
        pytest.importorskip("scapy")
        from src.dataloader import RawPcapIterableDataset

        pkt = _eth_ip_tcp("192.168.1.1", 0x02, b"\xAB" * 60)
        pcap_file = tmp_path / "tiny.pcap"
        pcap_file.write_bytes(_make_pcap_bytes([pkt]))

        seq_len = 1024  # longer than one packet → trailing pad guaranteed
        ds = RawPcapIterableDataset(
            pcap_path=str(pcap_file),
            max_sequence_length=seq_len,
            stride=seq_len,
        )
        tensors = list(ds)
        assert len(tensors) > 0, "Dataset yielded no tensors"
        last = tensors[-1]
        assert (-1 in last.tolist()), (
            "Trailing window must contain -1 sentinel. "
            "Padding was NOT changed from torch.zeros to torch.full(-1)."
        )

    def test_padding_last_element_is_minus_one(self, tmp_path):
        """The very last element of a padded window must be -1."""
        pytest.importorskip("scapy")
        from src.dataloader import RawPcapIterableDataset

        pkt = _eth_ip_tcp("192.168.1.2", 0x10, b"\xCC" * 60)
        pcap_file = tmp_path / "tiny2.pcap"
        pcap_file.write_bytes(_make_pcap_bytes([pkt]))

        seq_len = 1024
        ds = RawPcapIterableDataset(
            pcap_path=str(pcap_file),
            max_sequence_length=seq_len,
            stride=seq_len,
        )
        tensors = list(ds)
        last = tensors[-1]
        assert last[-1].item() == -1, (
            f"Last element of trailing window should be -1, got {last[-1].item()}"
        )

    def test_focal_loss_ignores_minus_one(self):
        """FocalLoss(ignore_index=-1) must produce exactly 0 loss for all-padding targets."""
        from src.losses import FocalLoss

        criterion = FocalLoss(gamma=2.0, ignore_index=-1)
        logits  = torch.randn(2, 4, 256)
        targets = torch.full((2, 4), -1, dtype=torch.long)

        loss = criterion(logits, targets)
        assert loss.item() == pytest.approx(0.0, abs=1e-6), (
            f"FocalLoss with all ignore_index=-1 targets should be 0.0, got {loss.item()}"
        )

    def test_focal_loss_old_ignore_index_regression(self):
        """
        Regression guard: ignore_index=-100 must NOT suppress loss for byte 0x00,
        confirming the original bug (zero-padding contributed to every gradient update).
        """
        from src.losses import FocalLoss

        broken_criterion = FocalLoss(gamma=2.0, ignore_index=-100)
        logits  = torch.randn(1, 4, 256)
        targets = torch.zeros(1, 4, dtype=torch.long)   # 0x00 padding (old behaviour)

        loss = broken_criterion(logits, targets)
        assert loss.item() > 0.0, (
            "With ignore_index=-100, byte 0 should NOT be ignored — "
            "confirming the original bug was real."
        )

    def test_focal_loss_real_bytes_still_contribute(self):
        """Non-padded byte positions must still produce non-zero loss."""
        from src.losses import FocalLoss

        criterion = FocalLoss(gamma=2.0, ignore_index=-1)
        logits  = torch.randn(2, 8, 256)
        targets = torch.tensor(
            [[10, 20, 30, 40, 50, 60, -1, -1],
             [11, 22, 33, 44, 55, 66, -1, -1]],
            dtype=torch.long,
        )
        loss = criterion(logits, targets)
        assert loss.item() > 0.0, "Loss should be > 0 when real byte targets are present"


# ---------------------------------------------------------------------------
# B_CKPT — Checkpoint resilience tests
# ---------------------------------------------------------------------------

class TestCheckpointResilience:

    @pytest.fixture
    def tiny_setup(self, tmp_path):
        from src.model import NetworkBytePatcher
        from src.dataloader import get_pcap_dataloader

        pcap_path = "local_test.pcap"
        if not os.path.exists(pcap_path):
            pytest.skip("local_test.pcap not found")

        device = torch.device("cpu")
        model = NetworkBytePatcher(d_model=32, nhead=2, num_layers=1).to(device)
        dataloader = get_pcap_dataloader(
            pcap_path=pcap_path,
            batch_size=2,
            num_workers=0,
            max_sequence_length=256,
        )
        ckpt_dir = str(tmp_path / "ckpts")
        return model, dataloader, ckpt_dir, device

    # ---

    def test_global_step_saved_in_epoch_checkpoint(self, tiny_setup):
        """Epoch-end checkpoint must contain global_step > 0."""
        from src.training import train_patcher_on_kosh

        model, dataloader, ckpt_dir, device = tiny_setup
        train_patcher_on_kosh(
            model=model, dataloader=dataloader, epochs=1,
            checkpoint_dir=ckpt_dir, lr=1e-3, checkpoint_interval_steps=999999,
        )

        ckpt = torch.load(os.path.join(ckpt_dir, "latest_patcher.pt"), map_location=device)
        assert "global_step" in ckpt, "Checkpoint missing 'global_step' key"
        assert ckpt["global_step"] > 0, f"global_step should be > 0; got {ckpt['global_step']}"

    def test_global_step_continues_on_resume(self, tiny_setup):
        """global_step after epoch 2 must be greater than after epoch 1."""
        from src.training import train_patcher_on_kosh
        from src.model import NetworkBytePatcher

        model, dataloader, ckpt_dir, device = tiny_setup

        train_patcher_on_kosh(
            model=model, dataloader=dataloader, epochs=1,
            checkpoint_dir=ckpt_dir, lr=1e-3, checkpoint_interval_steps=999999,
        )
        ckpt1 = torch.load(os.path.join(ckpt_dir, "latest_patcher.pt"), map_location=device)
        step1 = ckpt1["global_step"]

        resumed = NetworkBytePatcher(d_model=32, nhead=2, num_layers=1).to(device)
        train_patcher_on_kosh(
            model=resumed, dataloader=dataloader, epochs=2,
            checkpoint_dir=ckpt_dir, lr=1e-3, checkpoint_interval_steps=999999,
        )
        ckpt2 = torch.load(os.path.join(ckpt_dir, "latest_patcher.pt"), map_location=device)
        step2 = ckpt2["global_step"]

        assert step2 > step1, (
            f"global_step after epoch 2 ({step2}) should exceed epoch 1 ({step1})"
        )

    def test_mid_epoch_save_fires_at_interval(self, tiny_setup):
        """With checkpoint_interval_steps=1, torch.save must be called multiple times."""
        from src.training import train_patcher_on_kosh

        model, dataloader, ckpt_dir, device = tiny_setup
        save_calls = []

        _orig = torch.save
        def _spy(obj, path, **kw):
            save_calls.append({"gs": obj.get("global_step"), "epoch": obj.get("epoch")})
            _orig(obj, path, **kw)

        with patch("src.training.torch.save", side_effect=_spy):
            train_patcher_on_kosh(
                model=model, dataloader=dataloader, epochs=1,
                checkpoint_dir=ckpt_dir, lr=1e-3, checkpoint_interval_steps=1,
            )

        assert len(save_calls) >= 2, (
            f"Expected >= 2 torch.save calls with interval=1, got {len(save_calls)}"
        )
        # All saves must have a valid global_step
        for sc in save_calls:
            assert sc["gs"] is not None, f"Save missing global_step: {sc}"

    def test_interrupt_saves_current_epoch_not_start_epoch(self, tiny_setup):
        """
        BUG FIX regression: KeyboardInterrupt during epoch 1 must save epoch=1,
        NOT epoch=0 (start_epoch). This was the original bug.
        """
        from src.training import train_patcher_on_kosh
        from src.model import NetworkBytePatcher

        model, dataloader, ckpt_dir, device = tiny_setup

        # Complete epoch 0 so start_epoch becomes 1 on resume
        train_patcher_on_kosh(
            model=model, dataloader=dataloader, epochs=1,
            checkpoint_dir=ckpt_dir, lr=1e-3, checkpoint_interval_steps=999999,
        )

        # Resume for epoch 1 but interrupt after 1 batch
        resumed = NetworkBytePatcher(d_model=32, nhead=2, num_layers=1).to(device)
        call_count = {"n": 0}
        orig_iter = type(dataloader).__iter__

        def _interrupting_iter(self):
            for batch in orig_iter(self):
                call_count["n"] += 1
                yield batch
                if call_count["n"] >= 1:
                    raise KeyboardInterrupt("simulated wall-time eviction")

        with patch.object(type(dataloader), "__iter__", _interrupting_iter):
            train_patcher_on_kosh(
                model=resumed, dataloader=dataloader, epochs=2,
                checkpoint_dir=ckpt_dir, lr=1e-3, checkpoint_interval_steps=999999,
            )

        ckpt = torch.load(os.path.join(ckpt_dir, "latest_patcher.pt"), map_location=device)

        assert ckpt["epoch"] == 1, (
            f"Interrupt during epoch 1 must save epoch=1 (in-flight epoch), "
            f"NOT epoch=0 (start_epoch). Got {ckpt['epoch']}. "
            f"This is the original bug: 'start_epoch' was saved instead of 'epoch'."
        )
        assert "global_step" in ckpt
        assert ckpt["global_step"] >= 1

    def test_epoch_counter_increments_correctly(self, tiny_setup):
        """After 2 complete epochs checkpoint must record epoch=2."""
        from src.training import train_patcher_on_kosh

        model, dataloader, ckpt_dir, device = tiny_setup
        train_patcher_on_kosh(
            model=model, dataloader=dataloader, epochs=2,
            checkpoint_dir=ckpt_dir, lr=1e-3, checkpoint_interval_steps=999999,
        )
        ckpt = torch.load(os.path.join(ckpt_dir, "latest_patcher.pt"), map_location=device)
        assert ckpt["epoch"] == 2, (
            f"After 2 complete epochs checkpoint epoch should be 2. Got {ckpt['epoch']}"
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import subprocess, sys
    sys.exit(subprocess.call(["pytest", __file__, "-v"]))

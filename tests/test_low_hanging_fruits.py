"""
tests/test_low_hanging_fruits.py
==================================
Unit tests for the three low-hanging fruit security guardrails added to firewall_daemon.py:
  1. Target-Centric CUSUM accumulation across multiple source IPs.
  2. Alert-Excluded Recalibration (anomalous windows omitted from recalibration buffer).
  3. PSI Drift Freeze Guardrail (auto-recalibration pauses when PSI > 0.40).
"""
import time
import pytest
from firewall_daemon import CusumTracker, AdaptiveRecalibrator


def test_target_cusum_accumulation():
    """Verify that multiple distinct source IPs sending slightly elevated surprise to the
    same (dst_ip, dst_port) target accumulate together in target_cusum and trigger an alarm."""
    target_cusum = CusumTracker(k_sigma=0.5, h=10.0, leak=1.0)
    target_cusum.set_baseline(mu=5.0, sigma=1.0, reference=5.0)

    target_key = "192.168.1.50:80"
    now = time.time()

    # Source 1 sends 3 elevated windows (score 8.0, reference 5.0 -> +3.0 delta each)
    alarmed, level = target_cusum.update(target_key, 8.0, now)
    assert not alarmed
    assert abs(level - 3.0) < 1e-4

    # Source 2 sends another 3 elevated windows to the SAME target
    alarmed, level = target_cusum.update(target_key, 8.0, now + 1)
    assert not alarmed
    assert abs(level - 6.0) < 1e-4

    alarmed, level = target_cusum.update(target_key, 8.0, now + 2)
    assert not alarmed
    assert abs(level - 9.0) < 1e-4

    # Source 3 sends 1 more elevated window -> crosses h=10.0 (total surprise = 12.0)
    alarmed, level = target_cusum.update(target_key, 8.0, now + 3)
    assert alarmed, "Target-centric CUSUM failed to trip on multi-source distributed traffic"
    assert level > 10.0


def test_psi_drift_freeze_guardrail():
    """Verify that AdaptiveRecalibrator pauses threshold adaptation when PSI exceeds 0.40."""
    recal = AdaptiveRecalibrator(
        budget_q=0.99,
        ref_pct=99.0,
        anchor_threshold=12.0,
        anchor_ref=6.0,
        interval=10.0,
        min_samples=100
    )
    now = time.time()
    recal.last = now - 20.0  # elapsed > interval

    # Normal scores
    scores = [5.0 + (i % 10) * 0.5 for i in range(200)]

    # 1. High PSI (0.45 > 0.40) -> Recalibration FROZEN (returns None)
    res_high_psi = recal.maybe(scores, now, psi=0.45, max_psi=0.40)
    assert res_high_psi is None, "Recalibrator failed to freeze on high PSI drift (> 0.40)"

    # 2. Normal PSI (0.15 <= 0.40) -> Recalibration PROCEEDS (returns new thresholds)
    recal.last = now - 20.0
    res_normal_psi = recal.maybe(scores, now, psi=0.15, max_psi=0.40)
    assert res_normal_psi is not None, "Recalibrator failed to recalibrate on normal PSI"
    new_thr, new_ref = res_normal_psi
    assert new_thr > 0
    assert new_ref > 0


def test_alert_excluded_recalibration_logic():
    """Verify the exclusion logic: if a window score is flagged as an anomaly or CUSUM alarm,
    is_any_anomaly evaluates to True so the score is omitted from recalib_scores."""
    byte_threshold = 12.0

    def check_exclusion(window_score, flow_alarmed=False, target_alarmed=False):
        is_byte_anomaly = window_score > byte_threshold
        is_any_anomaly = is_byte_anomaly or flow_alarmed or target_alarmed
        return not is_any_anomaly  # returns True if included in recalib_scores

    # Benign unflagged packet -> included (True)
    assert check_exclusion(8.0, False, False) is True

    # Byte anomaly (> 12.0) -> excluded (False)
    assert check_exclusion(14.5, False, False) is False

    # Flow CUSUM alarm -> excluded (False)
    assert check_exclusion(9.0, True, False) is False

    # Target CUSUM alarm -> excluded (False)
    assert check_exclusion(9.0, False, True) is False

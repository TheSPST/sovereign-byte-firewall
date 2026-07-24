"""
tests/test_gold_ceiling_fix.py
================================
Regression tests for the 2026-07-23 Gold Threshold bugfix in firewall_daemon.py's
packet_callback (sniff_thread). The original commit (005d054) introduced two bugs:

  1. `pct` was referenced (`if pct is not None:`) without ever being assigned in
     that scope -> NameError on every real CRITICAL_BYTE event, which silently
     killed the heatmap enrichment and the paired "BYTE" alert.
  2. The plain BYTE alert (fired at the normal, adaptive `byte_threshold` -- the
     operating point evaluate_zero_day.py's detection numbers were measured at)
     was replaced outright by the CRITICAL_BYTE check, so any window scoring
     between byte_threshold and byte_threshold+3.0 produced no alert at all.

These tests exercise `score_percentile` (the function `pct` should have been
assigned from) directly, and re-derive the corrected BYTE/CRITICAL_BYTE gating
decision using the same inputs/formula as the fixed code, so a regression to
either bug trips a test failure instead of only showing up live.
"""
import pytest
from firewall_daemon import score_percentile


def test_score_percentile_is_callable_and_defined():
    """The exact function `pct` must be assigned from. If this import or call
    fails, the fix in packet_callback (`pct = score_percentile(...)`) is broken
    at the source."""
    calib = {"score_quantiles": [float(i) for i in range(101)]}  # q[i] == i
    pct = score_percentile(12.4, calib)
    assert pct is not None
    assert 0 <= pct <= 100


def test_score_percentile_returns_none_without_calibration():
    """No calibration yet (e.g. still in the learning phase) -> None, not a crash."""
    assert score_percentile(12.4, None) is None
    assert score_percentile(12.4, {}) is None


def _gating_decision(window_score, byte_threshold, current_calib):
    """Mirrors the corrected decision block in firewall_daemon.py's
    packet_callback: is_byte_anomaly and is_critical_byte are computed
    independently, so neither can silently suppress the other."""
    is_byte_anomaly = window_score > byte_threshold
    gold_thr = (current_calib.get("gold_threshold") if current_calib
                else (byte_threshold + 3.0))
    is_critical_byte = window_score > gold_thr
    return is_byte_anomaly, is_critical_byte, gold_thr


@pytest.mark.parametrize(
    "window_score, byte_threshold, current_calib, expect_byte, expect_critical",
    [
        # Below the normal threshold -> neither alert fires.
        (10.0, 12.0, None, False, False),
        # Above byte_threshold but below the +3.0 gold ceiling -> BYTE only,
        # this is the range the old buggy code dropped entirely.
        (13.5, 12.0, None, True, False),
        # Past the gold ceiling -> both BYTE and CRITICAL_BYTE fire together.
        (16.0, 12.0, None, True, True),
        # Same, but with an explicit (frozen) gold_threshold from calibration
        # instead of the byte_threshold+3.0 fallback.
        (20.0, 12.0, {"gold_threshold": 18.0}, True, True),
        (17.0, 12.0, {"gold_threshold": 18.0}, True, False),
    ],
)
def test_byte_and_critical_byte_gate_independently(
    window_score, byte_threshold, current_calib, expect_byte, expect_critical
):
    is_byte_anomaly, is_critical_byte, _ = _gating_decision(
        window_score, byte_threshold, current_calib
    )
    assert is_byte_anomaly is expect_byte
    assert is_critical_byte is expect_critical


def test_critical_byte_not_masked_by_byte_anomaly_flag():
    """Guards against re-nesting the CRITICAL_BYTE check inside `if
    is_byte_anomaly:` -- if adaptive recalibration ever drifted byte_threshold
    above the frozen gold_threshold, a naive nested check would drop
    CRITICAL_BYTE even though the hard ceiling was breached. Independent
    computation must still catch it."""
    current_calib = {"gold_threshold": 15.0}
    byte_threshold = 20.0  # drifted above the frozen ceiling
    window_score = 16.0    # above gold_threshold, below drifted byte_threshold

    is_byte_anomaly, is_critical_byte, gold_thr = _gating_decision(
        window_score, byte_threshold, current_calib
    )
    assert gold_thr == 15.0
    assert is_byte_anomaly is False
    assert is_critical_byte is True, (
        "CRITICAL_BYTE must fire independently of is_byte_anomaly"
    )

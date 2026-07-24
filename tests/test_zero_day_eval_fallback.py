"""
tests/test_zero_day_eval_fallback.py
======================================
Regression test for the 2026-07-24 Kaggle failure:

  Calibration attack files (0): []
  ERROR: Insufficient calibration data (need both benign and attack windows).

Root cause: the shipped "out-of-the-box" sample dataset (data/sample_pcaps/)
has exactly ONE attack pcap and ONE benign pcap. When that single attack file
is also passed as --holdout_attack_pcap (as the quickstart example does),
discover_attack_files() correctly excludes it from the calibration set to
avoid leakage -- but that left zero calibration attack files and the harness
hard-errored instead of degrading gracefully. Same problem existed silently
on the benign side (--benign_calibration_pcap == --benign_holdout_pcap gives
a meaningless "held-out" number, scored twice from identical data).

evaluate_zero_day.py now falls back to a same-file split (first half of a
file's windows for calibration, second half for holdout) instead of failing,
with a loud, recorded caveat. These tests cover the exact scenario from the
Kaggle run plus the boundary cases.
"""
import os
import tempfile

import pytest

from evaluate_zero_day import discover_attack_files, split_calibration_holdout


def test_discover_attack_files_reproduces_the_kaggle_bug_report():
    """With only one attack pcap in the dir, and that same file passed as the
    holdout attack file, discover_attack_files() must return zero calibration
    files -- this IS correct exclusion behavior (no leakage), and is exactly
    what the Kaggle log showed ('Calibration attack files (0): [])."""
    with tempfile.TemporaryDirectory() as d:
        attack_path = os.path.join(d, "attack_sample.pcap")
        open(attack_path, "wb").close()

        exclude = {"benign_sample.pcap", "benign_sample.pcap", "attack_sample.pcap"}
        files = discover_attack_files(d, exclude)
        assert files == [], (
            "Expected zero calibration attack files when the only file present "
            "is also the holdout file -- this reproduces the exact Kaggle failure."
        )


def test_discover_attack_files_finds_a_second_distinct_file():
    """Sanity check: a genuinely distinct second attack file IS picked up, so
    the fallback path only ever engages when it's actually needed."""
    with tempfile.TemporaryDirectory() as d:
        holdout_path = os.path.join(d, "attack_sample.pcap")
        calib_path = os.path.join(d, "another_attack.pcap")
        open(holdout_path, "wb").close()
        open(calib_path, "wb").close()

        exclude = {"benign_sample.pcap", "attack_sample.pcap"}
        files = discover_attack_files(d, exclude)
        assert [os.path.basename(f) for f in files] == ["another_attack.pcap"]


@pytest.mark.parametrize(
    "scores, expected_calib, expected_holdout",
    [
        ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0], [3.0, 4.0]),   # even count, matches the 4-window Kaggle log
        ([1.0, 2.0, 3.0], [1.0], [2.0, 3.0]),              # odd count
        ([1.0], [], [1.0]),                                 # single window -> calibration side empty
        ([], [], []),                                        # empty -> both empty
    ],
)
def test_split_calibration_holdout(scores, expected_calib, expected_holdout):
    calib, holdout = split_calibration_holdout(scores)
    assert calib == expected_calib
    assert holdout == expected_holdout


def test_split_calibration_holdout_with_too_few_windows_still_yields_empty_half():
    """A single-window file (e.g. a truly tiny sample pcap) splits into an
    empty calibration half. main()'s final 'if not benign_scores or not
    attack_scores' guard must still catch this -- the fallback degrades
    gracefully but doesn't fabricate data that isn't there."""
    calib, holdout = split_calibration_holdout([42.0])
    assert calib == []
    assert holdout == [42.0]
    # This is the condition main() checks right after the fallback:
    assert not calib, "an empty calibration half must still trip the insufficient-data guard"

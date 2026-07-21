"""Tests for the resilient capture supervisor in firewall_daemon.py.

Covers the two Wi-Fi-flap failure modes (dead sniffer thread, silent packet
stall) and the CAPTURE_GAP CSV marker that keeps incidents/day math honest
across capture downtime. Uses a fake sniffer so no root or live interface is
needed.
"""
import csv
import os
import tempfile
import threading
import time

from firewall_daemon import log_capture_gap, run_capture_supervised


class FakeThread:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


class FakeSniffer:
    """Configurable stand-in for scapy's AsyncSniffer."""

    def __init__(self, die_after_start=False, exception=None):
        self.die_after_start = die_after_start
        self.exception = exception
        self.thread = None
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        self.thread = FakeThread(alive=not self.die_after_start)

    def stop(self, join=True):
        self.stopped = True
        if self.thread is not None:
            self.thread._alive = False


def _tmp_csv():
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    with open(path, "w") as f:
        f.write("opened_at,closed_at,type,raw_alerts,peak_score,context\n")
    return path


def test_dead_sniffer_triggers_restart_and_gap_row():
    """A sniffer whose thread dies (interface down) must be restarted, and the
    outage must land in the incident CSV as a CAPTURE_GAP row."""
    log = _tmp_csv()
    made = []

    def factory():
        s = FakeSniffer(die_after_start=True, exception=OSError("en0: device down"))
        made.append(s)
        return s

    try:
        run_capture_supervised("en0", lambda p: None, [time.time()], log,
                               retry_secs=0.01, stall_timeout=0,
                               sniffer_factory=factory, poll_secs=0.01,
                               max_restarts=3)
        assert len(made) == 3, f"expected 3 restart attempts, got {len(made)}"
        with open(log) as f:
            rows = list(csv.DictReader(f))
        gaps = [r for r in rows if r["type"] == "CAPTURE_GAP"]
        assert gaps, "no CAPTURE_GAP row written for the outage"
        assert gaps[-1]["raw_alerts"] == "0"
        assert "gap_seconds" in gaps[-1]["context"]
    finally:
        os.unlink(log)


def test_stall_watchdog_restarts_silent_capture():
    """A sniffer that stays 'alive' but delivers no packets (silent stall)
    must be stopped and restarted once last_pkt goes stale."""
    log = _tmp_csv()
    made = []

    def factory():
        s = FakeSniffer(die_after_start=False)
        made.append(s)
        return s

    stale = [time.time() - 999.0]  # last packet long ago -> immediate stall
    try:
        run_capture_supervised("en0", lambda p: None, stale, log,
                               retry_secs=0.01, stall_timeout=0.05,
                               sniffer_factory=factory, poll_secs=0.01,
                               max_restarts=2)
        assert len(made) == 2, f"stall did not trigger restart (attempts={len(made)})"
        assert made[0].stopped, "stalled sniffer was not stopped before restart"
    finally:
        os.unlink(log)


def test_healthy_capture_resets_gap_and_marks_resume():
    """After a failure, a successful restart must close the gap: exactly one
    CAPTURE_GAP row spanning the outage, then the watchdog keeps running."""
    log = _tmp_csv()
    calls = {"n": 0}
    last_pkt = [time.time()]

    def factory():
        calls["n"] += 1
        # first sniffer dies, second one is healthy
        return FakeSniffer(die_after_start=(calls["n"] == 1))

    stop_flag = threading.Event()

    def keep_fresh():
        # simulate packets arriving so the healthy sniffer passes the watchdog
        while not stop_flag.is_set():
            last_pkt[0] = time.time()
            time.sleep(0.005)

    feeder = threading.Thread(target=keep_fresh, daemon=True)
    feeder.start()
    try:
        runner = threading.Thread(
            target=run_capture_supervised,
            args=("en0", lambda p: None, last_pkt, log),
            kwargs=dict(retry_secs=0.01, stall_timeout=5.0,
                        sniffer_factory=factory, poll_secs=0.01,
                        max_restarts=10),
            daemon=True,
        )
        runner.start()
        time.sleep(0.5)  # first sniffer fails, second runs under watchdog
        assert runner.is_alive(), "supervisor exited instead of monitoring the healthy sniffer"
        assert calls["n"] == 2, f"expected exactly one restart, got {calls['n'] - 1}"
        with open(log) as f:
            rows = list(csv.DictReader(f))
        gaps = [r for r in rows if r["type"] == "CAPTURE_GAP"]
        assert len(gaps) == 1, f"expected exactly 1 CAPTURE_GAP row, got {len(gaps)}"
    finally:
        stop_flag.set()
        os.unlink(log)


def test_permission_error_does_not_retry_forever():
    """PermissionError (needs sudo) is not recoverable by retrying - the
    supervisor must give up immediately instead of spinning."""
    made = []

    def factory():
        s = FakeSniffer(die_after_start=True, exception=PermissionError("BPF requires root"))
        made.append(s)
        return s

    run_capture_supervised("en0", lambda p: None, [time.time()], None,
                           retry_secs=0.01, stall_timeout=0,
                           sniffer_factory=factory, poll_secs=0.01,
                           max_restarts=50)
    assert len(made) == 1, f"supervisor retried a PermissionError {len(made)} times"


def test_log_capture_gap_row_is_csv_safe():
    """Gap rows must parse cleanly with the same schema as incident rows."""
    log = _tmp_csv()
    try:
        t = time.time()
        log_capture_gap(log, t - 42, t)
        with open(log) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        r = rows[0]
        assert r["type"] == "CAPTURE_GAP"
        assert r["peak_score"] == "0.00"
        assert "'gap_seconds':42" in r["context"]
    finally:
        os.unlink(log)


def test_log_capture_gap_none_log_is_noop():
    log_capture_gap(None, time.time() - 10, time.time())  # must not raise

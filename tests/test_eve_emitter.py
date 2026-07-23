"""
tests/test_eve_emitter.py
==========================
Unit tests for src/eve_emitter.py (Suricata-compatible EVE-JSON & RFC 5424 Syslog emitter).
"""
import json
import tempfile
import os
import time
import pytest

from src.eve_emitter import EveJsonEmitter, format_eve_json_record, format_syslog_rfc5424


def test_format_eve_json_record():
    """Verify EVE-JSON schema structure, severity mappings, and enrichment extraction."""
    enrichment = {
        "top_talkers": ["192.168.1.100 -> 10.0.0.1"],
        "top_ports": ["TCP/80"],
        "cusum_level": 27.5,
        "score_percentile": 99.9,
    }

    record = format_eve_json_record(
        incident_type="CRITICAL_BYTE",
        message="Hard static Gold Baseline ceiling breached!",
        score=14.25,
        enrichment=enrichment,
        timestamp=1700000000.0
    )

    assert record["event_type"] == "alert"
    assert record["src_ip"] == "192.168.1.100"
    assert record["dest_ip"] == "10.0.0.1"
    assert record["dest_port"] == 80
    assert record["proto"] == "TCP"

    alert = record["alert"]
    assert alert["severity"] == 1  # CRITICAL_BYTE -> Severity 1
    assert alert["signature_id"] == 900001
    assert "Sovereign Byte Firewall - CRITICAL_BYTE" in alert["signature"]
    assert alert["metadata"]["score"] == 14.25
    assert alert["metadata"]["cusum_level"] == 27.5


def test_format_syslog_rfc5424():
    """Verify RFC 5424 Syslog string formatting."""
    syslog_line = format_syslog_rfc5424(
        incident_type="SLOW_DISTRIBUTED",
        message="Distributed multi-source campaign detected targeting 10.0.0.1:80",
        score=28.5,
        hostname="test-sensor"
    )

    assert syslog_line.startswith("<")
    assert "test-sensor sovereign-firewall" in syslog_line
    assert "[SLOW_DISTRIBUTED score=28.50]" in syslog_line


def test_eve_emitter_file_writing():
    """Verify EveJsonEmitter writes valid atomic EVE-JSON lines to file."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    try:
        emitter = EveJsonEmitter(eve_log_path=path)
        emitter.emit(
            incident_type="BYTE",
            message="Payload anomaly detected! Surprise: 12.50 bits",
            score=12.50,
            enrichment={"top_talkers": ["1.1.1.1 -> 2.2.2.2"], "top_ports": ["UDP/53"]}
        )

        with open(path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1

        parsed = json.loads(lines[0])
        assert parsed["event_type"] == "alert"
        assert parsed["src_ip"] == "1.1.1.1"
        assert parsed["dest_ip"] == "2.2.2.2"
        assert parsed["dest_port"] == 53
        assert parsed["proto"] == "UDP"
        assert parsed["alert"]["severity"] == 2
    finally:
        if os.path.exists(path):
            os.unlink(path)

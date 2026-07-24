"""
tests/test_ips_enforcement.py
==============================
Unit tests for Active IPS Enforcement Engine (src/ips_enforcer.py).
"""
import time
import pytest
from src.ips_enforcer import IPSEnforcer


def test_ips_ban_and_unban():
    """Verify adding bans, checking active bans, and manual unbanning."""
    ips = IPSEnforcer(enabled=True, default_ttl=900.0, mock_mode=True)
    now = time.time()

    success = ips.ban_ip("192.168.1.100", reason="Payload anomaly detected! Surprise 14.5 bits", incident_type="CRITICAL_BYTE", now=now)
    assert success is True

    active = ips.get_active_bans(now=now)
    assert len(active) == 1
    assert active[0]["ip"] == "192.168.1.100"
    assert active[0]["incident_type"] == "CRITICAL_BYTE"
    assert "Payload anomaly" in active[0]["reason"]

    # Test manual unban
    unbanned = ips.unban_ip("192.168.1.100")
    assert unbanned is True
    assert len(ips.get_active_bans(now=now)) == 0


def test_ips_ttl_expiration():
    """Verify that bans naturally expire when their TTL has elapsed."""
    ips = IPSEnforcer(enabled=True, default_ttl=10.0, mock_mode=True)
    now = time.time()

    ips.ban_ip("10.0.0.50", reason="CUSUM flow alarm", incident_type="SLOW", ttl_secs=10.0, now=now - 15.0)  # 15s ago (> 10s TTL)
    ips.ban_ip("10.0.0.51", reason="CUSUM target alarm", incident_type="SLOW_DISTRIBUTED", ttl_secs=10.0, now=now)  # active now

    active = ips.get_active_bans(now=now)
    assert len(active) == 1
    assert active[0]["ip"] == "10.0.0.51"


def test_ips_ignored_ips():
    """Verify loopback and zero addresses are protected from accidental bans."""
    ips = IPSEnforcer(enabled=True, mock_mode=True)
    assert ips.ban_ip("127.0.0.1", reason="test", incident_type="BYTE") is False
    assert ips.ban_ip("0.0.0.0", reason="test", incident_type="BYTE") is False
    assert ips.ban_ip("::1", reason="test", incident_type="BYTE") is False
    assert len(ips.get_active_bans()) == 0

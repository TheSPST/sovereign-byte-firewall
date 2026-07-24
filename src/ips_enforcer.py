"""
src/ips_enforcer.py
===================
Active IPS Enforcement Engine.
Issues OS kernel drop rules (pfctl on macOS, iptables on Linux) to dynamically
block malicious source IPs and manages automatic TTL expiration.
"""
import os
import sys
import time
import subprocess
import logging
from datetime import datetime, timezone


class IPSEnforcer:
    """Manages active IP bans and interacts with OS firewall sub-systems."""

    def __init__(self, enabled=False, default_ttl=900.0, mock_mode=False):
        self.enabled = enabled
        self.default_ttl = default_ttl
        self.mock_mode = mock_mode or (os.geteuid() != 0 and sys.platform != "win32")
        self.banned_ips = {}  # ip -> {banned_at, ttl, reason, incident_type, count}
        self._init_os_firewall()

    def _init_os_firewall(self):
        if not self.enabled or self.mock_mode:
            return
        try:
            if sys.platform == "darwin":
                subprocess.run(["sudo", "pfctl", "-t", "sovereign_blacklist", "-T", "show"],
                               capture_output=True, check=False)
            elif sys.platform.startswith("linux"):
                subprocess.run(["sudo", "iptables", "-N", "SOVEREIGN_IPS"],
                               capture_output=True, check=False)
                subprocess.run(["sudo", "iptables", "-C", "INPUT", "-j", "SOVEREIGN_IPS"],
                               capture_output=True, check=False)
        except Exception as e:
            logging.warning(f"Could not initialize OS firewall rules: {e}")

    def ban_ip(self, ip, reason, incident_type, ttl_secs=None, now=None):
        """Bans a source IP address for a specified TTL (in seconds)."""
        if not ip or ip in ("0.0.0.0", "127.0.0.1", "::1", "localhost"):
            return False

        now = now if now is not None else time.time()
        ttl = ttl_secs if ttl_secs is not None else self.default_ttl

        existing = self.banned_ips.get(ip)
        if existing:
            existing["count"] += 1
            existing["last_seen"] = now
            existing["ttl"] = max(existing["ttl"], ttl)
            existing["reason"] = reason
            return True

        self.banned_ips[ip] = {
            "ip": ip,
            "banned_at": now,
            "ttl": ttl,
            "reason": reason,
            "incident_type": incident_type,
            "count": 1,
            "last_seen": now
        }

        logging.warning(f"[IPS BAN] Banning IP {ip} for {ttl:.0f}s. Reason: {reason} ({incident_type})")

        if self.enabled and not self.mock_mode:
            self._apply_os_ban(ip)

        return True

    def unban_ip(self, ip):
        """Removes an active ban for an IP address."""
        if ip not in self.banned_ips:
            return False

        info = self.banned_ips.pop(ip)
        logging.info(f"[IPS UNBAN] Unbanning IP {ip}")

        if self.enabled and not self.mock_mode:
            self._remove_os_ban(ip)

        return True

    def _apply_os_ban(self, ip):
        try:
            if sys.platform == "darwin":
                subprocess.run(["sudo", "pfctl", "-t", "sovereign_blacklist", "-T", "add", ip],
                               capture_output=True, check=False)
            elif sys.platform.startswith("linux"):
                subprocess.run(["sudo", "iptables", "-A", "SOVEREIGN_IPS", "-s", ip, "-j", "DROP"],
                               capture_output=True, check=False)
        except Exception as e:
            logging.warning(f"Could not apply OS kernel ban for {ip}: {e}")

    def _remove_os_ban(self, ip):
        try:
            if sys.platform == "darwin":
                subprocess.run(["sudo", "pfctl", "-t", "sovereign_blacklist", "-T", "delete", ip],
                               capture_output=True, check=False)
            elif sys.platform.startswith("linux"):
                subprocess.run(["sudo", "iptables", "-D", "SOVEREIGN_IPS", "-s", ip, "-j", "DROP"],
                               capture_output=True, check=False)
        except Exception as e:
            logging.warning(f"Could not remove OS kernel ban for {ip}: {e}")

    def check_expirations(self, now=None):
        """Evicts expired bans whose TTL has elapsed."""
        now = now if now is not None else time.time()
        expired = []
        for ip, info in list(self.banned_ips.items()):
            if (now - info["banned_at"]) >= info["ttl"]:
                expired.append(ip)

        for ip in expired:
            self.unban_ip(ip)

        return len(expired)

    def get_active_bans(self, now=None):
        """Returns formatted list of active bans for telemetry / Dashboard UI."""
        now = now if now is not None else time.time()
        self.check_expirations(now=now)
        bans = []
        for ip, info in self.banned_ips.items():
            elapsed = now - info["banned_at"]
            remaining = max(0.0, info["ttl"] - elapsed)
            banned_iso = datetime.fromtimestamp(info["banned_at"], tz=timezone.utc).strftime("%H:%M:%S")
            bans.append({
                "ip": ip,
                "reason": info["reason"],
                "incident_type": info["incident_type"],
                "banned_at": info["banned_at"],
                "banned_at_iso": banned_iso,
                "ttl_secs": info["ttl"],
                "remaining_secs": round(remaining, 1),
                "count": info["count"]
            })
        return sorted(bans, key=lambda x: x["remaining_secs"], reverse=True)

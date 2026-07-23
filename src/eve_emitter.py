"""
src/eve_emitter.py
===================
Suricata-compatible EVE-JSON & RFC 5424 Syslog Emitter.
Enables zero-friction ingestion by Security Onion, Splunk, Elastic (ELK), and Sentinel SIEMs.
"""
import os
import json
import time
import socket
import logging
from datetime import datetime, timezone


SEVERITY_MAP = {
    "CRITICAL_BYTE": 1,
    "BYTE": 2,
    "SLOW_DISTRIBUTED": 2,
    "SLOW": 3,
    "INFO": 4,
}

SIGNATURE_ID_MAP = {
    "CRITICAL_BYTE": 900001,
    "BYTE": 900002,
    "SLOW_DISTRIBUTED": 900003,
    "SLOW": 900004,
    "INFO": 900000,
}


def format_eve_json_record(incident_type, message, score=None, enrichment=None, timestamp=None):
    """
    Formats a firewall incident into a Suricata-compatible EVE-JSON dict.
    
    Args:
        incident_type (str): E.g. 'BYTE', 'SLOW', 'SLOW_DISTRIBUTED', 'CRITICAL_BYTE'.
        message (str): Incident description string.
        score (float): Anomaly score or CUSUM level.
        enrichment (dict): Enrichment dict containing top_talkers, top_ports, etc.
        timestamp (float): UNIX timestamp.
        
    Returns:
        dict: Standard EVE-JSON payload.
    """
    ts = timestamp if timestamp is not None else time.time()
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    iso_ts = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"

    enr = enrichment or {}
    top_talkers = enr.get("top_talkers") or []
    top_ports = enr.get("top_ports") or []

    src_ip = "0.0.0.0"
    dest_ip = "0.0.0.0"
    src_port = 0
    dest_port = 0
    proto = "TCP"

    if top_talkers:
        talker_str = top_talkers[0]
        if " -> " in talker_str:
            parts = talker_str.split(" -> ")
            src_ip, dest_ip = parts[0].strip(), parts[1].strip()

    if top_ports:
        port_str = top_ports[0]
        if "/" in port_str:
            p_parts = port_str.split("/")
            proto = p_parts[0].upper()
            try:
                dest_port = int(p_parts[1])
            except ValueError:
                dest_port = 0

    severity = SEVERITY_MAP.get(incident_type, 3)
    sig_id = SIGNATURE_ID_MAP.get(incident_type, 900000)

    eve_dict = {
        "timestamp": iso_ts,
        "event_type": "alert",
        "src_ip": src_ip,
        "src_port": src_port,
        "dest_ip": dest_ip,
        "dest_port": dest_port,
        "proto": proto,
        "alert": {
            "action": "allowed",
            "gid": 1,
            "signature_id": sig_id,
            "signature": f"Sovereign Byte Firewall - {incident_type}",
            "category": "Zero-Day Anomaly Detection",
            "severity": severity,
            "metadata": {
                "detector": "transformer_byte_level",
                "incident_type": incident_type,
                "message": message,
                "score": round(float(score), 4) if score is not None else 0.0,
                "cusum_level": enr.get("cusum_level", 0.0),
                "score_percentile": enr.get("score_percentile"),
            }
        }
    }
    return eve_dict


def format_syslog_rfc5424(incident_type, message, score=None, enrichment=None, hostname=None):
    """
    Formats a firewall incident into an RFC 5424 Syslog line.
    """
    dt = datetime.now(timezone.utc)
    iso_ts = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    host = hostname or socket.gethostname() or "sovereign-firewall"
    sev = SEVERITY_MAP.get(incident_type, 3)
    pri = 16 * 8 + (3 if sev == 1 else (4 if sev == 2 else 6))  # local0 facility
    score_str = f"{score:.2f}" if score is not None else "0.00"
    return f"<{pri}>1 {iso_ts} {host} sovereign-firewall - - - [{incident_type} score={score_str}] {message}"


class EveJsonEmitter:
    """Appends Suricata-compatible EVE-JSON records to file and optionally streams Syslog over UDP."""

    def __init__(self, eve_log_path="eve.json", syslog_host=None, syslog_port=514):
        self.eve_log_path = eve_log_path
        self.syslog_host = syslog_host
        self.syslog_port = syslog_port
        self._udp_sock = None
        if syslog_host:
            try:
                self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            except Exception as e:
                logging.warning(f"Could not open UDP socket for Syslog: {e}")

    def emit(self, incident_type, message, score=None, enrichment=None, timestamp=None):
        # 1. Write to EVE-JSON file
        if self.eve_log_path and str(self.eve_log_path).lower() != "none":
            try:
                record = format_eve_json_record(incident_type, message, score, enrichment, timestamp)
                line = json.dumps(record) + "\n"
                with open(self.eve_log_path, "a") as f:
                    f.write(line)
            except OSError as e:
                logging.warning(f"Could not write to EVE-JSON log {self.eve_log_path}: {e}")

        # 2. Stream to UDP Syslog if configured
        if self._udp_sock and self.syslog_host:
            try:
                syslog_line = format_syslog_rfc5424(incident_type, message, score, enrichment)
                self._udp_sock.sendto(syslog_line.encode("utf-8"), (self.syslog_host, self.syslog_port))
            except Exception as e:
                logging.warning(f"Could not send Syslog packet to {self.syslog_host}:{self.syslog_port}: {e}")

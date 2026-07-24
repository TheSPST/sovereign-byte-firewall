#!/usr/bin/env python3
"""
firewall_daemon.py
==================
Sovereign Byte-Level Firewall - Live Traffic Daemon.
Listens to a local network interface, extracts packets, and streams them 
through the OR-fused IDS (Byte-level Transformer + SYN Rate Detector).
Now with real-time WebSocket broadcasting for the Web Dashboard!
"""

import os
import sys
import math
import time
import bisect
import argparse
import logging
import json
import threading
import asyncio
import websockets
from collections import deque, Counter, defaultdict
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from scapy.all import AsyncSniffer, TCP, IP, UDP

from src.model import NetworkBytePatcher
from src.dataloader import RawPcapIterableDataset
from src.fast_sniffer import parse_packet_fast
from src.eve_emitter import EveJsonEmitter
from src.ips_enforcer import IPSEnforcer
from scapy.all import wrpcap, Ether, Raw

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Global variables for WebSockets
CONNECTED_CLIENTS = set()
loop = None

async def register(websocket):
    CONNECTED_CLIENTS.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        CONNECTED_CLIENTS.remove(websocket)

async def broadcast_alert(alert_data):
    if CONNECTED_CLIENTS:
        message = json.dumps(alert_data)
        websockets.broadcast(CONNECTED_CLIENTS, message)

def trigger_alert_async(alert_type, message, score=None, enrichment=None, banned_ips=None):
    if loop is not None:
        alert_data = {
            "timestamp": time.time(),
            "type": alert_type,
            "message": message,
            "score": score,
            "enrichment": enrichment or {},
            "banned_ips": banned_ips or [],
        }
        asyncio.run_coroutine_threadsafe(broadcast_alert(alert_data), loop)


# ---------------------------------------------------------------------------
# Incident enrichment (deterministic facts, no LLM)
# ---------------------------------------------------------------------------

def save_pcap_excerpt(pkt_raw_ring, incident_type, output_dir="incidents/excerpts"):
    """Saves recent raw packet bytes from the ring buffer into a standalone .pcap file.
    Returns the relative path to the created .pcap file."""
    if not pkt_raw_ring:
        return None
    try:
        os.makedirs(output_dir, exist_ok=True)
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"incident_{ts_str}_{incident_type.lower()}.pcap"
        filepath = os.path.join(output_dir, filename)

        pkts = []
        for raw in pkt_raw_ring:
            try:
                pkts.append(Ether(raw))
            except Exception:
                pkts.append(Raw(raw))

        wrpcap(filepath, pkts)
        return filepath
    except Exception as e:
        logging.warning(f"Could not write PCAP excerpt: {e}")
        return None


def compute_enrichment(pkt_meta, pkt_raw_ring=None, incident_type=None):
    """Summarize the recent packet-metadata ring buffer into human-readable
    facts an analyst can triage from: top talkers, ports, protocol mix, SYNs,
    and export a standalone .pcap excerpt if raw packets are provided.
    pkt_meta entries: (ts, src, dst, sport, dport, proto, size, is_syn)."""
    if not pkt_meta:
        return {}
    talkers, ports, protos = Counter(), Counter(), Counter()
    total_bytes = syns = 0
    for (_ts, src, dst, _sp, dport, proto, size, is_syn) in pkt_meta:
        talkers[f"{src} -> {dst}"] += size
        if dport:
            ports[dport] += 1
        protos[proto] += 1
        total_bytes += size
        if is_syn:
            syns += 1
    n = len(pkt_meta)

    pcap_path = None
    if pkt_raw_ring and incident_type:
        pcap_path = save_pcap_excerpt(pkt_raw_ring, incident_type)

    enr = {
        "packets": n,
        "bytes": total_bytes,
        "top_talkers": [{"pair": k, "bytes": v} for k, v in talkers.most_common(3)],
        "top_ports": [p for p, _ in ports.most_common(5)],
        "proto_mix_pct": {p: round(100.0 * c / n) for p, c in protos.most_common()},
        "syns": syns,
    }
    if pcap_path:
        enr["pcap_excerpt"] = pcap_path
    return enr


def score_percentile(score, calib):
    """Map a live surprise score to its percentile in the benign calibration
    distribution, using the stored quantile array. Returns None if unavailable."""
    if not calib:
        return None
    q = calib.get("score_quantiles")
    if not q:
        return None
    # q[i] is the i-th percentile value (i in 0..100); locate the score.
    return min(100, max(0, bisect.bisect_left(q, score)))

# Scoring is aligned with evaluate_zero_day.py --score_agg topk --topk_frac 0.1
# (the validated recipe). Score = mean of the top 10% per-byte surprise
# (-log2 P(actual next byte)), in bits. 12.0 ~= the Youden threshold found on
# the CIC eval (12.025 bits); per-environment calibration should replace it.
SCORE_METRIC = "surprise_topk"
DEFAULT_BYTE_THRESHOLD = 12.0

def parse_args():
    parser = argparse.ArgumentParser(description="Live Fused Firewall Daemon")
    parser.add_argument("--interface", type=str, default="en0", help="Network interface to sniff (e.g. en0, eth0)")
    parser.add_argument("--checkpoint", type=str, default="ckpt_best/checkpoints/latest_patcher_ep0_gs75000_mid_epoch.pt",
                        help="Path to best checkpoint (default: v2-masking gs75000 - the validated peak)")
    parser.add_argument("--byte_threshold", type=float, default=None,
                        help=f"Byte anomaly threshold in surprise bits. If omitted, uses saved calibration for the interface, else {DEFAULT_BYTE_THRESHOLD}")
    parser.add_argument("--topk_frac", type=float, default=0.1,
                        help="Fraction of bytes (by surprise, highest first) averaged into the window score (default 0.1, matching the validated eval recipe)")
    parser.add_argument("--learning_time", type=int, default=0,
                        help="Calibrate on benign traffic for X seconds, save threshold, then go LIVE automatically (0 = Disabled)")
    parser.add_argument("--target_incidents_per_hour", type=float, default=1.0,
                        help="Alert budget: pick the calibration threshold that yields at most this many "
                             "byte-alerts/hour on the benign baseline (0 = fall back to the 99.9th-percentile / "
                             "mean+3sigma rule). Actual incidents are <= this after dedup.")
    parser.add_argument("--rate_threshold", type=int, default=75, help="SYN rate threshold per 100ms window")
    parser.add_argument("--dedup_window", type=float, default=60.0,
                        help="Collapse repeated alerts of the same type within this many seconds into one incident (0 = disable)")
    parser.add_argument("--incident_log", type=str, default=None,
                        help="CSV file to append closed incidents to (default: incidents_<interface>.csv; 'none' to disable)")
    parser.add_argument("--enable_cusum", action="store_true", default=True,
                        help="Per-flow CUSUM accumulator for slow-and-low/APT detection (default on)")
    parser.add_argument("--no_cusum", dest="enable_cusum", action="store_false",
                        help="Disable the CUSUM slow-low detector")
    parser.add_argument("--cusum_k_sigma", type=float, default=0.5,
                        help="Fallback CUSUM reference = mean + k*sigma, used only if calibration "
                             "has no score quantiles (default 0.5)")
    parser.add_argument("--cusum_ref_pct", type=float, default=99.0,
                        help="CUSUM reference = this benign percentile of surprise (default 99). "
                             "Only flows persistently above it accumulate — keeps normal encrypted "
                             "flows (mildly-elevated but < p99) from tripping it.")
    parser.add_argument("--cusum_leak", type=float, default=0.98,
                        help="CUSUM leak factor <1: drains bounded elevation so a near-reference "
                             "flow can't slowly creep to an alarm (default 0.98)")
    parser.add_argument("--cusum_h", type=float, default=25.0,
                        help="CUSUM alarm bound in accumulated bits (default 25)")
    parser.add_argument("--summary_interval", type=float, default=3600.0,
                        help="Seconds between hourly meta-event summaries (default 3600; set lower for demos)")
    parser.add_argument("--adaptive_recalib", action="store_true", default=True,
                        help="Re-fit the threshold + CUSUM reference on recent traffic under drift (default on)")
    parser.add_argument("--no_adaptive_recalib", dest="adaptive_recalib", action="store_false",
                        help="Disable adaptive recalibration (static threshold)")
    parser.add_argument("--recalib_interval", type=float, default=900.0,
                        help="Seconds between adaptive recalibrations (default 900 = 15 min)")
    parser.add_argument("--recalib_max_step", type=float, default=1.0,
                        help="Max bits the threshold may move per recalibration (default 1.0)")
    parser.add_argument("--recalib_cap", type=float, default=3.0,
                        help="Max bits the adaptive threshold may drift from the original calibration "
                             "(poisoning guard: bounds how far a slow attacker can push it, default 3.0)")
    parser.add_argument("--recalib_min_samples", type=int, default=5000,
                        help="Minimum recent windows before an adaptive recalibration fires (default 5000)")
    parser.add_argument("--meta_log", type=str, default=None,
                        help="CSV for hourly meta-event summaries (default: meta_events_<interface>.csv; 'none' to disable)")
    parser.add_argument("--fast_sniffer", action="store_true", default=True,
                        help="Use C-struct zero-Scapy fast packet parser (default on)")
    parser.add_argument("--no_fast_sniffer", dest="fast_sniffer", action="store_false",
                        help="Disable fast sniffer and fall back to Scapy object trees")
    parser.add_argument("--eve_log", type=str, default="eve.json",
                        help="Path for Suricata-compatible EVE-JSON log (default: eve.json; 'none' to disable)")
    parser.add_argument("--syslog_host", type=str, default=None,
                        help="Optional UDP Syslog host (e.g. 192.168.1.50)")
    parser.add_argument("--syslog_port", type=int, default=514,
                        help="UDP Syslog port (default 514)")
    parser.add_argument("--enable_ips", action="store_true", default=False,
                        help="Enable active IPS kernel packet dropping (pfctl on macOS / iptables on Linux)")
    parser.add_argument("--ban_ttl", type=float, default=900.0,
                        help="TTL in seconds for dynamic IP bans (default: 900s = 15 min)")
    parser.add_argument("--sniff_retry_secs", type=float, default=10.0,
                        help="Seconds between capture restart attempts after the interface drops "
                             "(e.g. Wi-Fi flap during a long run; default 10)")
    parser.add_argument("--sniff_stall_timeout", type=float, default=300.0,
                        help="Restart capture if no packet arrives for this many seconds (catches the "
                             "silent-stall failure mode where the interface dies without an error; "
                             "0 = disable; default 300)")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length for byte patcher")
    parser.add_argument("--ws_port", type=int, default=8765, help="WebSocket port for dashboard")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-environment threshold calibration (persisted per interface)
# ---------------------------------------------------------------------------

def calibration_path(interface):
    return f"calibration_{interface}.json"

def load_calibration(interface):
    path = calibration_path(interface)
    if os.path.exists(path):
        try:
            with open(path) as f:
                calib = json.load(f)
            if calib.get("score_metric") != SCORE_METRIC:
                logging.warning(f"Ignoring {path}: calibrated with metric "
                                f"'{calib.get('score_metric', 'entropy_sum (legacy)')}', current metric is "
                                f"'{SCORE_METRIC}'. Re-run with --learning_time to recalibrate.")
                return None
            if "byte_threshold" in calib:
                return calib
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"Could not read {path}: {e}")
    return None

def save_calibration(interface, scores_tensor, elapsed_seconds=0.0, target_iph=0.0):
    """Derive a per-environment threshold from benign scores and persist it.

    If target_iph > 0 (an alert budget in byte-alerts/hour), the threshold is
    the score quantile that yields that many benign exceedances/hour given the
    measured window throughput - i.e. you tell it how many alerts/hour you can
    tolerate and it sets the bar. Raw exceedances upper-bound incidents (dedup
    only reduces them), so real incidents/hour <= target.

    If target_iph <= 0, falls back to max(99.9th percentile, mean+3*std).
    """
    quantile_thr = torch.quantile(scores_tensor, 0.999).item()
    mean_val = scores_tensor.mean().item()
    std_val = scores_tensor.std().item() if scores_tensor.numel() > 1 else 0.0
    sigma_thr = mean_val + 3 * std_val
    n = int(scores_tensor.numel())

    window_rate = (n / elapsed_seconds) if elapsed_seconds > 0 else 0.0  # windows/sec
    budget_thr = None
    budget_q = None
    if target_iph > 0 and window_rate > 0:
        windows_per_hour = window_rate * 3600.0
        frac = target_iph / windows_per_hour           # fraction of windows allowed over the bar
        frac = min(max(frac, 1e-6), 0.5)               # clamp to a sane range
        budget_q = 1.0 - frac
        budget_thr = torch.quantile(scores_tensor, budget_q).item()

    if budget_thr is not None:
        threshold = budget_thr
        method = f"alert-budget ({target_iph:g}/hr -> q{budget_q:.5f})"
    else:
        threshold = max(quantile_thr, sigma_thr)
        method = "max(q99.9, mean+3sigma)"

    # 0..100th percentile values, for mapping a live score to its percentile.
    try:
        qs = torch.tensor([i / 100.0 for i in range(101)])
        score_quantiles = torch.quantile(scores_tensor, qs).tolist()
    except Exception:
        score_quantiles = None

    calib = {
        "interface": interface,
        "score_metric": SCORE_METRIC,
        "byte_threshold": threshold,
        "anchor_threshold": threshold,  # frozen anchor for adaptive recalibration
        "gold_threshold": round(threshold + 3.0, 4),  # hard static ceiling (never adapts)
        "threshold_method": method,
        "score_quantiles": score_quantiles,
        "target_incidents_per_hour": target_iph,
        "window_rate_per_sec": round(window_rate, 3),
        "budget_quantile": budget_q,
        "budget_threshold": budget_thr,
        "quantile_999": quantile_thr,
        "mean_plus_3sigma": sigma_thr,
        "mean": mean_val,
        "std": std_val,
        "num_windows": n,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = calibration_path(interface)
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)
    logging.info(f"Calibration saved to {path} (method: {method})")
    return calib


# ---------------------------------------------------------------------------
# Incident aggregation (alert deduplication)
# ---------------------------------------------------------------------------

class IncidentAggregator:
    """Collapses repeated alerts of the same type within a time window into
    one incident: the first alert fires immediately, repeats are counted and
    summarized when the incident goes quiet. Turns alert storms (e.g. one
    sustained flood = hundreds of raw alerts) into a handful of incidents."""

    def __init__(self, window_seconds, alert_fn, incident_log=None, eve_emitter=None):
        self.window = window_seconds
        self.alert_fn = alert_fn
        self.incidents = {}  # key -> {first_ts, last_ts, count, max_score}
        self.incident_log = incident_log
        self.eve_emitter = eve_emitter
        if incident_log and not os.path.exists(incident_log):
            with open(incident_log, "w") as f:
                f.write("opened_at,closed_at,type,raw_alerts,peak_score,context\n")

    def report(self, key, message, score=None, enrichment=None):
        """Returns True if the alert was emitted, False if suppressed.
        Enrichment (computed facts) is captured on incident open and carried
        through to the WS payload and the incident-log context column."""
        if self.window <= 0:
            self.alert_fn(key, message, score, enrichment)
            if self.eve_emitter:
                self.eve_emitter.emit(key, message, score, enrichment)
            return True
        now = time.time()
        inc = self.incidents.get(key)
        if inc is not None and (now - inc["last_ts"]) > self.window:
            self._close(key, inc)
            inc = None
        if inc is None:
            self.incidents[key] = {"first_ts": now, "last_ts": now, "count": 1,
                                   "max_score": score if score is not None else 0.0,
                                   "enrichment": enrichment or {}}
            self.alert_fn(key, message, score, enrichment)
            if self.eve_emitter:
                self.eve_emitter.emit(key, message, score, enrichment, timestamp=now)
            return True
        inc["count"] += 1
        inc["last_ts"] = now
        if score is not None and score > inc["max_score"]:
            inc["max_score"] = score
        return False

    def flush(self):
        """Close and summarize incidents whose window has expired. Call periodically."""
        now = time.time()
        for key, inc in list(self.incidents.items()):
            if (now - inc["last_ts"]) > self.window:
                self._close(key, inc)

    def _close(self, key, inc):
        if inc["count"] > 1:
            duration = max(1.0, inc["last_ts"] - inc["first_ts"])
            msg = (f"Incident closed: {inc['count']} {key} alerts over {duration:.0f}s "
                   f"(peak score {inc['max_score']:.2f})")
            logging.warning(f"[{key} INCIDENT] {msg}")
            self.alert_fn(key, msg, inc["max_score"], inc.get("enrichment"))
            if self.eve_emitter:
                self.eve_emitter.emit(key, msg, inc["max_score"], inc.get("enrichment"), timestamp=inc["last_ts"])
        if self.incident_log:
            try:
                # context = compact JSON of the opening enrichment (CSV-safe: quoted).
                # Drop the heatmap array — it's for the live dashboard, not the CSV.
                ctx_data = {k: v for k, v in (inc.get("enrichment") or {}).items() if k != "heatmap"}
                ctx = json.dumps(ctx_data, separators=(",", ":")).replace('"', "'")
                with open(self.incident_log, "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(inc['first_ts']))},"
                            f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(inc['last_ts']))},"
                            f"{key},{inc['count']},{inc['max_score']:.2f},\"{ctx}\"\n")
            except OSError as e:
                logging.warning(f"Could not write incident log: {e}")
        self.incidents.pop(key, None)

# ---------------------------------------------------------------------------
# A.1 CUSUM per-flow accumulator (slow-and-low / APT detection)
# ---------------------------------------------------------------------------
class CusumTracker:
    """Per-flow cumulative-sum change detector. For each host-pair key:
        S = max(0, leak*S + (x - reference))     # x = window surprise (bits)
    alarm when S > h, then reset that flow's S to 0.

    The REFERENCE is a HIGH benign quantile (default 99th pct), not mu+k: every
    encrypted flow sits mildly above the global mean (its payload is masked, so
    it's consistently a little surprising), so a mu-based reference would make
    normal long-lived connections accumulate forever (observed live: benign
    Google/QUIC flows tripping the alarm). Referencing the 99th percentile means
    only a flow persistently in the benign top-1% tail accumulates. The LEAK
    (<1) drains bounded elevation so a flow that hovers near the reference can't
    slowly creep across h. Bounded memory: TTL evict + LRU cap."""

    MAX_FLOWS = 8192

    def __init__(self, k_sigma=0.5, h=25.0, ttl=172800.0, leak=0.98):
        self.k_sigma = k_sigma
        self.h = h
        self.ttl = ttl
        self.leak = leak
        self.reference = None
        self.enabled = False
        self.S = {}
        self.last_seen = {}

    def set_baseline(self, mu, sigma, reference=None):
        # reference = high benign quantile (preferred). Fall back to mu + k*sigma.
        self.reference = reference if reference is not None else mu + self.k_sigma * max(1e-6, sigma)
        self.enabled = True

    def update(self, key, x, now):
        """Feed one window's (flow_key, surprise). Returns (alarmed, level)."""
        if not self.enabled or key is None:
            return (False, 0.0)
        s = max(0.0, self.leak * self.S.get(key, 0.0) + (x - self.reference))
        self.last_seen[key] = now
        alarmed = s > self.h
        self.S[key] = 0.0 if alarmed else s
        self._evict(now)
        return (alarmed, s)

    def _evict(self, now):
        if len(self.last_seen) <= self.MAX_FLOWS:
            return
        cutoff = now - self.ttl
        for k in [k for k, t in self.last_seen.items() if t < cutoff]:
            self.last_seen.pop(k, None); self.S.pop(k, None)
        # still over cap -> drop least-recently-seen
        while len(self.last_seen) > self.MAX_FLOWS:
            oldest = min(self.last_seen, key=self.last_seen.get)
            self.last_seen.pop(oldest, None); self.S.pop(oldest, None)


# ---------------------------------------------------------------------------
# A.2 Hourly meta-event summary + concept-drift surface
# ---------------------------------------------------------------------------
class MetaEventReporter:
    """Rolls incidents over a rolling window (default 1h) into a single
    meta-event digest, and checks for concept drift by comparing the window's
    live score distribution to the calibration baseline (PSI). Emits to the
    dashboard, a meta_events CSV, and the log."""

    def __init__(self, interval, emit_fn, meta_log=None):
        self.interval = interval
        self.emit_fn = emit_fn
        self.meta_log = meta_log
        self.calib = None
        self.last_summary = time.time()
        self.incidents = []
        self.recent_scores = deque(maxlen=50000)
        if meta_log and not os.path.exists(meta_log):
            with open(meta_log, "w") as f:
                f.write("window_start,window_end,incidents,by_type,top_talkers,top_ports,drift_psi,drift_flag,verdict\n")

    def set_calib(self, calib):
        self.calib = calib

    def record_incident(self, key, enrichment):
        self.incidents.append((key, enrichment or {}))

    def record_score(self, x):
        self.recent_scores.append(x)

    def _psi(self):
        """Population Stability Index of recent live scores vs the calibration
        deciles. >0.25 conventionally signals a meaningful distribution shift."""
        q = self.calib.get("score_quantiles") if self.calib else None
        if not q or len(q) < 101 or len(self.recent_scores) < 100:
            return None
        edges = [q[i] for i in range(0, 101, 10)]  # 10 decile bins
        obs = [0] * 10
        for x in self.recent_scores:
            b = min(9, max(0, bisect.bisect_right(edges, x) - 1))
            obs[b] += 1
        n = len(self.recent_scores)
        psi = 0.0
        for o_count in obs:
            o = max(o_count / n, 1e-4)
            psi += (o - 0.1) * math.log(o / 0.1)
        return psi

    def maybe_summarize(self, now, force=False):
        if not force and (now - self.last_summary) < self.interval:
            return None
        start, end = self.last_summary, now
        n = len(self.incidents)
        by_type = Counter(k for k, _ in self.incidents)
        talkers, ports = Counter(), Counter()
        for _, e in self.incidents:
            for t in e.get("top_talkers", []):
                talkers[t["pair"]] += t.get("bytes", 0)
            for p in e.get("top_ports", []):
                ports[p] += 1
        psi = self._psi()
        drift_flag = psi is not None and psi > 0.25
        if drift_flag:
            verdict = f"DRIFT: baseline may be stale (PSI {psi:.2f}) - consider recalibrating"
        elif n == 0:
            verdict = "quiet hour - no incidents"
        else:
            verdict = f"{n} incident(s), " + ", ".join(f"{t}:{c}" for t, c in by_type.items())
        summary = {
            "window_start": start, "window_end": end, "incidents": n,
            "by_type": dict(by_type),
            "top_talkers": [p for p, _ in talkers.most_common(3)],
            "top_ports": [p for p, _ in ports.most_common(5)],
            "drift_psi": round(psi, 3) if psi is not None else None,
            "drift_flag": drift_flag, "verdict": verdict,
        }
        self.emit_fn("META", verdict, None, summary)
        logging.info(f"[META] {time.strftime('%H:%M', time.localtime(end))} summary: {verdict}")
        if self.meta_log:
            try:
                with open(self.meta_log, "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(start))},"
                            f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(end))},"
                            f"{n},\"{dict(by_type)}\",\"{summary['top_talkers']}\","
                            f"\"{summary['top_ports']}\",{summary['drift_psi']},"
                            f"{drift_flag},\"{verdict}\"\n")
            except OSError as e:
                logging.warning(f"Could not write meta log: {e}")
        # reset window
        self.last_summary = now
        self.incidents = []
        self.recent_scores.clear()
        return summary


# ---------------------------------------------------------------------------
# H.2 Adaptive recalibration (drift response, poisoning-safe)
# ---------------------------------------------------------------------------
def persist_adapted_threshold(interface, new_threshold, new_ref):
    """Rewrite the live byte_threshold in the calibration file so a restart keeps
    the adapted value. anchor_threshold is left untouched (the frozen anchor)."""
    path = calibration_path(interface)
    try:
        with open(path) as f:
            calib = json.load(f)
        calib["byte_threshold"] = round(new_threshold, 4)
        calib["adapted_ref"] = round(new_ref, 4)
        calib["adapted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(path, "w") as f:
            json.dump(calib, f, indent=2)
    except (OSError, json.JSONDecodeError) as e:
        logging.warning(f"Could not persist adapted threshold: {e}")


class AdaptiveRecalibrator:
    """Re-fits the byte threshold and the CUSUM reference on RECENT live scores so
    the alert budget keeps holding under concept drift (e.g. quiet-night baseline
    vs busy-day traffic). Guardrails against poisoning / over-reaction:
      - bounded step: threshold moves at most +/- max_step bits per update;
      - frozen anchor: it can never move more than +/- cap bits from the ORIGINAL
        calibration, so a slow attacker who drags traffic up cannot blind the
        detector without limit.
    Threshold re-fit uses the SAME alert-budget quantile as the original."""

    def __init__(self, budget_q, ref_pct, anchor_threshold, anchor_ref,
                 interval=900.0, max_step=1.0, cap=3.0, min_samples=5000, enabled=True):
        self.budget_q = budget_q
        self.ref_pct = ref_pct
        self.anchor_threshold = anchor_threshold
        self.anchor_ref = anchor_ref
        self.cur_threshold = anchor_threshold
        self.cur_ref = anchor_ref
        self.interval = interval
        self.max_step = max_step
        self.cap = cap
        self.min_samples = min_samples
        self.enabled = enabled and (budget_q is not None) and (anchor_threshold is not None)
        self.last = time.time()

    @staticmethod
    def _quantile(sorted_vals, q):
        i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
        return sorted_vals[i]

    def _bounded(self, cand, cur, anchor):
        cand = min(cur + self.max_step, max(cur - self.max_step, cand))      # per-update step
        cand = min(anchor + self.cap, max(anchor - self.cap, cand))          # frozen-anchor cap
        return cand

    def maybe(self, scores, now, psi=None, max_psi=0.40):
        """Returns (new_threshold, new_ref) if it recalibrated, else None."""
        if not self.enabled or (now - self.last) < self.interval or len(scores) < self.min_samples:
            return None
        if psi is not None and psi > max_psi:
            logging.warning(f"[ADAPTIVE] Recalibration FROZEN due to high structural drift (PSI {psi:.3f} > {max_psi:.2f})")
            self.last = now
            return None
        self.last = now
        s = sorted(scores)
        cand_t = self._bounded(self._quantile(s, self.budget_q), self.cur_threshold, self.anchor_threshold)
        cand_r = self._bounded(self._quantile(s, self.ref_pct / 100.0), self.cur_ref, self.anchor_ref)
        self.cur_threshold, self.cur_ref = cand_t, cand_r
        return (cand_t, cand_r)


class FlowBufferManager:
    """Manages per-flow byte queues to eliminate packet-interleaving noise.
    Keyed by 4-tuple TCP/UDP flow: (src_ip, sport, dst_ip, dport).
    Yields 512-byte windows strictly belonging to a single pure connection."""

    MAX_FLOWS = 4096

    def __init__(self, idle_ttl=300.0):
        self.idle_ttl = idle_ttl
        self.buffers = defaultdict(deque)
        self.last_seen = {}

    def add_bytes(self, flow_key, raw_bytes, now=None):
        if not flow_key or not raw_bytes:
            return
        now = now if now is not None else time.time()
        self.buffers[flow_key].extend(raw_bytes)
        self.last_seen[flow_key] = now
        self._evict(now)

    def pop_windows(self, seq_len):
        """Yields (flow_key, window_list) for all flows that have >= seq_len bytes."""
        for flow_key, buf in list(self.buffers.items()):
            while len(buf) >= seq_len:
                window = [buf.popleft() for _ in range(seq_len)]
                yield (flow_key, window)

    def _evict(self, now):
        cutoff = now - self.idle_ttl
        for k in [k for k, t in self.last_seen.items() if t < cutoff]:
            self.last_seen.pop(k, None)
            self.buffers.pop(k, None)
        while len(self.last_seen) > self.MAX_FLOWS:
            oldest = min(self.last_seen, key=self.last_seen.get)
            self.last_seen.pop(oldest, None)
            self.buffers.pop(oldest, None)


# ---------------------------------------------------------------------------
# Resilient capture supervisor (Wi-Fi flap / interface-down survival)
# ---------------------------------------------------------------------------

def log_capture_gap(incident_log, gap_start, gap_end):
    """Append a CAPTURE_GAP row to the incident CSV (same schema as incidents)
    so downtime can be excluded from incidents/day math instead of silently
    deflating it."""
    if not incident_log:
        return
    try:
        ctx = f"{{'gap_seconds':{gap_end - gap_start:.0f}}}"
        with open(incident_log, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(gap_start))},"
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(gap_end))},"
                    f"CAPTURE_GAP,0,0.00,\"{ctx}\"\n")
    except OSError as e:
        logging.warning(f"Could not write capture-gap marker: {e}")


def run_capture_supervised(iface, packet_callback, last_pkt, incident_log,
                           retry_secs=10.0, stall_timeout=300.0,
                           sniffer_factory=None, poll_secs=5.0, max_restarts=None):
    """Keep the capture alive across interface flaps (e.g. Wi-Fi drops during a
    long unattended run).

    scapy's blocking sniff() has two failure modes when the interface goes
    down: it raises (previously killing the capture thread while the WebSocket
    server kept the dashboard looking healthy), or it silently stops
    delivering packets. This supervisor uses AsyncSniffer and handles both:
    a dead sniffer thread or a packet stall (> stall_timeout with zero
    packets) triggers a restart with retry, and every outage is logged to the
    incident CSV as a CAPTURE_GAP row so incidents/day can be computed over
    actual capture uptime.

    last_pkt: single-element list holding the timestamp of the last packet
    seen (updated by packet_callback). max_restarts is for tests (None = run
    forever).
    """
    if sniffer_factory is None:
        # Force libpcap backend on macOS to avoid Scapy BPF empty-filter
        # attach_filter crash ('>' not supported between float and NoneType).
        # libpcap is always available on macOS via system-installed /usr/lib/libpcap.dylib.
        from scapy.config import conf as scapy_conf
        scapy_conf.use_pcap = True

        def sniffer_factory():
            return AsyncSniffer(
                iface=iface,
                prn=packet_callback,
                store=False,
            )
    gap_start = None
    restarts = 0
    while True:
        sniffer = sniffer_factory()
        try:
            sniffer.start()
            time.sleep(min(1.0, poll_secs))  # let a bad interface fail fast
            thread = getattr(sniffer, "thread", None)
            if thread is not None and not thread.is_alive():
                exc = getattr(sniffer, "exception", None)
                if isinstance(exc, PermissionError):
                    raise exc
                raise OSError(str(exc) if exc else "sniffer failed to start (interface down?)")
            if gap_start is not None:
                gap_end = time.time()
                log_capture_gap(incident_log, gap_start, gap_end)
                logging.warning(f"Capture RESUMED on '{iface}' after {gap_end - gap_start:.0f}s gap"
                                + (f" (CAPTURE_GAP row appended to {incident_log})" if incident_log else ""))
                gap_start = None
            last_pkt[0] = time.time()
            while True:  # watchdog loop while the sniffer runs
                time.sleep(poll_secs)
                thread = getattr(sniffer, "thread", None)
                if thread is not None and not thread.is_alive():
                    exc = getattr(sniffer, "exception", None)
                    if isinstance(exc, PermissionError):
                        raise exc
                    raise OSError(str(exc) if exc else "sniffer thread died (interface down?)")
                if stall_timeout and (time.time() - last_pkt[0]) > stall_timeout:
                    raise OSError(f"no packets for {stall_timeout:.0f}s (silent capture stall - interface flap?)")
        except PermissionError:
            logging.error("Permission denied! Sniffing live traffic requires root privileges. Try running with 'sudo'.")
            return
        except Exception as e:
            try:
                sniffer.stop(join=False)
            except Exception:
                pass
            if gap_start is None:
                gap_start = time.time()
                logging.error(f"Sniffer down: {e} - retrying every {retry_secs:.0f}s until '{iface}' returns "
                              "(gap will be marked in the incident CSV on resume).")
            restarts += 1
            if max_restarts is not None and restarts >= max_restarts:
                log_capture_gap(incident_log, gap_start, time.time())
                return
            time.sleep(retry_secs)


def sniff_thread(args, device, model, masker, calib=None):
    tls_state = {}
    byte_buffer = bytearray()
    current_bucket = int(time.time() * 10)
    syn_count = 0
    current_calib = calib  # updated in-place when calibration finishes

    # Rolling packet-metadata ring buffer (~last 10s) for incident enrichment.
    META_WINDOW = 10.0
    pkt_meta = deque()
    pkt_raw_ring = deque(maxlen=200)

    # Timestamp of the last captured packet (boxed for the capture supervisor's
    # stall watchdog).
    last_pkt = [time.time()]

    incident_log = args.incident_log
    if incident_log is None:
        incident_log = f"incidents_{args.interface}.csv"
    elif incident_log.lower() == "none":
        incident_log = None
    if incident_log:
        logging.info(f"Incident log: {incident_log}")

    eve_log = args.eve_log
    if eve_log and eve_log.lower() == "none":
        eve_log = None
    eve_emitter = EveJsonEmitter(eve_log_path=eve_log, syslog_host=args.syslog_host, syslog_port=args.syslog_port) if (eve_log or args.syslog_host) else None
    if eve_log:
        logging.info(f"EVE-JSON SIEM Log: {eve_log}")
    if args.syslog_host:
        logging.info(f"Syslog RFC 5424 streaming -> {args.syslog_host}:{args.syslog_port}")

    # Active IPS Enforcement Engine
    ips = IPSEnforcer(enabled=args.enable_ips, default_ttl=args.ban_ttl)

    def trigger_alert_with_ips(alert_type, message, score=None, enrichment=None):
        src_ip = None
        if enrichment:
            tt = enrichment.get("top_talkers")
            if tt and len(tt) > 0:
                pair = tt[0].get("pair") if isinstance(tt[0], dict) else str(tt[0])
                if " -> " in pair:
                    src_ip = pair.split(" -> ")[0].strip()
        if src_ip and alert_type in ("CRITICAL_BYTE", "SLOW_DISTRIBUTED", "BYTE", "SLOW", "RATE"):
            ips.ban_ip(src_ip, reason=message, incident_type=alert_type, ttl_secs=args.ban_ttl)
            if enrichment is not None:
                enrichment["banned_ips"] = ips.get_active_bans()

        trigger_alert_async(alert_type, message, score, enrichment, banned_ips=ips.get_active_bans())

    aggregator = IncidentAggregator(args.dedup_window, trigger_alert_with_ips, incident_log=incident_log, eve_emitter=eve_emitter)

    # A.1 CUSUM slow-low detector + A.2 hourly meta-event reporter
    cusum = CusumTracker(k_sigma=args.cusum_k_sigma, h=args.cusum_h,
                         leak=args.cusum_leak) if args.enable_cusum else None
    target_cusum = CusumTracker(k_sigma=args.cusum_k_sigma, h=args.cusum_h,
                                leak=args.cusum_leak) if args.enable_cusum else None
    last_pair = [None]  # dominant flow of the most recent packet (host-pair)
    last_dst = [None]   # dominant target of the most recent packet (dst_ip:dst_port)

    def cusum_ref(c):
        """CUSUM reference = the args.cusum_ref_pct benign percentile of surprise
        (from calibration quantiles), else None -> fall back to mean+k*sigma."""
        q = c.get("score_quantiles") if c else None
        if q and len(q) >= 101:
            return q[int(max(0, min(100, round(args.cusum_ref_pct))))]
        return None

    meta_log = args.meta_log
    if meta_log is None:
        meta_log = f"meta_events_{args.interface}.csv"
    elif meta_log.lower() == "none":
        meta_log = None
    meta = MetaEventReporter(args.summary_interval, trigger_alert_async, meta_log=meta_log)

    # Per-Flow Session Buffering (FlowBufferManager) to eliminate interleaved packet noise
    flow_buffers = FlowBufferManager(idle_ttl=300.0)

    # H.2 adaptive recalibration: rolling recent-score buffer + recalibrator.
    recalib_scores = deque(maxlen=200000)
    recal = [None]  # boxed so finish_calibration can (re)build it

    def build_recalibrator(c):
        if not args.adaptive_recalib or c is None:
            return None
        return AdaptiveRecalibrator(
            budget_q=c.get("budget_quantile"),
            ref_pct=args.cusum_ref_pct,
            anchor_threshold=c.get("anchor_threshold", args.byte_threshold),
            anchor_ref=cusum_ref(c),
            interval=args.recalib_interval, max_step=args.recalib_max_step,
            cap=args.recalib_cap, min_samples=args.recalib_min_samples,
        )

    if calib is not None:
        meta.set_calib(calib)
        if calib.get("mean") is not None:
            if cusum is not None:
                cusum.set_baseline(calib["mean"], calib.get("std", 1.0), reference=cusum_ref(calib))
            if target_cusum is not None:
                target_cusum.set_baseline(calib["mean"], calib.get("std", 1.0), reference=cusum_ref(calib))
        recal[0] = build_recalibrator(calib)
    if cusum is not None:
        logging.info(f"CUSUM slow-low detector ON (k={args.cusum_k_sigma}sigma, h={args.cusum_h} bits); "
                     f"meta-event summary every {args.summary_interval:.0f}s -> {meta_log}")
    if args.adaptive_recalib:
        logging.info(f"Adaptive recalibration ON (every {args.recalib_interval:.0f}s, "
                     f"step +/-{args.recalib_max_step} bits, cap +/-{args.recalib_cap} bits from anchor)")

    # --- CALIBRATION MODE STATE ---
    is_learning = args.learning_time > 0
    learning_start = time.time()
    learning_scores = []

    if is_learning:
        logging.info(f"CALIBRATING on '{args.interface}' for {args.learning_time} seconds...")
        logging.info("Use the network normally (benign traffic only) so the AI can learn this environment's baseline.")
        trigger_alert_async("INFO", f"Calibrating baseline for {args.learning_time}s...")
    else:
        _thr_display = f"{args.byte_threshold:.2f}" if args.byte_threshold is not None else "(from calibration)"
        logging.info(f"Firewall is LIVE. Monitoring traffic on '{args.interface}' "
                     f"(byte_threshold={_thr_display}, dedup_window={args.dedup_window:.0f}s)...")

    def finish_calibration():
        nonlocal is_learning
        if len(learning_scores) < 10:
            logging.error(f"Only {len(learning_scores)} windows seen during calibration - not enough. "
                          "Staying in calibration mode; generate more traffic.")
            return
        nonlocal current_calib
        elapsed = time.time() - learning_start
        calib = save_calibration(args.interface, torch.tensor(learning_scores),
                                 elapsed_seconds=elapsed, target_iph=args.target_incidents_per_hour)
        current_calib = calib
        meta.set_calib(calib)
        if cusum is not None:
            cusum.set_baseline(calib["mean"], calib.get("std", 1.0), reference=cusum_ref(calib))
        if target_cusum is not None:
            target_cusum.set_baseline(calib["mean"], calib.get("std", 1.0), reference=cusum_ref(calib))
        if cusum is not None or target_cusum is not None:
            ref_val = cusum.reference if cusum else target_cusum.reference
            logging.info(f" CUSUM reference = p{args.cusum_ref_pct:g} surprise "
                         f"({ref_val:.2f} bits), leak {args.cusum_leak}, h {args.cusum_h}")
        args.byte_threshold = calib["byte_threshold"]
        recal[0] = build_recalibrator(calib)   # fresh anchor for adaptive recalibration
        recalib_scores.clear()
        is_learning = False
        logging.info("=================================================")
        logging.info(f" CALIBRATION COMPLETE for '{args.interface}'")
        logging.info(f" Windows: {calib['num_windows']} | Mean: {calib['mean']:.2f} | Std: {calib['std']:.2f} "
                     f"| Rate: {calib['window_rate_per_sec']:.1f} win/s")
        logging.info(f" Method: {calib['threshold_method']} | q99.9: {calib['quantile_999']:.2f} "
                     f"| mean+3s: {calib['mean_plus_3sigma']:.2f}")
        logging.info(f" -> LIVE with byte_threshold = {args.byte_threshold:.2f} "
                     f"(budget {args.target_incidents_per_hour:g} alerts/hr) <-")
        logging.info("=================================================")
        trigger_alert_async("INFO", f"Calibration complete. LIVE with threshold {args.byte_threshold:.2f}")

    def packet_callback(packet):
        nonlocal byte_buffer, current_bucket, syn_count, tls_state

        # Guard: Scapy BPF on macOS can pass None on buffer edge cases
        if packet is None:
            return

        now = time.time()
        last_pkt[0] = now
        try:
            is_syn = bool(TCP in packet and packet[TCP].flags & 0x02)
        except Exception:
            is_syn = False

        # 1. Rate Detector (SYN Flood Check)
        t_bucket = int(now * 10)
        if t_bucket > current_bucket:
            if not is_learning and syn_count > args.rate_threshold:
                msg = f"Volumetric anomaly detected! {syn_count} SYNs in 100ms window."
                rate_enr = compute_enrichment(pkt_meta, pkt_raw_ring, "RATE")
                if aggregator.report("RATE", msg, score=syn_count, enrichment=rate_enr):
                    logging.warning(f"[RATE ALARM] {msg}")
                    meta.record_incident("RATE", rate_enr)
            aggregator.flush()
            if not is_learning:
                meta.maybe_summarize(now)  # hourly meta-event + drift check
                # H.2 adaptive recalibration: re-fit threshold + CUSUM ref on recent traffic
                if recal[0] is not None:
                    psi_val = meta._psi() if meta else None
                    res = recal[0].maybe(recalib_scores, now, psi=psi_val)
                    if res is not None:
                        new_thr, new_ref = res
                        old_thr = args.byte_threshold
                        args.byte_threshold = new_thr
                        if cusum is not None:
                            cusum.reference = new_ref
                        if target_cusum is not None:
                            target_cusum.reference = new_ref
                        persist_adapted_threshold(args.interface, new_thr, new_ref)
                        logging.info(f"[ADAPTIVE] recalibrated: byte_threshold {old_thr:.2f} -> "
                                     f"{new_thr:.2f}, CUSUM ref -> {new_ref:.2f} "
                                     f"(anchor {recal[0].anchor_threshold:.2f}, from {len(recalib_scores)} windows)")
                        trigger_alert_async("INFO", f"Adaptive recalibration: threshold now {new_thr:.2f} bits")
            syn_count = 0
            current_bucket = t_bucket

        if is_syn:
            syn_count += 1

        raw_bytes = bytes(packet)
        if not raw_bytes:
            return
        pkt_raw_ring.append(raw_bytes)

        # Capture packet metadata for incident enrichment (cheap, zero-copy fast parser).
        flow_key = None
        try:
            parsed = parse_packet_fast(raw_bytes) if args.fast_sniffer else None
            if parsed is not None:
                src_ip, sport, dst_ip, dport, proto, fast_syn = parsed
                if is_syn is False and fast_syn:
                    is_syn = True
                pkt_meta.append((now, src_ip, dst_ip, sport, dport, proto, len(raw_bytes), is_syn))
                last_pair[0] = frozenset((src_ip, dst_ip))   # flow key for CUSUM
                last_dst[0] = f"{dst_ip}:{dport}"             # target key for CUSUM
                flow_key = (src_ip, sport, dst_ip, dport)
            elif IP in packet:
                ipl = packet[IP]
                sport = packet[TCP].sport if TCP in packet else (packet[UDP].sport if UDP in packet else 0)
                dport = packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else 0)
                proto = "TCP" if TCP in packet else ("UDP" if UDP in packet else "other")
                pkt_meta.append((now, ipl.src, ipl.dst, sport, dport, proto, len(raw_bytes), is_syn))
                last_pair[0] = frozenset((ipl.src, ipl.dst))  # flow key for CUSUM
                last_dst[0] = f"{ipl.dst}:{dport}"            # target key for CUSUM
                flow_key = (ipl.src, sport, ipl.dst, dport)
            cutoff = now - META_WINDOW
            while pkt_meta and pkt_meta[0][0] < cutoff:
                pkt_meta.popleft()
        except Exception:
            pass

        if flow_key is None:
            flow_key = ("global", 0, "global", 0)

        # 2. Byte-level Payload Detector (Per-Flow Session Buffering)
        masked = masker._mask_packet_addresses(raw_bytes, stream_tls_state=tls_state)
        flow_buffers.add_bytes(flow_key, masked, now=now)
        
        for fk, window in flow_buffers.pop_windows(args.seq_len):
            with torch.no_grad():
                batch = torch.tensor([window], dtype=torch.long, device=device)

                # Protect against OOB indexing
                x_in = torch.clamp(batch[:, :-1], min=0)
                targets = batch[:, 1:]

                # Surprise scoring - mirrors evaluate_zero_day.py
                # (--score_agg topk --topk_frac 0.1), the recipe validated on
                # CIC: surprise_t = -log2 P(byte_t+1 | byte_<=t), window score
                # = mean of the top 10% most surprising bytes.
                logits = model(x_in)
                log_probs = F.log_softmax(logits, dim=-1)
                gather_idx = torch.clamp(targets, min=0).unsqueeze(-1)
                token_logprob = log_probs.gather(-1, gather_idx).squeeze(-1)
                surprise_bits = -token_logprob / math.log(2)
                k = max(1, min(surprise_bits.shape[1], int(round(args.topk_frac * surprise_bits.shape[1]))))
                topk_vals, _ = torch.topk(surprise_bits, k=k, dim=1)
                window_score = topk_vals.mean().item()

                if is_learning:
                    learning_scores.append(window_score)
                    if (time.time() - learning_start) > args.learning_time:
                        finish_calibration()  # transitions to LIVE in-place, no restart needed
                else:
                    meta.record_score(window_score)   # for the drift (PSI) check

                    # A.1 CUSUM: accumulate flow & target surprise for slow-low detection
                    flow_alarmed, flow_level = False, 0.0
                    target_alarmed, target_level = False, 0.0
                    if cusum is not None:
                        flow_alarmed, flow_level = cusum.update(last_pair[0], window_score, now)
                    if flow_alarmed:
                        msg = (f"CUSUM flow alarm! Cumulative surprise for flow {last_pair[0]} = "
                               f"{flow_level:.2f} bits (h={cusum.h:.1f})")
                        enr = compute_enrichment(pkt_meta, pkt_raw_ring, "SLOW")
                        enr["cusum_level"] = round(flow_level, 2)
                        if aggregator.report("SLOW", msg, score=flow_level, enrichment=enr):
                            logging.warning(f"[SLOW ALARM] {msg}")
                            meta.record_incident("SLOW", enr)

                    if target_cusum is not None:
                        target_alarmed, target_level = target_cusum.update(last_dst[0], window_score, now)
                        if target_alarmed:
                            msg = (f"CUSUM target alarm! Cumulative surprise for target {last_dst[0]} = "
                                   f"{target_level:.2f} bits (h={target_cusum.h:.1f})")
                            enr = compute_enrichment(pkt_meta, pkt_raw_ring, "SLOW_DISTRIBUTED")
                            enr["cusum_level"] = round(target_level, 2)
                            if aggregator.report("SLOW_DISTRIBUTED", msg, score=target_level, enrichment=enr):
                                logging.warning(f"[SLOW_DISTRIBUTED ALARM] {msg}")
                                meta.record_incident("SLOW_DISTRIBUTED", enr)

                    _thr = args.byte_threshold if args.byte_threshold is not None else DEFAULT_BYTE_THRESHOLD
                    is_byte_anomaly = window_score > _thr
                    # Static Gold Threshold Hard Ceiling: frozen at calibration time
                    # (byte_threshold + 3.0), never adapts. Checked independently of
                    # is_byte_anomaly below so a CRITICAL_BYTE can never be silently
                    # dropped if adaptive recalibration ever drifts byte_threshold
                    # past this frozen ceiling.
                    _base_thr = args.byte_threshold if args.byte_threshold is not None else DEFAULT_BYTE_THRESHOLD
                    gold_thr = current_calib.get("gold_threshold") if current_calib else (_base_thr + 3.0)
                    is_critical_byte = window_score > gold_thr
                    is_any_anomaly = is_byte_anomaly or flow_alarmed or target_alarmed

                    # Alert-Excluded Recalibration: only unflagged benign traffic votes on baseline re-fit
                    if not is_any_anomaly:
                        recalib_scores.append(window_score)

                    # Normal byte-level anomaly: the operating point the offline
                    # zero-day eval (evaluate_zero_day.py, --score_agg topk) was
                    # validated against. CRITICAL_BYTE below is an extra escalation
                    # layer on top of this, not a replacement for it.
                    if is_byte_anomaly or is_critical_byte:
                        pct = score_percentile(window_score, current_calib)
                        pct_str = f" ({pct}th pct of baseline)" if pct is not None else ""
                        inc_type_name = "CRITICAL_BYTE" if is_critical_byte else "BYTE"
                        enrich = compute_enrichment(pkt_meta, pkt_raw_ring, inc_type_name)
                        if pct is not None:
                            enrich["score_percentile"] = pct
                        # Per-byte surprise for the flagged window -> dashboard
                        # heatmap ("what the model saw"). surprise_bits[i] scores
                        # the prediction of window[i+1], so pair them 1:1.
                        sb = surprise_bits[0].tolist()
                        enrich["heatmap"] = {
                            "bytes": [int(b) for b in window[1:]],
                            "surprise": [round(float(x), 2) for x in sb],
                        }

                        if is_byte_anomaly:
                            msg = (f"Payload anomaly detected! Surprise: {window_score:.2f} bits > "
                                   f"{args.byte_threshold:.2f}{pct_str}")
                            if aggregator.report("BYTE", msg, score=window_score, enrichment=enrich):
                                logging.critical(f"[BYTE ALARM] {msg}")
                                meta.record_incident("BYTE", enrich)

                        if is_critical_byte:
                            crit_msg = (f"CRITICAL: Hard static Gold Baseline ceiling breached! Surprise: "
                                        f"{window_score:.2f} bits > gold_threshold {gold_thr:.2f}")
                            if aggregator.report("CRITICAL_BYTE", crit_msg, score=window_score, enrichment=enrich):
                                logging.critical(f"[CRITICAL BYTE ALARM] {crit_msg}")
                                meta.record_incident("CRITICAL_BYTE", enrich)

    run_capture_supervised(args.interface, packet_callback, last_pkt, incident_log,
                           retry_secs=args.sniff_retry_secs,
                           stall_timeout=args.sniff_stall_timeout)

async def main_async():
    global loop
    loop = asyncio.get_running_loop()
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    logging.info(f"Hardware Acceleration: {device}")

    # Load saved calibration (used for the threshold if not overridden, and for
    # score-percentile enrichment regardless).
    loaded_calib = load_calibration(args.interface)

    # Threshold resolution: explicit CLI flag > saved per-interface calibration > built-in default
    if args.byte_threshold is None:
        if loaded_calib is not None:
            args.byte_threshold = loaded_calib["byte_threshold"]
            logging.info(f"Loaded saved calibration for '{args.interface}' "
                         f"(threshold={args.byte_threshold:.2f}, calibrated {loaded_calib.get('calibrated_at', '?')}, "
                         f"{loaded_calib.get('num_windows', '?')} windows)")
        else:
            args.byte_threshold = DEFAULT_BYTE_THRESHOLD
            if args.learning_time == 0:
                logging.warning(f"No calibration found for '{args.interface}'. Using default threshold "
                                f"{DEFAULT_BYTE_THRESHOLD} - consider running with --learning_time 300 first.")
    
    if not os.path.exists(args.checkpoint):
        logging.error(f"Checkpoint not found at {args.checkpoint}. Exiting.")
        sys.exit(1)
        
    logging.info("Loading NetworkBytePatcher (gs75000 configuration)...")
    model = NetworkBytePatcher(d_model=128, nhead=4, num_layers=2, max_sequence_length=args.seq_len)
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    state_dict = ckpt['model_state']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    masker = RawPcapIterableDataset("dummy.pcap", mask_addresses=True)
    
    # Run the blocking sniffer in a background thread so asyncio can handle WebSockets
    sniffer_thread = threading.Thread(target=sniff_thread, args=(args, device, model, masker, loaded_calib), daemon=True)
    sniffer_thread.start()
    
    logging.info(f"Started WebSocket Broadcast Server on ws://localhost:{args.ws_port}")
    async with websockets.serve(register, "localhost", args.ws_port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logging.info("Shutting down Sovereign Firewall.")

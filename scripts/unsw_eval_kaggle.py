#!/usr/bin/env python3
"""
unsw_eval_kaggle.py
===================
Second-dataset validation on UNSW-NB15 (Kaggle runner).

Purpose: kill the "works on one dataset" objection. Runs the SAME validated
zero-day protocol as evaluate_zero_day.py (train on benign only, topk-surprise
scoring, Youden threshold, held-out attack category as the "zero day") on
UNSW-NB15 raw traffic - a different lab, different network, different tooling
from CIC-IDS2017.

HOW TO RUN ON KAGGLE
--------------------
1. New Notebook -> Add Data -> attach a UNSW-NB15 dataset that contains raw
   .pcap files and the ground-truth CSVs. Candidates (verify contents):
     - "UNSW-NB15 and CIC-IDS2017 Labelled PCAP Data" (yasiralifarrukh)
     - any mirror of the official pcaps (17-2-2015 / 18-2-2015 chunks)
       + UNSW-NB15_1.csv..UNSW-NB15_4.csv ground truth
2. Add-ons -> Secrets: add HF_TOKEN (read access).
3. Settings -> Accelerator: GPU.
4. In a cell:
       !git clone https://github.com/TheSPST/sovereign-byte-firewall.git
       %cd sovereign-byte-firewall
       !pip -q install dpkt huggingface_hub
       !python scripts/unsw_eval_kaggle.py
   The script prints an INVENTORY of the attached dataset first. If it can't
   find what it needs, it says exactly what's missing instead of guessing.

Environment overrides (all optional):
   HF_REPO_ID      HF repo holding the v2-masking checkpoint
                   (default: spst01/sovereign-byte-firewall-v2-masking-v2)
   CKPT_FILENAME   checkpoint file inside the repo
                   (default: checkpoints/latest_patcher_ep0_gs75000_mid_epoch.pt)
   LOCAL_CKPT      path to a checkpoint already on disk (skips HF download)
   HOLDOUT_CAT     UNSW attack category held out as the zero-day
                   (default: Shellcode - payload-centric, best zero-day analog)
   MAX_PCAPS       max raw pcap chunks to process (default: 2; each ~1GB)
   MAX_ATTACK_PKTS per-category packet cap for attack pcaps (default: 200000)
"""

import os
import sys
import csv
import glob
import socket
import struct
import subprocess
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_REPO_ID = os.environ.get("HF_REPO_ID", "spst01/sovereign-byte-firewall-v2-masking-v2")
CKPT_FILENAME = os.environ.get("CKPT_FILENAME", "checkpoints/latest_patcher_ep0_gs75000_mid_epoch.pt")
LOCAL_CKPT = os.environ.get("LOCAL_CKPT", "")
HOLDOUT_CAT = os.environ.get("HOLDOUT_CAT", "Shellcode").strip().lower()
MAX_PCAPS = int(os.environ.get("MAX_PCAPS", "2"))
MAX_ATTACK_PKTS = int(os.environ.get("MAX_ATTACK_PKTS", "200000"))

# Roots scanned for pcaps + GT CSVs. Override with INPUT_ROOTS (colon-separated).
# ./unsw_pcaps is included so chunks wget'ed in a notebook cell are picked up.
_default_roots = [r for r in ("/kaggle/input", "unsw_pcaps") if os.path.isdir(r)] or ["."]
INPUT_ROOTS = os.environ.get("INPUT_ROOTS", ":".join(_default_roots)).split(":")
WORK = "unsw_work"
os.makedirs(WORK, exist_ok=True)

# UNSW-NB15 ground-truth CSV column indices (49-column files, no header;
# order defined in NUSW-NB15_features.csv)
COL_SRCIP, COL_SPORT, COL_DSTIP, COL_DSPORT, COL_PROTO = 0, 1, 2, 3, 4
COL_STIME, COL_LTIME, COL_ATTACK_CAT, COL_LABEL = 28, 29, 47, 48

PROTO_NUM = {"tcp": 6, "udp": 17, "icmp": 1}


def fail(msg):
    print(f"\n[FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1: inventory the attached dataset(s)
# ---------------------------------------------------------------------------
def inventory():
    pcaps, gt_csvs, presplit = [], [], []
    paths = []
    for root in INPUT_ROOTS:
        paths.extend(glob.glob(os.path.join(root, "**", "*"), recursive=True))
    for path in paths:
        low = os.path.basename(path).lower()
        if low.endswith((".pcap", ".pcapng", ".cap")):
            # Pre-labelled per-class pcaps (some mirrors ship these)
            if any(t in low for t in ("attack", "normal", "benign", "exploit", "fuzz",
                                       "dos", "recon", "shellcode", "worm", "backdoor", "generic")):
                presplit.append(path)
            else:
                pcaps.append(path)
        elif low.startswith("unsw-nb15_") and low.endswith(".csv"):
            gt_csvs.append(path)
    pcaps.sort(); gt_csvs.sort(); presplit.sort()

    print("=" * 70)
    print("INVENTORY of attached data under", ", ".join(INPUT_ROOTS))
    print(f"  raw pcap chunks : {len(pcaps)}")
    for p in pcaps[:10]:
        print(f"     {p} ({os.path.getsize(p)/1e6:.0f} MB)")
    print(f"  pre-split pcaps : {len(presplit)}")
    for p in presplit[:20]:
        print(f"     {p} ({os.path.getsize(p)/1e6:.0f} MB)")
    print(f"  ground-truth CSVs: {len(gt_csvs)}")
    for p in gt_csvs:
        print(f"     {p}")
    print("=" * 70)
    return pcaps, gt_csvs, presplit


# ---------------------------------------------------------------------------
# Step 2: load attack flows from ground truth
# ---------------------------------------------------------------------------
def load_attack_flows(gt_csvs):
    """Parse label=1 rows into three matchers of decreasing strictness:
      flows        : (sip,sport,dip,dport,proto) -> [(st,lt,cat)] (both directions)
      attack_pairs : frozenset({sip,dip})        -> [(st,lt,cat)]
      attacker_ips : set of source IPs that appear in ANY attack row
    UNSW-NB15 note: the GT Stime/Ltime clock frequently does NOT align with the
    pcap capture clock, so time-window matching alone silently drops everything.
    The attacker IPs (e.g. 175.45.176.x) only ever generate attack traffic, so
    5-tuple / IP-pair matching is safe and robust when time fails."""
    flows = defaultdict(list)
    attack_pairs = defaultdict(list)
    attacker_ips = set()
    n = 0
    st_min, st_max = None, None
    for path in gt_csvs:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.reader(f):
                if len(row) < 49 or row[COL_LABEL].strip() != "1":
                    continue
                cat = (row[COL_ATTACK_CAT] or "unknown").strip().lower() or "unknown"
                proto = PROTO_NUM.get(row[COL_PROTO].strip().lower())
                if proto is None:
                    continue
                try:
                    sip, dip = row[COL_SRCIP].strip(), row[COL_DSTIP].strip()
                    sport = int(row[COL_SPORT], 0) if row[COL_SPORT].strip() else 0
                    dport = int(row[COL_DSPORT], 0) if row[COL_DSPORT].strip() else 0
                    st, lt = int(float(row[COL_STIME])), int(float(row[COL_LTIME]))
                except ValueError:
                    continue
                span = (st - 1, lt + 1, cat)
                flows[(sip, sport, dip, dport, proto)].append(span)
                flows[(dip, dport, sip, sport, proto)].append(span)
                attack_pairs[frozenset((sip, dip))].append(span)
                attacker_ips.add(sip)
                n += 1
                st_min = st if st_min is None else min(st_min, st)
                st_max = st if st_max is None else max(st_max, st)
    print(f"Loaded {n} attack flows across {len(flows)} directional 5-tuples, "
          f"{len(attack_pairs)} IP-pairs, {len(attacker_ips)} attacker IPs")
    if st_min is not None:
        print(f"  GT attack time range: {st_min}..{st_max}")
    if n == 0:
        fail("Ground-truth CSVs contained no label=1 rows - wrong files?")
    return flows, attack_pairs, attacker_ips


# ---------------------------------------------------------------------------
# Step 3: split raw pcap chunks into benign / per-category attack pcaps
# ---------------------------------------------------------------------------
def _make_l3_decoder(datalink):
    """Return a fn(buf) -> dpkt IP/IP6 object or None, matching the pcap's
    link-layer type. UNSW-NB15 was captured with `tcpdump -i any`, which yields
    Linux SLL (datalink 113), NOT Ethernet (1) - so Ethernet parsing silently
    fails on every packet. Handles Ethernet, SLL, SLL2, and raw-IP captures."""
    import dpkt
    # dpkt exposes DLT_* on its pcap Reader module in most versions.
    if datalink == 113 and hasattr(dpkt, "sll"):
        def dec(buf):
            try:
                return dpkt.sll.SLL(buf).data
            except Exception:
                return None
    elif datalink == 276 and hasattr(dpkt, "sll2"):
        def dec(buf):
            try:
                return dpkt.sll2.SLL2(buf).data
            except Exception:
                return None
    elif datalink in (101, 12, 14):  # DLT_RAW variants: buffer starts at IP
        def dec(buf):
            try:
                v = buf[0] >> 4
                return dpkt.ip.IP(buf) if v == 4 else (dpkt.ip6.IP6(buf) if v == 6 else None)
            except Exception:
                return None
    else:  # Ethernet (1) and fallback
        def dec(buf):
            try:
                return dpkt.ethernet.Ethernet(buf).data
            except Exception:
                return None
    return dec


def split_pcaps(pcaps, flows, attack_pairs, attacker_ips):
    import dpkt

    # MATCH_MODE: how strict to be. Default "auto" = try 5-tuple/pair with time,
    # but if the pcap/GT clocks don't overlap, fall back to time-agnostic
    # 5-tuple/pair matching (the robust choice for UNSW-NB15).
    mode = os.environ.get("MATCH_MODE", "auto").lower()

    def open_reader(fh, path):
        try:
            return dpkt.pcap.Reader(fh)
        except (ValueError, dpkt.NeedData):
            fh.seek(0)
            return dpkt.pcapng.Reader(fh)

    out = {}   # name -> [writer, filehandle, count]
    def writer(name):
        if name not in out:
            fh = open(os.path.join(WORK, f"{name}.pcap"), "wb")
            out[name] = [dpkt.pcap.Writer(fh), fh, 0]
        return out[name]

    stats = defaultdict(int)
    diag = defaultdict(int)  # 5tuple_hit, 5tuple_time_hit, pair_hit, ip_hit
    pcap_ts_min, pcap_ts_max = None, None
    from collections import Counter
    src_census = Counter()
    parseable = 0
    attacker_ip_pkts = 0

    # For UNSW: a packet whose src OR dst is a known attacker IP is attack.
    # We still want the category; take it from any flow/pair the IP participates
    # in (best-effort). Build attacker_ip -> representative category.
    ip_cat = {}
    for pair, spans in attack_pairs.items():
        for ip in pair:
            if ip in attacker_ips and ip not in ip_cat:
                ip_cat[ip] = spans[0][2]

    def categorize(key, pair, sip, dip, t, use_time):
        """Return attack category or None (cascade: 5-tuple -> IP-pair -> attacker-IP)."""
        if key is not None and key in flows:
            diag["5tuple_hit"] += 1
            spans = flows[key]
            if use_time:
                for st, lt, c in spans:
                    if st <= t <= lt:
                        diag["5tuple_time_hit"] += 1
                        return c
            else:
                return spans[0][2]
        if pair is not None and pair in attack_pairs:
            diag["pair_hit"] += 1
            spans = attack_pairs[pair]
            if use_time:
                for st, lt, c in spans:
                    if st <= t <= lt:
                        return c
            else:
                return spans[0][2]
        # Loosest tier: either endpoint is a known attacker IP.
        if sip in attacker_ips:
            diag["ip_hit"] += 1
            return ip_cat.get(sip, "unknown")
        if dip in attacker_ips:
            diag["ip_hit"] += 1
            return ip_cat.get(dip, "unknown")
        return None

    for ci, path in enumerate(pcaps[:MAX_PCAPS]):
        benign_name = f"benign_{ci}"
        print(f"[{ci+1}/{min(len(pcaps), MAX_PCAPS)}] splitting {path} ...")
        # Two-condition parse buffer so we can decide time-usefulness once.
        parsed = []  # (ts, buf, key, pair, sip, dip)
        with open(path, "rb") as fh:
            try:
                rd = open_reader(fh, path)
            except Exception as e:
                print(f"  SKIP (unreadable: {e})"); continue
            dl = rd.datalink()
            decode_l3 = _make_l3_decoder(dl)
            is_ethernet = (dl == 1)
            print(f"  datalink={dl} (1=Ethernet, 113=Linux SLL, 276=SLL2, 101=raw)"
                  + ("" if is_ethernet else " -> normalizing frames to Ethernet for the model"))
            for ts, buf in rd:
                key = pair = sip = dip = None
                wbuf = None  # Ethernet-framed bytes to persist (model trained on Ethernet)
                try:
                    ip = decode_l3(buf)
                    if isinstance(ip, dpkt.ip.IP) and isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                        l4 = ip.data
                        sip = socket.inet_ntoa(ip.src); dip = socket.inet_ntoa(ip.dst)
                        key = (sip, l4.sport, dip, l4.dport, ip.p)
                        pair = frozenset((sip, dip))
                        if is_ethernet:
                            wbuf = buf  # already Ethernet; keep exact bytes
                        else:
                            eth = dpkt.ethernet.Ethernet(
                                src=b"\x00" * 6, dst=b"\x00" * 6,
                                type=dpkt.ethernet.ETH_TYPE_IP, data=ip)
                            wbuf = bytes(eth)
                except Exception:
                    pass
                t = int(ts)
                pcap_ts_min = t if pcap_ts_min is None else min(pcap_ts_min, t)
                pcap_ts_max = t if pcap_ts_max is None else max(pcap_ts_max, t)
                if sip is not None:
                    parseable += 1
                    src_census[sip] += 1
                    if sip in attacker_ips or dip in attacker_ips:
                        attacker_ip_pkts += 1
                parsed.append((ts, wbuf, key, pair, sip, dip, t))

        # Decide whether to use the time window: only if any 5-tuple/pair key
        # exists AND at least one time-window hit occurs on a sample.
        use_time = (mode == "strict")
        if mode == "auto":
            time_hits = 0
            for (ts, buf, key, pair, sip, dip, t) in parsed:
                if key in flows:
                    for st, lt, c in flows[key]:
                        if st <= t <= lt:
                            time_hits += 1; break
                if time_hits >= 5:
                    break
            use_time = time_hits >= 5
            verdict = "USABLE" if use_time else "NOT usable (clock mismatch) -> time-agnostic 5-tuple/IP-pair matching"
            print(f"  MATCH_MODE=auto: time-window {verdict}")

        for (ts, wbuf, key, pair, sip, dip, t) in parsed:
            if wbuf is None:  # non-IP / unparseable — don't pollute the byte stream
                stats["skipped_non_ip"] += 1
                continue
            cat = categorize(key, pair, sip, dip, t, use_time)
            if cat is not None:
                w = writer(f"attack_{cat}")
                if w[2] < MAX_ATTACK_PKTS:
                    w[0].writepkt(wbuf, ts); w[2] += 1
                stats[cat] += 1
            else:
                w = writer(benign_name)
                w[0].writepkt(wbuf, ts); w[2] += 1
                stats["benign"] += 1

    print(f"  pcap time range: {pcap_ts_min}..{pcap_ts_max}")
    print(f"  parseable IP/TCP-UDP packets: {parseable}")
    print(f"  packets involving a known attacker IP {sorted(attacker_ips)}: {attacker_ip_pkts}")
    print(f"  top talkers (src IP): {src_census.most_common(10)}")
    print(f"  match diagnostics: {dict(diag)}")
    for name, (w, fh, cnt) in out.items():
        fh.close()
        print(f"  wrote {WORK}/{name}.pcap  ({cnt} packets)")
    print("Packet routing:", dict(stats))
    return [n for n in out if n.startswith("benign_")], \
           [n for n in out if n.startswith("attack_")]


# ---------------------------------------------------------------------------
# Step 4: fetch checkpoint and run the validated harness
# ---------------------------------------------------------------------------
def get_checkpoint():
    if LOCAL_CKPT:
        if not os.path.exists(LOCAL_CKPT):
            fail(f"LOCAL_CKPT={LOCAL_CKPT} does not exist")
        return LOCAL_CKPT
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    print(f"Downloading {CKPT_FILENAME} from {HF_REPO_ID} ...")
    return hf_hub_download(repo_id=HF_REPO_ID, filename=CKPT_FILENAME, token=token)


def main():
    pcaps, gt_csvs, presplit = inventory()

    if presplit and not pcaps:
        fail("Only pre-split pcaps found. This runner expects raw chunks + ground-truth "
             "CSVs (UNSW-NB15_1..4.csv). Inspect the pre-split files manually - if they are "
             "per-category attack/normal pcaps, point evaluate_zero_day.py at them directly:\n"
             "  --benign_calibration_pcap <normal_A.pcap> --benign_holdout_pcap <normal_B.pcap>\n"
             "  --attack_dir <dir with attack_*.pcap> --holdout_attack_pcap <attack_shellcode.pcap>\n"
             "  --score_agg topk --topk_frac 0.1")
    if not pcaps:
        fail("No raw .pcap files found under /kaggle/input. Attach a UNSW-NB15 mirror "
             "with raw pcaps (17-2-2015 / 18-2-2015 chunks).")
    if not gt_csvs:
        fail("No UNSW-NB15_*.csv ground-truth files found - attach them (4 files, 49 cols).")
    if len(pcaps) < 2 and MAX_PCAPS < 2:
        print("[WARN] Only one pcap chunk: benign holdout will come from the same chunk "
              "(weaker split). Attach 2+ chunks if possible.")

    flows, attack_pairs, attacker_ips = load_attack_flows(gt_csvs)
    benign, attacks = split_pcaps(pcaps, flows, attack_pairs, attacker_ips)

    if not benign:
        fail("No benign packets extracted - check ground truth / pcap pairing.")
    holdout_attack = f"attack_{HOLDOUT_CAT}"
    if holdout_attack not in attacks:
        fail(f"Held-out category '{HOLDOUT_CAT}' not present in extracted attacks "
             f"({sorted(attacks)}). Set HOLDOUT_CAT to one of those.")
    if len(attacks) < 2:
        fail("Need at least 2 attack categories (1 for calibration, 1 held out).")

    # Calibration attacks = every category EXCEPT the held-out one
    calib_dir = os.path.join(WORK, "calib_attacks")
    os.makedirs(calib_dir, exist_ok=True)
    for name in attacks:
        if name != holdout_attack:
            src = os.path.join(WORK, f"{name}.pcap")
            dst = os.path.join(calib_dir, f"{name}.pcap")
            if not os.path.exists(dst):
                os.link(src, dst)

    benign_calib = os.path.join(WORK, f"{benign[0]}.pcap")
    benign_holdout = os.path.join(WORK, f"{benign[-1] if len(benign) > 1 else benign[0]}.pcap")

    ckpt = get_checkpoint()

    cmd = [sys.executable, "evaluate_zero_day.py",
           "--checkpoint_path", ckpt,
           "--benign_calibration_pcap", benign_calib,
           "--benign_holdout_pcap", benign_holdout,
           "--attack_dir", calib_dir,
           "--holdout_attack_pcap", os.path.join(WORK, f"{holdout_attack}.pcap"),
           "--score_agg", "topk", "--topk_frac", "0.1",
           "--output_dir", f"results/unsw_eval_{HOLDOUT_CAT}"]
    print("\nRunning validated harness:\n ", " ".join(cmd), "\n")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

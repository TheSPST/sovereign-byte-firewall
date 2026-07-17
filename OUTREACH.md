# Outreach Copy — drafts
**Status:** ready except `[LIVE_FP]` (fill from the daemon's incidents/day once the overnight run completes). All numbers below are the honest ones from the brief; do not inflate. Tune voice to your own before sending.

---

## A. MSSP / boutique SOC — cold email (paid PoC)

**Subject:** Zero-day anomaly layer for your pcap — fixed-fee 2-week pilot

Hi {name},

You already have visibility your signature tools give you. The gap every SOC has is the novel payload that matches no signature — and that's the only thing my tool looks for.

It's a byte-level model that reads raw traffic (a span port or a pcap you send) and flags structurally anomalous flows — no agents, no log pipeline, no cloud, runs on one on-prem box. It self-calibrates to your network on day one and is silent by default.

What it's shown so far, stated honestly:
- On CIC-IDS2017 it caught 5/5 attack campaigns (Hulk, GoldenEye, Slowloris, Slowhttptest, Heartbleed) at ~1 false alarm/hour.
- Trained only on one dataset's benign traffic, it transferred to a completely separate dataset (UNSW-NB15) with near-identical separation (AUC 0.75 vs 0.73) — it learned general "normal," not one capture.
- On my own live network it runs at [LIVE_FP] incidents/day after a one-hour baseline.
- What it does *not* do: read inside encrypted tunnels, catch a patient sub-threshold attacker, or replace your stack. It's an additional layer. (I keep a written limitations doc; happy to share.)

The pilot: give me one week of pcap (or a span port), read-only, fixed fee. I return an incident report, the false-positive rate on *your* traffic, and a live demo of it catching replayed attacks against your baseline. If it surfaces nothing your stack didn't already catch, you've cheaply proven your coverage — also a result.

Worth a 20-minute call?

{signature}

---

## B. iDEX / MeitY / DSCI grant — problem-solution abstract (~200 words)

**Sovereign Byte-Level Anomaly Detection for Zero-Day Network Threats**

India's critical networks — defence, PSU, BFSI, and government — increasingly depend on foreign network-detection products that are cloud-connected and telemetry-hungry, an unacceptable dependency for sovereign and air-gapped environments. Signature-based tools miss novel and zero-day payloads by construction.

This project is an encoder-free, byte-level anomaly detection engine that processes raw network traffic as a sequence-prediction problem — learning the structural grammar of benign traffic and flagging never-before-seen attacks as statistical anomalies. It requires no attack labels, no cloud connectivity, and no telemetry egress; it runs fully offline on hardware from a single laptop to an A100 node, and self-calibrates to each deployment.

Validated on public benchmarks (CIC-IDS2017: 5/5 attack campaigns at ~1 false alarm/hour; cross-dataset transfer to UNSW-NB15 at AUC 0.75), the engine is a sovereign, on-premise complement to existing security stacks — a zero-day tripwire that keeps all data inside the network. We seek support to harden the system (long-horizon APT detection, drift-robust calibration, line-rate deployment via in-kernel steering) and pilot it with an Indian critical-infrastructure partner.

*(Trim/expand to the specific scheme's word limit; lead with the sovereignty angle for defence programs, the cost/coverage angle for BFSI.)*

---

## C. Public launch post (Show HN / r/netsec / LinkedIn) — hold until repo is public + numbers final

**Title:** Show HN: A byte-level language model that flags zero-day network attacks — offline, no signatures

I trained a small (~1.6M param) causal transformer to predict raw network traffic byte by byte, then score live packets by "surprise" — how improbable each byte is given the ones before it. Attacks it has never seen produce improbable byte sequences; that's the whole detection signal. No signatures, no cloud, no log pipeline; it self-calibrates to a network in about an hour and runs on a laptop.

Honest results: 5/5 attack campaigns on CIC-IDS2017 at ~1 false alarm/hour (fused with a rate detector); trained on one dataset's benign traffic only, it transferred to UNSW-NB15 at AUC 0.75 vs 0.73 on its own held-out — evidence it learned general "normal" rather than memorizing a capture. It's a tripwire/pre-filter layer, not a full IDS: it can't see inside encryption, a patient sub-threshold attacker can evade it, and it needs re-baselining as networks drift. I keep a limitations doc in the repo because an anomaly detector that claims no blind spots is lying.

Repo, methodology, and reproducible eval: {link}. Feedback from netsec folks especially welcome.

---

## D. Warm intro / one-liner (for LinkedIn DMs, referrals)

I built an offline, byte-level anomaly detector that flags zero-day network attacks with no signatures and no cloud — self-calibrates to a network in an hour, runs on a laptop. Validated cross-dataset (CIC→UNSW, AUC 0.75). Looking for MSSPs or critical-infra teams to run a fixed-fee pilot. Two minutes to see the demo?

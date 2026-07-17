# Future Roadmap — Pre-Filter Positioning & Open-Core Strategy
**Created: 2026-07-17 · Research-backed · Companion to `PRODUCT_ROADMAP.md` (features) and `PROJECT_REPORT_2026-07-16.md` (GTM)**

---

## Part 1 — Research findings

### The pre-filter economics are real, with one correction
- Splunk-class SIEM ingest runs **$100–180/GB/day** (~$150 rule of thumb), plus
  $20–45/GB/day for the security tier. At 5 TB/day that is **$3.6–7.3M/year**
  in ingest alone. (Cribl, Realm.Security, Expanso pricing guides, 2026.)
- An entire company category already monetizes "filter before the SIEM":
  **Cribl** publishes a typical **30–50% SIEM cost reduction** and built a
  multi-billion-dollar business on it. **Corelight** (open-core NDR) markets
  "slashing SIEM ingest costs" as a headline feature.
- **The correction:** SIEMs ingest *logs*, not raw packets — we are not a
  Cribl substitute. Our layer is one step deeper: **packet/pcap triage**.
  The nearest analog is Corelight's "Smart PCAP" (retain full packets only
  around interesting events). Full-packet capture retention is the cost we
  attack: storing everything is petabytes; storing "the surprising 1% plus
  context" is disks, not data centers.
- **Compliance framing matters:** "drop 99.9% of traffic" terrifies auditors.
  The sellable version is **tiered retention**: short-lived local ring buffer
  of everything + long-term retention and forwarding of only high-surprise
  windows. Nothing is unrecoverable within the ring-buffer horizon.

### Pre-filter positioning fits our current metrics BETTER than detection
This is the strategically important finding. Detection positioning demands a
threshold with high TPR at near-zero FPR — our hardest regime (32.6% @ 0.23%
FPR held-out). Triage positioning only demands good *ranking* — that attack
traffic sorts toward the top. From the gs75000 eval (`metrics.json`):

| Forward the top… | …and you capture (calibration attack windows) |
|---|---|
| ~21% most surprising | **51%** |
| ~1% most surprising | 9% (calibration) / **33% (held-out zero-day set)** |

Ranking is measured by AUC (0.73 calibration; the fused system is higher) and
improves with every model iteration — and a campaign producing thousands of
windows surfaces even at partial window capture. **Same engine, easier promise:
"we rank your traffic by improbability; your expensive tools only look at the
top of the list."** The brain metaphor is the pitch: eyes see everything,
attention goes only to what moves.

**KPI to institutionalize:** *attack-capture rate @ top-k% forwarded* (k = 0.1,
1, 5, 10). Requires persisting raw per-window scores — add `--save_scores` to
`evaluate_zero_day.py` (metrics.json currently stores only aggregates).

### Open-core precedent check — and a fact about our own repo
- **Snort** → Sourcefire → acquired by Cisco for **$2.7B** (2013); still the
  engine inside Cisco Secure Firewall. **Zeek** → **Corelight** (open-core
  NDR, venture-backed). **Wazuh** (OSSEC lineage) monetizes cloud + support.
  Stamus, Vectra and others build commercial layers on open engines. In
  network security, open core is not the alternative model — it is the
  *dominant* one, because analysts do not trust black boxes.
- **Fact discovered during research: `github.com/TheSPST/sovereign-byte-firewall`
  is ALREADY public** (MIT badge, 68 commits, README, eval results). The
  strategic question is not "should we open-source?" — it happened. The real
  decisions are the **boundary** (what stays open vs. paid), the **license**
  going forward, and the **launch** (a public repo nobody has seen is not
  open source as strategy; it's a backup).

---

## Part 2 — Decisions

### Open/paid boundary (the "engine vs. car" split)
**Open (already is):** model architecture, training pipeline, eval harness,
single-node daemon, basic dashboard. This is the trust/verifiability layer —
"read the code: it never calls home" turns the sovereign claim into a
provable fact.

**Paid:** central fleet dashboard (many sensors, one pane), enterprise
connectors (Splunk HEC, syslog/CEF, S3 tiering), tuned per-environment
models + calibration service, triage workflow (Phase 2 of PRODUCT_ROADMAP
grows into this), support/SLA, and pilots. Weights: publish research
checkpoints freely (reproducibility = credibility); commercial-grade tuned
weights per customer are a service, not a download.

### License
Stay **MIT for now**. Rationale: zero adoption friction while the project has
zero stars; the risk MIT carries (a vendor commercializing our code) is a
problem we would be lucky to have. As sole author you can relicense future
versions (e.g., AGPL like Grafana/Elastic-2024, or BSL like HashiCorp) the
moment there is traction worth protecting. Trigger to revisit: first external
production user, first serious fork, or first paid pilot.
**Housekeeping now:** the README badge claims MIT but the repo has no LICENSE
file — add one (badge currently links to nothing).

### Launch timing
Do NOT announce until the brief's two placeholder numbers exist (live FP +
UNSW). A public launch is a one-shot credibility event; "validated on two
datasets + live deployment" is the difference between a research toy and a
tool on HN/r/netsec. Target: early August.

---

## Part 3 — Phased plan (extends, never displaces, the GTM critical path)

### Phase A — OSS hygiene (July, ~half a day, alongside pilot prep)
1. Add LICENSE file (MIT), CONTRIBUTING stub, `pip install -e .` packaging.
2. Rewrite README top section for an outside reader: what it does in 3
   sentences, quickstart (`--learning_time` demo), honest results table
   (reuse TECHNICAL_BRIEF numbers), architecture diagram.
3. Tag `v0.1.0` release pinned to the v2 gs75000 checkpoint + eval command.
4. Add `--save_scores` to `evaluate_zero_day.py`; compute the capture-rate@k
   curve from existing checkpoints → becomes the pre-filter KPI baseline.

### Phase B — Pre-filter MVP (August, ~1–2 weeks, after first outreach wave)
1. `--prefilter` mode in the daemon: maintain a streaming quantile estimate
   of scores; forward windows above the top-k% cut (k configurable) instead
   of a fixed alert threshold.
2. Outputs that make it composable: (a) filtered `.pcap` stream/rotating
   files (feeds Wireshark, Zeek, any downstream NDR), (b) syslog/CEF alert
   line per forwarded window (feeds any SIEM), (c) retention tiering — ring
   buffer everything for N minutes, persist only forwarded windows.
3. Throughput benchmark: windows/sec and Mbps sustained on M2 Pro CPU/MPS
   and one A100 — the "what does it cost at the edge" number the pre-filter
   pitch needs. Publish it in the README.
4. Update brief + pitch with the triage framing: "rank-and-forward" for
   MSSPs (their analysts) and packet-retention savings for larger targets.

### Phase C — Community launch + enterprise seeds (Sept, gated on Phase B + numbers)
1. Launch: HN "Show HN", r/netsec, LinkedIn post, short technical blog
   ("byte-level language model as a network attention filter") — the Cribl
   cost numbers and capture-rate@k curve are the story.
2. Convert inbound: GitHub issues → discord/discussions; every serious
   deployment question is MSSP/pilot lead-gen.
3. Enterprise scaffolding ONLY on demonstrated demand: multi-sensor fleet
   view, HEC/S3 connectors as paid add-ons.

### Explicitly deferred
- Rewriting the engine in Rust/C for line-rate (>1 Gbps) capture — only if a
  pilot demands it; scapy + Python is fine for SPAN-port pilot volumes.
- SaaS/cloud anything — contradicts the sovereign story that wins our niche.
- Foundation/consortium governance — years away, if ever.

---

## Verdict
The pasted analysis is directionally right and the numbers back it: the
pre-filter framing is *easier to sell than detection* given current metrics,
the economics it appeals to are real ($100+/GB/day ingest, 30–50% proven
reduction market), and open-core is the proven path in exactly this product
category. The two corrections that matter: we triage *packets* (Smart-PCAP
analog), not SIEM logs directly; and the open-sourcing decision was already
made by the public repo — what remains is drawing the paid boundary, fixing
the license file, and not launching until the validation numbers exist.

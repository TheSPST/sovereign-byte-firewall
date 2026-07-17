# Product Roadmap — Daemon & Dashboard Workflow Features
**Created: 2026-07-17 · Companion to `roadmap.md` (research) and `PROJECT_REPORT_2026-07-16.md` (GTM)**

Guiding rule: nothing here may displace the outreach critical path
(live-FP measurement → UNSW eval → brief → outreach). Phase 1 ships before
outreach because it strengthens the pilot deliverable; Phase 2 is demo
polish; Phase 3 waits for a first pilot's real-world feedback.

---

## Phase 1 — Evidence in every alert (~1.5 days, pre-outreach)

### 1.1 One-click PCAP context (highest value/effort ratio)
Analysts live in Wireshark; hand them the exact packets.

**Implementation** (`firewall_daemon.py`):
- Ring buffer of raw `(timestamp, bytes)` packets covering the last
  ~15 seconds (`collections.deque`, cap by time + byte budget ~50 MB).
- On incident open (first alert of an incident, in `IncidentAggregator.report`),
  snapshot the buffer, keep capturing 5 more seconds, then write
  `alerts/incident_<ISO-ts>_<type>.pcap` via `scapy.wrpcap` on a worker
  thread (never block the sniffer callback).
- Incident CSV gains a `pcap_file` column; WS alert payload gains the path;
  dashboard renders it as a "open in Wireshark" filename.

**Acceptance:** replay a CIC attack pcap against the daemon → flagged
incident produces a .pcap openable in Wireshark containing the offending
window ±5s. Buffer overhead <100 MB RSS.

### 1.2 Deterministic incident enrichment (no LLM, no hallucination)
Every incident carries computed facts, not just a score:

- Top talkers in the window (src→dst pairs by bytes), ports involved,
  protocol mix (TCP/UDP/other %), SYN rate at the time.
- Score context: "13.4 bits — 99.97th percentile of this network's baseline"
  (percentile from the saved calibration distribution — store score
  histogram in `calibration_<iface>.json` at calibration time).
- Emit as extra fields in the WS payload + CSV columns.

**Acceptance:** alert message readable by a non-ML analyst; CSV row is
self-contained enough to triage without opening the pcap.

---

## Phase 2 — Show, don't tell (~2–3 days, demo polish before/during outreach)

### 2.1 Per-byte surprise heatmap ("what the model saw")
The single best CISO-demo asset: flagged window rendered as a hex/byte grid,
surprising bytes glowing red on a cool background.

**Implementation:**
- Daemon already computes `surprise_bits` per byte (shape [1, 511]) — on
  incident open, attach the top-scoring window's byte values + surprise
  array (quantized to uint8 0–255 scale) to the WS payload (~1 KB).
- Dashboard: new pane, `<canvas>` grid (32×16), color = surprise percentile
  vs calibration mean; hover shows offset/byte value/bits. Pure JS, no deps.

**Acceptance:** during a replayed Heartbleed capture, the heatmap visibly
lights up on the anomalous region; on marginal benign exceedances it shows
diffuse low-grade red (which is itself an honest triage signal).

### 2.2 Triage-grade dashboard
Upgrade `dashboard/` from alert ticker to incident manager:

- **Incident list** (not raw alerts): one row per incident — time, type,
  peak score, raw-alert count, enrichment summary, pcap link, heatmap link.
- **Filters:** severity (score percentile bands), type (BYTE/RATE), status.
- **Ack/resolve workflow:** status stored in a local `incidents_state.json`
  written by the daemon on WS message from the UI (dashboard stays static
  files; state persists across restarts).
- **Baseline vs anomaly chart:** rolling score sparkline (last hour, WS
  streamed every ~5s) with threshold line — makes "quiet by default" visible,
  which is the sales argument.

**Acceptance:** an analyst can open the dashboard cold and answer: what
happened, how bad, what's still open — without reading logs.

---

## Phase 3 — After first pilot feedback

### 3.1 Local LLM triage summaries (deliberately deferred)
Stack a small quantized local model (via `ollama`, e.g. Llama-3-8B-Q4) that
turns enriched incident facts into a plain-English paragraph.

**Why deferred, honestly:** today an incident carries score+ports+rates; an
LLM given thin context will produce fluent overconfident guesses — the exact
failure a security buyer catches instantly. It becomes valuable once Phase 1/2
context (talkers, heatmap stats, session features) exists to ground it.

**Design constraints when built:** strictly template-grounded prompt (facts
in, prose out, no speculation about attacker intent), output labeled
"AI summary — verify against pcap", fully offline (keeps the sovereign story),
optional flag `--llm_summaries`, off by default.

### 3.2 Session-table detector (closes the hydra_ftp gap)
Flow table counting connections per (src, dst, dport) over 60–120s windows;
catches slow brute-force both current detectors miss. Third input to the
OR-fusion. (Also feeds enrichment + LLM context.)

### 3.3 Packaging
`launchd`/`systemd` service files, single-command installer, config file
instead of CLI flags, log rotation. Prereq for any multi-machine MSSP pilot;
premature before one is signed.

---

## Sequencing vs. GTM (July–August 2026)

| When | GTM track (critical path) | Product track (this doc) |
|---|---|---|
| Jul 17–19 | FP measurement accumulating; UNSW eval on quota reset | Phase 1 (pcap context + enrichment) |
| Jul 20–24 | Brief finalized with both numbers; iDEX/MeitY applications | Phase 2 (heatmap + triage dashboard) |
| Jul 25–31 | MSSP outreach with demo video | Record demo video using Phase 2 UI |
| Aug | Pilot negotiation | Phase 3 only if a pilot demands it |

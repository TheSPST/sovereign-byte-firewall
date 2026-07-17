# Product Roadmap — Daemon & Dashboard Workflow Features
**Created: 2026-07-17 · Companion to `roadmap.md` (research) and `PROJECT_REPORT_2026-07-16.md` (GTM)**

Guiding rule: nothing here may displace the outreach critical path. UNSW eval
is done; the only remaining outreach blocker is the live-FP number. Everything
below is re-cut around that — **Phase 0 is the minimal outreach-ready sprint**
(low-effort, high-outreach-impact only); every heavier item is explicitly
deferred to *after a pilot is agreed*, so we don't gold-plate before we have a
customer.

---

## Phase 0 — Outreach-Ready sprint (do these first, ~2 days + the FP wait)

The single question this phase answers: *what is the minimum that makes the
demo compelling and the brief complete enough to start sending emails?*
Nothing here is speculative build; each item is hours, not days.

### 0.1 Finish the live-FP run + fill placeholders — *0 dev, just wait + edit*
Let the daemon (now on alert-budget calibration) run ~overnight; compute
incidents/day from `incidents_<iface>.csv`; drop the number into the `[LIVE_FP]`
slots in `TECHNICAL_BRIEF.md`, `README.md`, `OUTREACH.md`. **This is the last
outreach blocker.**

### 0.2 Push everything + tag v0.1.0 — *~15 min*
`git push` the full session; tag `v0.1.0` pinned to the v2 gs75000 checkpoint
with the reproducible eval command in the tag message. Gives outreach a stable
reference to point at.

### 0.3 Deterministic incident enrichment — *~half day* — makes alerts legible
Every incident carries computed facts, not just a score: top talkers
(src→dst by bytes), ports, protocol mix, SYN rate, and score percentile vs. the
saved baseline ("14.4 bits — 99.98th pct of this network"). No LLM, no
hallucination. Emit as WS payload + CSV columns. *Detail in 1.2 below.*
**Why now:** an alert a non-ML analyst can read is table stakes for the demo
and the pilot report; it's a few hours of work.

### 0.4 Per-byte surprise heatmap in the dashboard — *~half–1 day* — the demo moment
The single best CISO-demo asset, and the per-byte `surprise_bits` array
already exists in the daemon. Render the flagged window as a byte grid with
surprising bytes glowing red. *Detail in 2.1 below.*
**Why now:** turns an abstract score into "here's exactly what the model saw" —
the thing that makes a 2-minute video land.

### 0.5 Record the 2-minute demo video — *~half day* — the outreach attachment
Replay a CIC attack pcap through the daemon → dashboard lights up → heatmap
shows the anomalous bytes → incident is legible. Screen capture, no narration
needed. This is the asset every outreach email/DM/post links to.

**Phase 0 done = ready to send.** Brief complete, demo recorded, repo tagged.
Do NOT start Phase 1+ until the first batch of outreach is out.

---

## Deferred until a pilot is agreed (was Phase 1 — pilot deliverable polish)

### 1.1 One-click PCAP context — *~1 day* — DEFER (strong pilot feature, not needed to send the first email)
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
**Why deferred:** high value *once a pilot is running*, but the first outreach
email doesn't need it. Build the week a pilot says yes.

### 1.2 Deterministic incident enrichment — pulled into Phase 0.3 (full spec below)
Every incident carries computed facts, not just a score: top talkers (src→dst
by bytes), ports, protocol mix (TCP/UDP/other %), SYN rate, and score
percentile vs. the saved baseline ("14.4 bits — 99.98th pct"; store the score
histogram in `calibration_<iface>.json` at calibration time). Emit as WS payload
+ CSV columns. No LLM, no hallucination.
**Acceptance:** an alert a non-ML analyst can read; CSV row self-contained
enough to triage without opening the pcap.

---

## Phase 2 — Show, don't tell (demo polish)

### 2.1 Per-byte surprise heatmap — pulled into Phase 0.4 (full spec below)
The single best CISO-demo asset: flagged window rendered as a byte grid,
surprising bytes glowing red. The daemon already computes `surprise_bits` per
byte (shape [1, 511]); on incident open, attach the top-scoring window's byte
values + surprise array (quantized) to the WS payload (~1 KB). Dashboard renders
a `<canvas>` grid (32×16), colour = surprise percentile vs calibration mean,
hover shows offset/byte/bits. Pure JS, no deps.
**Acceptance:** on a replayed Heartbleed capture the heatmap lights up on the
anomalous region; on marginal benign exceedances it shows diffuse low-grade red.

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

## Phase H — Hardening & robustness (from LIMITATIONS.md / MITIGATION_PLAN.md)

Priority-ordered (cheapest-strongest first). These answer the five standing
limitations; each is scoped to a self-contained work session with a rough
effort estimate. None block first-customer outreach — they're what turn the
honest limitations into "managed, here's how."

### H.1 Cross-session CUSUM accumulator — *~3–4 days* — answers the APT / slow-low critique
Per-(src↔dst) bounded-memory accumulator that sums small per-window surprises
over a long horizon (24–48h) via CUSUM/EWMA, firing when cumulative deviation
crosses a bound even though no single window tripped. Add beaconing / rare-
destination detection (regular inter-arrival to unusual endpoints). Third input
to the OR-fusion, keyed by host-pair, evicted by LRU + time.
**Acceptance:** a synthetic "one improbable packet every 6h, each sub-threshold"
sequence is caught by the accumulator while the per-window scorer stays silent.
**Note:** also validate the plain-surprise CUSUM variant on the held-out eval
(the one untested scoring idea from memory).

### H.2 Drift monitoring + poisoning-safe baseline update — *~3 days* — answers concept drift
Builds on the shipped alert-budget calibration. (a) Track live score distribution
vs. the calibration baseline (PSI/KL); auto-flag when the baseline is stale.
(b) "Mark false positive" on the dashboard → fold confirmed-benign windows into
a baseline buffer and *re-derive the threshold* (never a full weight retrain).
(c) Keep a **frozen reference model** + bound the update rate so a slow-poisoning
attacker can't shift "normal" to accept malice.
**Acceptance:** a simulated environment change (score mean shifts +2 bits) raises
a "baseline stale" flag rather than a false-alarm storm; a slow-poison sequence
is rejected by the frozen-reference divergence check.

### H.3 Envelope/metadata upweighting — *~2 days* — sharpens the one signal we keep inside encryption
Explicitly weight packet timing intervals, sizes, and connection-sequence
features alongside the byte score for flows whose payload is masked (TLS/QUIC/
SSH). JA3/JA4-style handshake fingerprint + cert/SNI anomaly as cheap adds.
**Acceptance:** an in-tunnel exfil burst with abnormal size/timing is flagged by
the envelope signal even though its payload bytes are masked.

### H.4 Invariant checkers + private-weights policy — *~1 day code + policy* — defense-in-depth
Rule-based hard limits that alert regardless of ML score (e.g. connection to
known-bad ASN, impossible protocol state, egress to non-allowlisted port on a
locked-down segment). Plus formalize the open-core boundary: **customer-tuned
weights + calibration thresholds are private** (denies the white-box attacker
the model — the cheapest strong adversarial defense).
**Acceptance:** an invariant fires on a crafted low-surprise payload that the
byte model rates benign.

### H.5 XDP throughput benchmark → then Mamba backbone — *~2 days bench, Mamba ~1–2 wks* — answers line-rate
Before any line-rate claim, measure the **steer-down ratio** and sustained
windows/sec + Mbps on real traffic (M2 Pro CPU/MPS and one A100). Prototype the
XDP allowlist-pass / steer-unknown front-end; the byte-model stays the userspace
second stage. Then evaluate a NetMamba-style backbone (arXiv:2405.11449 /
MambaNetBurst 2605.11034) for the 1–60× inference speedup.
**Acceptance:** published throughput number + measured steered fraction; only
then does the brief mention line-rate scale.

---

## Validation experiments (decision-driving, not features)

### V.1 Byte n-gram baseline — *~2–3 hrs to run* — does the transformer earn its complexity?
**Question:** a cheap order-K byte n-gram also produces a next-byte surprise
score. If it matches the transformer on the same held-out zero-day protocol,
the transformer is over-engineered. **Built and unit-tested:** `ngram_baseline.py`
(same pcaps, same masking via `get_pcap_dataloader`, same topk-10% aggregation,
same benign-calibration / held-out-attack split — only the scorer differs).
**Run it on the split we already have cached** (UNSW Shellcode + Exploits, and
CIC) and compare AUC / held-out detection @ ~1% FPR to the transformer:
```
python ngram_baseline.py --benign_calibration_pcap unsw_work/benign_0.pcap \
  --benign_holdout_pcap unsw_work/benign_1_sub.pcap \
  --attack_dir unsw_work/calib_attacks \
  --holdout_attack_pcap unsw_work/attack_shellcode.pcap --order 3 --topk_frac 0.1
```
Sweep `--order 2,3,4,5`. **Decision rule:**
- n-gram within ~2 pts of the transformer → transformer NOT justified at current
  scale; shrink the model or simplify.
- transformer beats n-gram by >~5 pts → complexity justified; cite this ablation
  in technical due diligence (a reviewer *will* ask).
- middle → run a larger order + a Mamba backbone before deciding.
**Why it matters:** this is the sharpest "why not something simpler?" question,
and right now we can't answer it. The experiment settles it cheaply.

**VERDICT (run 2026-07-17, CIC split, transformer JUSTIFIED):** the n-gram
*loses decisively — via generalization, not detection.*

| model (same CIC split) | held-out benign FPR | held-out 0day detection |
|---|---|---|
| n-gram order 2 | 98.3% | ~100% (meaningless) |
| n-gram order 3 | 98.5% | 100% |
| n-gram order 4 | 98.7% | 100% |
| n-gram order 5 | 98.7% | 100% |
| **transformer gs75000** | **0.23%** | **32.6%** |

The n-gram's calibration AUC looks perfect (~0.999) but that's a memorization
artifact: it flags **98%+ of *held-out benign* traffic** (`normal2.pcap`),
making it unusable. It memorizes exact byte contexts; any context it hasn't seen
verbatim — including ordinary benign traffic — backs off to ~8 bits and trips.
The transformer assigns *low* surprise to unseen benign because it learned
general protocol structure. ~400× gap on the metric that decides deployability.

**Caveat (honest):** the n-gram trained on only the calibration pcap (~13.6k
windows) vs the transformer's full Monday corpus. But the failure is
*structural*, not data-starvation: n-grams have no notion of context similarity —
every unseen context is equally surprising (uniform backoff), whereas the
transformer's embeddings interpolate across similar contexts. On diverse /
encrypted traffic the byte-context space is effectively unbounded, so no finite
training set closes the gap. Airtight version: `--train_pcap Monday.pcap`
(expected to improve the n-gram but not to transformer-grade FPR). **This
ablation is the answer to "why not something simpler?" — cite it in due
diligence.**

---

## Sequencing vs. GTM (July–August 2026)

| When | GTM track (critical path) | Product track (this doc) |
|---|---|---|
| Jul 17–18 | Live-FP run finishing (last brief placeholder) | **Phase 0.1–0.2** (fill FP, push, tag v0.1.0) |
| Jul 19–21 | Brief + outreach copy finalized | **Phase 0.3–0.5** (enrichment, heatmap, demo video) |
| Jul 22–25 | **Send:** MSSP emails + iDEX/MeitY applications + warm DMs | — (outreach out; no new build) |
| Jul 26–31 | First replies / calls | 1.1 pcap-context only if a call asks for it |
| Aug | Pilot agreed → build pilot deliverables | Deferred Phase 1/2.2 (pilot polish), then Phase H hardening |
| Post-pilot | Renewal / enterprise features | Phase 3 (LLM triage), Phase H.1–H.5 |

**The cut:** Phase 0 is ~2 days of work plus the FP wait, and it's everything
needed to *start* selling. Triage dashboard (2.2), pcap-on-alert (1.1), LLM
(3.x), and all Phase H hardening wait until a real customer pulls them —
building them now is gold-plating a product nobody has bought yet.

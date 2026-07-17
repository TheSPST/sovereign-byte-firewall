# V2 Feature Integration + Distribution Strategy
**Date: 2026-07-17.** Two questions: (A) how do Mamba, the hourly meta-event summary, and CUSUM actually improve the system, and (B) how do we share this as a tool. Part B is grounded in how modern OSS security tools are really distributed (sources at the end).

---

## Part A — Using the three upgrades to improve the system

These are not three unrelated features; together they turn the current single-shot, per-window detector into a stateful, analyst-friendly, edge-deployable v2. Each maps to a standing limitation.

### A.1 CUSUM — gives the detector memory (closes the slow-and-low gap)
**Problem it fixes:** the per-window scorer is memoryless, so a patient attacker who keeps every packet just under the surprise threshold evades it entirely.
**How it improves the system:** a cumulative-sum accumulator, keyed per host-pair, sums small per-window surprises over a long horizon: `S_t = max(0, S_{t-1} + (x_t - (mu + k)))`, alarm when `S_t > h` (x = window surprise, mu = benign mean, k = slack ~0.5 sigma, h = bound). N individually-sub-threshold anomalies now sum into one detection. It becomes a **third fused detector** alongside byte and rate, with bounded LRU memory and reset-on-alarm.
**Two payoffs:** (1) a live per-flow accumulator in the daemon; (2) a scoring-level upgrade on the eval harness (accumulate surprise across windows) — the untested CUSUM idea already flagged internally.
**Effort:** ~3-4 days. Highest security value; cheapest to prototype on the existing eval. **Build first.**

### A.2 Hourly meta-event summary — reduces fatigue and surfaces drift
**Problem it fixes:** a stream of individual incidents is noisy, and there is nowhere the system notices its own baseline going stale.
**How it improves the system:** a reporter rolls all incidents in a rolling 1-hour window into one **meta-event**: count, severity distribution, top talkers/ports across the hour, and a steady-state-vs-spike verdict. Emitted to the dashboard, a `meta_events.csv`, and optionally a scheduled notification. This is also where **concept-drift detection** lives — compare the hour's score distribution to the calibration baseline (PSI/KL); when it shifts, flag "baseline stale" instead of firing a false-alarm storm. And it is the grounded input a future local-LLM layer would turn into a plain-English digest.
**Effort:** ~2 days. Builds directly on the existing `IncidentAggregator`. High analyst-UX value, low risk.

### A.3 Mamba backbone — the efficiency enabler for edge/pre-filter
**Problem it fixes:** a per-packet transformer cannot run at high line rates, and long-range structure is capped by the 512-byte context.
**How it improves the system:** a NetMamba-style state-space backbone processes sequences in linear time (vs the transformer's quadratic), reported at 1-60x faster inference with lower memory. That directly lowers the "cost at the edge" number the pre-filter positioning depends on, and makes a longer context (capturing longer-range protocol structure) affordable.
**Discipline:** same A/B as the n-gram ablation — same training data, same held-out protocol, compare AUC **and** throughput; keep only if it matches accuracy at materially better speed. Benchmark on CUDA/A100 (Mamba's fast kernel underperforms on Mac MPS).
**Effort:** ~1-2 weeks. Do **after** first customer — it optimizes a scale we do not need yet.

**How they compose:** CUSUM adds memory (better detection), the meta-event adds the analyst layer and drift safety (better usability + robustness), and Mamba adds throughput (better economics). Sequenced: CUSUM -> meta-event -> Mamba.

---

## Part B — How to share this as a tool

**Strategic finding from the research:** for a tool like this, standalone adoption is slow; the fast path is to be a *well-behaved citizen of the existing security ecosystem*. Security Onion alone has 2M+ downloads; teams already run Zeek/Suricata/Arkime/SIEM pipelines. If our alerts drop into that pipeline in a format it already understands, we inherit its distribution. So the strategy is layered from lowest-friction to highest-leverage.

### B.1 Source of truth — GitHub release (MIT)
Tag `v0.1.0`, reproducible eval command in the release notes, honest results table in the README. This is the trust anchor ("read the code, it never calls home"). Add supply-chain hygiene that a security audience expects: signed releases, an SBOM, and a Trivy/dependency scan in CI — a security tool with a sloppy supply chain loses credibility instantly.

### B.2 Frictionless install — PyPI + Docker
- **PyPI** (`pip install sovereign-byte-firewall`): the CLI, daemon, and eval as a standard Python package (like SigmaHQ's `sigma-cli`). Lowest barrier for researchers and tinkerers.
- **Docker image** (the real deployment unit): the research shows security sensors ship as containers that run anywhere from a Raspberry Pi to distributed cloud (this is how Security Onion and Malcolm deploy). `docker run` the daemon against a span port. This is what an MSSP or SOC will actually deploy in a pilot.

### B.3 The leverage move — speak the ecosystem's language
Adoption comes from *output format*, not features. Emit alerts as:
- **EVE-JSON** (Suricata's format) and/or **syslog/CEF** — so Security Onion, Malcolm, Elastic, Splunk, and any SIEM ingest our incidents with zero custom work. This realizes the "additional layer / pre-filter" positioning as literal distribution: we become another alert source in a stack the customer already runs.
- Consume **Zeek/Suricata logs or pcaps** as an alternate input, so we slot in behind an existing sensor rather than requiring our own tap.

### B.4 Ecosystem packages — inherit an existing user base
- **Zeek Package Manager (ZKG):** publish a package so Zeek's large community can add byte-level anomaly scoring with one command. Zeek has a package manager and an annual conference (ZeekWeek) — a real, reachable audience.
- **Security Onion / Malcolm:** get documented (then ideally bundled) as an optional add-on sensor. These integration-first distros are exactly where "an extra detection layer" gets discovered.

### B.5 Models — Hugging Face (already live)
Research checkpoints stay public on HF (`spst01/sovereign-byte-firewall*`) for reproducibility; **customer-tuned weights and calibration thresholds stay private** (the open-core + security boundary). Publishing research weights is credibility; the tuned per-environment model is the product.

### B.6 Detection-as-code posture
Version the model, thresholds, and calibration configs as code; ship the reproducible eval; treat a new checkpoint like a reviewed, tested, promotable artifact (the way detection teams now manage Sigma rules via CI/CD). This matches how the buyers already work and makes the tool feel native to a modern detection-engineering workflow.

### Recommended sharing sequence
1. **Now (pre-launch):** tag v0.1.0, PyPI package, Docker image, EVE-JSON/syslog output. These make the tool *installable and ingestible* — prerequisites for any pilot.
2. **At public launch (after live-FP + demo video):** Show HN / r/netsec, HF checkpoints, ZKG package. One-shot credibility event.
3. **On traction:** Security Onion / Malcolm integration docs, SIEM app listings, and (only if adoption warrants) a managed/enterprise tier.

**The through-line:** don't ask a SOC to adopt a new console. Give them a container that emits alerts their existing stack already reads, prove it on their pcap, and let the open engine build the trust. Distribution and the "additional-layer" positioning are the same move.

---
*Sources: Security Onion (securityonionsolutions.com, blog.securityonion.net — 2M+ downloads, Zeek+Suricata+Strelka+Elastic stack); Malcolm (Zeek+Arkime+Suricata+MISP, Docker/Podman/K8s); Zeek Package Manager + ZeekWeek (zeek.org, 2026 integration guide); SigmaHQ detection-as-code + sigma-cli via pip (sigmahq.io); Python/Docker supply-chain scanning with Trivy (pythonspeed.com).*

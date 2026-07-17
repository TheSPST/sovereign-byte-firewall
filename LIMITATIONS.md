# Limitations & Honest Scope
**Written for serious engineers evaluating this system.** Every limitation below is real. We publish them because an anomaly detector that claims no blind spots is lying, and because our positioning — a low-overhead *tripwire / pre-filter layer*, not an all-knowing shield — depends on being honest about the edges. Each item notes what's fundamental vs. what a roadmap item mitigates.

---

## 1. We are blind to semantics *inside* established encryption — FUNDAMENTAL

**The reality (correct):** we mask encrypted payloads (TLS app-data, QUIC, SSH transport, WireGuard) because ciphertext is high-entropy noise with no learnable structure. A zero-day exploit delivered *inside* a valid, established HTTPS session — a malicious command that fits the normal packet size and timing envelope — sails through. The model never sees its bytes.

**Precision / partial signal retained:** "blind" applies to *payload semantics*, not the whole flow. We still observe the cleartext handshake, certificate/SNI anomalies, and the **packet size + timing envelope** (a genuine side channel — beaconing, exfil bursts, and abnormal request/response rhythms remain visible). An exploit only evades us if it *also* mimics normal size/timing.

**Honest framing:** this ceiling is shared by every payload-blind sensor on the market; seeing inside TLS requires termination/MITM, which contradicts the sovereign, passive, zero-integration promise. We are a **complement to**, not a replacement for, TLS-terminating inspection. Sell us as the layer that catches structurally novel or badly-behaved traffic, explicitly not as an in-tunnel payload scanner.

---

## 2. Concept drift will cause false-positive spikes without re-baselining — MOSTLY FUNDAMENTAL, partially mitigated

**The reality (correct):** the day-one calibration is a *snapshot* of normal. A software rollout on day 5, or an engineering team suddenly pushing 2 a.m. backups, is genuinely improbable to the model → surprise spikes → FP burst → alert fatigue → team turns it off. This is the classic failure mode of *all* anomaly detection.

**What already helps:** (a) the new **alert-budget calibration** (`--target_incidents_per_hour`) makes the FP tolerance an explicit, tunable dial rather than a hidden statistical default; (b) per-environment calibration persists and can be re-run cheaply; (c) incidents carry the surprise score so a drift-driven FP burst is visibly "everything at 13.x," distinguishable from a real spike.

**Roadmap fix:** drift monitoring — track the live score distribution vs. the calibration baseline and auto-flag when the baseline is stale (KL/PSI shift), plus an analyst "mark normal" feedback loop that folds confirmed-benign windows back into the baseline. Scheduled rolling re-calibration (e.g. nightly) instead of a one-shot snapshot. **Until built, honest pilot language: "expect to re-baseline after major environment changes; the alert budget bounds the noise in the meantime."**

---

## 3. "Slow and low" / APT pacing evades a memoryless windowed detector — ARCHITECTURAL, CUSUM is the countermeasure

**The reality (correct):** to keep inference sub-millisecond we score each window in isolation — the model has **no long-term memory**. A patient attacker who sends one improbable packet, waits six hours, sends another, and keeps each individual window *just under* the surprise threshold evades detection entirely. This is a deeper problem than the brute-force gap already noted.

**Why it's real and not hand-wavable:** a memoryless detector cannot accumulate evidence across time by construction. Our rate/session companion detectors only cover volumetric pacing, not sub-threshold payload pacing.

**The direct countermeasure (already on our list):** **CUSUM / sequential change-point detection** accumulates small per-window surprises over time, so N individually-sub-threshold anomalies sum into a detection even when no single window trips. This is precisely why CUSUM is our next untested scoring experiment — this critique is its strongest justification. A stateful per-flow surprise accumulator (bounded memory, keyed by 5-tuple) is the concrete design. **Until built: we honestly do not catch a disciplined, sub-threshold-paced adversary.**

---

## 4. A per-packet model cannot run at core-router line rate — TRUE, defines our deployment envelope

**The reality (correct):** a MacBook on a mirror port handles a small office. At a medium-enterprise core router (millions of packets/sec), a per-packet transformer doing matrix multiplies will saturate CPU, drop packets, and fall over. Mamba improves the constant factor (1–60×) but does not change the fundamental: you cannot afford neural inference on *every* packet at Mpps.

**This validates the pre-filter architecture, it doesn't threaten it.** The correct deployment at scale is **eBPF/XDP in-kernel steering that drops/forwards ~99% of boring traffic before it reaches the model**; our byte-model is the userspace second-stage brain that sees only the steered suspicious fraction. We are explicitly a **span-port / small-segment / pre-filter** tool today, not an inline core-router appliance. **Do not claim line-rate core deployment.** The honest scaling story is "XDP does the cheap first pass; we do the expensive second look on what it flags."

---

## 5. A white-box adversary can craft low-surprise evasions — FUNDAMENTAL to likelihood detectors, raise the cost with defense-in-depth

**The reality (correct):** with access to the open-source model, an attacker runs the math backward — pads a malicious payload with **high-probability ("benign-grammar") bytes** so the surprise score stays near zero while the exploit executes. This is the textbook adversarial-ML attack on any likelihood-based detector.

**Precision — evasion is possible but not free:** (a) padding to benign grammar has a real cost — it constrains payload construction and can break the exploit's function or protocol validity; (b) the attacker must simultaneously evade the **fused** rate/session detectors, not just the byte model; (c) the **per-environment calibration threshold is not published** — it's derived from the customer's own traffic, so a white-box attacker still doesn't know the operating point; (d) masked regions can't be used as free padding space.

**This directly shapes the open-core boundary:** publish the architecture and research checkpoints (trust/verifiability), but **keep customer-tuned weights and calibration thresholds private** — that's the secret that raises adversarial cost, and it's also the paid-tier boundary. Longer term, detector ensembles and randomized scoring raise the evasion bar further. **Honest framing: we harden against opportunistic and novel attacks, not a determined white-box adversary with model access and unlimited attempts — no single-signal detector defends that threat model.**

---

## The honest one-paragraph scope

This is a low-overhead **tripwire / pre-filter** for loud, structurally unusual, or genuinely novel network behaviour that signature tools miss — deployed passively, on-prem, with near-zero integration. It does **not** see inside encrypted tunnels, it can be evaded by a patient sub-threshold adversary or a white-box attacker with model access, it needs re-baselining as the environment drifts, and it belongs behind an XDP pre-filter at high line rates. Positioned as a specialised *additional layer*, it is credible and useful. Positioned as an all-knowing security shield, it is not — and we won't sell it that way. Every serious engineer who reads this list should trust the numbers more, not less.

---
*Cross-references: [RESEARCH_LANDSCAPE.md](RESEARCH_LANDSCAPE.md) (byte-flattening critique, eBPF/XDP, Mamba), [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md) (pre-filter positioning, open-core boundary), [PRODUCT_ROADMAP.md](PRODUCT_ROADMAP.md) (session detector, drift/feedback).*

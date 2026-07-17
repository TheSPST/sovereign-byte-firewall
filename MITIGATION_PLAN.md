# Mitigation Plan — Graded Against Research
**Companion to [LIMITATIONS.md](LIMITATIONS.md).** Each limitation has a proposed mitigation; below is an honest grade of how far it actually gets us, what the literature says, and the refinement that makes it real. Verdicts are deliberately conservative — "Managed" and "Mitigated" almost never become "Solved" in anomaly detection.

---

## 1. Concept drift → online learning + human feedback
**Plan grade: Managed (not Solved) — and it introduces a new attack surface.**

Sliding-window online updates and a "mark false positive" feedback loop are the correct instincts and standard practice. But the plan omits the dominant risk:

- **Poisoning / boiling-frog.** A continuously self-updating "normal" model can be *slowly shifted to accept malicious traffic* by an attacker who introduces it gradually. This is a well-documented failure of online anomaly detection — you trade drift-robustness for poisoning-vulnerability. Naive online learning can train yourself blind.
- **Refinements that make it safe:** (a) bound the update rate and keep a **frozen reference model** to detect when the online model has drifted suspiciously far from it; (b) prefer updating the **calibration threshold** (cheap, reversible) over continuously retraining weights; (c) on "mark false positive," fold confirmed-benign windows into a **baseline buffer** rather than a full retrain; (d) only auto-absorb traffic that passes *both* the model and the rule-based guardrails.
- **Realistic verdict:** drift is *managed* with bounded, auditable updates — never fully solved, and the human-in-the-loop contradicts "runs itself" for the smallest customers (fine for MSSP/enterprise tiers).

## 2. Line rate → eBPF/XDP steering
**Plan grade: Architecturally right; the key number is unproven.**

Splitting into an in-kernel XDP first pass + userspace model second pass is exactly the deployment pattern the research points to (SmartX, in-kernel CNN work). Two precisions before claiming victory:

- **"Drop known-bad with zero compute" reintroduces signatures.** XDP can cheaply *allowlist-pass* known-good and *steer* unknown, but dropping known-bad needs rules/signatures at that tier. That's a fine complementary layer — just don't describe the whole system as signature-free once it's in place.
- **The steer-down ratio is the whole ballgame.** If XDP forwards even 10% of an Mpps link as "unknown/complex," the userspace model still faces enormous load. On diverse enterprise traffic the unknown fraction may be well above 10%. **This must be benchmarked** (windows/sec and sustained Mbps on real hardware — the FUTURE_ROADMAP throughput number). Until measured, we say "span-port / segment scale today; XDP-fronted core scale is the roadmap," not "runs comfortably on standard hardware."

## 3. Slow-and-low → dual-window / multi-granularity
**Plan grade: Mitigated; fix the specific long-window signal.**

Running the sub-ms per-packet scorer alongside a long-horizon (24–48h) tracker is the right structure and matches our CUSUM plan. One correction:

- **"Session persistence" is the wrong signal for the worst case.** A disciplined attacker opens a *new short connection each time*, hours apart — no long-lived session exists to flag. The signal that actually catches this is **cross-session surprise accumulation**: a CUSUM/EWMA accumulator keyed by host-pair (src↔dst) that sums small per-window surprises over the long horizon, plus **beaconing / rare-destination** detection (regular inter-arrival to an unusual endpoint). Persistence is one weak signal; accumulation is the real one.
- **Realistic verdict:** mitigated once cross-session accumulation ships; a memoryless system genuinely cannot catch a sub-threshold pacer, so this is a build item, not a claim we can make today.

## 4. Adversarial evasion → adversarial training + local variance + invariant checkers
**Plan grade: Well-reasoned; reorder the levers.**

- **Local variance (the "sovereign advantage") is the strongest lever — elevate it.** Because "normal" is calibrated to each customer's own traffic and the threshold is never published, **there is no universal evasion payload** — an attacker must tailor to each target's secret baseline, which they can't observe. This is genuine moving-target defense and it's your best adversarial story. Lead with it.
- **Adversarial training is the weak link — keep but temper.** Training against padded synthetics helps against evasions you anticipated; it's a known **arms race** and does not guarantee robustness to new adaptive attacks (adversarial robustness is broadly unsolved). It can also trade a little benign accuracy. Useful supplement, not a guarantee.
- **Invariant checkers (rule-based hard limits regardless of ML score) — excellent, keep.** Classic defense-in-depth.
- **Add the missing lever:** keep **per-customer tuned weights and thresholds private** (the open-core boundary). The white-box attack assumes model access; denying it is the cheapest strong defense.
- **Realistic verdict:** mitigated against opportunistic and non-adaptive adversaries; a determined white-box attacker with query access remains a hard threat model for any single detector.

## 5. Encryption blind spot → metadata weighting + SSL proxy
**Plan grade: Metadata = correct. SSL proxy = valid option, positioning trap.**

- **Upweight the envelope — correct and already our direction.** Explicitly weighting timing intervals, payload sizes, and connection sequences over ciphertext is the right engineering, and aligns with flow-metadata / JA3-JA4 fingerprinting work. Do this.
- **SSL decryption proxy — technically valid, strategically risky.** Sitting behind a TLS-terminating proxy lets us read cleartext, but it **contradicts the sovereign, passive, zero-integration, no-MITM promise** that is our entire differentiation — and the proxy termination point is exactly where incumbents (Zscaler, Palo Alto, etc.) already inspect, so we'd be fighting them on their turf having surrendered our angle. Offer it as "we can *also* sit behind your existing decryption proxy," never as the headline answer. The metadata/envelope story is what preserves the moat.
- **Realistic verdict:** mitigated via side channels; in-tunnel payload semantics remain out of reach without decryption, and we should not pretend otherwise.

---

## Net assessment
The plan is genuinely good and mostly aligns with both the research and our existing roadmap. The honest adjustments: (1) drift online-learning needs **poisoning guardrails** (frozen reference + bounded updates + threshold-not-weights); (2) line-rate hinges on an **unmeasured steer-down ratio** — benchmark before claiming it; (3) slow-low needs **cross-session accumulation (CUSUM)**, not session persistence; (4) **local per-environment variance** is the star adversarial defense, adversarial training is a tempered supplement, and keep tuned weights private; (5) metadata weighting yes, **SSL proxy as an option not a pillar**. None of these are blockers — they're the difference between "Solved" (which serious engineers won't believe) and "Managed, here's exactly how" (which they will).

## Priority order (cheapest-strongest first)
1. **Cross-session CUSUM accumulator** (#3) — highest security value, already on our list, directly answers the APT critique.
2. **Alert-budget + drift monitoring + frozen-reference guardrail** (#1) — builds on the calibration we just shipped; keeps FPs bounded without poisoning risk.
3. **Envelope/metadata upweighting** (#5) — sharpens the one signal we keep inside encryption; no positioning cost.
4. **Invariant checkers + keep-weights-private** (#4) — cheap defense-in-depth; the private-weights lever is a doc/policy decision, not code.
5. **XDP throughput benchmark** (#2) — measure the steer-down ratio before promising line-rate; Mamba backbone follows.

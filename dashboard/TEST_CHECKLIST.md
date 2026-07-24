# Dashboard Test Checklist — New Threat Type UI

Covers: `SLOW_DISTRIBUTED` (purple/cyan botnet-campaign badge) and
`CRITICAL_BYTE` (flashing red/gold banner on the heatmap panel).

## Setup

1. `pip install websockets` (if not already installed).
2. `python dashboard/mock_ws_server.py`
3. Open `dashboard/index.html` in a browser.
4. Watch it run through the full script once (~16s), then it loops automatically.

## Test cases

**TC1 — Connection**
Status orb turns green, header reads "Engine Connected (Secure Streaming)".
Pass/fail: ___

**TC2 — BYTE unaffected (regression check)**
`BYTE` alert still renders red, still increments "Byte Anomalies" stat, still
updates "Highest Surprise", heatmap still renders with hot cells.
Pass/fail: ___

**TC3 — RATE unaffected (regression check)**
`RATE` alert still renders amber, still increments "Rate Anomalies" stat.
Pass/fail: ___

**TC4 — SLOW_DISTRIBUTED badge**
Log entry border/left-edge and TYPE text render in a purple→cyan gradient
(not the default grey). A pill badge reading "⚡ BOTNET CAMPAIGN → 10.0.0.9:443"
appears under the message, and the enrichment line below it shows talkers,
ports, protocol mix, and cumulative bits.
Pass/fail: ___

**TC5 — CRITICAL_BYTE banner appears**
When the first `CRITICAL_BYTE` fires, a banner appears at the top of the
heatmap panel showing the message text and "9.12 bits", and is visibly
flashing between red and gold.
Pass/fail: ___

**TC6 — CRITICAL_BYTE log styling**
The corresponding log entry border is gold, TYPE text is gold with a red
glow (distinct from a plain `BYTE` entry).
Pass/fail: ___

**TC7 — Banner re-trigger**
The second `CRITICAL_BYTE` (fired 2s after the first, per the script) updates
the banner text/score to "9.40 bits" and the auto-dismiss timer restarts —
the banner should NOT disappear early because a second alert arrived.
Pass/fail: ___

**TC8 — Manual dismiss**
Click "Dismiss" on the banner. It hides immediately and does not reappear on
its own afterward (only a new `CRITICAL_BYTE` message should bring it back).
Pass/fail: ___

**TC9 — Auto-dismiss**
Let a `CRITICAL_BYTE` banner sit untouched. It should disappear on its own
~8 seconds after the last one fired.
Pass/fail: ___

**TC10 — Reduced motion**
In Chrome DevTools → Rendering tab → "Emulate CSS media feature
prefers-reduced-motion" → set to "reduce". Trigger a `CRITICAL_BYTE` again:
banner should appear solid (no flashing animation) but remain fully legible.
Pass/fail: ___

**TC11 — No console errors**
DevTools console stays clean (no errors/warnings) through a full script loop.
Pass/fail: ___

**TC12 — Log cap still works**
Let the mock server loop run for several cycles; confirm the log list still
caps at 100 entries (oldest entries drop off the bottom) regardless of the
mix of alert types.
Pass/fail: ___

## Notes
- The mock server only exercises the front end — it does not touch
  `firewall_daemon.py` or the model, so it's safe to run anytime without a
  live capture.
- Stop the dashboard's real daemon (if running) before starting the mock, or
  point one of them at a different port, since both bind `ws://localhost:8765`.

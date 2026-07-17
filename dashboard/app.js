const WS_URL = 'ws://localhost:8765';
let ws;

// DOM Elements
const statusOrb = document.getElementById('status-orb');
const connectionStatus = document.getElementById('connection-status');
const statByte = document.getElementById('stat-byte');
const statRate = document.getElementById('stat-rate');
const statMaxEntropy = document.getElementById('stat-max-entropy');
const logsContainer = document.getElementById('live-logs');
const clearBtn = document.getElementById('clear-logs');

// State
let byteCount = 0;
let rateCount = 0;
let maxEntropy = 0;

// Format timestamp
function formatTime(timestamp) {
    const d = new Date(timestamp * 1000);
    return d.toISOString().split('T')[1].slice(0, -1);
}

// --- Byte surprise heatmap ("what the model saw") ---
const heatmapCanvas = document.getElementById('heatmap-canvas');
const heatmapCaption = document.getElementById('heatmap-caption');
const heatmapTooltip = document.getElementById('heatmap-tooltip');
const HM_COLS = 32;
let lastHeatmap = null;  // {bytes, surprise}

function renderHeatmap(hm) {
    if (!heatmapCanvas || !hm || !hm.surprise || !hm.surprise.length) return;
    lastHeatmap = hm;
    const ctx = heatmapCanvas.getContext('2d');
    const n = hm.surprise.length;
    const rows = Math.ceil(n / HM_COLS);
    const cw = heatmapCanvas.width / HM_COLS;
    const ch = heatmapCanvas.height / rows;
    // Fixed scale: per-byte surprise ~0..8 bits; >=8 is fully "hot".
    const HOT = 8.0;
    ctx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
    for (let i = 0; i < n; i++) {
        const r = Math.floor(i / HM_COLS), c = i % HM_COLS;
        const t = Math.max(0, Math.min(1, hm.surprise[i] / HOT));
        // cool dark base, red glow rising with surprise
        const red = Math.round(255 * t);
        const grn = Math.round(40 + 40 * (1 - t));
        const blu = Math.round(70 * (1 - t) + 30 * t);
        ctx.fillStyle = `rgb(${red},${grn},${blu})`;
        ctx.fillRect(c * cw + 0.5, r * ch + 0.5, cw - 1, ch - 1);
    }
    const maxBits = Math.max(...hm.surprise);
    heatmapCaption.textContent = `${n} bytes · peak ${maxBits.toFixed(2)} bits · red = high surprise`;
}

if (heatmapCanvas) {
    heatmapCanvas.addEventListener('mousemove', (ev) => {
        if (!lastHeatmap) return;
        const rect = heatmapCanvas.getBoundingClientRect();
        const scaleX = heatmapCanvas.width / rect.width, scaleY = heatmapCanvas.height / rect.height;
        const n = lastHeatmap.surprise.length, rows = Math.ceil(n / HM_COLS);
        const cw = heatmapCanvas.width / HM_COLS, ch = heatmapCanvas.height / rows;
        const c = Math.floor(((ev.clientX - rect.left) * scaleX) / cw);
        const r = Math.floor(((ev.clientY - rect.top) * scaleY) / ch);
        const i = r * HM_COLS + c;
        if (i < 0 || i >= n) { heatmapTooltip.style.display = 'none'; return; }
        const b = lastHeatmap.bytes[i];
        heatmapTooltip.textContent = `#${i}  byte 0x${b.toString(16).padStart(2,'0')} (${b})  ${lastHeatmap.surprise[i].toFixed(2)} bits`;
        heatmapTooltip.style.left = (ev.clientX + 12) + 'px';
        heatmapTooltip.style.top = (ev.clientY + 12) + 'px';
        heatmapTooltip.style.display = 'block';
    });
    heatmapCanvas.addEventListener('mouseleave', () => { heatmapTooltip.style.display = 'none'; });
}

function formatEnrichment(e) {
    if (!e || Object.keys(e).length === 0) return '';
    const parts = [];
    if (e.top_talkers && e.top_talkers.length) {
        // talkers may be [{pair,bytes}] (incidents) or ["a -> b"] (meta summary)
        const names = e.top_talkers.map(t => (typeof t === 'string' ? t : t.pair));
        parts.push('talkers: ' + names.join(', '));
    }
    if (e.top_ports && e.top_ports.length) parts.push('ports: ' + e.top_ports.join('/'));
    if (e.proto_mix_pct) {
        parts.push(Object.entries(e.proto_mix_pct).map(([p, v]) => `${p} ${v}%`).join(' '));
    }
    if (typeof e.syns === 'number' && e.syns > 0) parts.push(e.syns + ' SYN');
    if (typeof e.score_percentile === 'number') parts.push(e.score_percentile + 'th pct');
    if (typeof e.cusum_level === 'number') parts.push('cumulative ' + e.cusum_level + ' bits');
    if (typeof e.drift_psi === 'number') parts.push('drift PSI ' + e.drift_psi);
    return parts.join('  •  ');
}

function addLog(type, score, message, timestamp, enrichment) {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type.toLowerCase()}-alert`;

    const timeSpan = document.createElement('span');
    timeSpan.className = 'time';
    timeSpan.textContent = formatTime(timestamp);

    const typeSpan = document.createElement('span');
    typeSpan.className = 'type';
    typeSpan.textContent = type;

    const scoreSpan = document.createElement('span');
    scoreSpan.className = 'score';
    scoreSpan.textContent = type === 'BYTE' ? parseFloat(score).toFixed(2) : (score ?? '');

    const msgSpan = document.createElement('span');
    msgSpan.className = 'msg';
    msgSpan.textContent = message;

    // Enrichment line (computed facts) shown under the message when present.
    const ctx = formatEnrichment(enrichment);
    if (ctx) {
        const ctxSpan = document.createElement('span');
        ctxSpan.className = 'context';
        ctxSpan.textContent = ctx;
        ctxSpan.style.cssText = 'display:block;color:var(--text-secondary);font-family:var(--font-mono);font-size:0.8em;margin-top:4px;';
        msgSpan.appendChild(ctxSpan);
    }

    entry.appendChild(timeSpan);
    entry.appendChild(typeSpan);
    entry.appendChild(scoreSpan);
    entry.appendChild(msgSpan);

    logsContainer.prepend(entry);

    // Keep max 100 logs in DOM
    if (logsContainer.children.length > 100) {
        logsContainer.removeChild(logsContainer.lastChild);
    }
}

function updateStats(type, score) {
    if (type === 'BYTE') {
        byteCount++;
        statByte.textContent = byteCount;
        statByte.classList.add('red-alert');
        
        const floatScore = parseFloat(score);
        if (floatScore > maxEntropy) {
            maxEntropy = floatScore;
            statMaxEntropy.textContent = maxEntropy.toFixed(2);
            statMaxEntropy.classList.add('red-alert');
        }
    } else if (type === 'RATE') {
        rateCount++;
        statRate.textContent = rateCount;
        statRate.classList.add('red-alert');
    }
    
    // Flash status orb
    statusOrb.className = 'status-orb alert';
    setTimeout(() => {
        if(ws.readyState === WebSocket.OPEN) {
            statusOrb.className = 'status-orb connected';
        }
    }, 1000);
}

function connect() {
    ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
        statusOrb.className = 'status-orb connected';
        connectionStatus.textContent = 'Engine Connected (Secure Streaming)';
        connectionStatus.style.color = 'var(--accent-green)';
        
        // Remove system messages
        logsContainer.innerHTML = '';
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.enrichment && data.enrichment.heatmap) {
            renderHeatmap(data.enrichment.heatmap);
        }
        addLog(data.type, data.score, data.message, data.timestamp, data.enrichment);
        updateStats(data.type, data.score);
    };
    
    ws.onclose = () => {
        statusOrb.className = 'status-orb';
        connectionStatus.textContent = 'Disconnected. Reconnecting...';
        connectionStatus.style.color = 'var(--accent-red)';
        setTimeout(connect, 3000);
    };
    
    ws.onerror = (err) => {
        console.error("WebSocket Error:", err);
    };
}

clearBtn.addEventListener('click', () => {
    logsContainer.innerHTML = '';
});

// Initialize
connect();

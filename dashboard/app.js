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

function formatEnrichment(e) {
    if (!e || Object.keys(e).length === 0) return '';
    const parts = [];
    if (e.top_talkers && e.top_talkers.length) {
        parts.push('talkers: ' + e.top_talkers.map(t => t.pair).join(', '));
    }
    if (e.top_ports && e.top_ports.length) parts.push('ports: ' + e.top_ports.join('/'));
    if (e.proto_mix_pct) {
        parts.push(Object.entries(e.proto_mix_pct).map(([p, v]) => `${p} ${v}%`).join(' '));
    }
    if (typeof e.syns === 'number' && e.syns > 0) parts.push(e.syns + ' SYN');
    if (typeof e.score_percentile === 'number') parts.push(e.score_percentile + 'th pct');
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

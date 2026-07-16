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

function addLog(type, score, message, timestamp) {
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
    scoreSpan.textContent = type === 'BYTE' ? parseFloat(score).toFixed(2) : score;
    
    const msgSpan = document.createElement('span');
    msgSpan.className = 'msg';
    msgSpan.textContent = message;
    
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
        addLog(data.type, data.score, data.message, data.timestamp);
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

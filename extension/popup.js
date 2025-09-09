// popup.js
// Handles UI, WebSocket connection and fallback REST request

const sendBtn = document.getElementById('sendBtn');
const restBtn = document.getElementById('restBtn');
const featuresEl = document.getElementById('features');
const resultText = document.getElementById('resultText');
const resultBox = document.getElementById('resultBox');
const statusEl = document.getElementById('status');
const cardIdEl = document.getElementById('card_id');
const endpointSelect = document.getElementById('endpointSelect');

function setStatus(s) {
    statusEl.textContent = 'Status: ' + s;
}

function showResult(obj) {
    resultBox.classList.remove('hidden');
    resultText.textContent = JSON.stringify(obj, null, 2);
}

// Build payload from UI
function buildPayload() {
    let features;
    try {
        features = JSON.parse(featuresEl.value);
    } catch (e) {
        throw new Error('Invalid JSON in features field.');
    }
    const card_id = cardIdEl.value ? String(cardIdEl.value) : null;
    const payload = { features };
    if (card_id) payload.card_id = card_id;
    return payload;
}

// WebSocket send
sendBtn.addEventListener('click', async () => {
    try {
        const payload = buildPayload();
        const wsUrl = endpointSelect.value; // expects ws://... for WebSocket
        if (!wsUrl.startsWith('ws://') && !wsUrl.startsWith('wss://')) {
            throw new Error('Select a WebSocket endpoint in the dropdown (ws:// or wss://).');
        }

        setStatus('connecting to WebSocket...');
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            setStatus('connected — sending payload');
            ws.send(JSON.stringify(payload));
        };

        ws.onmessage = (evt) => {
            try {
                const data = JSON.parse(evt.data);
                showResult(data);
                setStatus('received response');
            } catch (err) {
                setStatus('invalid JSON from server');
                resultText.textContent = evt.data;
                resultBox.classList.remove('hidden');
            } finally {
                ws.close();
            }
        };

        ws.onerror = (err) => {
            setStatus('WebSocket error');
            console.error('WS error', err);
            ws.close();
        };

        ws.onclose = () => {
            setStatus('ws closed');
        };
    } catch (err) {
        setStatus('error');
        resultBox.classList.remove('hidden');
        resultText.textContent = 'Error: ' + err.message;
    }
});

// REST fallback
restBtn.addEventListener('click', async () => {
    try {
        const payload = buildPayload();
        const restUrl = endpointSelect.value;
        if (!restUrl.startsWith('http://') && !restUrl.startsWith('https://')) {
            throw new Error('Select a REST endpoint in the dropdown (http:// or https://).');
        }

        setStatus('sending REST request...');
        const resp = await fetch(restUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(`HTTP ${resp.status}: ${text}`);
        }
        const data = await resp.json();
        showResult(data);
        setStatus('REST response received');
    } catch (err) {
        setStatus('error');
        resultBox.classList.remove('hidden');
        resultText.textContent = 'Error: ' + err.message;
    }
});

// Optionally: preload last features from chrome.storage
document.addEventListener('DOMContentLoaded', () => {
    chrome.storage.local.get(['lastFeatures', 'lastCardId'], (items) => {
        if (items.lastFeatures) featuresEl.value = items.lastFeatures;
        if (items.lastCardId) cardIdEl.value = items.lastCardId;
    });
});

// Save feature input when closing
window.addEventListener('unload', () => {
    try {
        chrome.storage.local.set({ lastFeatures: featuresEl.value, lastCardId: cardIdEl.value });
    } catch (e) { }
});

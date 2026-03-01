/* ═══════════════════════════════════════════════════════════════════
   TicketAI – Frontend JavaScript
   Handles tab switching, API calls, result rendering
   ═══════════════════════════════════════════════════════════════════ */

// ── Tab Navigation ─────────────────────────────────────────────────

document.querySelectorAll('.nav-pill').forEach(pill => {
    pill.addEventListener('click', () => {
        // Update active pill
        document.querySelectorAll('.nav-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');

        // Show correct tab
        const tabId = pill.getAttribute('data-tab');
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        document.getElementById('tab-' + tabId).classList.add('active');
    });
});

// ── Character Counter ──────────────────────────────────────────────

const ticketInput = document.getElementById('ticket-input');
const charCount = document.getElementById('char-count');

if (ticketInput && charCount) {
    ticketInput.addEventListener('input', () => {
        charCount.textContent = ticketInput.value.length;
    });
}

// ── Classify Single Ticket ─────────────────────────────────────────

async function classifyTicket() {
    const text = ticketInput.value.trim();
    if (!text) {
        showToast('Please enter a ticket description.', true);
        ticketInput.focus();
        return;
    }

    const btn = document.getElementById('classify-btn');
    btn.classList.add('loading');
    btn.disabled = true;

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || 'Something went wrong.', true);
            return;
        }

        renderResult(data);

    } catch (err) {
        showToast('Connection error. Is the server running?', true);
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

// Enter key to submit
if (ticketInput) {
    ticketInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            classifyTicket();
        }
    });
}

// ── Render Classification Result ───────────────────────────────────

const CATEGORY_COLORS = {
    'Technical Issue': { class: 'cat-technical', gradient: 'linear-gradient(135deg, #ef4444, #dc2626)' },
    'Billing Issue':   { class: 'cat-billing',   gradient: 'linear-gradient(135deg, #f59e0b, #d97706)' },
    'Account Access':  { class: 'cat-account',   gradient: 'linear-gradient(135deg, #6366f1, #4f46e5)' },
    'General Inquiry': { class: 'cat-general',   gradient: 'linear-gradient(135deg, #10b981, #059669)' },
};

const PRIORITY_COLORS = {
    'High':   'high',
    'Medium': 'medium',
    'Low':    'low',
};

const CONF_BAR_COLORS = {
    'Technical Issue': '#ef4444',
    'Billing Issue':   '#f59e0b',
    'Account Access':  '#6366f1',
    'General Inquiry': '#10b981',
};

function renderResult(data) {
    const card = document.getElementById('result-card');
    card.classList.remove('hidden');

    // Badges
    document.getElementById('result-category').textContent = data.category;
    const priBadge = document.getElementById('result-priority');
    priBadge.textContent = data.priority;
    priBadge.className = 'badge badge-priority ' + (PRIORITY_COLORS[data.priority] || '');

    // Category value
    const catText = document.getElementById('result-cat-text');
    catText.textContent = data.category;
    catText.className = 'result-value ' + (CATEGORY_COLORS[data.category]?.class || '');

    // Priority value
    const priText = document.getElementById('result-pri-text');
    priText.textContent = data.priority + ' Priority';
    priText.className = 'result-value pri-' + (data.priority || '').toLowerCase();

    // Confidence bars
    const barsEl = document.getElementById('confidence-bars');
    barsEl.innerHTML = '';

    // Sort confidence by value descending
    const sorted = Object.entries(data.confidence).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([cat, pct]) => {
        const row = document.createElement('div');
        row.className = 'conf-row';

        const color = CONF_BAR_COLORS[cat] || '#6366f1';

        row.innerHTML = `
            <span class="conf-label">${cat}</span>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="background: ${color};"></div>
            </div>
            <span class="conf-value">${pct.toFixed(1)}%</span>
        `;

        barsEl.appendChild(row);

        // Animate bar
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                row.querySelector('.conf-bar-fill').style.width = pct + '%';
            });
        });
    });

    // Cleaned text
    document.getElementById('cleaned-text').textContent = data.cleaned_text;

    // Scroll to result
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Batch Classification ───────────────────────────────────────────

async function batchClassify() {
    const raw = document.getElementById('batch-input').value.trim();
    if (!raw) {
        showToast('Please enter at least one ticket.', true);
        return;
    }

    const tickets = raw.split('\n').map(t => t.trim()).filter(Boolean);
    if (tickets.length === 0) {
        showToast('No valid tickets found.', true);
        return;
    }

    const btn = document.getElementById('batch-btn');
    btn.classList.add('loading');
    btn.disabled = true;

    try {
        const res = await fetch('/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickets }),
        });

        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || 'Something went wrong.', true);
            return;
        }

        renderBatchResults(data.results);

    } catch (err) {
        showToast('Connection error. Is the server running?', true);
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

function renderBatchResults(results) {
    const container = document.getElementById('batch-results');
    container.classList.remove('hidden');

    let rows = results.map((r, i) => {
        const priClass = PRIORITY_COLORS[r.priority] || '';
        const shortText = r.text.length > 70 ? r.text.substring(0, 67) + '...' : r.text;
        return `
            <tr>
                <td style="color: var(--text-muted); font-weight: 600;">${i + 1}</td>
                <td style="color: var(--text-primary);">${escapeHtml(shortText)}</td>
                <td><span class="badge badge-category">${r.category}</span></td>
                <td><span class="badge badge-priority ${priClass}">${r.priority}</span></td>
            </tr>
        `;
    }).join('');

    container.innerHTML = `
        <div class="batch-table">
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Ticket</th>
                        <th>Category</th>
                        <th>Priority</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;

    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Example Cards ──────────────────────────────────────────────────

function tryExample(card) {
    const text = card.querySelector('.example-text').textContent;

    // Switch to classify tab
    document.querySelectorAll('.nav-pill').forEach(p => p.classList.remove('active'));
    document.getElementById('nav-classify').classList.add('active');
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.getElementById('tab-classify').classList.add('active');

    // Fill in text and classify
    ticketInput.value = text;
    charCount.textContent = text.length;
    classifyTicket();
}

// ── Toast Notifications ────────────────────────────────────────────

function showToast(msg, isError = false) {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast' + (isError ? ' error' : '');
    toast.textContent = msg;
    document.body.appendChild(toast);

    setTimeout(() => toast.remove(), 3000);
}

// ── Utility ────────────────────────────────────────────────────────

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

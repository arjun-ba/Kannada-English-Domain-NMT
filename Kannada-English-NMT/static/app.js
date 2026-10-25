/**
 * IndicTrans NMT Studio - Core Application Logic
 * Vanilla ES6 JavaScript for fast, zero-dependency, ultra-smooth execution.
 */

// Global Application State
const state = {
    direction: "kn->en", // 'kn->en' or 'en->kn'
    currentDomain: "Agriculture",
    datasetSamples: [],
    lastTranslation: null
};

// DOM Content Loaded Initializer
document.addEventListener("DOMContentLoaded", () => {
    initEventListeners();
    fetchDataset();
    fetchSystemHealth();
});

// Setup Event Listeners
function initEventListeners() {
    const sourceInput = document.getElementById("source-input");
    const domainSelect = document.getElementById("domain-select");

    // Real-time character count
    sourceInput.addEventListener("input", () => {
        const len = sourceInput.value.length;
        document.getElementById("source-char-count").innerText = `${len} characters`;
    });

    // Keyboard shortcut: Ctrl + Enter / Cmd + Enter
    sourceInput.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            executeTranslation();
        }
    });

    domainSelect.addEventListener("change", (e) => {
        state.currentDomain = e.target.value;
    });
}

// Tab Switching
function switchTab(tabName) {
    // Update active tab buttons
    document.querySelectorAll(".nav-tab").forEach(tab => tab.classList.remove("active"));
    const activeBtn = document.getElementById(`tab-btn-${tabName}`);
    if (activeBtn) activeBtn.classList.add("active");

    // Update active sections
    document.querySelectorAll(".app-section").forEach(sec => sec.classList.remove("active"));
    const activeSection = document.getElementById(`section-${tabName}`);
    if (activeSection) activeSection.classList.add("active");

    // Refresh specific section data if required
    if (tabName === "microservices") {
        fetchSystemHealth();
    } else if (tabName === "dataset" && state.datasetSamples.length === 0) {
        fetchDataset();
    }
}

// Swap Direction (Kannada <-> English)
function swapLanguageDirection() {
    const sourceLabel = document.getElementById("source-lang-label");
    const targetLabel = document.getElementById("target-lang-label");
    const sourceInput = document.getElementById("source-input");
    const targetOutput = document.getElementById("target-output");

    if (state.direction === "kn->en") {
        state.direction = "en->kn";
        sourceLabel.innerText = "English";
        targetLabel.innerText = "Kannada (ಕನ್ನಡ)";
        sourceInput.classList.remove("kannada-text");
        sourceInput.placeholder = "Enter English sentence here (e.g., Farmers should get the soil tested)...";
    } else {
        state.direction = "kn->en";
        sourceLabel.innerText = "Kannada (ಕನ್ನಡ)";
        targetLabel.innerText = "English";
        sourceInput.classList.add("kannada-text");
        sourceInput.placeholder = "ಇಲ್ಲಿ ಕನ್ನಡ ವಾಕ್ಯವನ್ನು ನಮೂದಿಸಿ (ಉದಾಹರಣೆಗೆ: ರೈತರು ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಮಾಡಿಸಬೇಕು)...";
    }

    // Swap text values if current output has content
    const currentSource = sourceInput.value.trim();
    const currentTarget = targetOutput.innerText.trim();
    if (currentTarget && !targetOutput.classList.contains("placeholder-state")) {
        sourceInput.value = currentTarget;
        document.getElementById("source-char-count").innerText = `${currentTarget.length} characters`;
    }

    clearTarget();
    showToast(`Swapped translation direction: ${state.direction === "kn->en" ? "Kannada → English" : "English → Kannada"}`);
}

// Load Selected Quick Sample
function loadSelectedSample() {
    const select = document.getElementById("sample-select");
    const sampleVal = select.value;
    if (!sampleVal) return;

    if (state.direction !== "kn->en") {
        swapLanguageDirection();
    }

    const sourceInput = document.getElementById("source-input");
    sourceInput.value = sampleVal;
    document.getElementById("source-char-count").innerText = `${sampleVal.length} characters`;
    executeTranslation();
}

// Clear Source Input
function clearSource() {
    const sourceInput = document.getElementById("source-input");
    sourceInput.value = "";
    document.getElementById("source-char-count").innerText = "0 characters";
    clearTarget();
}

// Clear Target Output
function clearTarget() {
    const targetOutput = document.getElementById("target-output");
    targetOutput.className = "trans-output placeholder-state";
    targetOutput.innerHTML = `<span class="placeholder-text">Translation will appear here with neural confidence score and token alignment...</span>`;
    document.getElementById("confidence-badge").style.display = "none";
    document.getElementById("method-badge").style.display = "none";
    document.getElementById("domain-insights-panel").style.display = "none";
    document.getElementById("target-word-count").innerText = "0 words";
}

// Execute Translation (Calls API Gateway)
async function executeTranslation() {
    const sourceInput = document.getElementById("source-input");
    const text = sourceInput.value.trim();

    if (!text) {
        showToast("Please enter a sentence to translate.", "warning");
        sourceInput.focus();
        return;
    }

    const btn = document.getElementById("btn-translate");
    const spinner = document.getElementById("translate-spinner");
    const btnText = document.getElementById("translate-btn-text");

    // UI Loading state
    btn.disabled = true;
    spinner.style.display = "inline-block";
    btnText.innerText = "Processing...";

    try {
        const response = await fetch("/api/v1/translation/translate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: text,
                direction: state.direction,
                domain: state.currentDomain
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        state.lastTranslation = data;

        // Render target output
        renderTranslationResult(data);
        renderAttentionMatrix(data);
        renderDomainInsights(data);

        // Pre-fill evaluator hypothesis
        const evalHyp = document.getElementById("eval-hypothesis");
        if (evalHyp) evalHyp.value = data.translated_text;

        showToast("Translation completed successfully!", "success");
    } catch (err) {
        console.error("Translation error:", err);
        showToast(`Translation failed: ${err.message}`, "error");
    } finally {
        btn.disabled = false;
        spinner.style.display = "none";
        btnText.innerText = "Translate";
    }
}

// Render Translation Result
function renderTranslationResult(data) {
    const targetOutput = document.getElementById("target-output");
    targetOutput.className = "trans-output";
    if (state.direction === "en->kn") {
        targetOutput.classList.add("kannada-output");
    }
    targetOutput.innerText = data.translated_text;

    // Badges
    const confBadge = document.getElementById("confidence-badge");
    confBadge.innerText = `${data.confidence}% Confidence`;
    confBadge.style.display = "inline-block";

    const methodBadge = document.getElementById("method-badge");
    methodBadge.innerText = data.method;
    methodBadge.style.display = "inline-block";

    // Word count
    const words = data.translated_text.trim().split(/\s+/).filter(Boolean).length;
    document.getElementById("target-word-count").innerText = `${words} words`;
}

// Render Domain Entities Insights Panel
function renderDomainInsights(data) {
    const panel = document.getElementById("domain-insights-panel");
    const container = document.getElementById("entity-chips-container");
    container.innerHTML = "";

    if (!data.domain_terms || data.domain_terms.length === 0) {
        panel.style.display = "none";
        return;
    }

    panel.style.display = "block";
    data.domain_terms.forEach(term => {
        const chip = document.createElement("div");
        chip.className = "entity-chip";
        chip.innerHTML = `
            <span class="entity-source">${term.source_term}</span>
            <span class="entity-arrow">➔</span>
            <span class="entity-target">${term.target_meaning}</span>
            <span class="entity-domain-tag">${term.domain}</span>
        `;
        container.appendChild(chip);
    });
}

// Render Attention Alignment Matrix Heatmap
function renderAttentionMatrix(data) {
    const wrapper = document.getElementById("attention-matrix-wrapper");
    if (!data.attention_matrix || data.attention_matrix.length === 0) {
        wrapper.innerHTML = `<div class="empty-matrix-notice"><p>No attention data returned for this query.</p></div>`;
        return;
    }

    const srcTokens = data.source_tokens || [];
    const trgTokens = data.target_tokens || [];
    const matrix = data.attention_matrix; // [trg_len, src_len]

    let tableHtml = `<table class="matrix-table"><thead><tr><th>Target \\ Source</th>`;
    srcTokens.forEach(tok => {
        const isKn = /[\u0C80-\u0CFF]/.test(tok);
        tableHtml += `<th class="${isKn ? 'kannada-col' : ''}">${escapeHtml(tok)}</th>`;
    });
    tableHtml += `</tr></thead><tbody>`;

    for (let i = 0; i < trgTokens.length; i++) {
        const trgTok = trgTokens[i];
        tableHtml += `<tr><th>${escapeHtml(trgTok)}</th>`;
        const row = matrix[i] || [];
        for (let j = 0; j < srcTokens.length; j++) {
            const weight = row[j] !== undefined ? row[j] : 0.0;
            const alpha = Math.min(1.0, Math.max(0.04, weight));
            const bgRgba = `rgba(99, 102, 241, ${alpha})`;
            const textColor = alpha > 0.45 ? "#ffffff" : "#94a3b8";

            tableHtml += `
                <td class="matrix-cell" 
                    style="background-color: ${bgRgba}; color: ${textColor};"
                    title="Target: '${escapeHtml(trgTok)}' ⇄ Source: '${escapeHtml(srcTokens[j])}' (Weight: ${weight})">
                    ${weight.toFixed(2)}
                </td>
            `;
        }
        tableHtml += `</tr>`;
    }

    tableHtml += `</tbody></table>`;
    wrapper.innerHTML = tableHtml;
}

// Copy Target Text to Clipboard
function copyTargetText() {
    const text = document.getElementById("target-output").innerText.trim();
    if (!text || text.includes("Translation will appear here")) {
        showToast("Nothing to copy yet.", "warning");
        return;
    }
    navigator.clipboard.writeText(text).then(() => {
        showToast("Translation copied to clipboard!");
    }).catch(() => {
        showToast("Failed to copy to clipboard.", "error");
    });
}

// Speech Synthesis Pronunciation
function speakTargetText() {
    const text = document.getElementById("target-output").innerText.trim();
    if (!text || text.includes("Translation will appear here")) {
        showToast("No translated text to read out.", "warning");
        return;
    }

    if (!('speechSynthesis' in window)) {
        showToast("Text-to-speech is not supported in this browser.", "warning");
        return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = state.direction === "kn->en" ? "en-US" : "kn-IN";
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
    showToast("Speaking audio output...");
}

// BLEU Score Calculator
async function calculateBleuScore() {
    const hyp = document.getElementById("eval-hypothesis").value.trim();
    const ref = document.getElementById("eval-reference").value.trim();

    if (!hyp || !ref) {
        showToast("Both Candidate and Reference sentences are required.", "warning");
        return;
    }

    try {
        const res = await fetch("/api/v1/evaluation/bleu", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ hypothesis: hyp, reference: ref })
        });
        const data = await res.json();

        document.getElementById("metric-cumulative-bleu").innerText = `${data.bleu_cumulative}`;
        document.getElementById("metric-bleu-1").innerText = `${data.bleu_1}%`;
        document.getElementById("metric-bleu-2").innerText = `${data.bleu_2}%`;
        document.getElementById("metric-bleu-3").innerText = `${data.bleu_3}%`;
        document.getElementById("metric-brevity-penalty").innerText = `${data.brevity_penalty}`;

        // Render n-gram precision bars
        const barsContainer = document.getElementById("ngram-bars-container");
        barsContainer.innerHTML = "";

        if (data.ngram_precisions) {
            Object.entries(data.ngram_precisions).forEach(([ngram, val]) => {
                const barItem = document.createElement("div");
                barItem.className = "ngram-bar-item";
                barItem.innerHTML = `
                    <div class="ngram-bar-header">
                        <span>${ngram.toUpperCase()} Precision (${val.matched}/${val.total} matches)</span>
                        <span>${val.precision}%</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: ${val.precision}%"></div>
                    </div>
                `;
                barsContainer.appendChild(barItem);
            });
        }
        showToast("BLEU evaluation metrics computed!");
    } catch (err) {
        showToast(`Evaluation error: ${err.message}`, "error");
    }
}

// Run Validation Corpus Benchmark
async function runValidationBenchmark() {
    const btn = document.getElementById("btn-run-benchmark");
    const spinner = document.getElementById("benchmark-spinner");
    btn.disabled = true;
    spinner.style.display = "inline-block";

    try {
        const res = await fetch("/api/v1/evaluation/benchmark");
        const data = await res.json();
        showToast(`Benchmark complete! Average Validation BLEU: ${data.avg_bleu} across ${data.total_evaluated} samples.`, "success");
    } catch (err) {
        showToast(`Benchmark failed: ${err.message}`, "error");
    } finally {
        btn.disabled = false;
        spinner.style.display = "none";
    }
}

// Fetch Parallel Dataset
async function fetchDataset() {
    try {
        const res = await fetch("/api/v1/dataset/samples");
        const samples = await res.json();
        state.datasetSamples = samples;

        document.getElementById("stat-total-pairs").innerText = samples.length;
        renderDatasetTable(samples);
    } catch (err) {
        console.error("Dataset fetch error:", err);
    }
}

// Render Dataset Table
function renderDatasetTable(samples) {
    const tbody = document.getElementById("dataset-table-body");
    tbody.innerHTML = "";

    samples.forEach(sample => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${sample.id}</td>
            <td class="kn-col-text">${escapeHtml(sample.kannada)}</td>
            <td>${escapeHtml(sample.english)}</td>
            <td><span class="topic-pill">${escapeHtml(sample.topic)}</span></td>
            <td>
                <button class="use-sample-btn" onclick="useSampleInTranslator('${escapeHtml(sample.kannada)}', '${escapeHtml(sample.english)}')">
                    Test
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// Use Dataset Sample in Translator & Evaluator
function useSampleInTranslator(knText, enRef) {
    switchTab("translate");
    const sourceInput = document.getElementById("source-input");
    sourceInput.value = knText;
    document.getElementById("source-char-count").innerText = `${knText.length} characters`;

    // Also prefill evaluation reference
    const evalRef = document.getElementById("eval-reference");
    if (evalRef) evalRef.value = enRef;

    executeTranslation();
}

// Filter Dataset Search & Topics
function filterDataset() {
    const query = document.getElementById("dataset-search").value.toLowerCase();
    const filtered = state.datasetSamples.filter(item => {
        return item.kannada.toLowerCase().includes(query) ||
               item.english.toLowerCase().includes(query) ||
               item.topic.toLowerCase().includes(query);
    });
    renderDatasetTable(filtered);
}

function setTopicFilter(topic) {
    document.querySelectorAll(".topic-filter-btn").forEach(b => b.classList.remove("active"));
    event.target.classList.add("active");

    if (topic === "all") {
        renderDatasetTable(state.datasetSamples);
    } else {
        const filtered = state.datasetSamples.filter(item => item.topic.toLowerCase() === topic.toLowerCase());
        renderDatasetTable(filtered);
    }
}

// Fetch Microservices Health
async function fetchSystemHealth() {
    try {
        const res = await fetch("/api/v1/health");
        const data = await res.json();

        // Header indicator
        const healthBadge = document.getElementById("system-health-badge");
        const healthText = document.getElementById("system-health-text");
        healthText.innerText = "All 4 Services Active";

        // Gateway uptime
        const uptimeEl = document.getElementById("gateway-uptime");
        if (uptimeEl) uptimeEl.innerText = `${data.uptime_seconds}s`;

        // Microservices Grid
        const grid = document.getElementById("microservices-status-grid");
        if (!grid) return;
        grid.innerHTML = "";

        const services = [
            {
                name: "1. Preprocessing Microservice",
                status: data.microservices.preprocessing_service.status,
                details: data.microservices.preprocessing_service.features
            },
            {
                name: "2. Translation Microservice (NMT)",
                status: data.microservices.translation_service.status,
                details: [
                    `Model: ${data.microservices.translation_service.architecture}`,
                    `Runtime Device: ${data.microservices.translation_service.device}`,
                    `Seq2Seq Trained: ${data.microservices.translation_service.parameters_trained}`
                ]
            },
            {
                name: "3. Dataset Microservice",
                status: data.microservices.dataset_service.status,
                details: [
                    `Samples: ${data.microservices.dataset_service.samples_loaded} Domain Sentences`,
                    `Kannada Vocab: ${data.microservices.dataset_service.kn_vocab_size} tokens`,
                    `English Vocab: ${data.microservices.dataset_service.en_vocab_size} tokens`
                ]
            },
            {
                name: "4. Evaluation Microservice",
                status: data.microservices.evaluation_service.status,
                details: [
                    "Automated BLEU-1 to BLEU-4 Computation",
                    "Brevity Penalty (BP) & Smoothing",
                    "N-Gram Precision Analysis"
                ]
            }
        ];

        services.forEach(svc => {
            const card = document.createElement("div");
            card.className = "service-status-card";
            card.innerHTML = `
                <div class="service-card-header">
                    <span class="service-name">${svc.name}</span>
                    <span class="service-badge badge-healthy">${svc.status}</span>
                </div>
                <ul class="service-features-list">
                    ${svc.details.map(d => `<li>${d}</li>`).join("")}
                </ul>
            `;
            grid.appendChild(card);
        });
    } catch (err) {
        console.error("Health check error:", err);
    }
}

// Toast Notifications System
function showToast(msg, type = "info") {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    
    let icon = "✓";
    if (type === "warning") icon = "⚠️";
    if (type === "error") icon = "✕";

    toast.innerHTML = `<span>${icon}</span><span>${escapeHtml(msg)}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(100%)";
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// Security Helper
function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

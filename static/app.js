/* Transcriber frontend logic */

const socket = io();

// --- DOM refs ---
const dropzone     = document.getElementById("dropzone");
const fileInput    = document.getElementById("fileInput");
const fileInfo     = document.getElementById("fileInfo");
const fileNameEl   = document.getElementById("fileName");
const removeFile   = document.getElementById("removeFile");
const modelSelect  = document.getElementById("modelSelect");
const qualitySelect= document.getElementById("qualitySelect");
const langSelect   = document.getElementById("langSelect");
const tsToggle     = document.getElementById("tsToggle");
const btnStart     = document.getElementById("btnStart");
const btnStop      = document.getElementById("btnStop");
const btnCopy      = document.getElementById("btnCopy");
const btnSrt       = document.getElementById("btnSrt");
const btnTxt       = document.getElementById("btnTxt");
const progressWrap = document.getElementById("progressWrap");
const progressFill = document.getElementById("progressFill");
const progressLabel= document.getElementById("progressLabel");
const statsLine    = document.getElementById("statsLine");
const statusLine   = document.getElementById("statusLine");
const noticeLine   = document.getElementById("noticeLine");
const modelsList   = document.getElementById("modelsList");
const btnOpenModels    = document.getElementById("btnOpenModels");
const btnRefreshModels = document.getElementById("btnRefreshModels");
const btnOpenResults   = document.getElementById("btnOpenResults");
const output       = document.getElementById("output");
const metaInfo     = document.getElementById("metaInfo");
const deviceInfo   = document.getElementById("deviceInfo");

let uploadedFileId = null;
let uploadedName = "";
let uploading = false;
let running = false;
let segments = [];

const DEVICE_LABEL = { cuda: "GPU (CUDA, float16)", cpu: "процессор (int8)" };

// --- Settings (remembered between launches) ---

const SETTINGS_KEY = "transcriber.settings";
let wantedModel = null;
let firstModelsLoad = true;

function loadSettings() {
    try {
        const s = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
        if (s.model) wantedModel = s.model;
        if (s.quality) qualitySelect.value = s.quality;
        if (s.language) langSelect.value = s.language;
        if (typeof s.timestamps === "boolean") tsToggle.checked = s.timestamps;
        if (wantedModel) selectModel(wantedModel);
    } catch (_) {}
}

function saveSettings() {
    try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify({
            model: modelSelect.value,
            quality: qualitySelect.value,
            language: langSelect.value,
            timestamps: tsToggle.checked,
        }));
    } catch (_) {}
}

function selectModel(name) {
    if ([...modelSelect.options].some(o => o.value === name)) modelSelect.value = name;
}

[modelSelect, qualitySelect, langSelect].forEach(el => el.addEventListener("change", saveSettings));
tsToggle.addEventListener("change", () => { saveSettings(); renderOutput(); });

// --- Buttons state ---

function fullText() {
    return segments.map(s => s.text).join(" ").trim();
}

function updateButtons() {
    const hasText = segments.length > 0;
    btnStart.disabled = running || uploading || !uploadedFileId;
    btnStop.disabled = !running;
    btnCopy.disabled = !hasText;
    btnSrt.disabled = !hasText;
    btnTxt.disabled = !hasText;
    removeFile.disabled = running;
    [modelSelect, qualitySelect, langSelect].forEach(el => el.disabled = running);
}

// --- File upload ---

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { if (fileInput.files[0]) uploadFile(fileInput.files[0]); });

// Drag & drop
dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const f = e.dataTransfer.files[0];
    if (f) uploadFile(f);
});

function uploadFile(file) {
    const form = new FormData();
    form.append("audio", file);

    uploading = true;
    updateButtons();
    setStatus("Загрузка файла…", "");

    // XHR instead of fetch: shows progress for big files
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    xhr.upload.onprogress = e => {
        if (e.lengthComputable) setStatus(`Загрузка файла… ${Math.round(e.loaded / e.total * 100)}%`, "");
    };
    xhr.onload = () => {
        uploading = false;
        let data = {};
        try { data = JSON.parse(xhr.responseText); } catch (_) {}
        if (xhr.status !== 200 || data.error) {
            setStatus("Ошибка: " + (data.error || `HTTP ${xhr.status}`), "error");
            fileInput.value = "";
        } else {
            showFile(data.id, data.name);
            setStatus("Файл загружен. Можно начинать.", "success");
        }
        updateButtons();
    };
    xhr.onerror = () => {
        uploading = false;
        setStatus("Ошибка: сервер не отвечает", "error");
        updateButtons();
    };
    xhr.send(form);
}

function showFile(id, name) {
    uploadedFileId = id;
    uploadedName = name || "";
    fileNameEl.textContent = uploadedName;
    dropzone.classList.add("hidden");
    fileInfo.classList.remove("hidden");
}

removeFile.addEventListener("click", () => {
    uploadedFileId = null;
    uploadedName = "";
    fileInput.value = "";
    fileInfo.classList.add("hidden");
    dropzone.classList.remove("hidden");
    setStatus("", "");
    updateButtons();
});

// --- Start / Stop ---

btnStart.addEventListener("click", () => {
    if (!uploadedFileId || running) return;
    running = true;
    segments = [];
    renderOutput();
    metaInfo.textContent = "";
    statsLine.classList.add("hidden");
    hideNotice();
    setProgress(0);
    setStatus("Запуск…", "");
    updateButtons();

    socket.emit("start", {
        file_id: uploadedFileId,
        model: modelSelect.value,
        quality: qualitySelect.value,
        language: langSelect.value,
        timestamps: tsToggle.checked,
    });
});

btnStop.addEventListener("click", () => {
    btnStop.disabled = true;
    socket.emit("stop");
});

// --- SocketIO events ---

socket.on("sync", s => {
    // Fresh page or reconnect: restore whatever the server is doing / did last.
    if (s.file_ok && !uploadedFileId) showFile(s.file_id, s.file_name);
    if (s.running) {
        running = true;
        segments = s.segments || [];
        renderOutput();
        setProgress(s.progress || 0);
        setStatus(s.message, "");
        if (s.device) showDevice(s.device);
    } else if (s.result && !running) {
        finishRun(s.result, false);
    } else if (s.phase === "error") {
        setStatus(s.message, "error");
    }
    updateButtons();
});

let lostConnection = false;

socket.on("disconnect", () => {
    lostConnection = true;
    setStatus("Нет связи с сервером. Если это надолго, запустите Transcriber с рабочего стола снова.", "error");
});

socket.on("connect", () => {
    if (lostConnection) setStatus("", "");
    lostConnection = false;
});

socket.on("status", d => {
    if (d.device) showDevice(d.device);
    if (!running && d.phase !== "stopping") return;
    setStatus(d.message, "");
    if (d.phase === "loading" || d.phase === "decoding" || d.phase === "downloading") setProgress(0);
});

socket.on("loading_progress", d => {
    const mb = d.downloaded / 1048576, tot = d.total / 1048576;
    setStatus(`Скачивание модели ${d.model}: ${d.percent}% (${mb.toFixed(0)} / ${tot.toFixed(0)} МБ)`, "");
    setProgress(d.percent);
});

socket.on("notice", d => {
    noticeLine.textContent = "⚠ " + d.message;
    noticeLine.classList.remove("hidden");
});

socket.on("partial", d => {
    if (!running) return;
    const seg = { text: d.text, start: d.start, end: d.end };
    segments.push(seg);
    appendSegment(seg);
    setProgress(d.progress);

    if (d.speed > 0) {
        statsLine.classList.remove("hidden");
        statsLine.textContent = `⚡ ${d.speed.toFixed(1)}× реального времени · осталось ~${fmtTime(d.remaining)}`;
    }
    updateButtons();
});

socket.on("rollback", d => {
    // Server caught Whisper repeating itself: the last lines are being re-recognized.
    if (!running) return;
    segments.length = Math.min(segments.length, d.count);
    const nearBottom = output.scrollHeight - output.scrollTop - output.clientHeight < 60;
    renderOutput();
    if (nearBottom) output.scrollTop = output.scrollHeight;
    updateButtons();
});

socket.on("complete", r => finishRun(r, true));
socket.on("stopped", r => finishRun(r, true));

socket.on("error", d => {
    running = false;
    statsLine.classList.add("hidden");
    if (d.code === "no_file") removeFile.click();
    setStatus(d.message, "error");
    updateButtons();
});

function finishRun(r, live) {
    // live = the run just ended; false = restoring the last result when the window (re)opens
    running = false;
    segments = r.segments || [];
    renderOutput();
    statsLine.classList.add("hidden");
    if (r.device) showDevice(r.device);

    const parts = [];
    if (r.duration) parts.push(`⏱ ${fmtTime(r.duration)}`);
    if (r.language) parts.push(`язык: ${r.language} (${(r.language_probability * 100).toFixed(0)}%)`);
    const processed = r.duration;
    if (r.elapsed > 0 && processed) parts.push(`за ${fmtTime(r.elapsed)} (${(processed / r.elapsed).toFixed(1)}×)`);
    metaInfo.textContent = parts.join(" · ");

    const saved = r.saved ? ` Сохранено: results\\${r.saved}` : "";
    if (r.stopped) {
        const done = segments.length ? segments[segments.length - 1].end : 0;
        const of = r.duration ? ` из ${fmtTime(r.duration)}` : "";
        setStatus(`Остановлено. Распознано ${fmtTime(done)}${of}.${saved}`, "");
    } else {
        setProgress(100);
        setStatus(segments.length ? `Готово!${saved}` : "Готово, но речь в файле не найдена.", "success");
    }
    updateButtons();
}

// --- Output ---

function segLine(s) {
    const line = document.createElement("div");
    const ts = document.createElement("span");
    ts.className = "ts";
    ts.textContent = `[${fmtTime(s.start)} → ${fmtTime(s.end)}]`;
    line.appendChild(ts);
    line.appendChild(document.createTextNode(s.text));
    return line;
}

function renderOutput() {
    output.innerHTML = "";
    if (!segments.length) {
        output.innerHTML = '<span class="placeholder">Транскрипция появится здесь…</span>';
        return;
    }
    if (tsToggle.checked) {
        const frag = document.createDocumentFragment();
        segments.forEach(s => frag.appendChild(segLine(s)));
        output.appendChild(frag);
    } else {
        output.textContent = segments.map(s => s.text).join(" ").trim();
    }
}

function appendSegment(s) {
    // Follow the text only if the user has not scrolled up to read
    const nearBottom = output.scrollHeight - output.scrollTop - output.clientHeight < 60;
    if (segments.length === 1) output.innerHTML = "";
    if (tsToggle.checked) output.appendChild(segLine(s));
    else output.appendChild(document.createTextNode((segments.length > 1 ? " " : "") + s.text));
    if (nearBottom) output.scrollTop = output.scrollHeight;
}

// --- Copy / Download ---

btnCopy.addEventListener("click", () => {
    const text = fullText();
    const done = () => {
        btnCopy.textContent = "✓ Скопировано";
        setTimeout(() => btnCopy.textContent = "📋 Копировать", 1500);
    };
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done, () => copyFallback(text) && done());
    } else if (copyFallback(text)) {
        done();
    }
});

function copyFallback(text) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch (_) {}
    ta.remove();
    return ok;
}

function baseName() {
    const n = uploadedName.replace(/\.[^.]+$/, "").trim();
    return n || "transcript";
}

btnSrt.addEventListener("click", () => {
    const srt = segments.map((s, i) =>
        `${i + 1}\n${fmtSrt(s.start)} --> ${fmtSrt(s.end)}\n${s.text}\n`
    ).join("\n");
    downloadBlob(srt, baseName() + ".srt", "text/plain;charset=utf-8");
});

btnTxt.addEventListener("click", () => {
    let content = fullText();
    if (tsToggle.checked) {
        content = segments.map(s => `[${fmtTime(s.start)} → ${fmtTime(s.end)}] ${s.text}`).join("\n");
    }
    downloadBlob(content + "\n", baseName() + ".txt", "text/plain;charset=utf-8");
});

btnOpenResults.addEventListener("click", () => {
    fetch("/api/open_results_folder", { method: "POST" }).catch(() => {});
});

// --- Models panel ---

function loadModels() {
    fetch("/api/models")
        .then(r => r.json())
        .then(data => {
            const items = [];
            for (const [name, info] of Object.entries(data.builtin)) {
                const badge = info.downloaded
                    ? '<span class="model-badge ok">✓ в папке</span>'
                    : '<span class="model-badge">⬇ скачается при запуске</span>';
                items.push(`<div class="model-item"><span class="model-name">${esc(name)}</span><span class="model-desc">${esc(info.label)} · ~${info.size_mb} МБ</span>${badge}</div>`);
            }
            for (const [name, info] of Object.entries(data.custom)) {
                items.push(`<div class="model-item custom"><span class="model-name">${esc(name)}</span><span class="model-desc">своя модель · ~${info.size_mb} МБ</span></div>`);
            }
            modelsList.innerHTML = items.join("") || '<div class="model-item empty">Своих моделей пока нет</div>';

            // Add custom options to the model select
            const current = modelSelect.value;
            const optgroup = document.getElementById("customModelsGroup");
            if (optgroup) optgroup.remove();
            const names = Object.keys(data.custom);
            if (names.length) {
                const g = document.createElement("optgroup");
                g.id = "customModelsGroup";
                g.label = "Свои модели";
                names.forEach(n => {
                    const o = document.createElement("option");
                    o.value = n;
                    o.textContent = `${n} — своя модель`;
                    g.appendChild(o);
                });
                modelSelect.appendChild(g);
            }
            if (wantedModel) {
                selectModel(wantedModel);
            } else if (firstModelsLoad) {
                // No saved choice yet: start with the best model that is already downloaded
                // (the installer puts large-v3 on GPU computers and small on CPU-only ones).
                const best = ["large-v3", "medium", "small", "base", "tiny"].find(n => data.builtin[n] && data.builtin[n].downloaded)
                    || Object.keys(data.custom)[0];
                selectModel(best || current);
            } else {
                selectModel(current);
            }
            wantedModel = null;
            firstModelsLoad = false;
        })
        .catch(() => {});
}

btnOpenModels.addEventListener("click", () => {
    fetch("/api/open_models_folder", { method: "POST" }).then(r => r.json()).then(d => {
        if (d.ok) setStatus(`Папка моделей открыта: ${d.path}`, "");
    }).catch(() => {});
});

btnRefreshModels.addEventListener("click", loadModels);

// --- Device info ---

function showDevice(device) {
    deviceInfo.textContent = `Распознавание на: ${DEVICE_LABEL[device] || device} · VAD (пропуск тишины)`;
}

fetch("/api/info").then(r => r.json()).then(d => {
    if (d.device) showDevice(d.device);
    else deviceInfo.textContent = d.cuda
        ? "GPU найден: распознавание на CUDA (float16) · VAD (пропуск тишины)"
        : "GPU не найден: распознавание будет на процессоре (медленнее)";
}).catch(() => { deviceInfo.textContent = ""; });

loadSettings();
loadModels();
updateButtons();

// --- Helpers ---

function setProgress(pct) {
    progressWrap.classList.remove("hidden");
    progressFill.style.width = pct + "%";
    progressLabel.textContent = Math.round(pct) + "%";
}

function hideNotice() {
    noticeLine.textContent = "";
    noticeLine.classList.add("hidden");
}

function fmtTime(sec) {
    sec = Math.max(0, Math.floor(sec || 0));
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
    return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function fmtSrt(sec) {
    const total = Math.round(sec * 1000);
    const h = Math.floor(total / 3600000);
    const m = Math.floor((total % 3600000) / 60000);
    const s = Math.floor((total % 60000) / 1000);
    const ms = total % 1000;
    return `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")},${String(ms).padStart(3,"0")}`;
}

function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function downloadBlob(content, filename, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function setStatus(msg, cls) {
    statusLine.textContent = msg;
    statusLine.className = "status-line" + (cls ? " " + cls : "");
}

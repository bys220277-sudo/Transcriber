"""
Audio Transcriber — локальная транскрипция аудио через faster-whisper.
Работает на NVIDIA GPU (CUDA float16), при проблеме с GPU сам переходит на процессор.
Запуск: ярлык Transcriber на рабочем столе, start.bat или `python app.py` (`--no-window` — без окна).
"""

import os
import re
import sys
import time
import uuid
import shutil
import threading
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

# Console is cp1251/cp866: an unexpected character in print() must never crash the app.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

# --- Config ---
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODELS_FOLDER = BASE_DIR / "models"
RESULTS_FOLDER = BASE_DIR / "results"
for _d in (UPLOAD_FOLDER, MODELS_FOLDER, RESULTS_FOLDER):
    _d.mkdir(exist_ok=True)

PORT = 5678
HOST = "127.0.0.1"
URL = f"http://{HOST}:{PORT}"
IDLE_EXIT_SEC = 30  # server exits this long after the last app window is closed (if nothing is running)

ALLOWED_EXT = {
    ".mp3", ".wav", ".m4a", ".ogg", ".oga", ".flac", ".wma", ".aac", ".opus", ".webm",
    ".mp4", ".mkv", ".mov", ".avi", ".m4b", ".aif", ".aiff", ".amr", ".3gp",
}

MODEL_INFO = {
    "tiny":     {"label": "Tiny (быстрая, низкая точность)", "size_mb": 75},
    "base":     {"label": "Base (быстрая)", "size_mb": 142},
    "small":    {"label": "Small (хороший баланс)", "size_mb": 461},
    "medium":   {"label": "Medium (высокая точность)", "size_mb": 1536},
    "large-v3": {"label": "Large-v3 (максимальная точность)", "size_mb": 3086},
}

# "Точность" in the UI: decoding effort on top of the chosen model.
QUALITY = {
    "fast":   {"beam_size": 1, "best_of": 1},
    "normal": {"beam_size": 5, "best_of": 5},
    "max":    {"beam_size": 10, "best_of": 5, "patience": 1.5},
}


def _add_cuda_dll_dirs():
    """Make NVIDIA DLLs from pip packages (nvidia-cublas-cu12, nvidia-cudnn-cu12) visible, if installed."""
    if os.name != "nt":
        return
    for sp in sys.path:
        nv = Path(sp) / "nvidia"
        if not nv.is_dir():
            continue
        for b in nv.glob("*/bin"):
            try:
                os.add_dll_directory(str(b))
                os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
            except OSError:
                pass
        return


_add_cuda_dll_dirs()

# --- App ---
app = Flask(__name__)
app.config["SECRET_KEY"] = "transcriber-2024"
# Only our own page may talk to the server (not any website open in the browser).
socketio = SocketIO(app, cors_allowed_origins=[URL, f"http://localhost:{PORT}"], async_mode="threading")

# --- Model cache: one model at a time, so switching models does not pile up VRAM ---
_model_lock = threading.Lock()
_model = {"name": None, "obj": None, "device": None}

# --- Job / UI state (single user, single job) ---
_job_lock = threading.Lock()
_job = None            # dict with thread, stop Event and settings of the running/last job
_uploads = {}          # file_id -> original file name
_state = {
    "running": False,
    "phase": "idle",   # idle | starting | downloading | loading | decoding | transcribing | stopping | done | stopped | error
    "message": "",
    "model": None,
    "device": None,
    "file_id": None,
    "file_name": None,
    "timestamps": True,
    "segments": [],
    "progress": 0.0,
    "result": None,
}

_clients_lock = threading.Lock()
_clients = 0
_ever_connected = False
_cuda_available = None


def list_local_models():
    """All folders in models/ that look like a faster-whisper model (contain model.bin)."""
    return [d.name for d in sorted(MODELS_FOLDER.iterdir()) if d.is_dir() and (d / "model.bin").exists()]


def _model_ready(name: str) -> bool:
    """True if models/<name>/model.bin exists (all models live in models/)."""
    return (MODELS_FOLDER / name / "model.bin").exists()


def _download_builtin(name: str):
    """Download a built-in model into models/<name> (reuses HF cache if present)."""
    from huggingface_hub import snapshot_download
    dest = MODELS_FOLDER / name
    print(f"[model] Downloading '{name}' -> {dest} ...")
    snapshot_download(repo_id=f"Systran/faster-whisper-{name}", local_dir=str(dest))
    print(f"[model] '{name}' is now in {dest}")


def get_model(name: str, force_cpu: bool = False):
    """Return (model, device). Loads from models/<name>; CUDA float16 first, CPU int8 as a fallback."""
    with _model_lock:
        if _model["name"] == name and _model["obj"] is not None and not (force_cpu and _model["device"] != "cpu"):
            return _model["obj"], _model["device"]
        _model.update(name=None, obj=None, device=None)  # free the previous model first

        if not _model_ready(name):
            if name in MODEL_INFO:
                _download_builtin(name)
            else:
                raise FileNotFoundError(f"Модель '{name}' не найдена в папке models (нужен файл model.bin)")

        from faster_whisper import WhisperModel
        source = str(MODELS_FOLDER / name)
        obj, device = None, "cpu"
        if not force_cpu:
            try:
                print(f"[model] Loading '{name}' on CUDA (float16)...")
                obj, device = WhisperModel(source, device="cuda", compute_type="float16"), "cuda"
            except Exception as e:
                print(f"[model] CUDA failed: {e!r} -> falling back to CPU")
                socketio.emit("notice", {"message": f"GPU недоступен ({_short(e)}). Работаю на процессоре, это медленнее."})
        if obj is None:
            print(f"[model] Loading '{name}' on CPU (int8)...")
            obj = WhisperModel(source, device="cpu", compute_type="int8")
        _model.update(name=name, obj=obj, device=device)
        print(f"[model] '{name}' ready on {device}.")
        return obj, device


def _short(e: Exception, limit: int = 140) -> str:
    s = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return s if len(s) <= limit else s[:limit] + "…"


def _hf_download_bytes(model_name: str):
    """Best-effort: how many bytes of the model are downloaded so far."""
    cache = Path.home() / ".cache" / "huggingface"
    # 1) xet protocol log for this process (new huggingface download engine)
    xet_dir = cache / "xet" / "logs"
    if xet_dir.is_dir():
        logs = sorted(xet_dir.glob(f"xet_*_{os.getpid()}.log"))
        if logs:
            try:
                with open(logs[-1], "r", errors="ignore") as f:
                    data = f.read()
                m = None
                for m in re.finditer(r"observed bytes sent so far = (\d+)", data):
                    pass
                if m:
                    return int(m.group(1))
            except OSError:
                pass
    # 2) fallback: .incomplete files (HF cache repo, or the local_dir download folder)
    total, found = 0, False
    for folder in (cache / "hub" / f"models--Systran--faster-whisper-{model_name}" / "blobs",
                   MODELS_FOLDER / model_name / ".cache" / "huggingface" / "download"):
        if folder.is_dir():
            found = True
            for p in folder.rglob("*.incomplete"):
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    return total if found else None


def _watch_download(job: dict):
    """Emit download progress while the model is being fetched (first use of a model)."""
    model_name = job["model"]
    total = MODEL_INFO.get(model_name, {}).get("size_mb", 0) * 1024 * 1024
    last = -1
    while job["thread"].is_alive() and not _model_ready(model_name):
        got = _hf_download_bytes(model_name)
        if got is not None and got != last:
            last = got
            pct = min(99.0, got / total * 100) if total else 0.0
            socketio.emit("loading_progress", {
                "model": model_name,
                "downloaded": got,
                "total": total,
                "percent": round(pct, 1),
            })
        time.sleep(2)


def _upload_path(file_id: str):
    """Path of an uploaded file, or None. Only our own generated names are accepted."""
    if not re.fullmatch(r"[0-9a-f]{32}\.[a-z0-9]{1,5}", file_id or ""):
        return None
    p = UPLOAD_FOLDER / file_id
    return p if p.is_file() else None


def _clean_uploads(keep=()):
    for p in UPLOAD_FOLDER.iterdir():
        if p.is_file() and p.name not in keep:
            try:
                p.unlink()
            except OSError:
                pass


def _fmt_time(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _status(phase: str, message: str, **extra):
    _state.update(phase=phase, message=message, **extra)
    socketio.emit("status", {"phase": phase, "message": message, **extra})


def _snapshot():
    s = dict(_state)
    s["segments"] = list(_state["segments"])
    s["file_ok"] = _upload_path(s.get("file_id") or "") is not None
    return s


# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info")
def api_info():
    global _cuda_available
    if _cuda_available is None:
        try:
            import ctranslate2
            _cuda_available = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            _cuda_available = False
    return jsonify({"cuda": _cuda_available, "device": _model["device"]})


@app.route("/api/models")
def api_models():
    builtin = {}
    for name, info in MODEL_INFO.items():
        d = MODELS_FOLDER / name
        size_mb = info["size_mb"]
        if d.is_dir():
            try:
                size_mb = round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1048576)
            except OSError:
                pass
        builtin[name] = {**info, "downloaded": _model_ready(name), "size_mb": size_mb}
    custom = {}
    for name in list_local_models():
        if name in MODEL_INFO:
            continue
        p = MODELS_FOLDER / name
        size_mb = 0
        try:
            size_mb = round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1048576)
        except OSError:
            pass
        custom[name] = {"label": f"{name} (своя модель)", "size_mb": size_mb}
    return jsonify({"builtin": builtin, "custom": custom})


@app.route("/api/open_models_folder", methods=["POST"])
def api_open_models_folder():
    os.startfile(str(MODELS_FOLDER))  # Windows
    return jsonify({"ok": True, "path": str(MODELS_FOLDER)})


@app.route("/api/open_results_folder", methods=["POST"])
def api_open_results_folder():
    os.startfile(str(RESULTS_FOLDER))  # Windows
    return jsonify({"ok": True, "path": str(RESULTS_FOLDER)})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "audio" not in request.files:
        return jsonify({"error": "Файл не передан"}), 400
    f = request.files["audio"]
    if not f.filename:
        return jsonify({"error": "Пустое имя файла"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Формат {ext or '(без расширения)'} не поддерживается"}), 400

    # One file at a time: drop older uploads (except the one a running job is reading).
    busy = _job["file_id"] if _job is not None and _job["thread"].is_alive() else None
    _clean_uploads(keep={busy} if busy else ())

    fname = f"{uuid.uuid4().hex}{ext}"
    f.save(UPLOAD_FOLDER / fname)
    _uploads[fname] = f.filename
    if not _state["running"]:
        _state.update(file_id=fname, file_name=f.filename)
    return jsonify({"id": fname, "name": f.filename})


# --- SocketIO events ---

@socketio.on("connect")
def on_connect(auth=None):
    global _clients, _ever_connected
    with _clients_lock:
        _clients += 1
        _ever_connected = True
    emit("sync", _snapshot())


@socketio.on("disconnect")
def on_disconnect(*_args):
    global _clients
    with _clients_lock:
        _clients = max(0, _clients - 1)


@socketio.on("start")
def on_start(data=None):
    global _job
    data = data or {}
    with _job_lock:
        if _job is not None and _job["thread"].is_alive():
            emit("error", {"message": "Уже идёт транскрипция"})
            return

        path = _upload_path(str(data.get("file_id", "")))
        if path is None:
            emit("error", {"message": "Файл не найден, загрузите аудио ещё раз", "code": "no_file"})
            return

        model_name = str(data.get("model") or "large-v3")
        if Path(model_name).name != model_name or (model_name not in MODEL_INFO and not _model_ready(model_name)):
            emit("error", {"message": f"Модель '{model_name}' не найдена"})
            return

        language = str(data.get("language") or "ru")
        quality = data.get("quality") if data.get("quality") in QUALITY else "normal"
        job = {
            "file_id": path.name,
            "path": str(path),
            "file_name": _uploads.get(path.name, path.name),
            "model": model_name,
            "language": None if language == "auto" else language,  # None = Whisper detects the language
            "quality": quality,
            "timestamps": bool(data.get("timestamps", True)),
            "stop": threading.Event(),
        }
        job["thread"] = threading.Thread(target=_run_transcription, args=(job,), daemon=True)
        _job = job
        _state.update(running=True, phase="starting", message="Запуск…", model=model_name, device=None,
                      file_id=job["file_id"], file_name=job["file_name"], timestamps=job["timestamps"],
                      segments=[], progress=0.0, result=None)
        job["thread"].start()


@socketio.on("stop")
def on_stop(_data=None):
    job = _job
    if job is None or not job["thread"].is_alive() or job["stop"].is_set():
        return
    job["stop"].set()
    if _state["phase"] == "downloading":
        _status("stopping", "Остановлю, как только модель докачается…")
    else:
        _status("stopping", "Останавливаю…")


# --- Transcription worker ---

def _run_transcription(job: dict):
    try:
        name = job["model"]
        if not _model_ready(name):
            _status("downloading", f"Скачивание модели {name} (только в первый раз)…")
            threading.Thread(target=_watch_download, args=(job,), daemon=True).start()
        elif _model["name"] != name:
            _status("loading", f"Загрузка модели {name}…")
        model, device = get_model(name)
        if job["stop"].is_set():
            return _finish(job, None, stopped=True, device=device)

        _status("decoding", "Чтение аудиофайла…", device=device)
        from faster_whisper.audio import decode_audio
        try:
            audio = decode_audio(job["path"], sampling_rate=16000)
        except Exception as e:
            raise RuntimeError(f"Не удалось прочитать аудио: {_short(e)}") from e
        duration = len(audio) / 16000
        if duration < 0.1:
            raise RuntimeError("В файле нет звуковой дорожки")
        if job["stop"].is_set():
            return _finish(job, None, stopped=True, duration=duration, device=device)
        span = duration
        try:
            info, elapsed = _transcribe(job, model, device, audio, span)
        except Exception as e:
            # A broken CUDA setup often shows up only on the first real GPU work: retry on CPU.
            if device != "cuda" or _state["segments"] or job["stop"].is_set():
                raise
            print(f"[gpu] CUDA error during transcription: {e!r} -> retrying on CPU")
            socketio.emit("notice", {"message": f"Ошибка GPU ({_short(e)}). Продолжаю на процессоре, это медленнее."})
            model, device = get_model(name, force_cpu=True)
            info, elapsed = _transcribe(job, model, device, audio, span)

        _finish(job, info, stopped=job["stop"].is_set(), elapsed=elapsed, duration=duration, device=device)

    except Exception as e:
        print(f"[job] error: {e!r}")
        _state.update(running=False, phase="error", message=str(e))
        socketio.emit("error", {"message": str(e)})


LOOP_REPEATS = 3     # this many copies of the previous segment in a row = Whisper is stuck in a loop
WORD_RUN = 8         # this many one-word segments in a row = stuck in "one word per line" mode
LOOP_ROLLBACK = 10   # seconds re-decoded before a loop (text right before a loop is often already off)
MAX_RESTARTS = 20
SR = 16000


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def _one_word(text: str) -> bool:
    return len(text.split()) <= 1


def _prompt_from(segs, skip_norm: str = "", limit: int = 200) -> str:
    """Last normal sentences as a prompt: keeps punctuation and terms after a restart."""
    parts, size = [], 0
    for s in reversed(segs):
        if _one_word(s["text"]) or (skip_norm and _norm(s["text"]) == skip_norm):
            continue
        parts.append(s["text"])
        size += len(s["text"]) + 1
        if size >= limit:
            break
    return " ".join(reversed(parts))[-limit:]


def _transcribe(job: dict, model, device: str, audio, duration: float):
    """Stream segments to the UI. Returns (info of the first pass, elapsed seconds).

    Whisper conditions each window on the previous text, which keeps terms consistent but sometimes gets it stuck:
    it repeats one phrase ("Вот. Вот. Вот. …") over real speech, or switches to one word per segment for many
    minutes. When that happens, the bad segments and the last LOOP_ROLLBACK seconds before them are dropped and
    the rest of the file is decoded again from there, with the preceding normal sentences as a prompt (so
    punctuation and terms carry over). If it gets stuck again at the same spot, 30 s are decoded without
    conditioning, then normal decoding continues.
    """
    q = QUALITY[job["quality"]]
    language = job["language"]
    t_start = time.time()
    t0 = None
    first_info = None
    offset, limit, cond, prompt = 0.0, None, True, None   # current pass: audio[offset:limit]
    restarts, last_loop = 0, None
    floor = None  # end of the last unconditioned pass: its text is kept, rollbacks never go before it

    def emit_segment(entry):
        _state["segments"].append(entry)
        elapsed = time.time() - t0
        speed = (entry["end"] / elapsed) if elapsed > 0.5 else 0.0
        remaining = ((duration - entry["end"]) / speed) if speed > 0 else 0.0
        progress = min(100.0, entry["end"] / duration * 100) if duration > 0 else 0.0
        _state["progress"] = round(progress, 1)
        socketio.emit("partial", {
            **entry,
            "progress": round(progress, 1),
            "speed": round(speed, 2),
            "remaining": round(max(0.0, remaining), 1),
        })

    while True:
        if first_info is None:
            # transcribe() first runs VAD over the whole file (~10 s per hour of audio), then yields segments lazily.
            _status("transcribing", "Поиск речи в записи…", device=device)
        chunk = audio[int(offset * SR):int(limit * SR) if limit is not None else None]
        segments, info = model.transcribe(chunk, language=language, vad_filter=True,
                                          condition_on_previous_text=cond, initial_prompt=prompt or None, **q)
        if first_info is None:
            first_info = info
            language = language or info.language  # keep the detected language for restarts
            if not job["stop"].is_set():
                _status("transcribing", "Распознавание…" if device == "cuda" else
                        "Распознавание на процессоре (медленнее)…", device=device)
            t0 = time.time()  # speed/ETA are measured from here, without the VAD pass

        pending = []      # copies of the previous segment, held back until something different arrives
        loop_at, loop_norm, loop_kind = None, "", ""
        segs = _state["segments"]
        for seg in segments:
            text = seg.text.strip()
            if text:
                entry = {"text": text, "start": round(seg.start + offset, 2), "end": round(seg.end + offset, 2)}
                prev = pending[-1] if pending else (segs[-1] if segs else None)
                if prev is not None and _norm(text) == _norm(prev["text"]):
                    pending.append(entry)
                    if cond and len(pending) >= LOOP_REPEATS and restarts < MAX_RESTARTS:
                        loop_at, loop_norm, loop_kind = pending[0]["start"], _norm(text), "repeated segment"
                        break
                else:
                    for p in pending:
                        emit_segment(p)
                    pending = []
                    emit_segment(entry)
                    if (cond and restarts < MAX_RESTARTS and len(segs) >= WORD_RUN
                            and all(_one_word(s["text"]) for s in segs[-WORD_RUN:])):
                        loop_at, loop_kind = segs[-WORD_RUN]["start"], "one word per segment"
                        break
            if job["stop"].is_set():
                break

        if loop_at is not None and not job["stop"].is_set():
            restarts += 1
            # Stuck again at the same spot, or right after an unconditioned pass: the trouble is the audio itself,
            # so decode the next 30 s without conditioning instead of retrying with it.
            same_spot = ((last_loop is not None and abs(loop_at - last_loop) < 5)
                         or (floor is not None and loop_at < floor + 30))
            last_loop = loop_at
            # Drop the bad segments and the last LOOP_ROLLBACK seconds before them, tell the UI, decode again.
            keep = len(segs)
            back_to = loop_at - LOOP_ROLLBACK if floor is None else max(loop_at - LOOP_ROLLBACK, floor)
            while keep > 0 and segs[keep - 1]["start"] >= back_to:
                keep -= 1
            del segs[keep:]
            socketio.emit("rollback", {"count": keep})
            offset = segs[-1]["end"] if segs else 0.0
            prompt = _prompt_from(segs, skip_norm=loop_norm)
            limit, cond = (offset + 30 if same_spot else None), not same_spot
            print(f"[loop] {loop_kind} at {_fmt_time(loop_at)} -> re-decoding from {_fmt_time(offset)}"
                  f" (#{restarts}{', without conditioning' if same_spot else ''})")
            continue
        for p in pending:
            emit_segment(p)
        if job["stop"].is_set():
            break
        if limit is not None and limit < duration:
            # The unconditioned pass is done: continue normally after its last segment
            # (starting at a pause is cleaner than cutting at the 30 s mark).
            last_end = segs[-1]["end"] if segs else offset
            offset, limit, cond = (last_end if last_end > offset + 1 else limit), None, True
            floor = offset
            prompt = _prompt_from(segs)
            continue
        break
    return first_info, time.time() - t_start


def _finish(job: dict, info, stopped: bool, elapsed: float = 0.0, duration: float = 0.0, device=None):
    segs = list(_state["segments"])
    text = " ".join(s["text"] for s in segs).strip()
    result = {
        "text": text,
        "segments": segs,
        "duration": round(duration, 2),
        "language": info.language if info else None,
        "language_probability": round(info.language_probability, 3) if info else None,
        "device": device,
        "elapsed": round(elapsed, 1),
        "stopped": stopped,
        "saved": None,
    }
    if text:
        try:
            result["saved"] = _save_result(job, result)
        except OSError as e:
            print(f"[job] could not save result: {e!r}")
    progress = 100.0 if not stopped else _state["progress"]
    _state.update(running=False, phase="stopped" if stopped else "done", progress=progress, result=result,
                  message="Остановлено" if stopped else "Готово!")
    socketio.emit("stopped" if stopped else "complete", result)


def _save_result(job: dict, result: dict) -> str:
    """Save the transcript to results/<audio name>_<date>.txt so a closed window never loses it."""
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(job["file_name"]).stem).strip(" .") or "transcript"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    suffix = "_частично" if result["stopped"] else ""
    path = RESULTS_FOLDER / f"{stem[:80]}_{stamp}{suffix}.txt"
    if job["timestamps"]:
        body = "\n".join(f"[{_fmt_time(s['start'])} → {_fmt_time(s['end'])}] {s['text']}" for s in result["segments"])
    else:
        body = result["text"]
    path.write_text(body + "\n", encoding="utf-8")
    return path.name


# --- App window / lifecycle ---

def _find_edge():
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if base:
            p = Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            if p.exists():
                return str(p)
    return shutil.which("msedge")


def open_app_window():
    """Open the UI as a standalone app window (Edge app mode), or a normal browser tab as a fallback."""
    edge = _find_edge()
    if edge:
        try:
            subprocess.Popen([edge, f"--app={URL}", "--window-size=940,1040"])
            return
        except OSError:
            pass
    webbrowser.open(URL)


def _server_responds(timeout: float = 1.5) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(URL, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _idle_watchdog():
    """Exit when every app window has been closed for IDLE_EXIT_SEC and no transcription is running."""
    idle_ticks = 0
    while True:
        time.sleep(2)
        busy = _job is not None and _job["thread"].is_alive()
        if not _ever_connected or _clients > 0 or busy:
            idle_ticks = 0
            continue
        idle_ticks += 1  # counted in real ticks, so waking the laptop from sleep does not trigger an exit
        if idle_ticks * 2 >= IDLE_EXIT_SEC:
            print("[Transcriber] Window closed - shutting down.")
            os._exit(0)


# --- Main ---

if __name__ == "__main__":
    if "--download" in sys.argv:
        # Used by start.bat during installation: `python app.py --download large-v3`
        i = sys.argv.index("--download")
        name = sys.argv[i + 1] if i + 1 < len(sys.argv) else "large-v3"
        if name not in MODEL_INFO:
            print(f"[model] Unknown model '{name}'. Built-in: {', '.join(MODEL_INFO)}")
            sys.exit(2)
        if _model_ready(name):
            print(f"[model] '{name}' is already in {MODELS_FOLDER / name}")
            sys.exit(0)
        try:
            _download_builtin(name)
        except Exception as e:
            print(f"[model] Download failed: {e!r}. The model will be downloaded on first use.")
            sys.exit(1)
        sys.exit(0)

    no_window = "--no-window" in sys.argv or os.environ.get("TRANSCRIBER_NO_WINDOW") == "1"

    if _server_responds():
        print("[Transcriber] Already running - opening the window.")
        if not no_window:
            open_app_window()
        sys.exit(0)

    _clean_uploads()

    def _open_when_ready():
        # Wait until the server actually responds, then open the window.
        for _ in range(60):  # up to ~30 s
            if _server_responds(timeout=1):
                break
            time.sleep(0.5)
        open_app_window()

    if not no_window:
        threading.Thread(target=_open_when_ready, daemon=True).start()
    threading.Thread(target=_idle_watchdog, daemon=True).start()
    print(f"\n  [Transcriber] Running at {URL}  (closing the app window stops the server)\n")
    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)

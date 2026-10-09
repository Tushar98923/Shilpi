"""First-time setup test: a fresh user folder, in a real Blender window (~5 min, downloads ~110 MB).

Checks what a new user goes through: Set up installs llama.cpp into the add-on's folder (even though one may be
installed on this PC) and downloads the base model and Shilpi's adapter (each checked against its SHA-256);
Install voice pip-installs faster-whisper with Blender's Python; the first command starts the model server with
the adapter; Speak starts the speech server (which downloads Whisper on first start); quitting Blender stops both.
Also: a download whose hash doesn't match is deleted, and a 0.1.0 merged model is used as it is. The model files
are served from models/ by a local web server here, in place of the Hugging Face downloads.

Usage: python scripts/ui_check_setup.py      (no model or speech server running)
"""
import functools
import hashlib
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.headless_validator import _blender_executable  # noqa: E402

IN_BLENDER = r'''
import json, sys, time, bpy
sys.path.insert(0, ADDON_PARENT)
import shilpi as addon
from shilpi import runtime
addon.register()
OUT = open(RESULTS, "a", encoding="utf-8")
addon._DefaultPrefs.base_url, addon._DefaultPrefs.adapter_url = BASE_URL, ADAPTER_URL
runtime.KNOWN_SHA256.update({BASE_URL: BASE_SHA256, ADAPTER_URL: ADAPTER_SHA256})
addon._DefaultPrefs.server_url = "http://127.0.0.1:8180/v1"  # spare ports: never touch servers the user runs
addon._DefaultPrefs.asr_url = "http://127.0.0.1:8181"
# Pretend nothing is installed on this PC: only the add-on's own llama.cpp copy counts.
runtime.find_llama_server = lambda custom="": next(iter(sorted((runtime.data_dir() / "llama").rglob(
    "llama-server" + runtime.EXE))), None)

def record(name, ok, detail):
    OUT.write(json.dumps({"test": name, "ok": bool(ok), "detail": str(detail)[:300]}) + "\n"); OUT.flush()

def ctx():
    w = bpy.context.window_manager.windows[0]
    a = next(a for a in w.screen.areas if a.type == "VIEW_3D")
    return {"window": w, "area": a, "region": next(r for r in a.regions if r.type == "WINDOW")}

def wait(condition, seconds):
    end = time.time() + seconds
    while time.time() < end and not condition():
        yield

def steps():
    yield
    state = bpy.context.window_manager.bai
    record("data folder is isolated", str(runtime.data_dir()).startswith(TMP), runtime.data_dir())
    record("setup needed at first", runtime.find_llama_server() is None and not runtime.model_ready(), "")
    models = runtime.data_dir() / "models"
    models.mkdir(parents=True, exist_ok=True)
    merged = models / runtime.MERGED_FILE
    merged.write_bytes(b"0.1.0 model")
    record("a 0.1.0 merged model is used as it is (no download)",
           runtime.model_ready() and runtime.model_files() == (merged, None) and not runtime.model_downloads(
               addon._DefaultPrefs()), runtime.model_files())
    merged.unlink()
    damaged = models / "damaged.gguf.part"
    damaged.write_bytes(b"not the real file")
    runtime.KNOWN_SHA256["http://example.invalid/damaged.gguf"] = "0" * 64
    try:
        runtime._check_hash("http://example.invalid/damaged.gguf", damaged, runtime.Job(), "Test download")
        record("a download with the wrong hash is rejected", False, "no error")
    except RuntimeError as e:
        record("a download with the wrong hash is rejected", "damaged" in str(e) and not damaged.exists(), e)
    with bpy.context.temp_override(**ctx()):
        bpy.ops.bai.setup()
    yield from wait(lambda: not runtime.JOB.running, 900)
    record("Set up: llama.cpp installed in the add-on folder", runtime.find_llama_server() is not None,
           f"{runtime.JOB.error} {runtime.find_llama_server()}")
    base, adapter = runtime.model_files()
    record("Set up: base model and adapter downloaded and checked",
           base.name == runtime.BASE_FILE and adapter.name == runtime.ADAPTER_FILE and base.is_file() and
           adapter.is_file() and not list(models.glob("*.part")), runtime.JOB.error or state.last_status)
    with bpy.context.temp_override(**ctx()):
        bpy.ops.bai.setup(voice=True)
    yield from wait(lambda: not runtime.JOB.running, 900)
    record("Install voice: packages installed with Blender's pip", runtime.voice_installed(),
           runtime.JOB.error or state.last_status)
    for o in bpy.context.scene.objects:
        o.select_set(o.name == "Cube")
    bpy.context.view_layer.objects.active = bpy.data.objects["Cube"]
    state.request = "add a bevel modifier"
    with bpy.context.temp_override(**ctx()):
        bpy.ops.bai.run()
    record("first command starts the model server", state.last_status.startswith("Starting"), state.last_status)
    yield from wait(lambda: "llm" not in addon._WAITING, 180)
    record("queued command ran with the add-on's own llama.cpp",
           [m.type for m in bpy.data.objects["Cube"].modifiers] == ["BEVEL"], state.last_status)
    args = runtime.LLM.process.args if runtime.LLM.process else []
    record("model server runs the base model with the adapter (--lora)",
           "--lora" in args and str(adapter) in args and str(base) in args, args)
    with bpy.context.temp_override(**ctx()):
        bpy.ops.bai.listen()
    record("Speak starts the speech server", state.last_status.startswith("Starting voice"), state.last_status)
    yield from wait(lambda: "voice" not in addon._WAITING, 300)
    record("speech server up and listening", state.listening or "Listening" in state.last_status, state.last_status)
    yield from wait(lambda: not state.listening, 30)
    record("recording finished without an error (nobody spoke: nothing heard, or room noise transcribed)",
           not state.last_status.startswith(("Speech error", "Speech server")), state.last_status)
    OUT.close()
    bpy.ops.wm.quit_blender()

GEN = steps()
def tick():
    try:
        next(GEN)
    except StopIteration:
        return None
    except Exception:
        import traceback
        record("test script crashed", False, traceback.format_exc()[-600:])
        OUT.close()
        bpy.ops.wm.quit_blender()
        return None
    return 0.4
bpy.app.timers.register(tick, first_interval=1.5)
'''


def _answers(url):
    try:
        return urllib.request.urlopen(url, timeout=1).status == 200
    except OSError:
        return False


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    for url in ("http://127.0.0.1:8180/health", "http://127.0.0.1:8181/health"):
        if _answers(url):
            sys.exit(f"Something is already running at {url}; stop it first")
    base = ROOT / "models" / "base" / "Qwen3.5-2B-Q8_0.gguf"
    adapter = ROOT / "models" / "operator" / "shilpi-operator-v5-lora-f16.gguf"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "models"))
    handler.log_message = lambda *a: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix="bai_setup_test_") as tmp:
        results = Path(tmp) / "results.jsonl"
        results.touch()
        script = Path(tmp) / "test.py"
        script.write_text(f"ADDON_PARENT = {str(ROOT / 'addon')!r}\nRESULTS = {str(results)!r}\nTMP = {tmp!r}\n"
                          f"BASE_URL = 'http://127.0.0.1:{httpd.server_port}/base/{base.name}'\n"
                          f"ADAPTER_URL = 'http://127.0.0.1:{httpd.server_port}/operator/{adapter.name}'\n"
                          f"BASE_SHA256 = {_sha256(base)!r}\nADAPTER_SHA256 = {_sha256(adapter)!r}\n"
                          + IN_BLENDER, encoding="utf-8")
        env = {**os.environ, "BLENDER_USER_CONFIG": str(Path(tmp) / "config"),
               "BLENDER_USER_DATAFILES": str(Path(tmp) / "datafiles")}
        log = Path(tmp) / "blender.log"
        try:
            with log.open("w", encoding="utf-8", errors="replace") as f:
                subprocess.run([_blender_executable(), "--factory-startup", "--python", str(script)], env=env,
                               stdout=f, stderr=subprocess.STDOUT, timeout=1800)
        except subprocess.TimeoutExpired:
            print("Blender didn't finish; last log lines:\n" + log.read_text(encoding="utf-8", errors="replace")[-1500:])
        rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line]
        rows.append({"test": "Blender quit stops both servers", "detail": "a port still answers",
                     "ok": not _answers("http://127.0.0.1:8180/health") and not _answers("http://127.0.0.1:8181/health")})
    httpd.shutdown()
    for row in rows:
        print(f"  {'PASS' if row['ok'] else 'FAIL'} {row['test']:60} {'' if row['ok'] else row['detail']}")
    print(f"\n{sum(r['ok'] for r in rows)}/{len(rows)} setup checks passed")


if __name__ == "__main__":
    main()

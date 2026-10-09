"""Setup and servers: finds or installs llama.cpp, downloads the model, installs voice, starts/stops both servers.

Everything the add-on downloads goes into its own user folder (kept across add-on updates):
    llama/<build>/        llama.cpp release (llama-server + libraries), only if none is installed already
    models/<file>.gguf    the base model and Shilpi's adapter
    voice/packages/       faster-whisper + sounddevice, installed with Blender's own pip
    voice/whisper/        the Whisper model (downloaded by Install voice)
    logs/                 server logs, for when something goes wrong

Long jobs run in a background thread and only write to JOB and the server status; the UI polls them from a timer,
so Blender never freezes and bpy is never touched from a thread.
"""
import atexit
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import bpy

# llama.cpp build the operator model was tested with (newer builds renamed settings before; pinning avoids surprises)
LLAMA_BUILD = "b11193"
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{build}/llama-{build}-bin-{asset}"
# The model is the stock Qwen3.5-2B (Apache 2.0) plus Shilpi's LoRA adapter, which llama-server applies at load
# time. Later specialists add their own small adapters to the same base instead of another full model each.
# The base is pinned to the exact upload the adapter was tested on.
BASE_FILE = "Qwen3.5-2B-Q8_0.gguf"
BASE_URL = ("https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/f6d5376be1edb4d416d56da11e5397a961aca8ae/"
            + BASE_FILE)
ADAPTER_FILE = "shilpi-operator-v5-lora-f16.gguf"
ADAPTER_URL = f"https://huggingface.co/Tushar98923/shilpi-operator-GGUF/resolve/main/{ADAPTER_FILE}"
MODEL_SIZE_GB = 2.1  # base 2.01 GB + adapter 0.07 GB
# Downloads from these URLs must match these hashes (a custom URL in the preferences isn't checked).
KNOWN_SHA256 = {BASE_URL: "1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1",
                ADAPTER_URL: "99c9286ae6657d3acb49a3f72156d556c7873c5b6470493305ab2cc63a6b67ce"}
# Add-on 0.1.0 downloaded the operator merged into one file. If that's there, it's used as it is (same answers),
# so nobody downloads 2 GB again.
MERGED_FILE = "shilpi-operator-v5-q8_0.gguf"
VOICE_PACKAGES = ["faster-whisper==1.2.1", "sounddevice==0.5.6"]
WHISPER_MODEL = "base.en"  # ~145 MB, CPU; small next to the operator model, which gets the GPU
WHISPER_URL = "https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/{file}"
WHISPER_FILES = ["config.json", "tokenizer.json", "vocabulary.txt", "model.bin"]

IS_WINDOWS, IS_MAC = sys.platform == "win32", sys.platform == "darwin"
EXE = ".exe" if IS_WINDOWS else ""
ADDON_DIR = Path(__file__).parent


# ----------------------------------------------------------------------------------------------- folders

def data_dir() -> Path:
    """The add-on's own user folder (an extension's user folder survives updates)."""
    try:
        path = Path(bpy.utils.extension_path_user(__package__, create=True))
    except (ValueError, AttributeError):  # loaded from source (tests, development), not as an extension
        path = Path(bpy.utils.user_resource("DATAFILES", path="shilpi", create=True))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log_path(name: str) -> Path:
    folder = data_dir() / "logs"
    folder.mkdir(exist_ok=True)
    return folder / f"{name}.log"


def last_log_lines(name: str, count: int = 6) -> str:
    path = _log_path(name)
    if not path.exists():
        return ""
    lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    return " | ".join(lines[-count:])


# ----------------------------------------------------------------------------------------------- llama.cpp

def llama_asset() -> str | None:
    """The release file for this computer, or None if llama.cpp publishes no build for it."""
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if IS_WINDOWS:
        return "win-cpu-arm64.zip" if arm else "win-vulkan-x64.zip"  # Vulkan: NVIDIA, AMD and Intel GPUs, CPU fallback
    if IS_MAC:
        return "macos-arm64.tar.gz" if arm else "macos-x64.tar.gz"      # Metal on Apple Silicon
    if sys.platform.startswith("linux"):
        return "ubuntu-vulkan-arm64.tar.gz" if arm else "ubuntu-vulkan-x64.tar.gz"
    return None


def find_llama_server(custom: str = "") -> Path | None:
    """The preference path, then the add-on's own copy, then one already installed (PATH, winget, Homebrew)."""
    name = "llama-server" + EXE
    candidates = []
    if custom:
        candidates.append(Path(bpy.path.abspath(custom)))
    own = data_dir() / "llama" / LLAMA_BUILD
    if own.exists():
        candidates += sorted(own.rglob(name))
    on_path = shutil.which("llama-server")
    if on_path:
        candidates.append(Path(on_path))
    if IS_WINDOWS:
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        candidates.append(local / "Microsoft" / "WinGet" / "Links" / name)
        packages = local / "Microsoft" / "WinGet" / "Packages"
        if packages.exists():
            candidates += sorted(packages.glob(f"ggml.llamacpp*/{name}"))
    else:
        candidates += [Path("/opt/homebrew/bin/llama-server"), Path("/usr/local/bin/llama-server"),
                       Path.home() / ".local" / "bin" / "llama-server"]
    return next((c for c in candidates if c.is_file()), None)


def install_llama(job) -> Path:
    asset = llama_asset()
    if not asset:
        raise RuntimeError(f"No llama.cpp download for {sys.platform} {platform.machine()}. Install llama.cpp yourself "
                           "and set its path in the add-on preferences.")
    target = data_dir() / "llama" / LLAMA_BUILD
    archive = target.parent / f"llama-{LLAMA_BUILD}-{asset}"
    download(LLAMA_URL.format(build=LLAMA_BUILD, asset=asset), archive, job, "Downloading llama.cpp")
    job.update(step="Unpacking llama.cpp", progress=None)
    shutil.rmtree(target, ignore_errors=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            t.extractall(target, filter="data")  # keeps the libraries' relative symlinks, refuses anything outside
    archive.unlink(missing_ok=True)
    if not IS_WINDOWS:  # make sure the programs are executable
        for path in target.rglob("*"):
            if path.is_file() and not path.is_symlink():
                path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    found = find_llama_server()
    if not found:
        raise RuntimeError("llama.cpp was unpacked but llama-server wasn't in it")
    return found


# ----------------------------------------------------------------------------------------------- model

def model_files(custom: str = "") -> tuple[Path, Path | None]:
    """(model, adapter) for llama-server; adapter is None when the model file is complete on its own."""
    if custom:
        return Path(bpy.path.abspath(custom)), None
    folder = data_dir() / "models"
    base, adapter, merged = folder / BASE_FILE, folder / ADAPTER_FILE, folder / MERGED_FILE
    if merged.is_file() and not (base.is_file() and adapter.is_file()):
        return merged, None
    return base, adapter


def model_ready(custom: str = "") -> bool:
    model, adapter = model_files(custom)
    return model.is_file() and (adapter is None or adapter.is_file())


def model_downloads(prefs) -> list:
    """Setup steps for the model files that are missing."""
    model, adapter = model_files(prefs.model_path)
    if adapter is None:  # a custom or 0.1.0 model: nothing to download
        return []
    wanted = [(prefs.base_url, model, "Downloading the base model"),
              (prefs.adapter_url, adapter, "Downloading Shilpi's adapter")]
    return [lambda job, u=url, d=dest, l=label: download(u, d, job, l) for url, dest, label in wanted if not dest.is_file()]


def download(url: str, dest: Path, job, label: str, attempts: int = 5) -> None:
    """Download with progress. A dropped connection is retried, continuing from where it stopped."""
    for attempt in range(1, attempts + 1):
        try:
            return _download_once(url, dest, job, label)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            if job.cancelled or attempt == attempts or (isinstance(e, urllib.error.HTTPError) and e.code < 500):
                raise RuntimeError(f"{label} failed: {e}. Press Set up again to continue where it stopped.") from e
            job.update(step=f"{label}: connection problem, retrying ({attempt}/{attempts - 1})...")
            time.sleep(3 * attempt)


def _download_once(url: str, dest: Path, job, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "shilpi-addon",
                                                   **({"Range": f"bytes={have}-"} if have else {})})
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:  # already complete
            _check_hash(url, part, job, label)
            part.replace(dest)
            return
        if e.code == 404:
            raise RuntimeError(f"Not found (404): {url}. Check the download URL in the add-on preferences.") from e
        raise
    with response:
        if response.status != 206:  # the server ignored the resume request: start over
            have = 0
        total = int(response.headers.get("Content-Length") or 0) + have
        with open(part, "ab" if have else "wb") as f:
            done = have
            while True:
                if job.cancelled:
                    raise RuntimeError("Cancelled (the download continues where it stopped next time)")
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                job.update(step=f"{label}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB" if total else
                           f"{label}: {done / 1e6:,.0f} MB", progress=done / total if total else None)
    if total and part.stat().st_size != total:
        raise ConnectionError(f"download incomplete ({part.stat().st_size:,} of {total:,} bytes)")  # retried, resuming
    _check_hash(url, part, job, label)  # before the final name, so an unchecked file never looks finished
    part.replace(dest)


def _check_hash(url: str, part: Path, job, label: str) -> None:
    """A known file whose bytes don't match is deleted, so the next Set up downloads it again from the start."""
    expected = KNOWN_SHA256.get(url)
    if not expected:
        return
    import hashlib
    job.update(step=f"{label}: checking the file", progress=None)
    digest = hashlib.sha256()
    with open(part, "rb") as f:
        while chunk := f.read(1 << 22):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{label}: the downloaded file is damaged. Press Set up to download it again.")


# ----------------------------------------------------------------------------------------------- voice

def voice_packages_dir() -> Path:
    return data_dir() / "voice" / "packages"


def whisper_dir() -> Path:
    return data_dir() / "voice" / "whisper" / WHISPER_MODEL


def _packages_installed() -> bool:
    folder = voice_packages_dir()
    return (folder / "faster_whisper").is_dir() and (folder / "sounddevice.py").is_file()


def voice_installed() -> bool:
    return _packages_installed() and all((whisper_dir() / name).is_file() for name in WHISPER_FILES)


def install_voice(job) -> None:
    """faster-whisper + sounddevice into the add-on's folder with Blender's own Python, then the Whisper model."""
    if not _packages_installed():
        _install_voice_packages(job)
    for name in WHISPER_FILES:  # downloaded here (progress, retries) so the speech server starts in seconds
        if not (whisper_dir() / name).is_file():
            download(WHISPER_URL.format(file=name), whisper_dir() / name, job, f"Downloading Whisper ({name})")


def _install_voice_packages(job) -> None:
    job.update(step="Installing voice packages (faster-whisper, ~85 MB)", progress=None)
    folder = voice_packages_dir()
    folder.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    pip = [python, "-m", "pip"]
    flags = _no_window()
    if subprocess.run(pip + ["--version"], capture_output=True, **flags).returncode != 0:
        subprocess.run([python, "-m", "ensurepip"], capture_output=True, **flags)
    with open(_log_path("voice-install"), "w", encoding="utf-8") as log:
        result = subprocess.run(pip + ["install", "--upgrade", "--target", str(folder), "--only-binary=:all:",
                                       "--disable-pip-version-check", "--no-warn-script-location", "--retries", "10",
                                       "--timeout", "60", *VOICE_PACKAGES],
                                stdout=log, stderr=subprocess.STDOUT, env=_clean_env(), **flags)
    if result.returncode != 0 or not _packages_installed():
        raise RuntimeError("Installing voice failed: " + last_log_lines("voice-install", 3))


# ----------------------------------------------------------------------------------------------- background job

class Job:
    """One setup job at a time; the panel shows `step` and `progress`."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.running, self.cancelled, self.step, self.progress, self.error = False, False, "", None, ""

    def update(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def start(self, steps):
        if self.running:
            return False
        self.reset()
        self.running = True

        def work():
            try:
                for step in steps:
                    step(self)
                self.update(step="Done", progress=None)
            except Exception as e:
                self.update(error=str(e))
            finally:
                self.update(running=False)
        threading.Thread(target=work, daemon=True).start()
        return True


JOB = Job()


# ----------------------------------------------------------------------------------------------- servers

def _no_window() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if IS_WINDOWS else {}


def _clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["PYTHONNOUSERSITE"] = "1"
    return env


def healthy(url: str, timeout: float = 0.4) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def _process_name(pid: int) -> str:
    """The program name of a running process, or '' if there is none."""
    try:
        if IS_WINDOWS:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True,
                                 text=True, timeout=5, **_no_window()).stdout
            return out.split(",")[0].strip('"').lower() if out.startswith('"') else ""
        return subprocess.run(["ps", "-p", str(pid), "-o", "comm="], capture_output=True, text=True,
                              timeout=5).stdout.strip().lower()
    except (OSError, subprocess.SubprocessError):
        return ""


class Server:
    """A server process this add-on started (it never stops one it didn't start).

    Its process ID is saved in logs/<name>.pid, so a server left behind by a Blender crash (no atexit) is still
    recognised as ours and stopped by the next session's Stop button or quit, instead of holding GPU memory.
    """

    def __init__(self, name: str, program: str):
        self.name, self.program, self.process, self.error = name, program, None, ""

    def _pid_file(self) -> Path:
        return _log_path(self.name).with_suffix(".pid")

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, args: list, env: dict | None = None) -> None:
        self.stop()
        self.error = ""
        log = open(_log_path(self.name), "w", encoding="utf-8", errors="replace")
        self.process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        env=env, start_new_session=not IS_WINDOWS, **_no_window())
        log.close()  # the child keeps its own handle
        self._pid_file().write_text(str(self.process.pid))

    def _stop_leftover(self) -> None:
        """Stop a server an earlier, crashed Blender session started (only if that PID is still our program)."""
        try:
            pid = int(self._pid_file().read_text())
        except (OSError, ValueError):
            return
        if self.program in _process_name(pid):
            try:
                if IS_WINDOWS:
                    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, **_no_window())
                else:
                    os.kill(pid, 15)
            except OSError:
                pass

    def exited_with_error(self) -> bool:
        if self.process is not None and self.process.poll() is not None:
            self.error = f"{self.name} stopped (exit {self.process.returncode}): {last_log_lines(self.name)}"
            self.process = None
            return True
        return False

    def stop(self) -> None:
        if self.running():
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        elif self.process is None:
            self._stop_leftover()
        self.process = None
        self._pid_file().unlink(missing_ok=True)


LLM = Server("llama-server", "llama-server")
VOICE = Server("speech-server", "python")


def port_of(url: str, default: int) -> int:
    from urllib.parse import urlparse
    return urlparse(url).port or default


def start_llm(prefs) -> str | None:
    """Start llama-server with the operator model. Returns an error message, or None."""
    exe, (model, adapter) = find_llama_server(prefs.llama_path), model_files(prefs.model_path)
    if not exe:
        return "llama.cpp isn't installed: press Set up"
    for path in (model, adapter):
        if path is not None and not path.is_file():
            return f"Model not found ({path.name}): press Set up"
    env = {**os.environ, "LLAMA_ARG_CHAT_TEMPLATE_KWARGS": '{"enable_thinking": false}',  # trained without thinking
           "LLAMA_ARG_REASONING": "off"}
    port = port_of(prefs.server_url, 8080)
    LLM.start([str(exe), "-m", str(model), *(["--lora", str(adapter)] if adapter else []), "--host", "127.0.0.1",
               "--port", str(port), "--jinja", "-ngl", "99", "-c", "2048"], env=env)
    return None


def start_voice(prefs) -> str | None:
    if not voice_installed():
        return "Voice isn't installed: press Install voice"
    script = ADDON_DIR / "vendor" / "asr_server.py"
    args = [sys.executable, "-u", str(script), "--model", str(whisper_dir()),  # -u: the log shows progress live
            "--port", str(port_of(prefs.asr_url, 8081))]
    env = {**_clean_env(), "PYTHONPATH": str(voice_packages_dir())}
    if IS_MAC and prefs.voice_in_terminal:
        # macOS asks for microphone access per app; Terminal is allowed to ask, a process started by Blender may not be.
        command = data_dir() / "voice" / "start_speech_server.command"
        quoted = " ".join(f"'{a}'" for a in args)
        command.write_text(f"#!/bin/sh\nexport PYTHONPATH='{voice_packages_dir()}' PYTHONNOUSERSITE=1\nexec {quoted}\n")
        command.chmod(0o755)
        subprocess.Popen(["open", "-a", "Terminal", str(command)])
        return None
    VOICE.start(args, env=env)
    return None


def stop_all() -> None:
    LLM.stop()
    VOICE.stop()


atexit.register(stop_all)  # Blender quitting stops the servers this add-on started

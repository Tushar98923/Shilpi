"""Local speech-to-text server for the Blender add-on's mic button.

Records from the default microphone, stops by itself after a pause, and transcribes with Whisper
(faster-whisper, base.en by default: ~145 MB, runs on the CPU so the GPU stays free for the operator
model). The add-on polls it over HTTP on 127.0.0.1, so Blender never blocks and needs no audio libraries.

    POST /listen   start recording (stops on its own after ~0.8 s of silence, or after 12 s)
    POST /stop     stop now and transcribe what was heard
    GET  /result   {"state": "idle" | "listening" | "transcribing" | "done" | "error", "text": ..., "error": ...}

The add-on installs faster-whisper + sounddevice into its own folder and starts this file with Blender's
Python, so users don't run it by hand. By hand (the start script uses the .venv-asr environment):
    .\\scripts\\start_asr_server.ps1
    python scripts/asr_server.py --model small.en      # bigger (~250 MB), more accurate, slower
    python scripts/asr_server.py --test 4              # record 4 s and print the transcript, then exit
"""
import argparse
import json
import queue
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

RATE = 16000
BLOCK = 1600                 # 0.1 s
SILENCE_AFTER_SPEECH = 0.8   # seconds of quiet that end a command
MAX_SECONDS = 12
NO_SPEECH_TIMEOUT = 5        # give up if nothing is said
# Whisper treats this as text that came before the audio, so it prefers these spellings.
BLENDER_PROMPT = ("Blender commands: bevel, extrude, subdivide, inset, loop cut, solidify, decimate, remesh, "
                  "array, mirror, boolean, wireframe, shade smooth, shade flat, UV unwrap, keyframe, "
                  "Suzanne, icosphere, UV sphere, torus, cylinder, cone, cube, Edit Mode, Object Mode, "
                  "sculpt mode, origin, 3D cursor, modifier, material, collection, parent, rotate, scale.")


class Recorder:
    def __init__(self, model, threshold, wav=None):
        self.model, self.threshold, self.wav = model, threshold, wav
        self.lock = threading.Lock()
        self.state, self.text, self.error = "idle", "", ""
        self._stop = threading.Event()

    def result(self):
        with self.lock:
            return {"state": self.state, "text": self.text, "error": self.error}

    def listen(self):
        with self.lock:
            if self.state in ("listening", "transcribing"):
                return
            self.state, self.text, self.error = "listening", "", ""
        self._stop.clear()
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def _run(self):
        try:
            audio = self._record()
            self._set(state="transcribing")
            text = transcribe(self.model, audio) if audio.size else ""
            self._set(state="done", text=text)
        except Exception as e:  # report to the add-on instead of dying
            self._set(state="error", error=str(e))

    def _record(self):
        if self.wav:  # tests: "hear" this file instead of the mic
            return read_wav(self.wav)
        blocks, heard, quiet, start = [], False, 0.0, time.time()
        q = queue.Queue()
        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=BLOCK,
                            callback=lambda data, *_: q.put(data[:, 0].copy())):
            while not self._stop.is_set():
                try:
                    block = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                blocks.append(block)
                loud = float(np.sqrt(np.mean(block ** 2))) > self.threshold
                heard |= loud
                quiet = 0.0 if loud else quiet + BLOCK / RATE
                elapsed = time.time() - start
                if (heard and quiet >= SILENCE_AFTER_SPEECH) or elapsed > MAX_SECONDS:
                    break
                if not heard and elapsed > NO_SPEECH_TIMEOUT:
                    if not np.any(np.concatenate(blocks)):  # exact zeros: the OS is blocking the mic
                        raise RuntimeError("The microphone gives no sound. On macOS: System Settings > Privacy & "
                                           "Security > Microphone, allow Blender (or turn on 'Run speech server in "
                                           "Terminal' in the add-on preferences). Elsewhere: check the input device.")
                    return np.zeros(0, dtype=np.float32)
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)


def read_wav(path):
    """16 kHz mono 16-bit WAV -> float32 samples."""
    with wave.open(str(path)) as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, "need 16 kHz mono 16-bit"
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def transcribe(model, audio):
    segments, _ = model.transcribe(audio, language="en", beam_size=5, initial_prompt=BLENDER_PROMPT,
                                   condition_on_previous_text=False, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def make_handler(recorder):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, code=200):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path == "/listen":
                recorder.listen()
            elif self.path == "/stop":
                recorder.stop()
            else:
                return self._send({"error": "unknown path"}, 404)
            self._send(recorder.result())

        def do_GET(self):
            if self.path in ("/result", "/health"):
                return self._send(recorder.result())
            self._send({"error": "unknown path"}, 404)

        def log_message(self, *args):  # keep the console for transcripts
            pass
    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base.en", help="tiny.en (~76 MB), base.en (~145 MB), small.en (~484 MB)")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--threshold", type=float, default=0.012, help="mic level that counts as speech")
    ap.add_argument("--test", type=float, help="record this many seconds, print the transcript, exit")
    ap.add_argument("--wav", help="tests: every /listen 'hears' this 16 kHz mono WAV instead of the mic")
    ap.add_argument("--download-root", help="where the Whisper model is stored (default: the Hugging Face cache)")
    args = ap.parse_args()

    print(f"Loading Whisper {args.model} (CPU, int8; the first start downloads it)...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8", download_root=args.download_root)
    try:
        print("Microphone:", sd.query_devices(kind="input")["name"])
    except Exception as e:  # no input device: still serve, /listen reports the error
        print("No microphone found:", e)

    if args.test:
        print(f"Speak now ({args.test:g} s)...")
        audio = sd.rec(int(args.test * RATE), samplerate=RATE, channels=1, dtype="float32")
        sd.wait()
        t = time.time()
        print(f"-> {transcribe(model, audio[:, 0])!r}  ({time.time() - t:.2f} s)")
        return

    recorder = Recorder(model, args.threshold, args.wav)
    last = None

    def report():  # print each transcript once, so you can see what it heard
        nonlocal last
        while True:
            r = recorder.result()
            if r["state"] in ("done", "error") and r != last:
                print("heard:", repr(r["text"]) if r["state"] == "done" else "ERROR " + r["error"])
                last = r
            time.sleep(0.2)
    threading.Thread(target=report, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(recorder))
    print(f"Speech server on http://127.0.0.1:{args.port}  (Ctrl+C or close this window to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()

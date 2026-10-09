"""Shilpi, an AI assistant for Blender: type or say a command, the local operator model turns it into op/1 calls.

Sidebar (N) > Shilpi tab. The first time, press Set up: it installs llama.cpp (unless it's already installed) and
downloads the operator model into the add-on's folder. After that the model server starts by itself on the
first command and stops when Blender quits (runtime.py). Install voice adds the Speak button's speech
server (Whisper, CPU); Speak or Alt+Shift+V records until you pause, transcribes, and runs the command.
A server you started yourself on the same port is used as-is.

The scene serializer, output formats, executor, output schema and selection-free operator list are
vendored from shared/ by scripts/build_addon.py, so training data and the add-on can never drift apart.

Behaviour worth knowing:
- Constrained decoding: the server may only answer with shortlist operators and their real arguments.
- Each AI command is one undo step. "Undo/redo that" runs on the next tick (undo inside an operator,
  or in the same tick as other changes, crashed Blender in testing).
- If nothing is selected and the answer needs a selection, nothing runs and the panel says why.
- If a later call fails, the earlier ones stay as one undo step and the panel says Ctrl+Z reverts them.
- bai.ask / bai.decline show their question or reason and run nothing.
- Learning from corrections: see corrections.py.
"""
import json
import re
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

import bpy

from . import corrections, runtime
from .vendor import op_executor, output_formats, scene_serializer

# Reinstalling the add-on without restarting Blender reloads this file but keeps the old submodules in
# memory (new __init__ + old compact_scene crashed once), so always load the installed copies.
import importlib  # noqa: E402
for _module in (corrections, runtime, op_executor, output_formats, scene_serializer):
    importlib.reload(_module)

VENDOR = Path(__file__).with_name("vendor")
_SCHEMA = None
_SELECTION_FREE = None


def _schema():
    global _SCHEMA
    if _SCHEMA is None:
        path = VENDOR / "op_schema.json"
        _SCHEMA = json.loads(path.read_text(encoding="utf-8")) if path.exists() else False
    return _SCHEMA or None


def _selection_free():
    global _SELECTION_FREE
    if _SELECTION_FREE is None:
        path = VENDOR / "selection_free_ops.json"
        _SELECTION_FREE = set(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else set()
    return _SELECTION_FREE


def _chat(server_url: str, user_message: str, constrained: bool) -> str:
    payload = {
        "model": "operator",
        "messages": [
            {"role": "system", "content": output_formats.OPERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
        "max_tokens": 256,
    }
    if constrained and _schema():
        payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "op1", "schema": _schema()}}
    request = urllib.request.Request(f"{server_url.rstrip('/')}/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def needs_selection_but_none(calls: list, context) -> bool:
    """True if the answer acts on the selection, nothing is selected, and it doesn't select anything first."""
    if context.selected_objects or context.view_layer.objects.active is not None:
        return False
    first = calls[0]["op"]
    return first != "bai.select" and first not in _selection_free()


def _run_later(calls: list) -> None:
    """Undo/redo: run on the next tick with only the window overridden."""
    def run():
        window = bpy.context.window_manager.windows[0]
        with bpy.context.temp_override(window=window):
            try:
                op_executor.apply_calls(calls)
                bpy.context.window_manager.bai.last_status = "Done: " + ", ".join(c["op"] for c in calls)
            except op_executor.OpExecutionError as e:
                bpy.context.window_manager.bai.last_status = f"Failed: {e}"
        return None
    bpy.app.timers.register(run, first_interval=0.05)


def _asr(asr_url: str, path: str, method: str = "GET") -> dict:
    request = urllib.request.Request(f"{asr_url.rstrip('/')}{path}", method=method,
                                     data=b"" if method == "POST" else None)
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def _view3d_override():
    """Window + 3D View area/region, so an operator started from a timer behaves like a click in the viewport."""
    window = bpy.context.window_manager.windows[0]
    for area in window.screen.areas:
        if area.type == "VIEW_3D":
            region = next(r for r in area.regions if r.type == "WINDOW")
            return {"window": window, "area": area, "region": region}
    return {"window": window}


class _DefaultPrefs:
    server_url, constrained, learn = "http://127.0.0.1:8080/v1", True, True
    asr_url, voice_auto_run = "http://127.0.0.1:8081", True
    auto_start, model_path, llama_path, voice_in_terminal = True, "", "", False
    base_url, adapter_url = runtime.BASE_URL, runtime.ADAPTER_URL


def _prefs(context):
    """The add-on's preferences, or the defaults when it was loaded without being enabled in Preferences."""
    addon = context.preferences.addons.get(__package__)
    return addon.preferences if addon else _DefaultPrefs()


class BAI_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    auto_start: bpy.props.BoolProperty(name="Start the AI when needed", default=True,
                                       description="Start the model server on the first command and stop it when Blender "
                                                   "quits. Off: use Start AI / Stop AI in the panel")
    voice_auto_run: bpy.props.BoolProperty(name="Run spoken commands right away", default=True,
                                           description="Off: the transcript goes into the command box and you press Run")
    constrained: bpy.props.BoolProperty(name="Constrained decoding", default=True,
                                        description="Only allow real operators and arguments in the model's answer")
    learn: bpy.props.BoolProperty(name="Learn from my corrections", default=True,
                                  description="Ctrl+Z after an AI command, then doing it yourself, teaches it (stored locally)")
    model_path: bpy.props.StringProperty(name="Model file", subtype="FILE_PATH", default="",
                                         description="Use this complete GGUF instead of the downloaded base model and adapter "
                                                     "(leave empty normally)")
    llama_path: bpy.props.StringProperty(name="llama-server", subtype="FILE_PATH", default="",
                                         description="Use this llama-server program (leave empty: found or installed automatically)")
    base_url: bpy.props.StringProperty(name="Base model download URL", default=runtime.BASE_URL)
    adapter_url: bpy.props.StringProperty(name="Adapter download URL", default=runtime.ADAPTER_URL)
    server_url: bpy.props.StringProperty(name="Model server URL", default="http://127.0.0.1:8080/v1")
    asr_url: bpy.props.StringProperty(name="Speech server URL", default="http://127.0.0.1:8081")
    voice_in_terminal: bpy.props.BoolProperty(
        name="Run speech server in Terminal (macOS)", default=False,
        description="If macOS doesn't let Blender use the microphone: the speech server runs in a Terminal window, "
                    "which macOS asks microphone permission for")

    def draw(self, context):
        layout = self.layout
        for name in ("auto_start", "voice_auto_run", "constrained", "learn"):
            layout.prop(self, name)
        if runtime.IS_MAC:
            layout.prop(self, "voice_in_terminal")
        box = layout.box()
        box.label(text="Advanced (leave as they are normally)")
        for name in ("model_path", "llama_path", "base_url", "adapter_url", "server_url", "asr_url"):
            box.prop(self, name)
        box.label(text=f"Downloads and logs: {runtime.data_dir()}")


# ------------------------------------------------------------------ servers: start on demand, then run what waited

def _health_url(url: str) -> str:
    base = url.rstrip("/")
    return (base[:-3] if base.endswith("/v1") else base) + "/health"


_WAITING = {}  # "llm" / "voice" -> deadline while that server is starting
_QUEUED = {"run": False, "listen": False}


def _ensure(kind: str, prefs, force: bool = False) -> str:
    """'ready', 'starting' (the queued action runs when it is up), or an error message."""
    url = prefs.server_url if kind == "llm" else prefs.asr_url
    if runtime.healthy(_health_url(url)):
        return "ready"
    if kind in _WAITING:
        return "starting"
    if kind == "llm" and not (prefs.auto_start or force):
        return "The AI isn't running: press Start AI (or turn on 'Start the AI when needed' in Preferences)"
    error = runtime.start_llm(prefs) if kind == "llm" else runtime.start_voice(prefs)
    if error:
        return error
    _WAITING[kind] = time.time() + 180
    if not bpy.app.timers.is_registered(_watch_servers):
        bpy.app.timers.register(_watch_servers, first_interval=0.5)
    return "starting"


def _watch_servers():
    """Timer while a server starts: when it answers, run the command / start listening that waited for it."""
    wm = bpy.context.window_manager
    state, prefs = wm.bai, _prefs(bpy.context)
    for kind in list(_WAITING):
        server = runtime.LLM if kind == "llm" else runtime.VOICE
        url = prefs.server_url if kind == "llm" else prefs.asr_url
        if runtime.healthy(_health_url(url)):
            del _WAITING[kind]
            action = "run" if kind == "llm" else "listen"
            if _QUEUED[action]:
                _QUEUED[action] = False
                with bpy.context.temp_override(**_view3d_override()):
                    bpy.ops.bai.run() if action == "run" else bpy.ops.bai.listen()
            elif state.last_status.startswith("Starting"):
                state.last_status = "The AI is running." if kind == "llm" else "Voice is ready."
        elif server.exited_with_error() or time.time() > _WAITING[kind]:
            del _WAITING[kind]
            _QUEUED["run" if kind == "llm" else "listen"] = False
            state.listening = False
            state.last_status = server.error or f"The {'model' if kind == 'llm' else 'speech'} server didn't start in time."
    _redraw()
    return 0.5 if _WAITING else None


_SETUP = {"time": 0.0}


def setup_state(prefs) -> dict:
    """What is installed and running (recomputed at most every 2 s; the panel draws often)."""
    if time.time() - _SETUP["time"] > 2:
        _SETUP.update(time=time.time(), llama=runtime.find_llama_server(prefs.llama_path) is not None,
                      model=runtime.model_ready(prefs.model_path), voice=runtime.voice_installed(),
                      llm_up=runtime.healthy(_health_url(prefs.server_url), 0.2),
                      voice_up=runtime.healthy(_health_url(prefs.asr_url), 0.2))
    return _SETUP


def _watch_job():
    _SETUP["time"] = 0.0  # recheck what's installed
    _redraw()
    if runtime.JOB.running:
        return 0.5
    state = bpy.context.window_manager.bai
    state.last_status = f"Setup failed: {runtime.JOB.error}" if runtime.JOB.error else "Setup done. Type or say a command."
    return None


class BAI_OT_setup(bpy.types.Operator):
    """Install llama.cpp (if it isn't installed) and download the operator model into the add-on's folder"""

    bl_idname = "bai.setup"
    bl_label = "Set up"
    voice: bpy.props.BoolProperty(default=False, options={"SKIP_SAVE"})

    def execute(self, context):
        prefs = _prefs(context)
        steps = []
        if self.voice:
            steps.append(runtime.install_voice)
        else:
            if not runtime.find_llama_server(prefs.llama_path):
                steps.append(runtime.install_llama)
            steps += runtime.model_downloads(prefs)
        if not steps:
            context.window_manager.bai.last_status = "Everything is already set up."
            return {"FINISHED"}
        if not runtime.JOB.start(steps):
            self.report({"WARNING"}, "Setup is already running")
            return {"CANCELLED"}
        bpy.app.timers.register(_watch_job, first_interval=0.3)
        return {"FINISHED"}


class BAI_OT_cancel_setup(bpy.types.Operator):
    """Stop the download (it continues where it stopped next time)"""

    bl_idname = "bai.cancel_setup"
    bl_label = "Cancel"

    def execute(self, context):
        runtime.JOB.update(cancelled=True)
        return {"FINISHED"}


class BAI_OT_server(bpy.types.Operator):
    """Start or stop the AI servers (stopping frees the GPU memory)"""

    bl_idname = "bai.server"
    bl_label = "AI server"
    start: bpy.props.BoolProperty(default=True)

    def execute(self, context):
        state = context.window_manager.bai
        if self.start:
            result = _ensure("llm", _prefs(context), force=True)
            state.last_status = {"ready": "The AI is running.", "starting": "Starting the AI model..."}.get(result, result)
        else:
            runtime.stop_all()
            _WAITING.clear()
            state.last_status = "AI stopped." if not runtime.healthy(_health_url(_prefs(context).server_url)) else \
                "Stopped what this add-on started; another program is still serving the model on that port."
        _SETUP["time"] = 0.0
        return {"FINISHED"}


class BAI_OT_open_folder(bpy.types.Operator):
    """Open the folder with the add-on's downloads and server logs"""

    bl_idname = "bai.open_folder"
    bl_label = "Open folder"

    def execute(self, context):
        bpy.ops.wm.path_open(filepath=str(runtime.data_dir()))
        return {"FINISHED"}


class BAI_OT_run(bpy.types.Operator):
    """Send the command to the local operator model and run its answer"""

    bl_idname = "bai.run"
    bl_label = "Run"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        self._logged = {"scene": None, "calls": None}
        state = context.window_manager.bai
        state.last_reply = ""
        if state.request.strip():
            status = _ensure("llm", _prefs(context))
            if status == "starting":
                _QUEUED["run"] = True
                state.last_status = "Starting the AI model... (a few seconds; your command runs when it's ready)"
                return {"CANCELLED"}
            if status != "ready":
                state.last_status = status
                self.report({"WARNING"}, status)
                return {"CANCELLED"}
        result = self._execute(context, state)
        if state.request.strip() and _prefs(context).learn:
            corrections.log_command(state.request.strip(), self._logged["scene"], state.last_reply,
                                    self._logged["calls"], state.last_status)
        return result

    def _execute(self, context, state):
        prefs = _prefs(context)
        request = state.request.strip()
        if not request:
            self.report({"WARNING"}, "Type a command first")
            return {"CANCELLED"}
        if state.pending_question:  # this may be the answer to "Which one do you mean: A or B?"
            options = [o for o in state.pending_options.split("\n") if o]
            choice = match_answer(request, options)
            original = state.pending_request
            state.pending_question = state.pending_options = state.pending_request = ""
            if choice and choice in bpy.data.objects:
                # v5 acts on the selected one of two look-alikes (46/46 in a test); naming it exactly makes it ask again.
                for obj in context.view_layer.objects:
                    obj.select_set(obj.name == choice)
                context.view_layer.objects.active = bpy.data.objects[choice]
                request = state.request = original

        full_scene = scene_serializer.serialize_scene(context)
        scene = scene_serializer.compact_scene(full_scene, request=request)
        self._logged["scene"] = scene
        calls = corrections.shortcut_for(request, {o["name"] for o in full_scene["objects"]}) if prefs.learn else None
        source = "your shortcut" if calls else "model"
        if calls is None:
            try:
                reply = _chat(prefs.server_url, output_formats.build_user_message(request, scene), prefs.constrained)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                return self._fail(state, f"Server unreachable: {e}")
            state.last_reply = reply
            try:
                calls = output_formats.parse_op_output(reply)
            except output_formats.FormatError as e:
                return self._fail(state, f"Unusable answer: {e}")

        self._logged["calls"] = calls
        if any(c["op"] in op_executor.DATA_RELOADING_OPS for c in calls):
            _run_later(calls)
            state.last_status = "Running: " + ", ".join(c["op"] for c in calls)
            return {"CANCELLED"}  # no undo step of our own around an undo
        if needs_selection_but_none(calls, context):
            return self._fail(state, "Nothing is selected. Select an object, or say which one.", level="WARNING")

        try:
            op_executor.apply_calls(calls)
        except op_executor.AskUser as e:
            match = ASK_RE.match(e.question)
            state.pending_question, state.pending_request = e.question, request
            state.pending_options = "\n".join(match.groups()) if match else ""
            return self._fail(state, f"Question: {e.question}", level="WARNING")
        except op_executor.Declined as e:
            return self._fail(state, f"Can't do that: {e.reason}", level="WARNING")
        except op_executor.OpExecutionError as e:
            if e.completed:  # keep the calls that ran as one undo step, and say so
                state.last_status = f"Partly done ({e.completed} of {len(calls)} steps), Ctrl+Z reverts: {e}"
                self.report({"WARNING"}, state.last_status)
                return {"FINISHED"}
            return self._fail(state, f"Failed: {e}")

        corrections.TRACKER.ai_ran(request, scene, calls)
        state.last_status = f"Ran {len(calls)} call(s) ({source}): " + ", ".join(c["op"] for c in calls)
        self.report({"INFO"}, state.last_status)
        return {"FINISHED"}

    def _fail(self, state, message, level="ERROR"):
        state.last_status = message
        self.report({level}, message)
        return {"CANCELLED"}


ASK_RE = re.compile(r"Which one do you mean: (.+) or (.+)\?$")  # the only question v5 asks
_DIGITS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
           "seven": "7", "eight": "8", "nine": "9"}


_FILLER = {"the", "that", "this", "please", "um", "uh", "i", "mean", "it's", "its", "it", "is", "uhh",
            "umm", "erm", "yes", "yeah", "pick", "choose", "use", "do", "go", "with", "a", "an", "object", "named"}


def _norm(text: str) -> str:
    """'Cube point zero zero one' / 'cube.001' / 'Cube 001' -> 'cube001'."""
    words = re.findall(r"[a-z]+|\d+", text.lower())
    return "".join(_DIGITS.get(w, w) for w in words if w not in ("the", "point", "dot"))


def match_answer(answer: str, options: list) -> str | None:
    """Which option a typed or spoken answer picks, or None (then it is treated as a new command)."""
    if not options:
        return None
    # Only the choice plus filler counts as an answer: "um the second one please", "Cube.001".
    # "delete the cube" is a new command, not an answer.
    words = [w for w in re.findall(r"[a-z0-9.']+", answer.lower().replace("_", " ")) if w not in _FILLER]
    no_one = [w for w in words if w != "one"]  # "the first one"; but keep it for "cube zero zero one"
    if no_one in (["first"], ["1st"], ["former"]):
        return options[0]
    if no_one in (["second"], ["2nd"], ["other"], ["last"], ["latter"]):
        return options[-1]
    for candidate in (words, no_one):
        spoken = _norm(" ".join(candidate))
        found = next((o for o in options if _norm(o) == spoken), None)
        if found:
            return found
    return None


class BAI_OT_answer(bpy.types.Operator):
    """Pick this one and run the command on it"""

    bl_idname = "bai.answer"
    bl_label = "Answer"
    bl_options = {"REGISTER", "UNDO"}
    choice: bpy.props.StringProperty()

    def execute(self, context):
        context.window_manager.bai.request = self.choice
        return bpy.ops.bai.run()


class BAI_OT_listen(bpy.types.Operator):
    """Say a command: records until you pause, then runs it (click again to stop early)"""

    bl_idname = "bai.listen"
    bl_label = "Speak"

    def execute(self, context):
        state, prefs = context.window_manager.bai, _prefs(context)
        url = prefs.asr_url
        if not state.listening:
            status = _ensure("voice", prefs)
            if prefs.auto_start:
                _ensure("llm", prefs)  # warm the model up while you talk
            if status == "starting":
                _QUEUED["listen"] = True
                state.last_status = "Starting voice..."
                return {"FINISHED"}
            if status != "ready":
                state.last_status = status
                self.report({"WARNING"}, status)
                return {"CANCELLED"}
        try:
            if state.listening:  # second click: stop now; _poll_speech picks up the transcript
                _asr(url, "/stop", "POST")
                return {"FINISHED"}
            _asr(url, "/listen", "POST")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            state.last_status = f"Speech server unreachable ({e})"
            self.report({"ERROR"}, state.last_status)
            return {"CANCELLED"}
        state.listening = True
        state.last_status = "Listening... (stops when you pause)"
        if not bpy.app.timers.is_registered(_poll_speech):
            bpy.app.timers.register(_poll_speech, first_interval=0.2)
        return {"FINISHED"}


def _poll_speech():
    state, prefs = bpy.context.window_manager.bai, _prefs(bpy.context)
    try:
        result = _asr(prefs.asr_url, "/result")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        state.listening, state.last_status = False, f"Speech server stopped: {e}"
        _redraw()
        return None
    if result["state"] in ("listening", "transcribing"):
        if result["state"] == "transcribing" and state.last_status != "Transcribing...":
            state.last_status = "Transcribing..."
            _redraw()
        return 0.2
    state.listening = False
    if result["state"] == "error":
        state.last_status = "Speech error: " + result["error"]
    elif not result["text"]:
        state.last_status = "Didn't hear anything. Press Speak and talk."
    else:
        state.request = result["text"]
        if prefs.voice_auto_run:
            with bpy.context.temp_override(**_view3d_override()):
                bpy.ops.bai.run()
        else:
            state.last_status = f'Heard: "{result["text"]}". Press Run.'
    _redraw()
    return None


def _redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class BAI_OT_feedback(bpy.types.Operator):
    """Tell the assistant whether its last answer was right"""

    bl_idname = "bai.feedback"
    bl_label = "Feedback"
    good: bpy.props.BoolProperty()

    def execute(self, context):
        state = context.window_manager.bai
        corrections.mark_last_logged("right" if self.good else "wrong")
        if self.good:
            state.last_status = "Thanks, noted." if corrections.TRACKER.confirm() else "Nothing to rate yet."
        elif corrections.TRACKER.mark_wrong():
            state.last_status = "Got it. Undo it (Ctrl+Z) and do it yourself; I'll learn from what you do."
        else:
            state.last_status = "Nothing to rate yet."
        return {"FINISHED"}


class BAI_OT_forget_shortcut(bpy.types.Operator):
    """Stop using this shortcut"""

    bl_idname = "bai.forget_shortcut"
    bl_label = "Forget"
    key: bpy.props.StringProperty()

    def execute(self, context):
        corrections.disable(self.key)
        return {"FINISHED"}


class BAI_OT_export_corrections(bpy.types.Operator):
    """Save a copy of your commands and corrections to a file you choose (to share, if you want to)"""

    bl_idname = "bai.export_corrections"
    bl_label = "Export corrections"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH", default="shilpi_corrections.jsonl")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        rows = [{"kind": "correction", **r} for r in corrections.load()]
        log = corrections.store_path().with_name("commands.jsonl")
        if log.exists():
            rows += [{"kind": "command", **json.loads(line)} for line in log.read_text(encoding="utf-8").splitlines() if line]
        Path(self.filepath).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.report({"INFO"}, f"Exported {len(rows)} commands and corrections")
        return {"FINISHED"}


class BAI_State(bpy.types.PropertyGroup):
    request: bpy.props.StringProperty(name="Command", description="What should happen in the scene")
    last_reply: bpy.props.StringProperty(name="Model reply")
    last_status: bpy.props.StringProperty(name="Status")
    listening: bpy.props.BoolProperty(name="Listening")
    pending_question: bpy.props.StringProperty()  # the model asked this and waits for an answer
    pending_options: bpy.props.StringProperty()   # its choices, one per line
    pending_request: bpy.props.StringProperty()   # the command to run again once answered


def _wrapped(layout, text, width, icon="NONE"):
    """A label that wraps instead of being cut off at the sidebar's edge."""
    column = layout.column(align=True)
    for i, line in enumerate(textwrap.wrap(text, width) or [""]):
        column.label(text=line, icon=icon if i == 0 else "BLANK1" if icon != "NONE" else "NONE")


class BAI_PT_panel(bpy.types.Panel):
    bl_label = "Shilpi"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Shilpi"

    def draw(self, context):
        state = context.window_manager.bai
        layout = self.layout
        if not bpy.app.version_string.startswith(output_formats.PINNED_BLENDER_VERSION):
            layout.label(text=f"Trained on Blender {output_formats.PINNED_BLENDER_VERSION}; this is {bpy.app.version_string}",
                         icon="ERROR")
        width = max(20, int(context.region.width / (9 * context.preferences.system.ui_scale)))  # characters per line
        if not self._draw_setup(context, layout, width):
            return
        layout.prop(state, "request", text="")
        row = layout.row(align=True)
        row.operator(BAI_OT_run.bl_idname, icon="PLAY")
        row.operator(BAI_OT_listen.bl_idname, text="Stop" if state.listening else "Speak",
                     icon="CANCEL" if state.listening else "REC", depress=state.listening)
        if state.pending_question:
            box = layout.box()
            _wrapped(box, state.pending_question, width, icon="QUESTION")
            options = [o for o in state.pending_options.split("\n") if o]
            row = box.row(align=True)
            for option in options:
                row.operator(BAI_OT_answer.bl_idname, text=option).choice = option
            if options:
                _wrapped(box, "Or say / type which one (\"the first one\", a name).", width)
        elif state.last_status:
            _wrapped(layout, state.last_status, width)
        row = layout.row(align=True)
        row.operator(BAI_OT_feedback.bl_idname, text="Right", icon="CHECKMARK").good = True
        row.operator(BAI_OT_feedback.bl_idname, text="Wrong", icon="X").good = False
        if state.last_reply:
            box = layout.box()
            box.label(text="Model reply:")
            _wrapped(box, state.last_reply[:400], width)

    def _draw_setup(self, context, layout, width) -> bool:
        """Setup / server status. Returns False while the add-on can't take commands yet."""
        prefs, job = _prefs(context), runtime.JOB
        s = setup_state(prefs)
        if job.running:
            box = layout.box()
            box.progress(factor=job.progress or 0.0, type="BAR", text=job.step or "Working...")
            box.operator(BAI_OT_cancel_setup.bl_idname, icon="CANCEL")
            return s["llm_up"] or (s["llama"] and s["model"])
        if job.error:
            _wrapped(layout.box(), f"Setup failed: {job.error}", width, icon="ERROR")
        if not s["llm_up"] and not (s["llama"] and s["model"]):
            box = layout.box()
            box.label(text="First-time setup", icon="IMPORT")
            box.label(text=f"llama.cpp: {'installed' if s['llama'] else 'will be installed (~30 MB)'}",
                      icon="CHECKMARK" if s["llama"] else "BLANK1")
            box.label(text=f"Model: {'ready' if s['model'] else f'will be downloaded ({runtime.MODEL_SIZE_GB} GB)'}",
                      icon="CHECKMARK" if s["model"] else "BLANK1")
            box.operator(BAI_OT_setup.bl_idname, text="Set up", icon="IMPORT")
            return False
        row = layout.row(align=True)
        starting = "llm" in _WAITING
        row.label(text="AI: starting..." if starting else "AI: running" if s["llm_up"] else "AI: starts on first command"
                  if prefs.auto_start else "AI: stopped", icon="SOLO_ON" if s["llm_up"] else "SOLO_OFF")
        if s["llm_up"] or starting:
            row.operator(BAI_OT_server.bl_idname, text="Stop", icon="PAUSE").start = False
        else:
            row.operator(BAI_OT_server.bl_idname, text="Start", icon="PLAY").start = True
        row.operator(BAI_OT_open_folder.bl_idname, text="", icon="FILE_FOLDER")
        if not s["voice"] and not s["voice_up"]:
            layout.operator(BAI_OT_setup.bl_idname, text="Install voice (~230 MB)", icon="REC").voice = True
        return True


class BAI_PT_learning(bpy.types.Panel):
    bl_label = "Learned shortcuts"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Shilpi"
    bl_parent_id = "BAI_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        items = corrections.shortcuts()
        if not items:
            layout.label(text="None yet. Ctrl+Z a wrong answer, then do it yourself.")
        for key, calls in items.items():
            row = layout.row()
            row.label(text=f"'{key}' -> " + ", ".join(c["op"] for c in calls))
            row.operator(BAI_OT_forget_shortcut.bl_idname, text="", icon="TRASH").key = key
        layout.operator(BAI_OT_export_corrections.bl_idname, icon="EXPORT")


def _on_undo(*_):
    corrections.TRACKER.on_undo()


def _poll_corrections():
    try:
        message = corrections.TRACKER.poll(_prefs(bpy.context).learn)
        if message:
            bpy.context.window_manager.bai.last_status = message
    except (KeyError, AttributeError):
        pass
    return 1.0


CLASSES = (BAI_Preferences, BAI_State, BAI_OT_setup, BAI_OT_cancel_setup, BAI_OT_server, BAI_OT_open_folder, BAI_OT_run,
           BAI_OT_answer, BAI_OT_listen, BAI_OT_feedback, BAI_OT_forget_shortcut, BAI_OT_export_corrections,
           BAI_PT_panel, BAI_PT_learning)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.bai = bpy.props.PointerProperty(type=BAI_State)
    bpy.app.handlers.undo_post.append(_on_undo)
    bpy.app.timers.register(_poll_corrections, first_interval=1.0, persistent=True)
    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig:  # Alt+Shift+V in the 3D View: talk without opening the sidebar (Alt+V is Rip in Edit Mode)
        km = keyconfig.keymaps.new(name="3D View", space_type="VIEW_3D")
        _KEYMAPS.append((km, km.keymap_items.new(BAI_OT_listen.bl_idname, "V", "PRESS", alt=True, shift=True)))


_KEYMAPS = []


def unregister():
    for km, kmi in _KEYMAPS:
        km.keymap_items.remove(kmi)
    _KEYMAPS.clear()
    runtime.JOB.update(cancelled=True)
    runtime.stop_all()  # never leave a model server holding GPU memory after the add-on is gone
    _WAITING.clear()
    for timer in (_poll_speech, _watch_servers, _watch_job):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    if _on_undo in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(_on_undo)
    if bpy.app.timers.is_registered(_poll_corrections):
        bpy.app.timers.unregister(_poll_corrections)
    del bpy.types.WindowManager.bai
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)

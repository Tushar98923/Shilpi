"""Learning from corrections (plan section 2b). Local only; nothing leaves the PC unless exported.

How a correction is caught:
1. The add-on runs an AI command and remembers it (request, scene, calls).
2. The user presses Ctrl+Z within two minutes (undo_post handler), or clicks the thumbs-down button.
3. The operators the user then runs by hand (Blender's operator history, window_manager.operators)
   are the right answer for that request in that scene. Once they stop for a few seconds, the
   correction is saved to corrections.jsonl in Blender's config folder.

Shortcuts: when the same request (normalised) has been corrected to the same calls twice, it
becomes a rule that runs without the model, as long as every object it names still exists.

Operators run from scripts are not recorded in the operator history, so step 3 only works with real
user input; it can't be exercised by an automated test.
"""
import json
import re
import time
from pathlib import Path

import bpy

UNDO_WINDOW_S = 120     # Ctrl+Z this soon after an AI command marks it wrong
SETTLE_S = 6            # the correction is saved once the user has stopped for this long
GIVE_UP_S = 180         # forget a pending correction after this long without any manual operator
SHORTCUT_AFTER = 2      # same request corrected the same way this many times -> shortcut

# Operators that are navigation, UI or selection clicks, not the "answer".
IGNORED_PREFIXES = ("view3d.rotate", "view3d.move", "view3d.zoom", "view3d.view_", "view3d.navigate", "view3d.dolly",
                    "screen.", "wm.", "ui.", "ed.undo", "ed.redo", "bai.", "outliner.", "file.", "buttons.",
                    "view2d.", "anim.change_frame", "object.select_", "view3d.select", "mesh.select_", "info.")


def _store_dir() -> Path:
    folder = Path(bpy.utils.user_resource("CONFIG", path="shilpi"))
    old = Path(bpy.utils.user_resource("CONFIG", path="blender_ai"))  # before the project was named Shilpi
    if old.is_dir() and not folder.exists():
        old.rename(folder)  # keep the corrections and shortcuts learned so far
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def store_path() -> Path:
    return _store_dir() / "corrections.jsonl"


def disabled_path() -> Path:
    return _store_dir() / "disabled_shortcuts.json"


def normalize(request: str) -> str:
    return " ".join(re.findall(r"[a-z0-9.]+", request.lower()))


def _plain(value):
    """Operator property values -> JSON (vectors, colours and sets become lists)."""
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "bl_rna"):  # macro sub-operator properties, e.g. TRANSFORM_OT_translate
        return _set_properties(value)
    try:
        return [_plain(v) for v in value]
    except TypeError:
        return str(value)


def _set_properties(props) -> dict:
    out = {}
    for prop in props.bl_rna.properties:
        name = prop.identifier
        if name != "rna_type" and props.is_property_set(name):
            out[name] = _plain(getattr(props, name))
    return out


def operator_to_call(op) -> dict | None:
    """A window_manager.operators entry -> an op/1 call, or None for navigation/selection/UI."""
    category, _, name = op.bl_idname.partition("_OT_")
    op_name = f"{category.lower()}.{name}"
    if not name or op_name.startswith(IGNORED_PREFIXES):
        return None
    return {"op": op_name, "args": _set_properties(op.properties)}


def log_command(request: str, scene: dict, reply: str, calls, status: str) -> None:
    """Every command typed, its scene and what happened (local only). With the Right/Wrong buttons, this is
    how real human commands become a test set: export it and they can be scored like any eval item."""
    with (_store_dir() / "commands.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"request": request, "scene": scene, "reply": reply, "calls": calls, "status": status,
                            "time": time.time()}) + "\n")


def mark_last_logged(verdict: str) -> None:
    path = _store_dir() / "commands.jsonl"
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines:
        row = json.loads(lines[-1])
        row["verdict"] = verdict
        lines[-1] = json.dumps(row)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load() -> list[dict]:
    path = store_path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def save(row: dict) -> None:
    with store_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def disabled() -> set:
    path = disabled_path()
    return set(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else set()


def disable(key: str) -> None:
    keys = disabled() | {key}
    disabled_path().write_text(json.dumps(sorted(keys)), encoding="utf-8")


def shortcuts() -> dict:
    """{normalised request: calls} for requests corrected the same way SHORTCUT_AFTER times."""
    counts = {}
    for row in load():
        if row.get("source") != "manual_fix" or not row.get("right"):
            continue
        key = (normalize(row["request"]), json.dumps(row["right"], sort_keys=True))
        counts[key] = counts.get(key, 0) + 1
    off = disabled()
    return {request: json.loads(calls) for (request, calls), n in counts.items() if n >= SHORTCUT_AFTER and request not in off}


def shortcut_for(request: str, scene_names: set) -> list | None:
    calls = shortcuts().get(normalize(request))
    if not calls:
        return None
    named = {n for c in calls for key in ("objects", "object") for n in ([c["args"][key]] if isinstance(c["args"].get(key), str)
                                                                          else c["args"].get(key, []) or [])}
    return calls if named <= scene_names else None  # a shortcut that names a missing object would fail


class Tracker:
    """Remembers the last AI command and turns a Ctrl+Z + manual fix into a saved correction."""

    def __init__(self):
        self.last = None

    def ai_ran(self, request: str, scene: dict, calls: list) -> None:
        self.last = {"request": request, "scene": scene, "calls": calls, "time": time.time(),
                     "wrong": False, "mark": None, "last_op_time": None, "seen": 0}

    def mark_wrong(self) -> bool:
        if self.last is None or self.last["wrong"]:
            return False
        self.last.update(wrong=True, wrong_time=time.time(), mark=len(bpy.context.window_manager.operators),
                         selection=sorted(o.name for o in bpy.context.selected_objects))
        return True

    def on_undo(self) -> None:
        if self.last and not self.last["wrong"] and time.time() - self.last["time"] < UNDO_WINDOW_S:
            self.mark_wrong()

    def poll(self, learning_enabled: bool) -> str | None:
        """Called every second. Returns a status message when a correction was saved."""
        last = self.last
        if not (last and last["wrong"] and learning_enabled):
            return None
        ops = list(bpy.context.window_manager.operators)[last["mark"]:]
        calls = [c for c in (operator_to_call(op) for op in ops) if c]
        now = time.time()
        if len(calls) != last["seen"]:
            last.update(seen=len(calls), last_op_time=now)
            return None
        if not calls:
            if now - last["wrong_time"] > GIVE_UP_S:
                self.last = None
            return None
        if now - last["last_op_time"] < SETTLE_S:
            return None
        selected_now = sorted(o.name for o in bpy.context.selected_objects)
        if selected_now and selected_now != last["scene"]["selected"]:
            calls = [{"op": "bai.select", "args": {"objects": selected_now}}] + calls
        save({"request": last["request"], "scene": last["scene"], "wrong": last["calls"], "right": calls,
              "source": "manual_fix", "time": now})
        self.last = None
        return f"Learned: '{last['request']}' -> " + ", ".join(c["op"] for c in calls)

    def confirm(self) -> bool:
        """Thumbs up: the last answer was right (kept as a positive example, not a shortcut)."""
        if self.last is None:
            return False
        save({"request": self.last["request"], "scene": self.last["scene"], "right": self.last["calls"],
              "source": "thumbs_up", "time": time.time()})
        self.last = None
        return True


TRACKER = Tracker()

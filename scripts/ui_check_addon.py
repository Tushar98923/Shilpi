"""End-to-end test of the add-on in a real Blender window (opens one for ~30 s).

Loads addon/shilpi from source, then drives bai.run with canned model answers (so each
behaviour is tested exactly) and finally with the live v5 model, which the add-on starts by itself.
Uses a temporary Blender config folder, so your real preferences and corrections are untouched.

Usage: python scripts/ui_check_addon.py     (with no model server running: the live part checks that the
add-on starts one with models/operator/qwen3.5-2b-operator-v5-q8_0.gguf and stops it when Blender quits)
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.headless_validator import _blender_executable  # noqa: E402

IN_BLENDER = r'''
import json, sys, time, bpy
sys.path.insert(0, ADDON_PARENT)
import shilpi as addon
from shilpi import corrections
addon.register()
OUT = open(RESULTS, "a", encoding="utf-8")
canned = {"reply": None}
addon._chat = lambda url, msg, constrained: canned["reply"]
addon._ensure = lambda kind, prefs, force=False: "ready"  # canned part: no server needed
LIVE_CHAT = None

def ctx():
    w = bpy.context.window_manager.windows[0]
    a = next(a for a in w.screen.areas if a.type == "VIEW_3D")
    return {"window": w, "area": a, "region": next(r for r in a.regions if r.type == "WINDOW")}

def run(request, reply=None):
    bpy.context.window_manager.bai.request = request
    canned["reply"] = None if reply is None else json.dumps({"calls": reply})
    with bpy.context.temp_override(**ctx()):
        result = bpy.ops.bai.run()
        # A button click gives a FINISHED operator its undo step; a call from a script doesn't, so add it here.
        if "FINISHED" in result:
            bpy.ops.ed.undo_push(message="AI: " + request)
    return bpy.context.window_manager.bai.last_status

def objs():
    return {o.name: [m.type for m in o.modifiers] for o in bpy.context.scene.objects}

def select(*names):
    for o in bpy.context.scene.objects:
        o.select_set(o.name in names)
    bpy.context.view_layer.objects.active = bpy.data.objects[names[0]] if names else None

def record(name, ok, detail):
    OUT.write(json.dumps({"test": name, "ok": bool(ok), "detail": str(detail)[:200]}) + "\n"); OUT.flush()

def steps():
    yield
    select("Cube")
    s = run("bevel it", [{"op": "object.modifier_add", "args": {"type": "BEVEL"}}])
    record("normal command", objs()["Cube"] == ["BEVEL"], s); yield
    select()
    s = run("shade smooth", [{"op": "object.shade_smooth", "args": {}}])
    record("guard: nothing selected", s.startswith("Nothing is selected"), s); yield
    s = run("subsurf the cube", [{"op": "bai.select", "args": {"objects": ["Cube"]}}, {"op": "object.modifier_add", "args": {"type": "SUBSURF"}}])
    record("guard lets a select-first answer through", "SUBSURF" in objs()["Cube"], s); yield
    before = objs()
    s = run("bevel the cube", [{"op": "bai.ask", "args": {"question": "Which one do you mean: Cube or Cube.001?"}}])
    record("ask: shows question, runs nothing", s.startswith("Question:") and objs() == before, s); yield
    s = run("what's the weather", [{"op": "bai.decline", "args": {"reason": "That's outside what I can do in Blender."}}])
    record("decline: shows reason, runs nothing", s.startswith("Can't do that") and objs() == before, s); yield
    select("Cube")
    s = run("add a mirror and select the ghost", [{"op": "object.modifier_add", "args": {"type": "MIRROR"}}, {"op": "bai.select", "args": {"objects": ["Ghost"]}}])
    record("partial failure keeps an undo step and says so", s.startswith("Partly done") and "MIRROR" in objs()["Cube"], s); yield
    select("Cube")
    s = run("add a solidify", [{"op": "object.modifier_add", "args": {"type": "SOLIDIFY"}}])
    yield
    s = run("undo that", [{"op": "ed.undo", "args": {}}])
    yield; yield; yield
    s = bpy.context.window_manager.bai.last_status
    record("undo runs on the next tick without crashing", "SOLIDIFY" not in objs()["Cube"], f"{s} | {objs()['Cube']}")
    record("undo right after an AI command marks it wrong", corrections.TRACKER.last and corrections.TRACKER.last["wrong"], corrections.TRACKER.last); yield
    corrections.TRACKER.last = None
    fix = [{"op": "object.modifier_add", "args": {"type": "BEVEL"}}]
    for _ in range(2):
        corrections.save({"request": "round off the edges", "scene": {}, "wrong": [], "right": fix, "source": "manual_fix", "time": 0})
    select("Cube")
    bpy.data.objects["Cube"].modifiers.clear()
    def no_model(*a):
        raise AssertionError("model was called")
    addon._chat = no_model
    s = run("Round off the edges!")
    record("shortcut after two identical corrections, no model call", objs()["Cube"] == ["BEVEL"] and "shortcut" in s, s); yield
    corrections.disable("round off the edges")
    record("forgotten shortcut is not used", "round off the edges" not in corrections.shortcuts(), corrections.shortcuts()); yield
    # live model, constrained decoding. No server is running: the first command starts one with the v5 model.
    addon._chat = LIVE
    addon._ensure = ENSURE_LIVE
    addon._DefaultPrefs.model_path = MODEL
    bpy.data.objects["Cube"].modifiers.clear()
    select("Cube")
    try:
        s = run("give the cube a wireframe modifier")
        record("auto-start: first command starts the model server", s.startswith("Starting the AI model"), s); yield
        for _ in range(400):  # up to ~160 s
            if "llm" not in addon._WAITING:
                break
            yield
        s = bpy.context.window_manager.bai.last_status
        record("live: queued command runs once the server is up (wireframe)", "WIREFRAME" in objs()["Cube"], s); yield
        select()
        s = run("add a uv sphere at 0, 0, 3")
        sphere = [o for o in bpy.context.scene.objects if o.name.startswith("Sphere")]
        record("live: add sphere at a location", sphere and [round(v, 2) for v in sphere[0].location] == [0, 0, 3], s); yield
        select()
        s = run("delete the light")
        record("live: nothing selected, names the light -> select + delete", "Light" not in objs(), s); yield
        # Look-alikes: the model asks which one; answering selects it and re-runs the command.
        cube = bpy.data.objects["Cube"]
        twin = cube.copy(); twin.data = cube.data.copy(); bpy.context.collection.objects.link(twin)
        twin.location.x += 3
        cube.modifiers.clear(); twin.modifiers.clear()
        state = bpy.context.window_manager.bai
        select()
        s = run("make the cube twice as big")
        record("live ask: question and both choices stored", state.pending_options.split("\n") == ["Cube", twin.name], s); yield
        with bpy.context.temp_override(**ctx()):
            bpy.ops.bai.answer(choice=twin.name)
        record("live ask: answer button runs it on that one", round(twin.scale.x, 2) == 2 and round(cube.scale.x, 2) == 1
               and not state.pending_question, f"{state.last_status} | {tuple(cube.scale)} {tuple(twin.scale)}"); yield
        select()
        s = run("add a bevel modifier to the cube")
        s = run("um the first one") if state.pending_question else s
        record("live ask: typed/spoken 'the first one' answers it", objs()["Cube"] == ["BEVEL"] and objs()[twin.name] == [], s); yield
    except Exception as e:
        record("live model", False, f"{type(e).__name__}: {e}")
    OUT.close()
    bpy.ops.wm.quit_blender()

LIVE = addon.__dict__["_chat_live"]
ENSURE_LIVE = addon.__dict__["_ensure_live"]
GEN = steps()
def tick():
    try:
        next(GEN)
    except StopIteration:
        return None
    except Exception:
        import traceback
        record("test script crashed", False, traceback.format_exc()[-400:])
        OUT.close()
        bpy.ops.wm.quit_blender()
        return None
    return 0.4
bpy.app.timers.register(tick, first_interval=1.5)
'''


def _answers(url):
    import urllib.request
    try:
        return urllib.request.urlopen(url, timeout=1).status == 200
    except OSError:
        return False


def main():
    if _answers("http://127.0.0.1:8080/health"):
        sys.exit("A model server is already running on port 8080; stop it first (this test starts its own)")
    with tempfile.TemporaryDirectory(prefix="bai_addon_test_") as tmp:
        results = Path(tmp) / "results.jsonl"
        results.touch()
        script = Path(tmp) / "test.py"
        header = (f"ADDON_PARENT = {str(ROOT / 'addon')!r}\nRESULTS = {str(results)!r}\n"
                  "import importlib, sys\nsys.path.insert(0, ADDON_PARENT)\nimport shilpi as _a\n"
                  "_a._chat_live = _a._chat\n_a._ensure_live = _a._ensure\n"
                  f"MODEL = {str(ROOT / 'models' / 'operator' / 'qwen3.5-2b-operator-v5-q8_0.gguf')!r}\n")
        script.write_text(header + IN_BLENDER, encoding="utf-8")
        env = {**os.environ, "BLENDER_USER_CONFIG": str(Path(tmp) / "config")}
        log = Path(tmp) / "blender.log"
        try:
            with log.open("w", encoding="utf-8", errors="replace") as f:
                subprocess.run([_blender_executable(), "--factory-startup", "--python", str(script)], env=env,
                               stdout=f, stderr=subprocess.STDOUT, timeout=300)
        except subprocess.TimeoutExpired:
            print("Blender didn't finish; last log lines:\n" + log.read_text(encoding="utf-8", errors="replace")[-1500:])
        rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line]
        if not any(r["test"].startswith("live ask: typed") for r in rows):  # stopped before the last check
            print("Blender stopped early; last log lines:\n" + log.read_text(encoding="utf-8", errors="replace")[-2500:])
    rows.append({"test": "Blender quit stops the model server it started", "detail": "port 8080 still answers",
                 "ok": not _answers("http://127.0.0.1:8080/health")})
    for row in rows:
        print(f"  {'PASS' if row['ok'] else 'FAIL'} {row['test']:52} {'' if row['ok'] else row['detail']}")
    print(f"\n{sum(r['ok'] for r in rows)}/{len(rows)} add-on checks passed")


if __name__ == "__main__":
    main()

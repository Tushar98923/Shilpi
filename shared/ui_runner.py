"""In-Blender side of the UI harness: runs op/1 calls in a real 3D viewport. Do not import outside Blender.

The headless harness (blender_runner.py) can't run viewport operators (views, snapping, local view,
undo, render...). This runs Blender WITH its interface, executes each job inside a 3D View context
from a timer, records the UI state before and after, writes one JSON line per job and quits.

Launched by shared/ui_harness.py as:
    blender --factory-startup --python ui_runner.py -- <job_file.json>
Job: {"id", "setup": bpy code | null, "calls": [...], "prep": bpy code run after setup | null}
"""
import json
import os
import sys
import traceback

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import op_executor  # noqa: E402
import scene_serializer  # noqa: E402

# Undo/redo reload Blender's data: any Python reference to areas, objects or the context dict made before
# them is dangling afterwards (holding one crashed Blender). They run with only the window overridden and
# everything is looked up again afterwards.
DATA_RELOADING_OPS = {"ed.undo", "ed.redo", "ed.undo_history"}

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
SPEC = json.load(open(ARGS[0], encoding="utf-8"))
OUT = open(SPEC["results_path"], "a", encoding="utf-8")


def view3d_context():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                return {"window": window, "screen": window.screen, "area": area,
                        "region": next(r for r in area.regions if r.type == "WINDOW")}
    return None


def ui_state(ctx):
    space = ctx["area"].spaces.active if ctx and ctx["area"].type == "VIEW_3D" else None
    r3d = space.region_3d if space else None
    scene = bpy.context.scene
    render = bpy.data.images.get("Render Result")
    return {
        "view_perspective": r3d.view_perspective if r3d else None,
        "view_rotation": [round(v, 3) for v in r3d.view_rotation] if r3d else None,
        "view_distance": round(r3d.view_distance, 3) if r3d else None,
        "view_location": [round(v, 3) for v in r3d.view_location] if r3d else None,
        "shading": space.shading.type if space else None,
        "xray": bool(space.shading.show_xray) if space else None,
        "local_view": bool(space.local_view) if space else None,
        "cursor": [round(v, 3) for v in scene.cursor.location],
        "playing": bool(bpy.context.screen.is_animation_playing) if bpy.context.screen else None,
        "screen": ctx["window"].screen.name if ctx else None,
        "render_result": bool(render and render.has_data),
        "objects": sorted(o.name for o in scene.objects),
        "locations": {o.name: [round(v, 3) for v in o.location] for o in scene.objects},
        "camera": scene.camera.name if scene.camera else None,
        "hidden": sorted(o.name for o in scene.objects if o.hide_get()),
        "saved": bool(bpy.data.is_saved),
        "windows": len(bpy.context.window_manager.windows),
        "render_running": bool(bpy.app.is_job_running("RENDER")),
        "selected": sorted(o.name for o in bpy.context.selected_objects) if bpy.context.view_layer else [],
        "scene": [{k: o[k] for k in ("name", "data", "modifiers", "constraints")}
                  for o in scene_serializer.serialize_scene()["objects"]],
    }


def reset(ctx):
    with bpy.context.temp_override(**ctx):
        if bpy.context.object and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        if bpy.context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel()
        space = ctx["area"].spaces.active
        if space.local_view:
            bpy.ops.view3d.localview()
        space.shading.type, space.shading.show_xray = "SOLID", False
        space.region_3d.view_perspective = "PERSP"


def run_job(job):
    """A generator: each `yield` hands control back to Blender for one timer tick.

    Setup, the calls and the after-state each get their own tick. Undo/redo crash Blender when run in the
    same tick as the change they undo, and they reload data, so they run with only the window overridden.
    """
    result = {"id": job["id"], "ok": False, "error": None}
    try:
        ctx = view3d_context()
        reset(ctx)
        with bpy.context.temp_override(**ctx):
            if job.get("setup"):
                exec(compile(job["setup"], "<setup>", "exec"), {"__name__": "__main__"})
            if job.get("prep"):
                exec(compile(job["prep"], "<prep>", "exec"), {"__name__": "__main__"})
        yield
        ctx = view3d_context()
        result["before"] = ui_state(ctx)
        if any(c["op"] in DATA_RELOADING_OPS for c in job["calls"]):
            with bpy.context.temp_override(window=bpy.context.window_manager.windows[0]):
                op_executor.apply_calls(job["calls"])
        else:
            with bpy.context.temp_override(**ctx):
                op_executor.apply_calls(job["calls"])
        yield
        result["after"] = ui_state(view3d_context())
        result["ok"] = True
    except op_executor.AskUser as e:
        result.update(ok=True, asked=e.question)
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-600:]
    OUT.write(json.dumps(result) + "\n")
    OUT.flush()


def all_jobs():
    for job in SPEC["jobs"]:
        yield from run_job(job)
    OUT.close()
    bpy.ops.wm.quit_blender()


STEPS = all_jobs()


def tick():
    try:
        next(STEPS)
    except StopIteration:
        return None
    return 0.2


bpy.app.timers.register(tick, first_interval=1.5)

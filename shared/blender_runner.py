"""In-Blender side of the verification harness. Do not import this outside Blender.

Launched by shared/headless_validator.py as:
    blender --background --factory-startup --python blender_runner.py -- <job_file.json>

The job file holds {"jobs": [...], "results_path": str, "thumbnail_size": int}.
Each job is run on a fresh factory-startup scene:

    {"id": str,
     "kind": "script" | "ops",
     "code": str,                 # kind=script: the candidate bpy script
     "calls": [...],              # kind=ops: op/1 calls, already format-validated
     "setup": str | null,         # optional bpy code that builds the starting scene
     "sanity": str | null,        # optional legacy snippet printing __VALIDATION_RESULT__:PASS|FAIL:...
     "thumbnail": str | null}     # optional PNG path

One JSON line per job is appended to results_path as soon as the job
finishes, so if Blender hangs or crashes mid-batch the host knows exactly
which job was in flight and restarts from the next one.
"""
import contextlib
import io
import json
import linecache
import math
import os
import sys
import traceback

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import op_executor  # noqa: E402
import scene_serializer  # noqa: E402

RESULT_MARKER = "__VALIDATION_RESULT__:"
MAX_STDOUT = 4000


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=False)


def _candidate_traceback(exc: BaseException, filename: str) -> str:
    """Traceback trimmed to frames from the candidate code, so training pairs carry the real error."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == filename]
    lines = [f'  line {f.lineno}, in {f.name}\n    {f.line or ""}' for f in frames[-3:]]
    return "\n".join(lines + [f"{type(exc).__name__}: {exc}"])


def _exec(code: str, filename: str):
    # Register the source so tracebacks can show the failing line's text.
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    namespace = {"__name__": "__main__", "__file__": filename}
    exec(compile(code, filename, "exec"), namespace)


def _render_thumbnail(path: str, size: int):
    scene = bpy.context.scene
    points = []
    for obj in scene.objects:
        if obj.type in {"MESH", "CURVE", "SURFACE", "FONT", "META"} and obj.visible_get():
            points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if points:
        lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
        hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
        center, radius = (lo + hi) / 2, max((hi - lo).length / 2, 0.5)
    else:
        center, radius = Vector((0, 0, 0)), 2.0

    cam_data = bpy.data.cameras.new("__bai_thumb_cam")
    cam_data.lens = 50
    cam_data.clip_end = max(1000.0, radius * 20)
    cam = bpy.data.objects.new("__bai_thumb_cam", cam_data)
    scene.collection.objects.link(cam)
    direction = Vector((1.0, -1.0, 0.8)).normalized()
    distance = radius / math.sin(math.radians(18)) * 1.1
    cam.location = center + direction * distance
    cam.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()

    previous_camera = scene.camera
    render = scene.render
    render.engine = "BLENDER_WORKBENCH"
    render.resolution_x = render.resolution_y = size
    render.resolution_percentage = 100
    render.image_settings.file_format = "PNG"
    render.filepath = path
    scene.camera = cam
    try:
        bpy.ops.render.render(write_still=True)
    finally:
        scene.camera = previous_camera
        bpy.data.objects.remove(cam)
        bpy.data.cameras.remove(cam_data)


def run_job(job: dict, thumbnail_size: int) -> dict:
    result = {
        "id": job["id"],
        "ok": False,
        "stage": None,
        "error": None,
        "traceback": None,
        "scene_before": None,
        "scene_after": None,
        "diff": None,
        "thumbnail": None,
        "stdout": "",
    }
    _reset_scene()

    if job.get("setup"):
        try:
            _exec(job["setup"], "<setup>")
        except Exception as e:  # noqa: BLE001 -- anything the setup raises is a harness-side bug worth reporting
            result.update(stage="setup", error=f"{type(e).__name__}: {e}", traceback=_candidate_traceback(e, "<setup>"))
            return result

    result["scene_before"] = scene_serializer.serialize_scene()

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            if job["kind"] == "ops":
                op_executor.apply_calls(job["calls"])
            else:
                _exec(job["code"], "<candidate>")
    except op_executor.OpExecutionError as e:
        result.update(stage="exec", error=str(e))
    except BaseException as e:  # noqa: BLE001 -- includes SystemExit from candidate code
        result.update(stage="exec", error=f"{type(e).__name__}: {e}", traceback=_candidate_traceback(e, "<candidate>"))

    try:
        result["scene_after"] = scene_serializer.serialize_scene()
        result["diff"] = scene_serializer.diff_scenes(result["scene_before"], result["scene_after"])
    except Exception as e:  # noqa: BLE001 -- e.g. candidate left the scene in a state we cannot read
        result.update(stage=result["stage"] or "serialize", error=result["error"] or f"serializer failed: {e}")

    if result["error"] is None and job.get("sanity"):
        sanity_out = io.StringIO()
        try:
            with contextlib.redirect_stdout(sanity_out):
                _exec(job["sanity"], "<sanity>")
        except Exception as e:  # noqa: BLE001
            result.update(stage="sanity", error=f"sanity check raised {type(e).__name__}: {e}")
        else:
            verdicts = [line[len(RESULT_MARKER):].strip() for line in sanity_out.getvalue().splitlines() if line.startswith(RESULT_MARKER)]
            if not verdicts:
                result.update(stage="sanity", error="sanity check did not report a result")
            elif verdicts[-1] != "PASS":
                result.update(stage="sanity", error=verdicts[-1])

    if result["error"] is None and job.get("thumbnail"):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _render_thumbnail(job["thumbnail"], thumbnail_size)
            result["thumbnail"] = job["thumbnail"]
        except Exception as e:  # noqa: BLE001 -- a failed thumbnail should not fail the pair
            result["thumbnail_error"] = f"{type(e).__name__}: {e}"

    result["stdout"] = stdout.getvalue()[-MAX_STDOUT:]
    result["ok"] = result["error"] is None
    return result


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    with open(argv[0], encoding="utf-8") as f:
        spec = json.load(f)

    with open(spec["results_path"], "a", encoding="utf-8") as out:
        header = {"id": "__runner__", "blender": bpy.app.version_string.split(" ")[0]}
        out.write(json.dumps(header) + "\n")
        out.flush()
        for job in spec["jobs"]:
            result = run_job(job, spec.get("thumbnail_size", 128))
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()


main()

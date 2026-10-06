"""Run every viewport-only shortlist operator in a real Blender window and check it did what it should.

These operators can't run in the headless harness, so Way 1 trusts their code-written answers. This
checks that those answers actually work from the add-on's point of view (inside a 3D View).

Usage: python scripts/ui_check_viewport.py      (opens a Blender window for ~a minute)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "domains" / "operator_agent"))

from shared.ui_harness import run_ui_jobs

from operator_shortlist import SETUPS

SHORTLIST = json.loads((ROOT / "shared" / "data" / "operator_shortlist.json").read_text(encoding="utf-8"))["operators"]

ADD_AND_PUSH = "import bpy\nbpy.ops.mesh.primitive_cube_add(location=(5, 5, 5))\nbpy.context.active_object.name = 'Undo_Me'\nbpy.ops.ed.undo_push(message='add')\n"
PREP = {
    "ed.undo": ADD_AND_PUSH,
    "ed.redo": ADD_AND_PUSH + "bpy.ops.ed.undo()\n",
    "screen.animation_cancel": "import bpy\nbpy.ops.screen.animation_play()\n",
    "object.hide_view_clear": "import bpy\nbpy.data.objects['Sphere'].hide_set(True)\n",
    "view3d.view_selected": "import bpy\nbpy.context.region_data.view_location = (20, 20, 0)\n",
    "render.render": "import bpy\nr = bpy.context.scene.render\nr.resolution_x = r.resolution_y = 64\nr.engine = 'BLENDER_WORKBENCH'\n",
    "render.opengl": "import bpy\nr = bpy.context.scene.render\nr.resolution_x = r.resolution_y = 64\n",
}


def expectation(op, before, after):
    """None if the operator did its job, else what's wrong."""
    def changed(*keys):
        return any(before[k] != after[k] for k in keys)
    checks = {
        "view3d.view_axis": lambda: changed("view_rotation"),
        "view3d.view_all": lambda: changed("view_location", "view_distance"),
        "view3d.view_selected": lambda: changed("view_location", "view_distance"),
        "view3d.view_camera": lambda: after["view_perspective"] == "CAMERA",
        "view3d.view_persportho": lambda: changed("view_perspective"),
        "view3d.toggle_xray": lambda: changed("xray"),
        "view3d.toggle_shading": lambda: after["shading"] == "WIREFRAME",
        "view3d.localview": lambda: after["local_view"],
        "view3d.snap_cursor_to_selected": lambda: changed("cursor"),
        "view3d.snap_cursor_to_center": lambda: after["cursor"] == [0, 0, 0],
        "view3d.snap_cursor_to_active": lambda: changed("cursor"),
        "view3d.snap_selected_to_cursor": lambda: changed("locations"),
        "view3d.snap_selected_to_grid": lambda: changed("locations"),
        "view3d.snap_selected_to_active": lambda: changed("locations"),
        "view3d.camera_to_view": lambda: changed("locations"),
        "view3d.camera_to_view_selected": lambda: changed("locations"),
        "view3d.object_as_camera": lambda: changed("camera"),
        "object.hide_view_clear": lambda: not after["hidden"],
        "ed.undo": lambda: "Undo_Me" not in after["objects"],
        "ed.redo": lambda: "Undo_Me" in after["objects"],
        "screen.animation_play": lambda: after["playing"],
        "screen.animation_cancel": lambda: not after["playing"],
        "screen.screen_full_area": lambda: changed("screen"),
        "render.render": lambda: after["render_result"] or after["render_running"] or changed("windows"),
        "render.opengl": lambda: after["render_result"] or after["render_running"] or changed("windows"),
        "wm.save_mainfile": lambda: after["saved"] or changed("windows"),  # unsaved file: the Save dialog opens
        "object.select_grouped": lambda: changed("selected"),
        "sculpt.symmetrize": lambda: changed("scene"),
        "paint.mask_flood_fill": lambda: changed("scene"),
        "sculpt.face_sets_init": lambda: changed("scene"),
    }
    check = checks.get(op, lambda: before != after)
    return None if check() else "ran, but the expected change didn't happen"


def main():
    ops = {op: e for op, e in SHORTLIST.items() if e["status"] == "viewport"}
    jobs = [{"id": op, "setup": SETUPS[e["setup"]], "prep": PREP.get(op), "calls": [{"op": op, "args": e["args"]}]}
            for op, e in ops.items()]
    print(f"Running {len(jobs)} viewport-only operators in a Blender window...", flush=True)
    # Undo/redo get a fresh Blender each: after a long session of mixed operations (sculpt mode in
    # particular), calling ed.undo from a script crashed Blender, while it works on a normal history.
    alone = [j for j in jobs if j["id"] in ("ed.undo", "ed.redo")]
    together = [j for j in jobs if j not in alone]
    results = run_ui_jobs(together) + [r for j in alone for r in run_ui_jobs([j])]
    jobs = together + alone
    passed, report = 0, {}
    for job, result in zip(jobs, results):
        op = job["id"]
        if not result["ok"]:
            status = f"ERROR {result['error'][:110]}"
        elif result.get("asked"):
            status = f"asked {result['asked']}"
        else:
            problem = expectation(op, result["before"], result["after"])
            status = "ok" if problem is None else problem
        passed += status == "ok"
        report[op] = status
        print(f"  {'PASS' if status == 'ok' else 'FAIL'} {op:36} {'' if status == 'ok' else status}")
    print(f"\n{passed}/{len(jobs)} viewport operators work from a 3D View")
    (ROOT / "shared" / "data" / "ui_check_viewport.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

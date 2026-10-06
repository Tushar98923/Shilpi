"""UI harness (host side): run op/1 calls in Blender WITH its interface, in a real 3D viewport.

For what the headless harness can't do: viewport operators (views, shading, snapping, local view),
undo/redo, playback, render, save. Opens a Blender window for a few seconds, so it only works on a
machine with a display.

    from shared.ui_harness import run_ui_jobs
    results = run_ui_jobs([{"id": "top", "setup": None, "calls": [{"op": "view3d.view_axis", "args": {"type": "TOP"}}]}])
"""
import json
import subprocess
import tempfile
from pathlib import Path

from shared.headless_validator import _blender_executable

RUNNER = Path(__file__).with_name("ui_runner.py")


def run_ui_jobs(jobs: list[dict], timeout: int = 600) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="bai_ui_") as tmp:
        job_file = Path(tmp) / "jobs.json"
        results_path = Path(tmp) / "results.jsonl"
        results_path.touch()
        job_file.write_text(json.dumps({"jobs": jobs, "results_path": str(results_path)}), encoding="utf-8")
        log = Path(tmp) / "blender.log"
        with log.open("w", encoding="utf-8", errors="replace") as log_file:
            try:
                subprocess.run([_blender_executable(), "--factory-startup", "--python", str(RUNNER), "--", str(job_file)],
                               stdout=log_file, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        done = {r["id"]: r for r in map(json.loads, results_path.read_text(encoding="utf-8").splitlines()) if r}
        tail = log.read_text(encoding="utf-8", errors="replace")[-800:]
    return [done.get(job["id"], {"id": job["id"], "ok": False, "error": "not run (Blender exited early)", "log": tail})
            for job in jobs]

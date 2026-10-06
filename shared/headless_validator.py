"""Headless verification harness (host side).

Runs candidate bpy scripts or op/1 calls through Blender in background mode
via shared/blender_runner.py, and returns for each one: pass/fail, the error
and trimmed traceback, the scene state before and after (scene/1 JSON), their
diff, and optionally a thumbnail render. A training pair is kept only if it
passes.

Many jobs share one Blender process (startup costs ~2s, a typical job ~50ms).
If a job hangs or crashes Blender, it is marked failed and a fresh Blender
picks up from the next job.
"""
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from shared.output_formats import PINNED_BLENDER_VERSION

load_dotenv()

RESULT_MARKER = "__VALIDATION_RESULT__:"
RUNNER = Path(__file__).with_name("blender_runner.py")


@dataclass
class ValidationResult:
    ok: bool
    error: str | None = None
    stdout: str = ""
    stderr: str = ""
    id: str = ""
    stage: str | None = None          # setup / exec / serialize / sanity / no_change / check / timeout / crash
    traceback: str | None = None
    scene_before: dict | None = None
    scene_after: dict | None = None
    diff: dict | None = None
    thumbnail: str | None = None
    extra: dict = field(default_factory=dict)


def _blender_executable() -> str:
    exe = os.environ.get("BLENDER_EXECUTABLE")
    if not exe:
        raise RuntimeError(
            "BLENDER_EXECUTABLE is not set. Copy .env.example to .env and set it "
            r"to your Blender install path, e.g. D:\Blender\blender.exe"
        )
    return exe


def _check_version(found: str) -> None:
    if found != PINNED_BLENDER_VERSION and os.environ.get("BAI_ALLOW_VERSION_MISMATCH") != "1":
        raise RuntimeError(
            f"Blender {found} found, but all data is pinned to {PINNED_BLENDER_VERSION} "
            "(shared/output_formats.py). Point BLENDER_EXECUTABLE at the pinned build, "
            "or set BAI_ALLOW_VERSION_MISMATCH=1 for a throwaway experiment."
        )


def _from_record(record: dict) -> ValidationResult:
    known = {k: record.get(k) for k in ("ok", "error", "stdout", "id", "stage", "traceback",
                                        "scene_before", "scene_after", "diff", "thumbnail")}
    known["stdout"] = known["stdout"] or ""
    extra = {k: v for k, v in record.items() if k not in known}
    return ValidationResult(**known, extra=extra)


def _run_batch(jobs: list[dict], timeout: int, thumbnail_size: int) -> tuple[list[dict], str, bool]:
    """Run jobs in one Blender process until done, hung or crashed. Returns (records, stderr, killed)."""
    with tempfile.TemporaryDirectory(prefix="bai_harness_") as tmp:
        job_file = Path(tmp) / "jobs.json"
        results_path = Path(tmp) / "results.jsonl"
        results_path.touch()
        job_file.write_text(json.dumps({"jobs": jobs, "results_path": str(results_path),
                                        "thumbnail_size": thumbnail_size}), encoding="utf-8")
        stderr_path = Path(tmp) / "stderr.txt"
        with stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_file:
            proc = subprocess.Popen(
                [_blender_executable(), "--background", "--factory-startup",
                 "--python", str(RUNNER), "--", str(job_file)],
                stdout=subprocess.DEVNULL, stderr=stderr_file,
            )
            # Startup gets extra slack; after that each job must finish within `timeout`.
            deadline = time.monotonic() + timeout + 30
            seen_lines = 0
            killed = False
            while proc.poll() is None:
                time.sleep(0.2)
                lines = results_path.read_text(encoding="utf-8").count("\n")
                if lines != seen_lines:
                    seen_lines = lines
                    deadline = time.monotonic() + timeout
                elif time.monotonic() > deadline:
                    proc.kill()
                    proc.wait()
                    killed = True
                    break
        records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return records, stderr_path.read_text(encoding="utf-8", errors="replace"), killed


def run_jobs(
    jobs: list[dict],
    timeout: int = 60,
    thumbnail_size: int = 128,
    require_change: bool = True,
    check: Callable[[ValidationResult], str | None] | None = None,
) -> list[ValidationResult]:
    """Verify a list of jobs (see blender_runner.py for the job schema).

    require_change: fail jobs that leave the scene exactly as they found it.
    check: optional host-side verdict, called on each otherwise-passing result;
           return None to accept or a short reason string to reject.
    """
    results: dict[str, ValidationResult] = {}
    pending = list(jobs)
    version_checked = False

    while pending:
        records, stderr, killed = _run_batch(pending, timeout, thumbnail_size)
        header = next((r for r in records if r["id"] == "__runner__"), None)
        if header is None:
            tail = "\n".join(stderr.strip().splitlines()[-8:])
            raise RuntimeError(f"Blender failed to start the harness runner:\n{tail}")
        if not version_checked:
            _check_version(header["blender"])
            version_checked = True

        for record in records:
            if record["id"] != "__runner__":
                results[record["id"]] = _from_record(record)

        remaining = [job for job in pending if job["id"] not in results]
        if remaining:
            # The first unfinished job is the one Blender was running when it hung or died.
            culprit = remaining[0]
            results[culprit["id"]] = ValidationResult(
                ok=False, id=culprit["id"],
                stage="timeout" if killed else "crash",
                error=f"timeout after {timeout}s" if killed else "Blender crashed",
                stderr="\n".join(stderr.strip().splitlines()[-8:]),
            )
            remaining = remaining[1:]
        pending = remaining

    ordered = [results[job["id"]] for job in jobs]
    for result in ordered:
        if not result.ok:
            continue
        if require_change and result.diff is not None and not any(result.diff.values()):
            result.ok, result.stage, result.error = False, "no_change", "scene unchanged"
        elif check is not None:
            reason = check(result)
            if reason:
                result.ok, result.stage, result.error = False, "check", reason
    return ordered


def run_headless(
    code: str,
    sanity_check_code: str | None = None,
    timeout: int = 60,
    thumbnail: str | None = None,
    setup: str | None = None,
    require_change: bool = True,
) -> ValidationResult:
    """Verify one candidate script. Kept for the original single-script call sites."""
    job = {"id": "single", "kind": "script", "code": code, "setup": setup,
           "sanity": sanity_check_code, "thumbnail": thumbnail}
    return run_jobs([job], timeout=timeout, require_change=require_change)[0]

"""Check every operator on the Phase 1 shortlist: valid arguments, and (for
"harness" entries) that it runs headless in its setup scene and changes it.

Writes shared/data/operator_shortlist.json, which the data generators read:
    {"operators": {op: {group, setup, args, verify, status, error}}, "counts": {...}}
status: "ok" (verified in Blender), "viewport" (Way 1 only, not run), "failed".

Usage: python scripts/probe_shortlist.py [--only mesh.bevel,uv.unwrap]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "domains" / "operator_agent"))

from shared.headless_validator import run_jobs
from shared.output_formats import FormatError, parse_op_output

from operator_shortlist import SETUPS, SHORTLIST

REGISTRY = ROOT / "shared" / "data" / "operator_registry.json"
OUT = ROOT / "shared" / "data" / "operator_shortlist.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="Comma-separated operators to probe (prints only; doesn't write)")
    args = parser.parse_args()
    only = [op for op in args.only.split(",") if op]
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    results = {}
    jobs = []
    for op, (group, setup, op_args, verify) in SHORTLIST.items():
        if only and op not in only:
            continue
        entry = {"group": group, "setup": setup, "args": op_args, "verify": verify, "status": None, "error": None}
        results[op] = entry
        try:
            parse_op_output(json.dumps({"calls": [{"op": op, "args": op_args}]}), registry)
        except FormatError as e:
            entry.update(status="failed", error=f"format: {e}")
            continue
        if setup not in SETUPS:
            entry.update(status="failed", error=f"unknown setup {setup!r}")
            continue
        if verify == "viewport":
            entry["status"] = "viewport"
            continue
        jobs.append({"id": op, "kind": "ops", "calls": [{"op": op, "args": op_args}], "setup": SETUPS[setup]})

    print(f"Running {len(jobs)} operators in Blender...", flush=True)
    for result in run_jobs(jobs, timeout=60):
        entry = results[result.id]
        if result.ok:
            entry["status"] = "ok"
        else:
            entry.update(status="failed", error=f"{result.stage}: {(result.error or '')[:160]}")

    counts = {}
    for entry in results.values():
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    for op, entry in results.items():
        if entry["status"] == "failed":
            print(f"  FAIL {op:40} [{entry['setup']}] {entry['error']}")
    print(counts)
    if not only:
        OUT.write_text(json.dumps({"operators": results, "counts": counts}, indent=1), encoding="utf-8")
        print(f"-> {OUT}")


if __name__ == "__main__":
    main()

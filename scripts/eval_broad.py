"""Score an operator-agent model on evals/operator/broad.jsonl, reported by situation and operator group.

How an answer is graded:
- reference bai.ask / bai.decline: right if the model answers with that same kind of call.
- harness references: the reference and the model's answer both run from the item's setup in headless
  Blender, and the final scenes are compared (objects, transforms with rotations compared as
  orientations, modifiers, constraints, mesh data, mode...). Different-but-equivalent answers count.
  Selection is only compared when the reference changes it.
- viewport-only references can't run headless: compared call by call, ignoring selects of objects
  that are already the selection.

Usage:
    python scripts/eval_broad.py --constrained --out models/operator/broad_v4.jsonl
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from shared.dataset_io import read_jsonl
from shared.headless_validator import run_jobs
from shared.output_formats import OPERATOR_SYSTEM_PROMPT, FormatError, build_user_message, parse_op_output

from eval_operator_agent import chat

REGISTRY = json.loads((ROOT / "shared" / "data" / "operator_registry.json").read_text(encoding="utf-8"))
SCHEMA = ROOT / "shared" / "data" / "op_schema.json"
TOL = 2e-3
COMPARED_FIELDS = ("type", "parent", "scale", "hidden", "hide_render", "materials", "modifiers", "constraints",
                   "vertex_groups", "shape_keys", "particle_systems", "rigid_body", "force_field", "keyframes", "collections")


def _matrix(euler):
    x, y, z = euler
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    return [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz,
            cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz,
            -sy, sx * cy, cx * cy]


def _close(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= TOL
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    return a == b


def scene_difference(ref: dict, got: dict, before: dict) -> str | None:
    """None if the model's final scene matches the reference's, else what differs first."""
    r_objs, g_objs = {o["name"]: o for o in ref["objects"]}, {o["name"]: o for o in got["objects"]}
    if r_objs.keys() != g_objs.keys():
        return f"objects differ: {sorted(set(r_objs) ^ set(g_objs))[:4]}"
    for name, r in r_objs.items():
        g = g_objs[name]
        if not _close(r["location"], g["location"]):
            return f"{name}.location {g['location']} != {r['location']}"
        if not _close(_matrix(r["rotation"]), _matrix(g["rotation"])):
            return f"{name}.rotation {g['rotation']} != {r['rotation']}"
        for field in COMPARED_FIELDS:
            if not _close(r.get(field), g.get(field)):
                return f"{name}.{field} differs"
        r_data = {k: v for k, v in (r.get("data") or {}).items() if k != "selected"}
        g_data = {k: v for k, v in (g.get("data") or {}).items() if k != "selected"}
        if not _close(r_data, g_data):
            keys = [k for k in r_data if not _close(r_data.get(k), g_data.get(k))]
            return f"{name}.data {keys[:3]} differs"
    for field in ("mode", "frame", "collections", "rigidbody_world", "mesh_select_mode", "cursor"):
        if not _close(ref.get(field), got.get(field)):
            return f"scene {field} differs"
    if (ref["selected"], ref["active"]) != (before["selected"], before["active"]):  # the command was about selection
        if (ref["selected"], ref["active"]) != (got["selected"], got["active"]):
            return f"selection {got['selected']} != {ref['selected']}"
    return None


def _without_noop_selects(calls, scene):
    sole = scene["selected"] if scene["active"] in scene["selected"] else None
    return [c for c in calls if not (c["op"] == "bai.select" and sole and sorted(c["args"]["objects"]) == sorted(sole))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--file", default=str(ROOT / "evals" / "operator" / "broad.jsonl"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reuse", default=None, help="Re-grade the replies saved in this results file (no model calls)")
    args = parser.parse_args()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8")) if args.constrained else None
    items = [i for i in read_jsonl(Path(args.file)) if i.get("status") != "rejected"][: args.limit]

    saved = {r["id"]: r["reply"] for r in read_jsonl(Path(args.reuse))} if args.reuse else None
    print(f"{len(items)} items; {'re-grading saved replies' if saved else 'asking the model'}...", flush=True)
    results, started = [], time.time()
    for n, item in enumerate(items, 1):
        if n % 25 == 0:
            print(f"  {n}/{len(items)} answered", flush=True)
        messages = [{"role": "system", "content": OPERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(item["prompt"], item["scene"])}]
        reply = saved[item["id"]] if saved else chat(args.base_url, "operator", messages, schema=schema)
        try:
            calls, error = parse_op_output(reply, REGISTRY), None
        except FormatError as e:
            calls, error = None, f"format: {e}"
        results.append({"id": item["id"], "prompt": item["prompt"], "reply": reply, "calls": calls, "error": error,
                        "tags": item["tags"], "group": item["group"], "verify": item["verify"]})
    per_reply = (time.time() - started) / max(1, len(items))

    by_id = {i["id"]: i for i in items}
    jobs = []
    for r in results:
        item = by_id[r["id"]]
        ref_first = item["reference"][0]["op"]
        if r["calls"] is None:
            r["correct"] = False
        elif ref_first in ("bai.ask", "bai.decline"):
            r["correct"] = len(r["calls"]) == 1 and r["calls"][0]["op"] == ref_first
            if not r["correct"]:
                r["error"] = f"expected {ref_first}, got {[c['op'] for c in r['calls']]}"
        elif r["calls"][0]["op"] in ("bai.ask", "bai.decline"):
            r["correct"], r["error"] = False, f"answered {r['calls'][0]['op']} on a clear request"
        elif item["verify"] in ("viewport", "calls"):  # can't replay headless: compare calls
            want, got = (_without_noop_selects(c, item["scene"]) for c in (item["reference"], r["calls"]))
            r["correct"] = _close(want, got)
            if not r["correct"]:
                r["error"] = f"calls differ: {json.dumps(got)[:120]}"
        else:
            jobs += [{"id": f"ref:{r['id']}", "kind": "ops", "calls": item["reference"], "setup": item["setup"]},
                     {"id": f"got:{r['id']}", "kind": "ops", "calls": r["calls"], "setup": item["setup"]}]
    print(f"Running {len(jobs) // 2} answer pairs in Blender...", flush=True)
    ran = {res.id: res for res in run_jobs(jobs, timeout=60, require_change=False)}
    for r in results:
        if "correct" in r:
            continue
        ref, got = ran[f"ref:{r['id']}"], ran[f"got:{r['id']}"]
        if not got.ok:
            r["correct"], r["error"] = False, f"{got.stage}: {(got.error or '')[:100]}"
        elif not ref.ok:
            r["correct"], r["error"] = False, f"reference failed: {ref.error}"  # shouldn't happen: references were verified
        else:
            problem = scene_difference(ref.scene_after, got.scene_after, ref.scene_before)
            r["correct"], r["error"] = problem is None, problem

    total = sum(r["correct"] for r in results)
    print(f"\n{len(results)} items, {per_reply:.2f}s per reply")
    print(f"  correct: {total}/{len(results)} ({total / len(results):.1%})")
    for label, key in (("by situation", "tags"), ("by operator group", "group")):
        buckets = defaultdict(list)
        for r in results:
            for k in (r[key] if isinstance(r[key], list) else [r[key]]):
                buckets[k].append(r["correct"])
        print(f"  {label}: " + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in sorted(buckets.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))))
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
        print(f"\nPer-item results -> {args.out}")


if __name__ == "__main__":
    main()

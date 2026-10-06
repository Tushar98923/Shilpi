"""Score an operator-agent model on evals/operator/heldout.jsonl.

Works against any OpenAI-compatible server: llama.cpp's llama-server (the
shipping runtime) or Ollama. Each held-out prompt is sent with its starting
scene; the reply must parse as op/1, and the calls are then run through the
harness on that scene and checked against the item's expectations -- so an
answer that differs from the reference but does the right thing still counts.

Usage:
    llama-server -m models/dryrun/qwen3.5-0.8b-operator-q8_0.gguf --port 8080
    python scripts/eval_operator_agent.py --base-url http://127.0.0.1:8080/v1

    python scripts/eval_operator_agent.py --base-url http://127.0.0.1:11434/v1 --model qwen2.5:7b-instruct --limit 30
"""
import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "domains" / "operator_agent"))

from shared.eval_sets import load_heldout
from shared.headless_validator import run_jobs
from shared.output_formats import OPERATOR_SYSTEM_PROMPT, FormatError, build_user_message, parse_op_output

from operator_tasks import check_expectations

REGISTRY = ROOT / "shared" / "data" / "operator_registry.json"


def chat(base_url: str, model: str, messages: list[dict], timeout: int = 120, schema: dict | None = None) -> str:
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 256}
    if schema is not None:  # constrained decoding: only shortlist operators and their real arguments
        payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "op1", "schema": schema}}
    body = json.dumps(payload).encode()
    request = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="operator", help="Model name; llama-server ignores it")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--approved-only", action="store_true", help="Score only hand-approved items (the real number)")
    parser.add_argument("--out", default=None, help="Write per-item results as JSONL")
    parser.add_argument("--constrained", action="store_true", help="Constrained decoding with shared/data/op_schema.json")
    args = parser.parse_args()

    schema = json.loads((ROOT / "shared" / "data" / "op_schema.json").read_text(encoding="utf-8")) if args.constrained else None
    items = load_heldout("operator", approved_only=args.approved_only)[: args.limit]
    if not items:
        print("No held-out items found in evals/operator/heldout.jsonl.")
        sys.exit(1)
    if not args.approved_only:
        print("Note: scoring drafts too. Pass --approved-only once the set is hand-checked.\n")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    rows, jobs = {}, []
    started = time.monotonic()
    for item in items:
        messages = [{"role": "system", "content": OPERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(item["prompt"], item["scene"])}]
        reply = chat(args.base_url, args.model, messages, schema=schema)
        row = {"id": item["id"], "family": item["category"], "prompt": item["prompt"], "reply": reply,
               "format_ok": False, "exact": False, "verified": False, "error": None}
        try:
            calls = parse_op_output(reply, registry)
        except FormatError as e:
            row["error"] = f"format: {e}"
        else:
            row["format_ok"] = True
            row["exact"] = calls == item["reference"]
            jobs.append({"id": item["id"], "kind": "ops", "calls": calls, "setup": item["setup"]})
        rows[item["id"]] = row
    latency = (time.monotonic() - started) / len(items)

    expect = {item["id"]: item["expect"] for item in items}

    def check(result):
        return check_expectations(expect[result.id], result.scene_before, result.scene_after, result.diff)

    for result in run_jobs(jobs, timeout=30, check=check):
        rows[result.id]["verified"] = result.ok
        if not result.ok:
            rows[result.id]["error"] = f"{result.stage}: {result.error}"

    by_family = defaultdict(lambda: [0, 0])
    for row in rows.values():
        by_family[row["family"]][0] += row["verified"]
        by_family[row["family"]][1] += 1

    n = len(rows)
    pct = lambda k: 100 * sum(r[k] for r in rows.values()) / n  # noqa: E731
    print(f"{n} prompts, {latency:.2f}s per reply")
    print(f"  valid op/1 JSON:   {pct('format_ok'):5.1f}%")
    print(f"  exact match:       {pct('exact'):5.1f}%")
    print(f"  verified correct:  {pct('verified'):5.1f}%   <- the headline number")
    print("  by family: " + ", ".join(f"{f} {ok}/{total}" for f, (ok, total) in sorted(by_family.items())))
    failures = [r for r in rows.values() if not r["verified"]][:8]
    for row in failures:
        print(f"  FAIL {row['id']} {row['prompt']!r}\n       -> {row['reply'][:140]!r}\n       {row['error']}")

    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows.values()), encoding="utf-8")


if __name__ == "__main__":
    main()

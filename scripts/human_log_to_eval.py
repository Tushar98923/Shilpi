"""Turn an exported add-on log (Learned shortcuts > Export) into evals/operator/human.jsonl.

- commands marked Right: the model's own answer, confirmed by you, is the reference
- commands marked Wrong and then fixed by hand: your fix is the reference
- commands marked Wrong without a fix: written to human_to_label.jsonl, to be labelled by hand

The log has the scene as the model saw it but not a replayable setup, so these items are graded by
comparing calls (scripts/eval_broad.py --file evals/operator/human.jsonl), not by replaying the scene.

Usage: python scripts/human_log_to_eval.py shilpi_corrections.jsonl
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.eval_sets import normalize  # noqa: E402


def main():
    rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
    fixes = {normalize(r["request"]): r["right"] for r in rows if r.get("kind") == "correction" and r.get("source") == "manual_fix"}
    items, to_label = [], []
    for r in rows:
        if r.get("kind") != "command" or not r.get("scene"):
            continue
        reference = r.get("calls") if r.get("verdict") == "right" else fixes.get(normalize(r["request"]))
        if reference is None:
            if r.get("verdict") == "wrong":
                to_label.append(r)
            continue
        items.append({"id": f"human_{len(items):03d}", "specialist": "operator", "prompt": r["request"], "scene": r["scene"],
                      "setup": None, "reference": reference, "intended": None, "verify": "calls", "op": reference[-1]["op"],
                      "group": "human", "tags": ["human", r.get("verdict", "?")], "source": "add-on log",
                      "status": "approved", "reviewer": "user (Right/Wrong buttons)", "notes": ""})
    out = ROOT / "evals" / "operator" / "human.jsonl"
    out.write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    (out.with_name("human_to_label.jsonl")).write_text("".join(json.dumps(r) + "\n" for r in to_label), encoding="utf-8")
    print(f"{len(items)} human test items -> {out}; {len(to_label)} wrong answers without a fix need labelling")


if __name__ == "__main__":
    main()

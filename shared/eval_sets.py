"""Held-out evaluation sets: loading, and keeping them out of training data.

Layout: evals/<specialist>/heldout.jsonl, one prompt per line:

    {"id": str, "specialist": str, "prompt": str,
     "setup": str | null,          # bpy code that builds the starting scene, if any
     "reference": ... | null,      # a known-good answer (op/1 calls or a script), if any
     "category": str | null, "source": str,
     "status": "draft" | "approved" | "rejected",
     "reviewer": str | null, "notes": str}

Drafts are machine-picked; a human flips them to approved.
Only approved items count for evaluation, but drafts AND approved items are
both excluded from training, so a prompt never leaks while it waits for review.
"""
import re
from pathlib import Path

from shared.dataset_io import read_jsonl

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
NEAR_DUPLICATE_JACCARD = 0.9

# Public benchmarks kept next to our own sets: (file under evals/<specialist>/, prompt field).
# They are scored separately, but their prompts are excluded from training just the same.
EXTERNAL_BENCHMARKS = {
    "modeling": [("cadbench/CADBench.jsonl", "instruction")],  # BlenderLLM's CADBench, Apache-2.0
    "operator": [("broad.jsonl", "prompt"), ("human.jsonl", "prompt")],  # every operator group + real-use situations
}

_WORD_RE = re.compile(r"[a-z0-9.]+")


def normalize(prompt: str) -> str:
    return " ".join(_WORD_RE.findall(prompt.lower()))


def _words(prompt: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(prompt.lower()))


def heldout_path(specialist: str) -> Path:
    return EVALS_DIR / specialist / "heldout.jsonl"


def load_heldout(specialist: str | None = None, approved_only: bool = False) -> list[dict]:
    paths = [heldout_path(specialist)] if specialist else sorted(EVALS_DIR.glob("*/heldout.jsonl"))
    items = [item for path in paths for item in read_jsonl(path) if item.get("status") != "rejected"]
    if approved_only:
        items = [item for item in items if item.get("status") == "approved"]
    return items


def load_external(specialist: str | None = None) -> list[dict]:
    specialists = [specialist] if specialist else list(EXTERNAL_BENCHMARKS)
    items = []
    for spec in specialists:
        for rel_path, field in EXTERNAL_BENCHMARKS.get(spec, []):
            path = EVALS_DIR / spec / rel_path
            if not path.exists():
                continue
            for record in read_jsonl(path):
                items.append({"id": f"{rel_path.split('/')[0]}:{record.get('id')}", "prompt": record[field]})
    return items


def leakage_items(specialist: str | None = None) -> list[dict]:
    """Everything a training set for this specialist must not contain."""
    return load_heldout(specialist) + load_external(specialist)


class LeakageIndex:
    """Answers "is this training prompt (nearly) one of the held-out prompts?"."""

    def __init__(self, items: list[dict]):
        self._exact = {normalize(item["prompt"]): item["id"] for item in items}
        self._word_sets = [(_words(item["prompt"]), item["id"]) for item in items]

    @classmethod
    def for_specialist(cls, specialist: str | None = None) -> "LeakageIndex":
        return cls(leakage_items(specialist))

    def match(self, prompt: str) -> str | None:
        """Id of the held-out item this prompt leaks, or None."""
        hit = self._exact.get(normalize(prompt))
        if hit:
            return hit
        words = _words(prompt)
        if not words:
            return None
        for held_words, item_id in self._word_sets:
            overlap = len(words & held_words) / len(words | held_words)
            if overlap >= NEAR_DUPLICATE_JACCARD:
                return item_id
        return None

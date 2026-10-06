"""Dump every bpy.ops operator with its properties, for the pinned Blender version.

Run inside Blender (not with plain Python):
    blender --background --factory-startup --python shared/operator_registry.py -- shared/data/operator_registry.json

The output is the ground truth for the operator agent: which operators exist,
their argument names, types, enum values and defaults, and whether each one's
poll() passes headless in Object Mode and in Edit Mode (with the default cube
active). output_formats.parse_op_output() validates model output against it.
"""
import json
import os
import sys

import bpy

SKIP_PROPS = {"rna_type"}


def _default(prop):
    if prop.type == "ENUM":
        return sorted(prop.default_flag) if prop.is_enum_flag else prop.default
    if getattr(prop, "array_length", 0) > 0:
        return [round(v, 6) if isinstance(v, float) else v for v in prop.default_array]
    if prop.type == "FLOAT":
        return round(prop.default, 6)
    if prop.type in {"POINTER", "COLLECTION"}:
        return None
    return prop.default


def _describe_prop(prop):
    out = {"type": prop.type, "default": _default(prop)}
    if prop.subtype not in {"NONE", ""}:
        out["subtype"] = prop.subtype
    if prop.description:
        out["description"] = prop.description
    if getattr(prop, "array_length", 0) > 0:
        out["array_length"] = prop.array_length
    if prop.type in {"INT", "FLOAT"}:
        out["soft_min"], out["soft_max"] = prop.soft_min, prop.soft_max
    if prop.type == "ENUM":
        out["enum_items"] = [item.identifier for item in prop.enum_items]
        out["is_flag"] = prop.is_enum_flag
    if prop.is_hidden or prop.is_skip_save:
        out["hidden"] = True
    return out


def _poll_all():
    polls = {}
    for category in dir(bpy.ops):
        if category.startswith("_"):
            continue
        for name in dir(getattr(bpy.ops, category)):
            if name.startswith("_"):
                continue
            op = getattr(getattr(bpy.ops, category), name)
            try:
                polls[f"{category}.{name}"] = bool(op.poll())
            except Exception:  # noqa: BLE001 -- some polls raise without a window; that means "not usable headless"
                polls[f"{category}.{name}"] = False
    return polls


def main():
    out_path = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "operator_registry.json"

    object_mode_polls = _poll_all()
    bpy.ops.object.mode_set(mode="EDIT")
    edit_mode_polls = _poll_all()
    bpy.ops.object.mode_set(mode="OBJECT")

    operators = {}
    for full_name in sorted(object_mode_polls):
        category, name = full_name.split(".")
        op = getattr(getattr(bpy.ops, category), name)
        try:
            rna = op.get_rna_type()
        except (KeyError, RuntimeError):
            continue
        operators[full_name] = {
            "label": rna.name,
            "description": rna.description,
            "properties": {p.identifier: _describe_prop(p) for p in rna.properties if p.identifier not in SKIP_PROPS},
            "poll_object_mode": object_mode_polls[full_name],
            "poll_edit_mode": edit_mode_polls.get(full_name, False),
        }

    registry = {
        "blender": bpy.app.version_string.split(" ")[0],
        "count": len(operators),
        "operators": operators,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=1, sort_keys=True)
    print(f"Wrote {len(operators)} operators -> {out_path}")


main()

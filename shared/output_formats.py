"""Frozen model output formats. See docs/output_formats.md for the full spec.

Every dataset is built in these formats, so changing one means regenerating
data. Bump the version tag (op/1 -> op/2, bai-script/1 -> ...) instead of
editing a format in place.

Pure Python, no third-party deps: the add-on vendors this file as-is.

- Operator agent  -> op/1 JSON:      {"calls": [{"op": "mesh.primitive_cube_add", "args": {...}}]}
- Script specialists -> bai-script/1: raw bpy script whose first two lines are
                                     "# bai-script/1 specialist=<name>" and "import bpy"
- Planner         -> plan/1 JSON:    {"steps": [{"specialist": "<name>", "request": "..."}]}
- Router          -> one label from ROUTER_LABELS, as plain text
"""
import json
import re

PINNED_BLENDER_VERSION = "5.2.1"

OP_FORMAT = "op/1"
SCRIPT_FORMAT = "bai-script/1"
PLAN_FORMAT = "plan/1"

# Specialists that emit bai-script/1. Names are frozen: they appear in every
# script header, plan step and router label.
SCRIPT_SPECIALISTS = (
    "code_repair",
    "materials",
    "lighting",
    "camera",
    "render",
    "modeling",
    "geometry_nodes",
    "animation",
    "rigging",
    "simulation",
    "uv",
    "set_dressing",
)
# Specialists whose outputs are prose or their own JSON, defined in their phase.
OTHER_SPECIALISTS = (
    "planner",
    "scene_understanding",
    "visual_critic",
    "reference_reader",
    "art_director",
    "tutor",
)
ROUTER_LABELS = ("operator",) + SCRIPT_SPECIALISTS + ("planner", "scene_understanding", "reference_reader", "art_director", "tutor")

# Built-in pseudo-operators the operator agent may call in addition to real
# bpy.ops. They cover things that have no headless-safe operator.
BUILTIN_OPS = {
    "bai.select": {"objects": list, "active": (str, type(None)), "extend": bool},
    "bai.set": {"object": str, "property": str, "value": object},
    # Keyframes: the real anim.keyframe_* operators need an editor context. property "all" = I key LocRotScale.
    "bai.keyframe": {"object": str, "property": str, "frame": int, "delete": bool},
    "bai.frame": {"frame": int},  # jump to a frame
    # Move (or with link=true, also link) the selected objects into a collection by name, creating it if needed.
    # object.move_to_collection needs a per-session collection id a model can't know.
    "bai.collection": {"name": str, "link": bool},
    # Ask the user instead of acting, when the request is ambiguous ("bevel the cube" with two cubes and
    # neither selected). The add-on shows the question and runs nothing. Must be the only call.
    "bai.ask": {"question": str},
    # Loop cut by voice (Edit Mode): cut the edge ring that runs along `axis` N times. "axis" is the direction
    # of the edges being cut, so Z = a horizontal ring. mesh.loopcut_slide needs a mouse position and
    # crashes headless Blender even when given an edge.
    "bai.loopcut": {"cuts": int, "axis": str},
    # Say why nothing will happen: the request is outside what the operator agent does ("what's the
    # weather", "make it look like marble" = a materials job) or names an object that isn't there.
    # Must be the only call. (Idea from Hammer / xLAM's irrelevance data.)
    "bai.decline": {"reason": str},
}
BAI_LOOPCUT_AXES = ("X", "Y", "Z")
BAI_SET_PROPERTIES = ("location", "rotation_euler", "scale", "name", "hidden", "hide_render")  # hidden = the eye icon / H key
BAI_KEYFRAME_PROPERTIES = ("location", "rotation_euler", "scale", "all")

_OP_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$")
_HEADER_RE = re.compile(r"^# bai-script/1 specialist=([a-z_]+)\s*$")
_FENCE_RE = re.compile(r"^```(?:python|json)?\s*\n(.*?)\n?```\s*$", re.DOTALL)


class FormatError(ValueError):
    pass


def _strip_fence(text: str) -> str:
    text = text.strip()
    match = _FENCE_RE.match(text)
    return match.group(1).strip() if match else text


def _is_json_value(value) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(_is_json_value(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_value(v) for k, v in value.items())
    return False


# --- op/1 ---------------------------------------------------------------------

def parse_op_output(text: str, registry: dict | None = None) -> list[dict]:
    """Parse and validate an operator-agent response. Returns the list of calls.

    `registry` is the dict written by shared/operator_registry.py; when given,
    operator names and argument names/enum values are checked against it.
    Raises FormatError with a short reason on any problem.
    """
    try:
        payload = json.loads(_strip_fence(text))
    except json.JSONDecodeError as e:
        raise FormatError(f"not valid JSON: {e.msg}") from None
    if not isinstance(payload, dict) or set(payload) != {"calls"}:
        raise FormatError('top level must be exactly {"calls": [...]}')
    calls = payload["calls"]
    if not isinstance(calls, list) or not calls:
        raise FormatError('"calls" must be a non-empty list')
    for i, call in enumerate(calls):
        validate_op_call(call, registry, where=f"calls[{i}]")
    if len(calls) > 1 and any(c["op"] in ("bai.ask", "bai.decline") for c in calls):
        raise FormatError("bai.ask / bai.decline must be the only call")
    return calls


def validate_op_call(call, registry: dict | None = None, where: str = "call") -> None:
    if not isinstance(call, dict) or set(call) - {"op", "args"} or "op" not in call:
        raise FormatError(f'{where} must be {{"op": str, "args": {{...}}}}')
    op, args = call["op"], call.get("args", {})
    if not isinstance(op, str) or not _OP_NAME_RE.match(op):
        raise FormatError(f"{where}: bad operator name {op!r}")
    if not isinstance(args, dict) or not _is_json_value(args):
        raise FormatError(f"{where}: args must be a JSON object")

    if op in BUILTIN_OPS:
        spec = BUILTIN_OPS[op]
        for key, value in args.items():
            if key not in spec:
                raise FormatError(f"{where}: {op} has no argument {key!r}")
            if not isinstance(value, spec[key]):
                raise FormatError(f"{where}: {op}.{key} has the wrong type")
        if op == "bai.select" and "objects" not in args:
            raise FormatError(f"{where}: bai.select needs 'objects'")
        if op == "bai.set":
            if {"object", "property", "value"} - args.keys():
                raise FormatError(f"{where}: bai.set needs object, property and value")
            if args["property"] not in BAI_SET_PROPERTIES:
                raise FormatError(f"{where}: bai.set cannot set {args['property']!r}")
        if op == "bai.keyframe":
            if {"object", "property"} - args.keys():
                raise FormatError(f"{where}: bai.keyframe needs object and property")
            if args["property"] not in BAI_KEYFRAME_PROPERTIES:
                raise FormatError(f"{where}: bai.keyframe cannot key {args['property']!r}")
        if op == "bai.frame" and "frame" not in args:
            raise FormatError(f"{where}: bai.frame needs frame")
        if op == "bai.collection" and not args.get("name"):
            raise FormatError(f"{where}: bai.collection needs a name")
        if op == "bai.ask" and not str(args.get("question", "")).strip():
            raise FormatError(f"{where}: bai.ask needs a question")
        if op == "bai.decline" and not str(args.get("reason", "")).strip():
            raise FormatError(f"{where}: bai.decline needs a reason")
        if op == "bai.loopcut" and args.get("axis", "Z") not in BAI_LOOPCUT_AXES:
            raise FormatError(f"{where}: bai.loopcut axis must be X, Y or Z")
        return

    if op.startswith("bai."):
        raise FormatError(f"{where}: unknown built-in {op!r}")
    if registry is None:
        return
    entry = registry["operators"].get(op)
    if entry is None:
        raise FormatError(f"{where}: unknown operator {op!r}")
    props = entry["properties"]
    for key, value in args.items():
        prop = props.get(key)
        if prop is None:
            raise FormatError(f"{where}: {op} has no argument {key!r}")
        # Dynamic enums (e.g. constraint types) introspect as empty; the harness checks those instead.
        if prop["type"] == "ENUM" and not prop.get("is_flag") and prop["enum_items"] and value not in prop["enum_items"]:
            raise FormatError(f"{where}: {op}.{key}={value!r} is not one of {prop['enum_items']}")


def format_op_output(calls: list[dict]) -> str:
    """Canonical serialization used for training targets (compact, stable key order)."""
    return json.dumps({"calls": [{"op": c["op"], "args": c.get("args", {})} for c in calls]}, separators=(", ", ": "))


# --- bai-script/1 ---------------------------------------------------------------

def script_header(specialist: str) -> str:
    if specialist not in SCRIPT_SPECIALISTS:
        raise ValueError(f"not a script specialist: {specialist!r}")
    return f"# bai-script/1 specialist={specialist}\nimport bpy\n"


def with_header(specialist: str, body: str) -> str:
    """Put the frozen header on a bpy script body (drops fences and a leading `import bpy`)."""
    body = _strip_fence(body)
    lines = body.splitlines()
    while lines and (not lines[0].strip() or _HEADER_RE.match(lines[0])):
        lines.pop(0)
    if lines and lines[0].strip() == "import bpy":
        lines.pop(0)
    return script_header(specialist) + "\n".join(lines).rstrip() + "\n"


def parse_script_output(text: str, expected_specialist: str | None = None) -> tuple[str, str]:
    """Validate a specialist response. Returns (specialist, script). Raises FormatError."""
    script = _strip_fence(text)
    lines = script.splitlines()
    if len(lines) < 2:
        raise FormatError("script is shorter than the header")
    match = _HEADER_RE.match(lines[0])
    if not match:
        raise FormatError("first line must be '# bai-script/1 specialist=<name>'")
    specialist = match.group(1)
    if specialist not in SCRIPT_SPECIALISTS:
        raise FormatError(f"unknown specialist {specialist!r}")
    if expected_specialist and specialist != expected_specialist:
        raise FormatError(f"header says {specialist!r}, expected {expected_specialist!r}")
    if lines[1].strip() != "import bpy":
        raise FormatError("second line must be 'import bpy'")
    try:
        compile(script, f"<{specialist}>", "exec")
    except SyntaxError as e:
        raise FormatError(f"syntax error on line {e.lineno}: {e.msg}") from None
    return specialist, script + ("\n" if not script.endswith("\n") else "")


# --- plan/1 ------------------------------------------------------------------------

def parse_plan_output(text: str) -> list[dict]:
    try:
        payload = json.loads(_strip_fence(text))
    except json.JSONDecodeError as e:
        raise FormatError(f"not valid JSON: {e.msg}") from None
    if not isinstance(payload, dict) or set(payload) != {"steps"} or not isinstance(payload["steps"], list) or not payload["steps"]:
        raise FormatError('top level must be exactly {"steps": [...]} with at least one step')
    allowed = ("operator",) + SCRIPT_SPECIALISTS
    for i, step in enumerate(payload["steps"]):
        if not isinstance(step, dict) or set(step) != {"specialist", "request"}:
            raise FormatError(f'steps[{i}] must be {{"specialist": str, "request": str}}')
        if step["specialist"] not in allowed:
            raise FormatError(f"steps[{i}]: planner cannot call {step['specialist']!r}")
        if not isinstance(step["request"], str) or not step["request"].strip():
            raise FormatError(f"steps[{i}]: empty request")
    return payload["steps"]


# --- prompt framing (inputs are frozen too) -------------------------------------------

# Kept short on purpose: the operator agent is a 0.8B-2B model, and every
# training example carries this text.
OPERATOR_SYSTEM_PROMPT = (
    "You are the Blender operator agent. Turn the request into Blender operator calls. "
    'Reply with JSON only: {"calls": [{"op": "<bpy.ops name>", "args": {...}}]}. '
    "Use bai.select to select objects by name and bai.set to set location, rotation_euler "
    "(radians), scale, name, hidden or hide_render."
)

def build_user_message(request: str, scene: dict | None = None) -> str:
    """The one user-message layout every model sees: compact scene JSON, then the request.

    `scene` should be the output of scene_serializer.compact_scene().
    """
    if scene is None:
        return f"Request: {request.strip()}"
    return f"Scene: {json.dumps(scene, separators=(',', ':'))}\nRequest: {request.strip()}"

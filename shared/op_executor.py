"""Execute validated op/1 calls inside Blender. Used by the harness and the add-on.

Call output_formats.parse_op_output() first; this module assumes the calls are
well-formed and only deals with running them.
"""


class OpExecutionError(RuntimeError):
    completed = 0  # how many calls ran before this one failed (set by apply_calls)


class AskUser(Exception):
    """Raised by bai.ask: the model wants the user to clarify. Callers show `question` and run nothing."""

    def __init__(self, question: str):
        super().__init__(question)
        self.question = question


class Declined(Exception):
    """Raised by bai.decline: nothing to do here. Callers show `reason` and run nothing."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _builtin_select(args):
    import bpy

    view_layer = bpy.context.view_layer
    names = args["objects"]
    missing = [n for n in names if n not in bpy.context.scene.objects]
    if missing:
        raise OpExecutionError(f"bai.select: no object named {missing[0]!r}")
    if not args.get("extend", False):
        for obj in bpy.context.scene.objects:
            obj.select_set(False, view_layer=view_layer)
    for name in names:
        bpy.context.scene.objects[name].select_set(True, view_layer=view_layer)
    active = args.get("active", names[0] if names else None)
    if active is not None:
        if active not in bpy.context.scene.objects:
            raise OpExecutionError(f"bai.select: no object named {active!r}")
        view_layer.objects.active = bpy.context.scene.objects[active]


def _builtin_set(args):
    import bpy

    obj = bpy.context.scene.objects.get(args["object"])
    if obj is None:
        raise OpExecutionError(f"bai.set: no object named {args['object']!r}")
    try:
        if args["property"] == "hidden":
            obj.hide_set(bool(args["value"]))
        else:
            setattr(obj, args["property"], args["value"])
    except (TypeError, ValueError, AttributeError) as e:
        raise OpExecutionError(f"bai.set: {e}") from None


def _builtin_keyframe(args):
    import bpy

    obj = bpy.context.scene.objects.get(args["object"])
    if obj is None:
        raise OpExecutionError(f"bai.keyframe: no object named {args['object']!r}")
    frame = args.get("frame", bpy.context.scene.frame_current)
    paths = ("location", "rotation_euler", "scale") if args["property"] == "all" else (args["property"],)
    for path in paths:
        if args.get("delete", False):
            try:
                obj.keyframe_delete(data_path=path, frame=frame)
            except RuntimeError as e:
                raise OpExecutionError(f"bai.keyframe: {str(e).strip()}") from None
        else:
            obj.keyframe_insert(data_path=path, frame=frame)


def _builtin_frame(args):
    import bpy

    bpy.context.scene.frame_set(int(args["frame"]))


def _builtin_collection(args):
    import bpy

    objects = list(bpy.context.selected_objects)
    if not objects:
        raise OpExecutionError("bai.collection: nothing is selected")
    scene = bpy.context.scene
    collection = bpy.data.collections.get(args["name"])
    if collection is None:
        collection = bpy.data.collections.new(args["name"])
    if collection.name not in {c.name for c in scene.collection.children_recursive}:
        scene.collection.children.link(collection)
    for obj in objects:
        if not args.get("link", False):
            for old in list(obj.users_collection):
                if old != collection:
                    old.objects.unlink(obj)
        if obj.name not in collection.objects:
            collection.objects.link(obj)


def _edge_ring(start):
    """Edges across a strip of quads from `start` (what a loop cut cuts through)."""
    ring, todo = {start}, [start]
    while todo:
        edge = todo.pop()
        for loop in edge.link_loops:
            if len(loop.face.verts) != 4:
                continue
            opposite = loop.link_loop_next.link_loop_next.edge
            if opposite not in ring:
                ring.add(opposite)
                todo.append(opposite)
    return list(ring)


def _builtin_loopcut(args):
    import bmesh
    import bpy
    from mathutils import Vector

    obj = bpy.context.edit_object
    if obj is None or obj.type != "MESH":
        raise OpExecutionError("bai.loopcut: needs a mesh in Edit Mode")
    axis = "XYZ".index(args.get("axis", "Z"))
    bm = bmesh.from_edit_mesh(obj.data)

    def along_axis(edge):
        direction = edge.verts[1].co - edge.verts[0].co
        return direction.length > 1e-9 and abs(direction[axis]) / direction.length > 0.9

    candidates = [e for e in bm.edges if along_axis(e) and not e.hide]
    if not candidates:
        raise OpExecutionError(f"bai.loopcut: no edges run along {args.get('axis', 'Z')} to cut through")
    center = sum((v.co for v in bm.verts), Vector()) / len(bm.verts)
    start = min(candidates, key=lambda e: ((e.verts[0].co + e.verts[1].co) / 2 - center).length)
    bmesh.ops.subdivide_edgering(bm, edges=_edge_ring(start), cuts=int(args.get("cuts", 1)))
    bmesh.update_edit_mesh(obj.data)


def _builtin_ask(args):
    raise AskUser(args["question"])


def _builtin_decline(args):
    raise Declined(args["reason"])


_BUILTINS = {"bai.select": _builtin_select, "bai.set": _builtin_set, "bai.keyframe": _builtin_keyframe,
             "bai.frame": _builtin_frame, "bai.collection": _builtin_collection, "bai.ask": _builtin_ask,
             "bai.loopcut": _builtin_loopcut, "bai.decline": _builtin_decline}


def _flag_enums_as_sets(op, args: dict) -> dict:
    """JSON has no sets, but enum-flag properties (e.g. object.align align_axis) only accept one."""
    props = op.get_rna_type().properties
    return {k: set(v) if isinstance(v, list) and k in props and getattr(props[k], "is_enum_flag", False) else v
            for k, v in args.items()}


INVOKE_WITH_UI = {"render.render", "render.opengl", "wm.save_mainfile"}


def apply_call(call: dict) -> None:
    import bpy

    op_name, args = call["op"], call.get("args", {})
    if op_name in _BUILTINS:
        _BUILTINS[op_name](args)
        return

    category, name = op_name.split(".")
    op = getattr(getattr(bpy.ops, category, None), name, None)
    if op is None:
        raise OpExecutionError(f"unknown operator {op_name!r}")
    args = _flag_enums_as_sets(op, args)
    if not op.poll():
        raise OpExecutionError(f"{op_name}: poll() failed in the current context (mode={bpy.context.mode})")
    # With a UI, these behave like their hotkey: F12 opens the render window (EXEC renders invisibly),
    # Ctrl+S on a never-saved file opens the Save dialog (EXEC fails). Headless, they stay EXEC.
    mode = "INVOKE_DEFAULT" if op_name in INVOKE_WITH_UI and bpy.context.window is not None else "EXEC_DEFAULT"
    try:
        result = op(mode, **args)
    except (TypeError, RuntimeError, ValueError) as e:
        raise OpExecutionError(f"{op_name}: {str(e).strip()}") from None
    if "CANCELLED" in result:
        raise OpExecutionError(f"{op_name}: operator was cancelled")


# Undo/redo reload Blender's data, so they must not run inside another operator (the add-on runs them
# on the next timer tick). Calling them in the same tick as other changes crashed Blender in testing.
DATA_RELOADING_OPS = {"ed.undo", "ed.redo"}


def apply_calls(calls: list[dict]) -> None:
    for i, call in enumerate(calls):
        try:
            apply_call(call)
        except OpExecutionError as e:
            error = OpExecutionError(f"calls[{i}] {e}")
            error.completed = i
            raise error from None

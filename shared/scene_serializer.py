"""Frozen scene-state JSON format, shared by training data and the shipped add-on.

`serialize_scene()` runs inside Blender (imports bpy lazily). `diff_scenes()` and
`compact_scene()` are pure Python and also run outside Blender, so the data
pipeline can diff and trim states that the harness wrote to disk.

This module must stay importable from inside Blender with no third-party
dependencies -- the add-on vendors it as-is.

Format "scene/1" (do not change field meanings without bumping SCENE_FORMAT;
every dataset built on it would need regenerating):

    {
      "format": "scene/1",
      "blender": "5.2.1",
      "mode": "OBJECT",                 # context.mode, e.g. OBJECT / EDIT_MESH
      "active": "Cube" | null,
      "selected": ["Cube", ...],        # sorted
      "cursor": {"location": [x,y,z], "rotation": [x,y,z]},
      "frame": 1,
      "objects": [ {object}, ... ],     # sorted by name
      "materials": [ {material}, ... ]  # sorted by name, only materials with users
    }

    object = {
      "name", "type", "parent" | null, "collections": [...],
      "location", "rotation", "scale", "dimensions",     # rotation is XYZ euler radians
      "hidden": bool,                   # eye icon or viewport-disabled
      "hide_render": bool, "selected": bool,
      "materials": [slot material names, null for empty slots],
      "modifiers": [{"name", "type"}],
      "data": {...}                     # type-specific, see _object_data()
    }

    mesh data = {"mesh", "vertices", "edges", "faces", "smooth_faces"}

Verification-only fields (added 2026-10-01; additive, so scene/1 data stays valid).
They let the harness see changes the fields above miss. compact_scene() never
shows them to a model:

    object += {"constraints": [types], "vertex_groups": n, "shape_keys": n, "particle_systems": n,
               "rigid_body": type | null, "force_field": type | null, "keyframes": n}
    mesh data += {"selected": [v, e, f], "hidden": [v, e, f], "seams": n, "sharp_edges": n,
                  "attributes": [names], "geometry": fingerprint, "dyntopo": bool}
    armature data = {"bones": n, "pose": fingerprint}
    curve data += {"points": n, "cyclic": n, "geometry": fingerprint}
    scene += {"collections": [names], "rigidbody_world": bool, "mesh_select_mode": [vert, edge, face]}

Floats are rounded to FLOAT_DIGITS so identical scenes serialize identically
across runs (no 1e-9 noise in diffs or in training targets).
"""

SCENE_FORMAT = "scene/1"
FLOAT_DIGITS = 4


def _r(value):
    return round(float(value), FLOAT_DIGITS) + 0.0  # + 0.0 turns -0.0 into 0.0


def _vec(values):
    return [_r(v) for v in values]


def _color(values):
    return [_r(v) for v in values][:4]


_HANDLE_TYPES = ("FREE", "VECTOR", "ALIGNED", "AUTO")
_SPLINE_TYPES = ("POLY", "BEZIER", "NURBS", "CATMULL_ROM")
FINGERPRINT_MAX_POINTS = 200_000
_WORDS_RE = __import__("re").compile(r"[a-z0-9]+")
_GENERIC_WORDS = {"the", "and", "obj", "object", "mesh", "geo", "low", "high", "new", "old", "copy"}  # skip fingerprinting huge meshes; counts still catch most changes


def _fingerprint(values) -> str:
    """Short stable hash of rounded numbers: changes whenever geometry moves or reorders."""
    import hashlib

    digest = hashlib.md5()
    for v in values:
        digest.update(b"%.4f," % v)
    return digest.hexdigest()[:12]


def _mesh_fingerprint(data, in_edit_mode: bool):
    """Order-independent: Blender 5 builds some primitives (UV sphere) with a different face order on
    every run, so faces are described by their vertex positions, each rotated to start at its smallest
    corner (winding still counts, so flipped normals change it), then sorted. UVs are sorted too."""
    if len(data.vertices) > FINGERPRINT_MAX_POINTS:
        return None
    keys = [tuple(_r(c) for c in v.co) for v in data.vertices]
    faces = []
    for p in data.polygons:
        corners = [keys[i] for i in p.vertices]
        start = corners.index(min(corners))
        faces.append(tuple(corners[start:] + corners[:start]))
    faces.sort()
    loose = sorted(keys)  # also catches vertices that belong to no face
    values = [c for face in faces for corner in face for c in corner] + [-1.0] + [c for k in loose for c in k]
    uvs = sorted(_r(v) for v in _uv_values(data, in_edit_mode))
    return _fingerprint(values + [-2.0] + uvs)


def _uv_values(data, in_edit_mode: bool) -> list:
    # In Edit Mode, UVs live in the BMesh: update_from_editmode() doesn't sync them to the mesh.
    if in_edit_mode:
        import bmesh

        bm = bmesh.from_edit_mesh(data)
        return [c for layer in bm.loops.layers.uv.values() for f in bm.faces for loop in f.loops for c in loop[layer].uv]
    return [c for layer in data.uv_layers if layer.name in data.attributes
            for d in data.attributes[layer.name].data for c in d.vector]


def _mesh_data(data, in_edit_mode: bool = False):
    verts, edges, faces = data.vertices, data.edges, data.polygons
    return {
        "mesh": data.name,
        "vertices": len(verts),
        "edges": len(edges),
        "faces": len(faces),
        "smooth_faces": sum(1 for p in faces if p.use_smooth),
        "selected": [sum(1 for v in verts if v.select), sum(1 for e in edges if e.select), sum(1 for p in faces if p.select)],
        "hidden": [sum(1 for v in verts if v.hide), sum(1 for e in edges if e.hide), sum(1 for p in faces if p.hide)],
        "seams": sum(1 for e in edges if e.use_seam),
        "sharp_edges": sum(1 for e in edges if e.use_edge_sharp),
        "attributes": sorted(a.name for a in data.attributes if not a.name.startswith(".") or a.name.startswith(".sculpt")),
        "geometry": _mesh_fingerprint(data, in_edit_mode),
    }


def _object_data(obj):
    data = obj.data
    if data is None:
        return {}
    if obj.type == "MESH":
        if obj.mode == "EDIT":
            obj.update_from_editmode()  # edit-mode changes live in the BMesh until synced
        return {**_mesh_data(data, obj.mode == "EDIT"), "dyntopo": bool(getattr(obj, "use_dynamic_topology_sculpting", False))}
    if obj.type == "ARMATURE":
        bones = data.edit_bones if obj.mode == "EDIT" else data.bones
        pose = [c for pb in obj.pose.bones for vec in (pb.location, pb.rotation_quaternion, pb.rotation_euler, pb.scale)
                for c in vec] if obj.pose else []
        if obj.mode == "EDIT":
            shape = [c for b in bones for c in (*b.head, *b.tail, b.roll)]
            flags = [float(b.select + 2 * b.select_head + 4 * b.select_tail) for b in bones]
        else:  # Blender 5 moved bone selection from Bone to PoseBone
            shape, flags = [], [float(pb.select) for pb in obj.pose.bones] if obj.pose else []
        return {"bones": len(bones), "pose": _fingerprint(pose + [-1.0] + shape + [-2.0] + flags)}
    if obj.type == "LIGHT":
        out = {"light_type": data.type, "energy": _r(data.energy), "color": _color(data.color)}
        if data.type == "SPOT":
            out["spot_size"] = _r(data.spot_size)
        if data.type == "AREA":
            out["size"] = _r(data.size)
        return out
    if obj.type == "CAMERA":
        return {
            "camera_type": data.type,
            "lens": _r(data.lens),
            "sensor_width": _r(data.sensor_width),
            "clip_start": _r(data.clip_start),
            "clip_end": _r(data.clip_end),
        }
    if obj.type in {"CURVE", "FONT", "SURFACE"}:
        if obj.type == "FONT":
            return {"splines": len(data.splines), "body": data.body}
        points = [p for s in data.splines for p in list(s.bezier_points) + list(s.points)]
        coords = [c for p in points for c in p.co]
        handles = [float(_HANDLE_TYPES.index(t)) for s in data.splines for p in s.bezier_points
                   for t in (p.handle_left_type, p.handle_right_type)]
        types = [float(_SPLINE_TYPES.index(s.type)) for s in data.splines]
        types += [float(p.select_control_point) for s in data.splines for p in s.bezier_points]
        types += [float(p.select) for s in data.splines for p in s.points]
        return {"splines": len(data.splines), "points": len(points),
                "cyclic": sum(1 for s in data.splines if s.use_cyclic_u),
                "geometry": _fingerprint(coords + [-1.0] + handles + [-2.0] + types)}
    return {}


def _principled_inputs(mat):
    if not mat.use_nodes or mat.node_tree is None:
        return {"base_color": _color(mat.diffuse_color)}
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            inputs = node.inputs
            out = {}
            for key, socket_name in (("base_color", "Base Color"), ("metallic", "Metallic"), ("roughness", "Roughness")):
                socket = inputs.get(socket_name)
                if socket is None:
                    continue
                value = socket.default_value
                out[key] = _color(value) if key == "base_color" else _r(value)
            return out
    return {}


def _material(mat):
    return {
        "name": mat.name,
        "use_nodes": bool(mat.use_nodes),
        "nodes": len(mat.node_tree.nodes) if mat.use_nodes and mat.node_tree else 0,
        **_principled_inputs(mat),
    }


def _object(obj, selected_names):
    return {
        "name": obj.name,
        "type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "collections": sorted(c.name for c in obj.users_collection),
        "location": _vec(obj.location),
        "rotation": _vec(obj.rotation_euler),
        "scale": _vec(obj.scale),
        "dimensions": _vec(obj.dimensions),
        "hidden": bool(obj.hide_get() or obj.hide_viewport),
        "hide_render": bool(obj.hide_render),
        "selected": obj.name in selected_names,
        "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
        "modifiers": [{"name": m.name, "type": m.type} for m in obj.modifiers],
        "data": _object_data(obj),
        "constraints": [c.type for c in obj.constraints],
        "vertex_groups": len(obj.vertex_groups),
        "shape_keys": len(obj.data.shape_keys.key_blocks) if getattr(obj.data, "shape_keys", None) else 0,
        "particle_systems": len(obj.particle_systems),
        "rigid_body": obj.rigid_body.type if obj.rigid_body else None,
        "force_field": obj.field.type if obj.field and obj.field.type != "NONE" else None,
        "keyframes": _keyframe_count(obj),
    }


def _keyframe_count(obj) -> int:
    anim = obj.animation_data
    if anim is None or anim.action is None:
        return 0
    try:  # Blender 4.4+ layered actions keep curves in channelbags
        from bpy_extras import anim_utils
        channelbag = anim_utils.action_get_channelbag_for_slot(anim.action, anim.action_slot)
        fcurves = channelbag.fcurves if channelbag else []
    except (ImportError, AttributeError):
        fcurves = getattr(anim.action, "fcurves", [])
    return sum(len(fc.keyframe_points) for fc in fcurves)


def serialize_scene(context=None) -> dict:
    """Snapshot the current scene into the frozen scene/1 dict. Runs inside Blender only."""
    import bpy

    context = context or bpy.context
    scene = context.scene
    view_layer = context.view_layer

    selected = sorted(obj.name for obj in scene.objects if obj.select_get(view_layer=view_layer))
    active = view_layer.objects.active
    selected_set = set(selected)

    materials = sorted((m for m in bpy.data.materials if m.users > 0), key=lambda m: m.name)

    return {
        "format": SCENE_FORMAT,
        "blender": bpy.app.version_string.split(" ")[0],
        "mode": context.mode,
        "active": active.name if active else None,
        "selected": selected,
        "cursor": {
            "location": _vec(scene.cursor.location),
            "rotation": _vec(scene.cursor.rotation_euler),
        },
        "frame": scene.frame_current,
        "collections": sorted(c.name for c in scene.collection.children_recursive),
        "rigidbody_world": scene.rigidbody_world is not None,
        "mesh_select_mode": [bool(v) for v in scene.tool_settings.mesh_select_mode],
        "objects": [_object(obj, selected_set) for obj in sorted(scene.objects, key=lambda o: o.name)],
        "materials": [_material(m) for m in materials],
    }


# --- pure-Python helpers (usable outside Blender) --------------------------------

def mentioned_objects(request: str, names: list[str]) -> list[str]:
    """Object names a request refers to: whole name ("Crate", "Cube.001") or a word of it ("the chair" -> SM_Chair_01)."""
    text = " ".join(_WORDS_RE.findall(request.lower()))
    found = []
    for name in names:
        words = [w for w in _WORDS_RE.findall(name.lower().replace("_", " ").replace(".", " ")) if len(w) >= 3 and not w.isdigit()]
        if name.lower() in request.lower() or any(f" {w} " in f" {text} " for w in words if w not in _GENERIC_WORDS):
            found.append(name)
    return found


def compact_scene(scene: dict, max_objects: int = 40, request: str | None = None) -> dict:
    """The trimmed form of a scene/1 dict that goes into model prompts.

    Small models get a short context, so prompts carry only what commands
    usually refer to: names, types, rough placement, selection and mode.
    The full dict stays the harness's source of truth for diffs.

    Objects stay in name order. When there are more than `max_objects`, the active
    and selected objects and the ones `request` mentions are always kept, so the
    target can't fall off the end of the list in a big scene.
    """
    kept = scene["objects"]
    if len(kept) > max_objects:
        names = [o["name"] for o in kept]
        must = {scene["active"], *scene["selected"], *(mentioned_objects(request, names) if request else [])}
        must = [o for o in kept if o["name"] in must][:max_objects]
        rest = [o for o in kept if o not in must][: max_objects - len(must)]
        kept = sorted(must + rest, key=lambda o: o["name"])
    objects = []
    for obj in kept:
        entry = {"name": obj["name"], "type": obj["type"], "location": [round(v, 2) for v in obj["location"]]}
        if obj["modifiers"]:
            entry["modifiers"] = [m["type"] for m in obj["modifiers"]]
        if any(obj["materials"]):
            entry["materials"] = [m for m in obj["materials"] if m]
        objects.append(entry)
    return {
        "mode": scene["mode"],
        "active": scene["active"],
        "selected": scene["selected"],
        "objects": objects,
        **({"truncated": len(scene["objects"]) - len(kept)} if len(scene["objects"]) > len(kept) else {}),
    }


def diff_scenes(before: dict, after: dict) -> dict:
    """Structured difference between two scene/1 dicts.

    Returns {"added": [names], "removed": [names], "changed": {name: {field: [old, new]}},
             "scene": {field: [old, new]}, "materials_added": [...], "materials_removed": [...]}.
    Empty lists/dicts mean no change; `is_empty_diff()` checks for "nothing happened".
    """
    b_objs = {o["name"]: o for o in before["objects"]}
    a_objs = {o["name"]: o for o in after["objects"]}

    changed = {}
    for name in sorted(b_objs.keys() & a_objs.keys()):
        fields = {}
        for key in a_objs[name]:
            if key == "name":
                continue
            if b_objs[name].get(key) != a_objs[name][key]:
                fields[key] = [b_objs[name].get(key), a_objs[name][key]]
        if fields:
            changed[name] = fields

    scene_fields = {}
    for key in ("mode", "active", "selected", "cursor", "frame", "collections", "rigidbody_world", "mesh_select_mode"):
        if before.get(key) != after.get(key):
            scene_fields[key] = [before.get(key), after.get(key)]

    b_mats = {m["name"] for m in before["materials"]}
    a_mats = {m["name"] for m in after["materials"]}

    return {
        "added": sorted(a_objs.keys() - b_objs.keys()),
        "removed": sorted(b_objs.keys() - a_objs.keys()),
        "changed": changed,
        "scene": scene_fields,
        "materials_added": sorted(a_mats - b_mats),
        "materials_removed": sorted(b_mats - a_mats),
    }


def is_empty_diff(diff: dict) -> bool:
    return not any(diff.values())

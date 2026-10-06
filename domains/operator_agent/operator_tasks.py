"""Random scenes + spoken commands + expected op/1 calls, for the operator agent.

Each task family picks a target in a random scene, writes a command in one of
its phrasings, and states what the scene must look like afterwards. The
harness runs the calls and keeps the pair only if that expectation holds.

Phrasings are split: TRAIN phrasings build the training set, EVAL phrasings
only ever appear in the held-out set, so eval measures unseen wording.

This is the Phase 0 dry-run subset (~20 operators, template phrasings). Phase 1
widens it to the full registry with teacher-model paraphrases.
"""
import math
import random
import re
from dataclasses import dataclass, field

PRIMITIVES = {
    # kind: (operator, default object name, spoken names, size arg)
    "cube": ("mesh.primitive_cube_add", "Cube", ["cube", "box"], "size"),
    "sphere": ("mesh.primitive_uv_sphere_add", "Sphere", ["sphere", "ball"], "radius"),
    "cylinder": ("mesh.primitive_cylinder_add", "Cylinder", ["cylinder"], "radius"),
    "cone": ("mesh.primitive_cone_add", "Cone", ["cone"], "radius1"),
    "torus": ("mesh.primitive_torus_add", "Torus", ["torus", "donut"], "major_radius"),
    "plane": ("mesh.primitive_plane_add", "Plane", ["plane"], "size"),
    "monkey": ("mesh.primitive_monkey_add", "Suzanne", ["monkey", "suzanne"], "size"),
}
CUSTOM_NAMES = ["Table", "Pillar", "Rock", "Crate", "Tower", "Wheel", "Base", "Dome", "Post", "Block"]

MODIFIERS = {
    "BEVEL": ["bevel {t}", "add a bevel to {t}", "give {t} bevelled edges", "round off the edges of {t}"],
    "SUBSURF": ["add a subdivision surface to {t}", "subsurf {t}", "put a subdivision modifier on {t}"],
    "ARRAY": ["add an array modifier to {t}", "array {t}", "make an array of {t}"],
    "MIRROR": ["mirror {t}", "add a mirror modifier to {t}"],
    "SOLIDIFY": ["solidify {t}", "give {t} some thickness", "add solidify to {t}"],
    "WIREFRAME": ["turn {t} into a wireframe", "add a wireframe modifier to {t}"],
    "TRIANGULATE": ["triangulate {t}", "add a triangulate modifier to {t}"],
    "DECIMATE": ["decimate {t}", "add a decimate modifier to {t}"],
}
LIGHT_TYPES = {"POINT": ["point light", "point lamp"], "SUN": ["sun", "sun light"],
               "SPOT": ["spot light", "spotlight"], "AREA": ["area light"]}


@dataclass
class SceneObject:
    kind: str          # primitive kind, "light" or "camera"
    name: str
    spoken: str
    location: tuple
    rotation: tuple = (0, 0, 0)   # radians
    scale: tuple = (1, 1, 1)
    smooth: bool = False


@dataclass
class Scene:
    objects: list
    active: str | None
    selected: list
    mode: str = "OBJECT"

    def meshes(self):
        return [o for o in self.objects if o.kind in PRIMITIVES]

    def get(self, name):
        return next(o for o in self.objects if o.name == name)


@dataclass
class Task:
    family: str
    request: str
    calls: list
    expect: list = field(default_factory=list)


# --- scene generation ---------------------------------------------------------------

def _coord(rng):
    return rng.choice([rng.randint(-5, 5), round(rng.uniform(-5, 5), 1)])


def random_scene(rng: random.Random, mode: str = "OBJECT") -> Scene:
    kinds = rng.sample(sorted(PRIMITIVES), rng.randint(1, 4))
    objects, used_custom = [], set()
    for kind in kinds:
        _, default_name, spoken, _ = PRIMITIVES[kind]
        if rng.random() < 0.3:
            name = rng.choice([n for n in CUSTOM_NAMES if n not in used_custom])
            used_custom.add(name)
            objects.append(SceneObject(kind, name, name.lower(), (_coord(rng), _coord(rng), 0)))
        else:
            objects.append(SceneObject(kind, default_name, rng.choice(spoken), (_coord(rng), _coord(rng), 0)))
    if rng.random() < 0.5:
        objects.append(SceneObject("light", "Light", "light", (4, -4, 5)))
    if rng.random() < 0.4:
        objects.append(SceneObject("camera", "Camera", "camera", (7, -7, 5)))

    meshes = [o.name for o in objects if o.kind in PRIMITIVES]
    active = rng.choice(meshes) if (mode == "EDIT" or rng.random() < 0.7) else None
    selected = [active] if active and rng.random() < 0.8 else []
    return Scene(objects, active, selected, mode)


def setup_code(scene: Scene) -> str:
    lines = ["import bpy", "for o in list(bpy.data.objects):", "    bpy.data.objects.remove(o, do_unlink=True)"]
    for obj in scene.objects:
        loc = tuple(float(v) for v in obj.location)
        if obj.kind == "light":
            lines.append(f"bpy.ops.object.light_add(type='POINT', location={loc})")
        elif obj.kind == "camera":
            lines.append(f"bpy.ops.object.camera_add(location={loc})")
        else:
            lines.append(f"bpy.ops.{PRIMITIVES[obj.kind][0]}(location={loc})")
        lines.append(f"bpy.context.active_object.name = {obj.name!r}")
        if any(obj.rotation):
            lines.append(f"bpy.context.active_object.rotation_euler = {tuple(float(v) for v in obj.rotation)}")
        if tuple(obj.scale) != (1, 1, 1):
            lines.append(f"bpy.context.active_object.scale = {tuple(float(v) for v in obj.scale)}")
        if obj.smooth:
            lines.append("bpy.ops.object.shade_smooth()")
    lines.append("for o in bpy.context.scene.objects:\n    o.select_set(False)")
    lines.append("bpy.context.view_layer.objects.active = " + (f"bpy.data.objects[{scene.active!r}]" if scene.active else "None"))
    for name in scene.selected:
        lines.append(f"bpy.data.objects[{name!r}].select_set(True)")
    if scene.mode == "EDIT":
        lines.append("bpy.ops.object.mode_set(mode='EDIT')")
        lines.append("bpy.ops.mesh.select_all(action='SELECT')")
    return "\n".join(lines) + "\n"


# --- phrasing helpers ---------------------------------------------------------------

def _num(v):
    return str(int(v)) if float(v).is_integer() else str(v)


def _loc_phrase(rng, loc, prep="at"):
    x, y, z = (_num(v) for v in loc)
    return rng.choice([f"{prep} {x}, {y}, {z}", f"{prep} ({x}, {y}, {z})", f"{prep} x {x} y {y} z {z}", f"{prep} {x} {y} {z}"])


def _target(rng, scene: Scene, obj: SceneObject):
    """How the command refers to obj, and the calls needed to make it the active selection.

    Canonical rule the model learns: a named target gets a bai.select first,
    unless it is already the only selected object and the active one.
    """
    is_sole_active = scene.active == obj.name and scene.selected == [obj.name]
    if is_sole_active and rng.random() < 0.5:
        return rng.choice(["it", "this", "the selected object", "the active object"]), []
    ref = rng.choice([f"the {obj.spoken}", f"the {obj.spoken}", obj.name])
    return ref, ([] if is_sole_active else [{"op": "bai.select", "args": {"objects": [obj.name]}}])


def _pick(rng, phrasings: tuple, split: str):
    train, evals = phrasings
    return rng.choice(train if split == "train" else evals)


# --- task families ------------------------------------------------------------------
# Each takes (rng, scene, split) and returns a Task, or None if the scene doesn't suit it.

ADD_PHRASINGS = (
    ["add a {n}", "add a {n} {loc}", "create a {n} {loc}", "put a {n} {loc}", "new {n}",
     "add a {n} with {sz} {v}", "i need a {n}", "make a {n} {loc}"],
    ["gimme a {n} {loc}", "can you drop a {n} {loc}", "spawn a {n}", "throw in a {n} of {sz} {v}"],
)


def task_add(rng, scene, split):
    kind = rng.choice(sorted(PRIMITIVES))
    op, _, spoken, size_arg = PRIMITIVES[kind]
    args, words = {}, {"n": rng.choice(spoken), "loc": "", "sz": size_arg.rstrip("1").replace("_", " "), "v": ""}
    phrase = _pick(rng, ADD_PHRASINGS, split)
    if "{loc}" in phrase:
        loc = (_coord(rng), _coord(rng), rng.choice([0, 0, 1, 2]))
        args["location"] = [float(v) for v in loc]
        words["loc"] = _loc_phrase(rng, loc)
    if "{v}" in phrase:
        value = round(rng.uniform(0.5, 4), 1)
        args[size_arg] = value
        words["v"] = _num(value)
    request = phrase.format(**words)
    return Task("add_primitive", request, [{"op": op, "args": args}], [("added_types", ["MESH"])])


DELETE_PHRASINGS = (["delete {t}", "remove {t}", "get rid of {t}", "delete {t} please"],
                    ["nuke {t}", "erase {t} from the scene", "i don't want {t} anymore"])


def task_delete(rng, scene, split):
    obj = rng.choice(scene.meshes())
    ref, pre = _target(rng, scene, obj)
    request = _pick(rng, DELETE_PHRASINGS, split).format(t=ref)
    return Task("delete", request, pre + [{"op": "object.delete", "args": {}}], [("removed", [obj.name])])


def task_modifier(rng, scene, split):
    obj = rng.choice(scene.meshes())
    mod = rng.choice(sorted(MODIFIERS))
    phrasings = MODIFIERS[mod]
    cut = max(1, len(phrasings) - 1)
    ref, pre = _target(rng, scene, obj)
    request = rng.choice(phrasings[:cut] if split == "train" else phrasings[cut:]).format(t=ref)
    return Task("modifier", request, pre + [{"op": "object.modifier_add", "args": {"type": mod}}],
                [("modifier", obj.name, mod)])


SHADE_PHRASINGS = {
    True: (["shade {t} smooth", "smooth shade {t}", "make {t} smooth shaded"], ["make {t} look smooth, not faceted"]),
    False: (["shade {t} flat", "flat shade {t}", "make {t} flat shaded"], ["make {t} look faceted"]),
}


def task_shade(rng, scene, split):
    obj = rng.choice([o for o in scene.meshes() if o.kind != "plane"] or scene.meshes())
    smooth = obj.kind != "plane" and rng.random() < 0.8  # primitives start flat, so flat-shading needs a smooth start
    if not smooth:
        return None
    ref, pre = _target(rng, scene, obj)
    request = _pick(rng, SHADE_PHRASINGS[True], split).format(t=ref)
    return Task("shade", request, pre + [{"op": "object.shade_smooth", "args": {}}], [("smooth", obj.name)])


MOVE_PHRASINGS = (["move {t} {loc}", "put {t} {loc}", "set the location of {t} {loc}", "place {t} {loc}"],
                  ["shift {t} over {loc}", "relocate {t} {loc}"])


def task_move(rng, scene, split):
    obj = rng.choice(scene.meshes())
    loc = (_coord(rng), _coord(rng), rng.choice([0, 1, 2, 3]))
    ref = rng.choice([f"the {obj.spoken}", obj.name])
    request = _pick(rng, MOVE_PHRASINGS, split).format(t=ref, loc=_loc_phrase(rng, loc, prep=rng.choice(["to", "at"])))
    value = [float(v) for v in loc]
    return Task("move", request, [{"op": "bai.set", "args": {"object": obj.name, "property": "location", "value": value}}],
                [("prop", obj.name, "location", value)])


SCALE_PHRASINGS = (["scale {t} to {v}", "set the scale of {t} to {v}", "make {t} scale {v}"],
                   ["resize {t} to {v} times its default size"])


def task_scale(rng, scene, split):
    obj = rng.choice(scene.meshes())
    v = rng.choice([0.5, 1.5, 2, 3, 0.25, 4])
    ref = f"the {obj.spoken}"
    request = _pick(rng, SCALE_PHRASINGS, split).format(t=ref, v=_num(v))
    value = [float(v)] * 3
    return Task("scale", request, [{"op": "bai.set", "args": {"object": obj.name, "property": "scale", "value": value}}],
                [("prop", obj.name, "scale", value)])


ROTATE_PHRASINGS = (["rotate {t} {d} degrees around {a}", "rotate {t} {d} degrees on the {a} axis", "turn {t} {d} degrees about {a}"],
                    ["spin {t} {d} degrees on {a}"])


def task_rotate(rng, scene, split):
    obj = rng.choice(scene.meshes())
    degrees = rng.choice([15, 30, 45, 60, 90, 180, 270])
    axis = rng.choice("xyz")
    value = [0.0, 0.0, 0.0]
    value["xyz".index(axis)] = round(math.radians(degrees), 4)
    request = _pick(rng, ROTATE_PHRASINGS, split).format(t=f"the {obj.spoken}", d=degrees, a=axis if rng.random() < 0.5 else axis.upper())
    return Task("rotate", request, [{"op": "bai.set", "args": {"object": obj.name, "property": "rotation_euler", "value": value}}],
                [("prop", obj.name, "rotation", value)])


RENAME_PHRASINGS = (["rename {t} to {n}", "call {t} {n}", "name {t} {n}"], ["change the name of {t} to {n}"])


def task_rename(rng, scene, split):
    obj = rng.choice(scene.meshes())
    taken = {o.name for o in scene.objects}
    new_name = rng.choice([n for n in CUSTOM_NAMES + ["Hero", "Prop", "Detail", "Floor"] if n not in taken])
    request = _pick(rng, RENAME_PHRASINGS, split).format(t=f"the {obj.spoken}", n=new_name)
    return Task("rename", request, [{"op": "bai.set", "args": {"object": obj.name, "property": "name", "value": new_name}}],
                [("renamed", obj.name, new_name)])


HIDE_PHRASINGS = (["hide {t}", "hide {t} in the viewport"], ["make {t} invisible in the viewport"])


def task_hide(rng, scene, split):
    obj = rng.choice(scene.meshes())
    request = _pick(rng, HIDE_PHRASINGS, split).format(t=f"the {obj.spoken}")
    return Task("hide", request, [{"op": "bai.set", "args": {"object": obj.name, "property": "hidden", "value": True}}],
                [("hidden", obj.name)])


LIGHT_PHRASINGS = (["add a {l}", "add a {l} {loc}", "put a {l} {loc}", "create a {l}"], ["light it with a {l} {loc}", "i want a {l} up {loc}"])


def task_light(rng, scene, split):
    light_type = rng.choice(sorted(LIGHT_TYPES))
    phrase = _pick(rng, LIGHT_PHRASINGS, split)
    args = {"type": light_type}
    loc_text = ""
    if "{loc}" in phrase:
        loc = (_coord(rng), _coord(rng), rng.choice([3, 4, 5, 6]))
        args["location"] = [float(v) for v in loc]
        loc_text = _loc_phrase(rng, loc)
    request = phrase.format(l=rng.choice(LIGHT_TYPES[light_type]), loc=loc_text)
    return Task("light", request, [{"op": "object.light_add", "args": args}], [("added_types", ["LIGHT"])])


CAMERA_PHRASINGS = (["add a camera", "add a camera {loc}", "put a camera {loc}"], ["set up a new camera {loc}"])


def task_camera(rng, scene, split):
    phrase = _pick(rng, CAMERA_PHRASINGS, split)
    args, loc_text = {}, ""
    if "{loc}" in phrase:
        loc = (_coord(rng), _coord(rng), rng.choice([1, 2, 3, 5]))
        args["location"] = [float(v) for v in loc]
        loc_text = _loc_phrase(rng, loc)
    return Task("camera", phrase.format(loc=loc_text), [{"op": "object.camera_add", "args": args}], [("added_types", ["CAMERA"])])


SELECT_PHRASINGS = (["select {t}", "select {t} only", "click on {t}"], ["pick {t}"])
SELECT_ALL_PHRASINGS = {
    "SELECT": (["select all", "select everything", "select all objects"], ["grab everything"]),
    "DESELECT": (["deselect all", "deselect everything", "clear the selection"], ["unselect everything"]),
}


def task_select(rng, scene, split):
    roll = rng.random()
    if roll < 0.5:
        candidates = [o for o in scene.meshes() if scene.selected != [o.name]]
        if not candidates:
            return None
        obj = rng.choice(candidates)
        request = _pick(rng, SELECT_PHRASINGS, split).format(t=rng.choice([f"the {obj.spoken}", obj.name]))
        return Task("select", request, [{"op": "bai.select", "args": {"objects": [obj.name]}}],
                    [("selected", [obj.name]), ("active", obj.name)])
    action = "SELECT" if roll < 0.75 else "DESELECT"
    if action == "DESELECT" and not scene.selected:
        return None
    if action == "SELECT" and len(scene.selected) == len(scene.objects):
        return None
    request = _pick(rng, SELECT_ALL_PHRASINGS[action], split)
    expected = sorted(o.name for o in scene.objects) if action == "SELECT" else []
    return Task("select", request, [{"op": "object.select_all", "args": {"action": action}}], [("selected", expected)])


EDIT_PHRASINGS = (["go into edit mode", "enter edit mode", "edit mode", "tab into edit mode"], ["switch to edit mode please"])
EDIT_TARGET_PHRASINGS = (["edit {t}", "go into edit mode on {t}", "enter edit mode for {t}"], ["let me edit the mesh of {t}"])


def task_edit_mode(rng, scene, split):
    if scene.active and scene.selected == [scene.active] and rng.random() < 0.5:
        return Task("mode", _pick(rng, EDIT_PHRASINGS, split), [{"op": "object.mode_set", "args": {"mode": "EDIT"}}], [("mode", "EDIT_MESH")])
    obj = rng.choice(scene.meshes())
    ref, pre = _target(rng, scene, obj)
    if not pre and ref in {"it", "this"}:
        ref = f"the {obj.spoken}"
    request = _pick(rng, EDIT_TARGET_PHRASINGS, split).format(t=ref)
    return Task("mode", request, pre + [{"op": "object.mode_set", "args": {"mode": "EDIT"}}], [("mode", "EDIT_MESH"), ("active", obj.name)])


SUBDIVIDE_PHRASINGS = (["subdivide", "subdivide it {c} times", "subdivide with {c} cuts", "add {c} cuts"], ["chop the mesh up with {c} cuts"])
TRIS_PHRASINGS = (["triangulate the faces", "convert quads to tris", "make it all triangles"], ["turn the quads into triangles"])
EXIT_PHRASINGS = (["go back to object mode", "exit edit mode", "object mode"], ["leave edit mode"])


def task_edit_ops(rng, scene, split):
    """Only for scenes set up in Edit Mode with the whole mesh selected."""
    obj = scene.get(scene.active)
    roll = rng.random()
    if roll < 0.45:
        phrase = _pick(rng, SUBDIVIDE_PHRASINGS, split)
        cuts = rng.randint(1, 5) if "{c}" in phrase else 1
        args = {"number_cuts": cuts} if "{c}" in phrase else {}
        return Task("edit_subdivide", phrase.format(c=cuts), [{"op": "mesh.subdivide", "args": args}], [("faces_increase", obj.name)])
    if roll < 0.75 and obj.kind not in {"plane"}:
        return Task("edit_triangulate", _pick(rng, TRIS_PHRASINGS, split), [{"op": "mesh.quads_convert_to_tris", "args": {}}],
                    [("faces_increase", obj.name)])
    return Task("mode", _pick(rng, EXIT_PHRASINGS, split), [{"op": "object.mode_set", "args": {"mode": "OBJECT"}}], [("mode", "OBJECT")])


CHAIN_PHRASINGS = (["add a {n} and {m}", "create a {n} then {m}", "make a {n} and {m}"], ["drop in a {n} and {m}"])
CHAIN_MODS = {"BEVEL": "bevel it", "SUBSURF": "subsurf it", "SOLIDIFY": "solidify it", "ARRAY": "array it", "MIRROR": "mirror it"}


def task_chain(rng, scene, split):
    kind = rng.choice([k for k in PRIMITIVES if k != "plane"])
    op, _, spoken, _ = PRIMITIVES[kind]
    if rng.random() < 0.6:
        mod = rng.choice(sorted(CHAIN_MODS))
        request = _pick(rng, CHAIN_PHRASINGS, split).format(n=rng.choice(spoken), m=CHAIN_MODS[mod])
        return Task("chain", request, [{"op": op, "args": {}}, {"op": "object.modifier_add", "args": {"type": mod}}],
                    [("added_types", ["MESH"]), ("modifier", None, mod)])
    request = _pick(rng, CHAIN_PHRASINGS, split).format(n=rng.choice(spoken), m=rng.choice(["shade it smooth", "smooth shade it"]))
    return Task("chain", request, [{"op": op, "args": {}}, {"op": "object.shade_smooth", "args": {}}],
                [("added_types", ["MESH"]), ("smooth", None)])


# --- relative transforms (Phase 1) ---------------------------------------------------
# "Move it 2 on X" / "rotate it 45 more degrees" / "twice as big" use the transform
# operators, which act on the selection relative to where things are now. The model
# never sees rotation or scale in the compact scene, so relative wording must map to
# these, and bai.set is kept for absolute "set it to" wording.

def task_move_relative(rng, scene, split):
    obj = rng.choice(scene.meshes())
    axis = rng.randrange(3)
    amount = rng.choice([0.5, 1, 1, 2, 2, 3, 4, 5, 1.5, 0.25]) * rng.choice([1, -1])
    delta = [0.0, 0.0, 0.0]
    delta[axis] = float(amount)
    ref, pre = _target(rng, scene, obj)
    request = f"move {ref} {_num(amount)} along {'xyz'[axis]}"
    final = [round(float(v) + d, 4) for v, d in zip(obj.location, delta)]
    return Task("move_relative", request, pre + [{"op": "transform.translate", "args": {"value": delta}}],
                [("prop", obj.name, "location", final)])


def task_rotate_relative(rng, scene, split):
    obj = rng.choice(scene.meshes())
    axis = rng.choices([0, 1, 2], weights=[0.15, 0.15, 0.7])[0]  # people mostly turn things around Z
    start = [0.0, 0.0, 0.0]
    start[axis] = round(math.radians(rng.choice([0, 15, 30, 45, 90])), 4)
    obj.rotation = tuple(start)  # the task sets its own starting pose, so "more" is meaningful
    degrees = rng.choice([5, 10, 15, 30, 45, 60, 90, 120, 180])
    ref, pre = _target(rng, scene, obj)
    request = f"rotate {ref} {degrees} more degrees around {'xyz'[axis]}"
    final = list(start)
    final[axis] = round(start[axis] + math.radians(degrees), 4)
    return Task("rotate_relative", request,
                pre + [{"op": "transform.rotate", "args": {"value": round(math.radians(degrees), 4), "orient_axis": "XYZ"[axis]}}],
                [("prop", obj.name, "rotation", final)])


def task_scale_relative(rng, scene, split):
    obj = rng.choice(scene.meshes())
    start = rng.choice([1, 1, 0.5, 1.5, 2])
    obj.scale = (start, start, start)
    factor = rng.choice([2, 2, 3, 0.5, 0.5, 1.5, 0.25, 4])
    ref, pre = _target(rng, scene, obj)
    request = f"scale {ref} by {_num(factor)}"
    final = [round(start * factor, 4)] * 3
    return Task("scale_relative", request, pre + [{"op": "transform.resize", "args": {"value": [float(factor)] * 3}}],
                [("prop", obj.name, "scale", final)])


OBJECT_MODE_FAMILIES = [
    (task_add, 3), (task_delete, 2), (task_modifier, 3), (task_shade, 1), (task_move, 2), (task_scale, 1),
    (task_rotate, 1), (task_rename, 1), (task_hide, 1), (task_light, 1), (task_camera, 1), (task_select, 2),
    (task_edit_mode, 1), (task_chain, 2),
]
EDIT_MODE_SHARE = 0.1


def _typo(rng, text):
    words = text.split(" ")
    long_words = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not long_words:
        return text
    i = rng.choice(long_words)
    w = words[i]
    j = rng.randrange(len(w) - 1)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def _surface_noise(rng, request):
    """Casing, punctuation and occasional typos -- spoken input arrives messy."""
    request = " ".join(request.split())
    roll = rng.random()
    if roll < 0.35:
        request = request[0].upper() + request[1:]
    if rng.random() < 0.2:
        request += rng.choice([".", "!", " please", " pls"])
    if rng.random() < 0.08:
        request = _typo(rng, request)
    return request


def sample(rng: random.Random, split: str, families=OBJECT_MODE_FAMILIES, noise: bool = True) -> tuple[Scene, Task]:
    while True:
        if rng.random() < EDIT_MODE_SHARE:
            scene = random_scene(rng, mode="EDIT")
            task = task_edit_ops(rng, scene, split)
        else:
            scene = random_scene(rng)
            fns, weights = zip(*families)
            task = rng.choices(fns, weights=weights)[0](rng, scene, split)
        if task is not None:
            if noise:
                task.request = _surface_noise(rng, task.request)
            return scene, task


# --- scenes described by the teacher (Phase 1 Way 2) ---------------------------------

SPEC_KINDS = sorted(PRIMITIVES) + ["light", "camera"]
_SPEC_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_. -]{0,30}$")


def scene_from_spec(spec: dict) -> Scene:
    """Build a Scene from the teacher's JSON description. Raises ValueError on anything off-spec.

    The teacher never writes setup code: it names objects, kinds and transforms,
    and setup_code() turns that into bpy, so a bad spec fails here, not in Blender.
    """
    if not isinstance(spec, dict) or not isinstance(spec.get("objects"), list) or not 1 <= len(spec["objects"]) <= 12:
        raise ValueError("scene needs 1-12 objects")
    objects, names = [], set()
    for raw in spec["objects"]:
        if not isinstance(raw, dict):
            raise ValueError("object must be a JSON object")
        kind, name = raw.get("kind"), raw.get("name")
        if kind not in SPEC_KINDS:
            raise ValueError(f"unknown object kind {kind!r}")
        if not isinstance(name, str) or not _SPEC_NAME_RE.match(name) or name in names:
            raise ValueError(f"bad or duplicate object name {name!r}")
        names.add(name)
        location = _spec_vector(raw.get("location", [0, 0, 0]), "location")
        rotation = tuple(round(math.radians(v), 4) for v in _spec_vector(raw.get("rotation_degrees", [0, 0, 0]), "rotation_degrees"))
        scale = _spec_vector(raw.get("scale", [1, 1, 1]), "scale")
        if any(v <= 0 for v in scale):
            raise ValueError("scale must be positive")
        smooth = bool(raw.get("smooth", False)) and kind in PRIMITIVES
        objects.append(SceneObject(kind, name, name.lower(), location, rotation, scale, smooth))

    mode = spec.get("mode", "OBJECT")
    if mode not in ("OBJECT", "EDIT"):
        raise ValueError(f"mode must be OBJECT or EDIT, not {mode!r}")
    active = spec.get("active")
    selected = spec.get("selected", [])
    if active is not None and active not in names:
        raise ValueError(f"active object {active!r} is not in the scene")
    if not isinstance(selected, list) or any(n not in names for n in selected):
        raise ValueError("selected lists an object that is not in the scene")
    if mode == "EDIT":
        if active is None or next(o for o in objects if o.name == active).kind not in PRIMITIVES:
            raise ValueError("edit mode needs an active mesh object")
        selected = [active]
    return Scene(objects, active, sorted(set(selected)), mode)


def _spec_vector(value, what) -> tuple:
    if (not isinstance(value, list) or len(value) != 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) <= 1000 for v in value)):
        raise ValueError(f"{what} must be three numbers")
    return tuple(value)


# --- expectation check (host side, on the harness result) ----------------------------

def _close(a, b, tol=1e-3):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def check_expectations(expect: list, before: dict, after: dict, diff: dict) -> str | None:
    objs = {o["name"]: o for o in after["objects"]}
    added = diff["added"]
    for rule in expect:
        kind = rule[0]
        if kind == "added_types":
            types = sorted(objs[n]["type"] for n in added)
            if types != sorted(rule[1]):
                return f"expected new {rule[1]}, got {types}"
        elif kind == "removed":
            if sorted(diff["removed"]) != sorted(rule[1]):
                return f"expected {rule[1]} removed, got {diff['removed']}"
        elif kind == "modifier":
            name = rule[1] or (added[0] if added else None)
            if name not in objs or rule[2] not in [m["type"] for m in objs[name]["modifiers"]]:
                return f"{name} has no {rule[2]} modifier"
        elif kind == "smooth":
            name = rule[1] or (added[0] if added else None)
            data = objs.get(name, {}).get("data", {})
            if not data or data["smooth_faces"] != data["faces"]:
                return f"{name} is not smooth shaded"
        elif kind == "prop":
            _, name, field_name, value = rule
            if name not in objs or not _close(objs[name][field_name], value):
                return f"{name}.{field_name} != {value}"
        elif kind == "renamed":
            if rule[1] in objs or rule[2] not in objs:
                return f"{rule[1]} was not renamed to {rule[2]}"
        elif kind == "hidden":
            if not objs.get(rule[1], {}).get("hidden"):
                return f"{rule[1]} is not hidden"
        elif kind == "selected":
            if after["selected"] != sorted(rule[1]):
                return f"selection is {after['selected']}, expected {sorted(rule[1])}"
        elif kind == "active":
            if after["active"] != rule[1]:
                return f"active is {after['active']}, expected {rule[1]}"
        elif kind == "mode":
            if after["mode"] != rule[1]:
                return f"mode is {after['mode']}, expected {rule[1]}"
        elif kind == "faces_increase":
            b = {o["name"]: o for o in before["objects"]}[rule[1]]["data"]["faces"]
            if objs[rule[1]]["data"]["faces"] <= b:
                return f"{rule[1]} face count did not grow"
        else:
            raise ValueError(f"unknown expectation {kind!r}")
    return None

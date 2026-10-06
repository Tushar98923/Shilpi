# Frozen formats

Every dataset is built in these formats. Changing one means regenerating data,
so don't edit a format in place: add a new version (`op/2`) and migrate.
The code lives in `shared/output_formats.py` and `shared/scene_serializer.py`,
and the add-on vendors both byte for byte (`scripts/build_addon.py`).

**Pinned Blender: 5.2.1 LTS** (`PINNED_BLENDER_VERSION`). The harness refuses
any other build unless `BAI_ALLOW_VERSION_MISMATCH=1` is set.

## Model input: the same for every model

```
system: <the model's fixed system prompt>
user:   Scene: {compact scene JSON}
        Request: <what the user said>
```

`build_user_message(request, compact_scene(serialize_scene()))` builds it. The
compact scene holds mode, active object, selection, and each object's name,
type, location, modifiers and materials (at most 40 objects).

## Scene state: `scene/1`

`serialize_scene()` captures mode, active object, selection, cursor, frame, and for
every object its type, parent, collections, transforms, dimensions, visibility,
material slots, modifiers and type-specific data (mesh counts, light/camera settings).
It also records the materials in use. Floats are rounded to 4 digits, so the same
scene always serializes identically. `diff_scenes()` reports what was added,
removed and changed.

## Operator agent: `op/1`

```json
{"calls": [{"op": "bai.select", "args": {"objects": ["Cone"]}},
           {"op": "object.modifier_add", "args": {"type": "BEVEL"}}]}
```

- `op` is a `bpy.ops` name without the `bpy.ops.` prefix, or a built-in:
  - `bai.select {objects: [..], active?: str, extend?: bool}`: select by name (there is no headless-safe operator for this)
  - `bai.set {object, property, value}`: `property` is one of `location`, `rotation_euler` (radians), `scale`, `name`, `hidden` (eye icon), `hide_render`
  - `bai.keyframe {object, property, frame?: int, delete?: bool}` (added 2026-10-01): insert or delete a keyframe; `property` is `location`, `rotation_euler`, `scale` or `all` (the I key's LocRotScale). The `anim.keyframe_*` operators need an editor context
  - `bai.frame {frame: int}` (added 2026-10-01): jump to a frame
  - `bai.collection {name, link?: bool}` (added 2026-10-01): move the selected objects into a collection by name, creating it if needed; `link: true` adds without removing from other collections. `object.move_to_collection` needs a per-session collection id a model can't know
  - `bai.ask {question}` (added 2026-10-02): ask the user instead of acting when the request is ambiguous (two look-alikes, neither selected). Must be the only call; the executor raises `AskUser`
  - `bai.decline {reason}` (added 2026-10-03): say why nothing will happen (unrelated request, another specialist's job, object not in the scene). Must be the only call; the executor raises `Declined`
  - `bai.loopcut {cuts, axis}` (added 2026-10-03): loop cut through the edge ring along `axis` (Z = a horizontal ring), in Edit Mode. Replaces `mesh.loopcut_slide`, which needs a mouse position
- Constrained decoding: `shared/data/op_schema.json` (built by `scripts/build_op_schema.py`) lists every allowed operator with its argument names and enum values, in the order the training data writes them (llama-server's grammar enforces schema order)
- With a UI, `render.render`, `render.opengl` and `wm.save_mainfile` run as INVOKE (like F12 / Ctrl+S); headless they stay EXEC
- Adding a built-in is additive: existing data stays valid. Changing one is a format bump.
- Argument names and enum values are checked against `shared/data/operator_registry.json`
  (all 2,499 operators in 5.2.1, with properties, defaults and headless poll results).
  Enum-flag arguments are JSON lists (the executor turns them into sets); dynamic enums that
  introspect as empty (e.g. constraint types) are checked by running, not by the registry.
- Which operators the operator agent learns: `domains/operator_agent/operator_shortlist.py`,
  probed by `scripts/probe_shortlist.py` into `shared/data/operator_shortlist.json`.
- Convention the data teaches: a named target gets a `bai.select` first, unless it is
  already the only selected object and the active one. Relative wording ("move it 2 on x",
  "twice as big") uses `transform.translate/rotate/resize`; absolute ("set it to") uses `bai.set`.
- Training targets are written by `format_op_output()` (stable key order).

## Script specialists: `bai-script/1`

The raw script, without a markdown fence. The first two lines are fixed:

```python
# bai-script/1 specialist=modeling
import bpy
...
```

Specialist names are frozen in `SCRIPT_SPECIALISTS`: code_repair, materials, lighting,
camera, render, modeling, geometry_nodes, animation, rigging, simulation, uv, set_dressing.
`with_header()` converts existing fenced scripts. All 19,085 verified mesh_generation
scripts convert cleanly, so no data needs regenerating.

## Planner: `plan/1`

```json
{"steps": [{"specialist": "modeling", "request": "a round oak table"},
           {"specialist": "materials", "request": "oak wood on the table"}]}
```

A step can only name `operator` or a script specialist.

## Router

One label from `ROUTER_LABELS`, as plain text.

Vision, critic, art director and tutor outputs are defined when their phases start.

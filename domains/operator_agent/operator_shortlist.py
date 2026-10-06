"""The Phase 1 operator shortlist: the everyday operators the operator agent learns.

Scope rule: one-shot actions a user would say out loud ("bevel it", "unwrap it",
"top view"). Multi-step jobs belong to the Phase 5 script specialists; node
editors, sequencer, clip, graph/NLA, outliner and grease pencil are out; so are
dangerous operators (quit, open/revert, factory reset, purge), UI internals and
mouse-only tools (brush strokes, knife, rip).

Each entry: op -> (group, setup, example args, verify)
    setup   -- a key of SETUPS: the starting scene the operator is tested in
    args    -- one representative call; generators sample their own values later
    verify  -- "harness": must run headless and change the scene
               "viewport": needs a real 3D viewport, so it can't run headless.
                           Way 1 only (code writes the answer); never Way 2.

scripts/probe_shortlist.py runs every "harness" entry and writes the results to
shared/data/operator_shortlist.json, which the generators read.
"""

_SELECT_ONLY = """
for o in bpy.context.scene.objects:
    o.select_set(False)
"""

_BASE = """import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.context.scene.cursor.location = (1.0, 1.0, 0.0)
bpy.ops.mesh.primitive_uv_sphere_add(location=(3, 0, 0))
bpy.ops.mesh.primitive_cylinder_add(location=(-3, 0, 0))
bpy.ops.object.light_add(type='POINT', location=(4, -4, 5))
bpy.context.active_object.name = 'Light'  # Blender 5 names a new point light 'Point'; the default scene's is 'Light'
bpy.ops.object.camera_add(location=(7, -7, 5))
bpy.context.scene.camera = bpy.context.active_object
bpy.ops.mesh.primitive_cube_add(location=(0.5, 0, 0), rotation=(0, 0, 0.3), scale=(1.2, 1.2, 1.2))
""" + _SELECT_ONLY + """
cube = bpy.data.objects['Cube']
cube.scale = (1.2, 1.2, 1.2)  # the add operator's scale argument doesn't set object scale
cube.select_set(True)
bpy.context.view_layer.objects.active = cube
"""


def _edit(select: str = "SELECT", extra: str = "") -> str:
    return _BASE + extra + f"bpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.mesh.select_all(action='{select}')\n"


def _replace_active(add: str) -> str:
    """The base scene, but with the active object swapped for another primitive."""
    return _BASE + "bpy.data.objects.remove(cube, do_unlink=True)\n" + add + """
obj = bpy.context.active_object
""" + _SELECT_ONLY + """
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
"""


_FACE_TOP = """import bmesh
bm = bmesh.from_edit_mesh(bpy.context.object.data)
for f in bm.faces:
    f.select_set(False)
top = max(bm.faces, key=lambda f: f.calc_center_median().z)
top.select_set(True)
bm.faces.active = top
bmesh.update_edit_mesh(bpy.context.object.data)
"""
_TWO_VERTS = """import bmesh
bm = bmesh.from_edit_mesh(bpy.context.object.data)
for v in bm.verts:
    v.select_set(False)
bm.faces.ensure_lookup_table()
face = bm.faces[0]
face.verts[0].select_set(True)
face.verts[2].select_set(True)
bm.select_flush_mode()
bmesh.update_edit_mesh(bpy.context.object.data)
"""
_ONE_VERT = """import bmesh
bm = bmesh.from_edit_mesh(bpy.context.object.data)
for v in bm.verts:
    v.select_set(False)
bm.verts.ensure_lookup_table()
bm.verts[0].select_set(True)
bm.select_flush_mode()
bmesh.update_edit_mesh(bpy.context.object.data)
"""
_ARMATURE = """import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.object.armature_add(location=(0, 0, 0))
arm = bpy.context.active_object
"""
_CURVE = """import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.curve.primitive_bezier_curve_add(location=(0, 0, 0))
"""

SETUPS = {
    "object": _BASE,
    "object_two": _BASE + "bpy.data.objects['Sphere'].select_set(True)\n",  # Sphere selected, Cube active
    "object_linked": _BASE + "bpy.ops.object.duplicate_move_linked()\n",
    "object_parented": _BASE + "bpy.data.objects['Sphere'].parent = cube\n",
    "object_child": _BASE + """sphere = bpy.data.objects['Sphere']
sphere.parent = cube
cube.select_set(False)
sphere.select_set(True)
bpy.context.view_layer.objects.active = sphere
""",
    "object_smooth": _BASE + "bpy.ops.object.shade_smooth()\n",
    "object_material": _BASE + """mat = bpy.data.materials.new('Red')
cube.data.materials.append(mat)
cube.data.materials.append(None)
bpy.data.objects['Sphere'].select_set(True)
""",
    "object_modifiers": _BASE + """cube.modifiers.new('Bevel', 'BEVEL')
cube.modifiers.new('Subdivision', 'SUBSURF')
""",
    "object_constraint": _BASE + "c = cube.constraints.new('TRACK_TO')\nc.target = bpy.data.objects['Sphere']\n",
    "object_physics": _BASE + "bpy.ops.rigidbody.object_add()\nbpy.ops.object.particle_system_add()\n",
    "object_keyed": _BASE + """cube.keyframe_insert('location', frame=1)
cube.keyframe_insert('location', frame=20)
cube.keyframe_insert('rotation_euler', frame=20)
""",
    "object_curve": _replace_active("bpy.ops.curve.primitive_bezier_curve_add(location=(0, 0, 0))"),
    "object_text": _replace_active("bpy.ops.object.text_add(location=(0, 0, 0))"),
    "object_armature": _replace_active("bpy.ops.object.armature_add(location=(0, 0, 0))"),
    "object_rig": _BASE + """bpy.ops.object.armature_add(location=(0.5, 0, -1))
arm = bpy.context.active_object
arm.select_set(True)
cube.select_set(True)
bpy.context.view_layer.objects.active = arm
""",
    "object_uv": _BASE + "cube.data.uv_layers.new(name='Extra')\n",
    "mesh_edit": _edit(),
    "mesh_edit_none": _edit("DESELECT"),
    "mesh_edit_face": _edit(extra="") + _FACE_TOP,
    "mesh_edit_face_grown": _edit() + _FACE_TOP + "bpy.ops.mesh.select_more()\n",
    "mesh_edit_face_mode": _edit() + _FACE_TOP + "bpy.ops.mesh.select_mode(type='FACE')\n",
    "mesh_edit_smooth": _edit(extra="bpy.ops.object.shade_smooth()\n"),
    "mesh_edit_all_active": _edit() + _FACE_TOP + "bpy.ops.mesh.select_all(action='SELECT')\n",  # all selected, top face active
    "mesh_edit_uv_uneven": _edit() + """import bmesh
bm = bmesh.from_edit_mesh(bpy.context.object.data)
uv = bm.loops.layers.uv.active
bm.faces.ensure_lookup_table()
for loop in bm.faces[0].loops:  # shrink one face's UVs so island scales differ
    loop[uv].uv *= 0.3
bmesh.update_edit_mesh(bpy.context.object.data)
""",
    "mesh_edit_two_verts": _edit() + _TWO_VERTS,
    "mesh_edit_vert": _edit() + _ONE_VERT,
    "mesh_edit_hidden": _edit() + _FACE_TOP + "bpy.ops.mesh.hide(unselected=False)\n",
    "mesh_edit_flipped": _edit() + "bpy.ops.mesh.flip_normals()\n",
    "mesh_edit_tris": _edit() + "bpy.ops.mesh.quads_convert_to_tris()\n",
    "mesh_edit_tris_none": _edit() + "bpy.ops.mesh.quads_convert_to_tris()\nbpy.ops.mesh.select_all(action='DESELECT')\n",
    "mesh_edit_doubles": _edit() + "bpy.ops.mesh.duplicate()\nbpy.ops.mesh.select_all(action='SELECT')\n",
    "mesh_edit_hole": _edit() + _FACE_TOP + "bpy.ops.mesh.delete(type='FACE')\nbpy.ops.mesh.select_all(action='SELECT')\n",
    "mesh_edit_asym": _edit(extra="cube.data.vertices[0].co.x -= 0.4\n"),
    "grid_edit": _replace_active("bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8, size=2)")
                 + "bpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.mesh.select_all(action='SELECT')\n",
    "grid_edit_none": _replace_active("bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8, size=2)")
                      + "bpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.mesh.select_all(action='DESELECT')\n",
    "grid_edit_face": _replace_active("bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8, size=2)")
                      + "bpy.ops.object.mode_set(mode='EDIT')\n" + _FACE_TOP,
    "grid_edit_face_grown": _replace_active("bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8, size=2)")
                            + "bpy.ops.object.mode_set(mode='EDIT')\n" + _FACE_TOP + "bpy.ops.mesh.select_more()\n",
    "grid_edit_subdivided": _replace_active("bpy.ops.mesh.primitive_plane_add(size=2)")
                            + "bpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.mesh.select_all(action='SELECT')\n"
                            + "bpy.ops.mesh.subdivide(number_cuts=3)\n",
    "circle_edit": _replace_active("bpy.ops.mesh.primitive_circle_add(vertices=16, location=(0, 0, 0))")
                   + "bpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.mesh.select_all(action='SELECT')\n",
    "two_loops_edit": _replace_active("bpy.ops.mesh.primitive_circle_add(vertices=16, location=(0, 0, 0))")
                      + "bpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.mesh.primitive_circle_add(vertices=16, location=(0, 0, 2))\n"
                      + "bpy.ops.mesh.select_all(action='SELECT')\n",
    "two_boxes_edit": _edit() + "bpy.ops.mesh.select_all(action='DESELECT')\n"
                      + "bpy.ops.mesh.primitive_cube_add(size=1, location=(1.7, 0.0, 0.5))\n",  # straddles the cube's +X face
    "monkey_edit": _replace_active("bpy.ops.mesh.primitive_monkey_add(location=(0, 0, 0))")
                   + "bpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.mesh.select_all(action='SELECT')\n",
    "monkey_sculpt": _replace_active("bpy.ops.mesh.primitive_monkey_add(location=(0, 0, 0))")
                     + "bpy.ops.object.mode_set(mode='SCULPT')\n",
    "monkey_sculpt_asym": _replace_active("bpy.ops.mesh.primitive_monkey_add(location=(0, 0, 0))")
                          + "bpy.context.object.data.vertices[0].co.x += 0.3\nbpy.ops.object.mode_set(mode='SCULPT')\n",
    "armature_edit": _ARMATURE + "bpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.armature.select_all(action='SELECT')\n",
    "armature_edit_tip": _ARMATURE + """bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.armature.select_all(action='DESELECT')
bone = arm.data.edit_bones[0]
bone.select_tail = True
""",
    "pose": _ARMATURE + """bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.select_all(action='SELECT')
pb = arm.pose.bones[0]
pb.location = (0.3, 0, 0)
pb.rotation_quaternion = (0.9, 0.3, 0, 0)
pb.scale = (1.5, 1.5, 1.5)
""",
    "curve_edit": _CURVE + "bpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.curve.select_all(action='SELECT')\n",
    "curve_edit_dense": _CURVE + "bpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.curve.select_all(action='SELECT')\n"
                        + "bpy.ops.curve.subdivide(number_cuts=4)\n",
    "curve_edit_end": _CURVE + """bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.curve.select_all(action='DESELECT')
bpy.context.object.data.splines[0].bezier_points[-1].select_control_point = True
""",
}

H, V = "harness", "viewport"

SHORTLIST = {
    # --- adding things --------------------------------------------------------------------
    "mesh.primitive_cube_add": ("add", "object", {"size": 2.0, "location": [0.0, 0.0, 2.0]}, H),
    "mesh.primitive_uv_sphere_add": ("add", "object", {"radius": 1.0}, H),
    "mesh.primitive_ico_sphere_add": ("add", "object", {"subdivisions": 2}, H),
    "mesh.primitive_cylinder_add": ("add", "object", {"vertices": 16, "depth": 2.0}, H),
    "mesh.primitive_cone_add": ("add", "object", {"radius1": 1.0}, H),
    "mesh.primitive_torus_add": ("add", "object", {"major_radius": 1.0}, H),
    "mesh.primitive_plane_add": ("add", "object", {"size": 4.0}, H),
    "mesh.primitive_circle_add": ("add", "object", {"vertices": 32}, H),
    "mesh.primitive_grid_add": ("add", "object", {"x_subdivisions": 10, "y_subdivisions": 10}, H),
    "mesh.primitive_monkey_add": ("add", "object", {}, H),
    "curve.primitive_bezier_curve_add": ("add", "object", {}, H),
    "curve.primitive_bezier_circle_add": ("add", "object", {"radius": 1.0}, H),
    "curve.primitive_nurbs_curve_add": ("add", "object", {}, H),
    "curve.primitive_nurbs_circle_add": ("add", "object", {}, H),
    "curve.primitive_nurbs_path_add": ("add", "object", {}, H),
    "object.empty_add": ("add", "object", {"type": "PLAIN_AXES"}, H),
    "object.light_add": ("add", "object", {"type": "SUN"}, H),
    "object.camera_add": ("add", "object", {"location": [0.0, -8.0, 2.0]}, H),
    "object.text_add": ("add", "object", {}, H),
    "object.armature_add": ("add", "object", {}, H),
    "object.metaball_add": ("add", "object", {"type": "BALL"}, H),
    "object.speaker_add": ("add", "object", {}, H),
    "object.lightprobe_add": ("add", "object", {"type": "SPHERE"}, H),
    "object.volume_add": ("add", "object", {}, H),
    "object.effector_add": ("add", "object", {"type": "WIND"}, H),

    # --- delete / duplicate / join ----------------------------------------------------------
    "object.delete": ("object", "object", {}, H),
    "object.duplicate_move": ("object", "object", {"TRANSFORM_OT_translate": {"value": [2.0, 0.0, 0.0]}}, H),
    "object.duplicate_move_linked": ("object", "object", {"TRANSFORM_OT_translate": {"value": [0.0, 2.0, 0.0]}}, H),
    "object.join": ("object", "object_two", {}, H),

    # --- selecting objects ------------------------------------------------------------------
    "object.select_all": ("select", "object", {"action": "SELECT"}, H),
    "object.select_by_type": ("select", "object", {"type": "LIGHT"}, H),
    "object.select_random": ("select", "object", {"ratio": 0.5, "seed": 3}, H),
    "object.select_pattern": ("select", "object", {"pattern": "Sph*"}, H),
    "object.select_camera": ("select", "object", {}, H),
    "object.select_hierarchy": ("select", "object_parented", {"direction": "CHILD"}, H),
    "object.select_grouped": ("select", "object_parented", {"type": "CHILDREN_RECURSIVE"}, V),  # poll needs a viewport

    # --- transforms ---------------------------------------------------------------------------
    "transform.translate": ("transform", "object", {"value": [2.0, 0.0, 0.0]}, H),
    "transform.rotate": ("transform", "object", {"value": 0.7854, "orient_axis": "Z"}, H),
    "transform.resize": ("transform", "object", {"value": [2.0, 2.0, 2.0]}, H),
    "transform.mirror": ("transform", "object", {"constraint_axis": [True, False, False]}, H),
    "object.location_clear": ("transform", "object", {}, H),
    "object.rotation_clear": ("transform", "object", {}, H),
    "object.scale_clear": ("transform", "object", {}, H),
    "object.origin_set": ("transform", "object", {"type": "ORIGIN_CURSOR"}, H),
    "object.transform_apply": ("transform", "object", {"location": False, "rotation": True, "scale": True}, H),
    "object.randomize_transform": ("transform", "object", {"loc": [1.0, 1.0, 0.0]}, H),
    "object.align": ("transform", "object_two", {"align_axis": ["X"]}, H),

    # --- parenting / links ----------------------------------------------------------------------
    "object.parent_set": ("relations", "object_two", {"type": "OBJECT"}, H),
    "object.parent_no_inverse_set": ("relations", "object_two", {}, H),
    "object.parent_clear": ("relations", "object_child", {"type": "CLEAR"}, H),
    "object.track_set": ("relations", "object_two", {"type": "TRACKTO"}, H),
    "object.track_clear": ("relations", "object_constraint", {"type": "CLEAR"}, H),
    "object.make_links_data": ("relations", "object_material", {"type": "MATERIAL"}, H),
    "object.make_single_user": ("relations", "object_linked", {"object": True, "obdata": True}, H),

    # --- shading ----------------------------------------------------------------------------------
    "object.shade_smooth": ("shading", "object", {}, H),
    "object.shade_flat": ("shading", "object_smooth", {}, H),
    "object.shade_auto_smooth": ("shading", "object", {"angle": 0.5236}, H),
    "object.shade_smooth_by_angle": ("shading", "object", {"angle": 0.5236}, H),

    # --- modifiers ------------------------------------------------------------------------------------
    "object.modifier_add": ("modifiers", "object", {"type": "BEVEL"}, H),
    "object.modifier_apply": ("modifiers", "object_modifiers", {"modifier": "Bevel"}, H),
    "object.modifier_remove": ("modifiers", "object_modifiers", {"modifier": "Bevel"}, H),
    "object.modifier_move_up": ("modifiers", "object_modifiers", {"modifier": "Subdivision"}, H),
    "object.modifier_move_down": ("modifiers", "object_modifiers", {"modifier": "Bevel"}, H),
    "object.modifier_copy": ("modifiers", "object_modifiers", {"modifier": "Bevel"}, H),
    "object.modifiers_clear": ("modifiers", "object_modifiers", {}, H),
    "object.subdivision_set": ("modifiers", "object", {"level": 2}, H),

    # --- convert / remesh ----------------------------------------------------------------------------
    "object.convert": ("convert", "object_curve", {"target": "MESH"}, H),
    "object.voxel_remesh": ("convert", "object", {}, H),

    # --- modes ------------------------------------------------------------------------------------------
    "object.mode_set": ("modes", "object", {"mode": "EDIT"}, H),
    "object.editmode_toggle": ("modes", "object", {}, H),
    "object.posemode_toggle": ("modes", "object_armature", {}, H),
    "sculpt.sculptmode_toggle": ("modes", "object", {}, H),
    "paint.weight_paint_toggle": ("modes", "object", {}, H),
    "paint.vertex_paint_toggle": ("modes", "object", {}, H),
    "paint.texture_paint_toggle": ("modes", "object", {}, H),

    # --- collections -----------------------------------------------------------------------------------
    "bai.collection": ("collections", "object_two", {"name": "Props"}, H),

    # --- materials --------------------------------------------------------------------------------------
    "material.new": ("materials", "object", {}, H),
    "object.material_slot_add": ("materials", "object", {}, H),
    "object.material_slot_remove": ("materials", "object_material", {}, H),
    "object.material_slot_remove_unused": ("materials", "object_material", {}, H),

    # --- object data ------------------------------------------------------------------------------------
    "object.vertex_group_add": ("data", "object", {}, H),
    "object.vertex_group_assign_new": ("data", "mesh_edit_face", {}, H),
    "object.shape_key_add": ("data", "object", {"from_mix": False}, H),
    "mesh.uv_texture_add": ("data", "object", {}, H),
    "mesh.uv_texture_remove": ("data", "object_uv", {}, H),
    "geometry.color_attribute_add": ("data", "object", {"name": "Color"}, H),

    # --- physics ----------------------------------------------------------------------------------------
    "rigidbody.object_add": ("physics", "object", {"type": "ACTIVE"}, H),
    "rigidbody.objects_add": ("physics", "object_two", {"type": "PASSIVE"}, H),
    "rigidbody.object_remove": ("physics", "object_physics", {}, H),
    "rigidbody.world_add": ("physics", "object", {}, H),
    "object.particle_system_add": ("physics", "object", {}, H),
    "object.particle_system_remove": ("physics", "object_physics", {}, H),
    "object.forcefield_toggle": ("physics", "object", {}, H),
    "object.quick_smoke": ("physics", "object", {}, H),
    "object.quick_liquid": ("physics", "object", {}, H),
    "object.quick_fur": ("physics", "object", {}, H),
    "object.quick_explode": ("physics", "object", {}, H),

    # --- constraints -------------------------------------------------------------------------------------
    "object.constraint_add": ("constraints", "object", {"type": "COPY_LOCATION"}, H),
    "object.constraints_clear": ("constraints", "object_constraint", {}, H),
    "constraint.delete": ("constraints", "object_constraint", {"constraint": "Track To", "owner": "OBJECT"}, H),
    "constraint.apply": ("constraints", "object_constraint", {"constraint": "Track To", "owner": "OBJECT"}, H),

    # --- mesh edit: selection -------------------------------------------------------------------------
    "mesh.select_all": ("mesh_select", "mesh_edit", {"action": "DESELECT"}, H),
    "mesh.select_mode": ("mesh_select", "mesh_edit_face", {"type": "EDGE"}, H),
    "mesh.select_more": ("mesh_select", "mesh_edit_face", {}, H),
    "mesh.select_less": ("mesh_select", "grid_edit_face_grown", {}, H),
    "mesh.select_random": ("mesh_select", "grid_edit_none", {"ratio": 0.3, "seed": 1}, H),
    "mesh.select_nth": ("mesh_select", "grid_edit", {"skip": 1}, H),
    "mesh.select_linked": ("mesh_select", "mesh_edit_face", {}, H),
    "mesh.select_non_manifold": ("mesh_select", "grid_edit_none", {}, H),
    "mesh.select_face_by_sides": ("mesh_select", "mesh_edit_tris_none", {"number": 3, "type": "EQUAL"}, H),
    "mesh.select_similar": ("mesh_select", "mesh_edit_face_mode", {"type": "FACE_SIDES"}, H),
    "mesh.edges_select_sharp": ("mesh_select", "mesh_edit_none", {}, H),
    "mesh.region_to_loop": ("mesh_select", "mesh_edit_face", {}, H),
    "mesh.faces_select_linked_flat": ("mesh_select", "grid_edit_face", {}, H),

    # --- mesh edit: modelling ---------------------------------------------------------------------------
    "mesh.extrude_region_move": ("mesh_model", "mesh_edit_face", {"TRANSFORM_OT_translate": {"value": [0.0, 0.0, 1.0]}}, H),
    "mesh.extrude_region_shrink_fatten": ("mesh_model", "mesh_edit_face", {"TRANSFORM_OT_shrink_fatten": {"value": 0.5}}, H),
    "mesh.extrude_faces_move": ("mesh_model", "mesh_edit", {"TRANSFORM_OT_shrink_fatten": {"value": 0.3}}, H),
    "mesh.extrude_vertices_move": ("mesh_model", "mesh_edit_vert", {"TRANSFORM_OT_translate": {"value": [0.0, 0.0, 1.0]}}, H),
    "mesh.extrude_repeat": ("mesh_model", "mesh_edit_face", {"steps": 3}, H),
    "mesh.inset": ("mesh_model", "mesh_edit_face", {"thickness": 0.2}, H),
    "mesh.bevel": ("mesh_model", "mesh_edit", {"offset": 0.1, "segments": 2}, H),
    "mesh.subdivide": ("mesh_model", "mesh_edit", {"number_cuts": 2}, H),
    "mesh.unsubdivide": ("mesh_model", "grid_edit_subdivided", {}, H),
    "mesh.bridge_edge_loops": ("mesh_model", "two_loops_edit", {}, H),
    "mesh.fill": ("mesh_model", "circle_edit", {}, H),
    "mesh.fill_grid": ("mesh_model", "circle_edit", {}, H),
    "mesh.fill_holes": ("mesh_model", "mesh_edit_hole", {"sides": 0}, H),
    "mesh.edge_face_add": ("mesh_model", "circle_edit", {}, H),
    "mesh.spin": ("mesh_model", "mesh_edit_face", {"steps": 6, "angle": 1.5708, "center": [0.0, 3.0, 0.0], "axis": [1.0, 0.0, 0.0]}, H),
    "mesh.poke": ("mesh_model", "mesh_edit_face", {}, H),
    "mesh.bisect": ("mesh_model", "mesh_edit", {"plane_co": [0.0, 0.0, 0.0], "plane_no": [0.0, 0.0, 1.0]}, H),
    "mesh.solidify": ("mesh_model", "mesh_edit_face", {"thickness": 0.1}, H),
    "mesh.wireframe": ("mesh_model", "mesh_edit", {"thickness": 0.05}, H),
    "mesh.symmetrize": ("mesh_model", "mesh_edit_asym", {"direction": "NEGATIVE_X"}, H),
    "mesh.symmetry_snap": ("mesh_model", "mesh_edit_asym", {"threshold": 0.5}, H),
    "mesh.convex_hull": ("mesh_model", "monkey_edit", {}, H),
    "mesh.intersect_boolean": ("mesh_model", "two_boxes_edit", {"operation": "UNION"}, H),
    "mesh.intersect": ("mesh_model", "two_boxes_edit", {}, H),
    "mesh.duplicate_move": ("mesh_model", "mesh_edit_face", {"TRANSFORM_OT_translate": {"value": [0.0, 0.0, 1.0]}}, H),
    "mesh.separate": ("mesh_model", "mesh_edit_face", {"type": "SELECTED"}, H),
    "mesh.split": ("mesh_model", "mesh_edit_face", {}, H),
    "mesh.edge_split": ("mesh_model", "mesh_edit", {}, H),
    "mesh.vert_connect_path": ("mesh_model", "mesh_edit_two_verts", {}, H),
    "mesh.vert_connect": ("mesh_model", "mesh_edit_two_verts", {}, H),
    # Loop cut by voice: mesh.loopcut_slide needs a mouse position (and crashes headless), so loop cuts use
    # the bai.loopcut built-in, which cuts the edge ring along an axis. Harness-verified.
    "bai.loopcut": ("mesh_model", "mesh_edit", {"cuts": 1, "axis": "Z"}, H),

    # --- mesh edit: delete / cleanup ---------------------------------------------------------------------
    "mesh.delete": ("mesh_cleanup", "mesh_edit_face", {"type": "FACE"}, H),
    "mesh.dissolve_verts": ("mesh_cleanup", "mesh_edit_vert", {}, H),
    "mesh.dissolve_edges": ("mesh_cleanup", "grid_edit", {}, H),
    "mesh.dissolve_faces": ("mesh_cleanup", "grid_edit", {}, H),
    "mesh.dissolve_limited": ("mesh_cleanup", "grid_edit", {}, H),
    "mesh.remove_doubles": ("mesh_cleanup", "mesh_edit_doubles", {"threshold": 0.0001}, H),
    "mesh.merge": ("mesh_cleanup", "mesh_edit_face", {"type": "CENTER"}, H),
    "mesh.edge_collapse": ("mesh_cleanup", "mesh_edit_face", {}, H),
    "mesh.decimate": ("mesh_cleanup", "monkey_edit", {"ratio": 0.5}, H),
    "mesh.quads_convert_to_tris": ("mesh_cleanup", "mesh_edit", {}, H),
    "mesh.tris_convert_to_quads": ("mesh_cleanup", "mesh_edit_tris", {}, H),

    # --- mesh edit: normals, marks, shading, visibility -----------------------------------------------------
    "mesh.flip_normals": ("mesh_normals", "mesh_edit", {}, H),
    "mesh.normals_make_consistent": ("mesh_normals", "mesh_edit_flipped", {"inside": False}, H),
    "mesh.faces_shade_smooth": ("mesh_normals", "mesh_edit", {}, H),
    "mesh.faces_shade_flat": ("mesh_normals", "mesh_edit_smooth", {}, H),
    "mesh.mark_sharp": ("mesh_normals", "mesh_edit", {}, H),
    "mesh.mark_seam": ("mesh_normals", "mesh_edit_face", {}, H),
    "mesh.hide": ("mesh_normals", "mesh_edit_face", {"unselected": False}, H),
    "mesh.reveal": ("mesh_normals", "mesh_edit_hidden", {}, H),

    # --- mesh edit: deform ----------------------------------------------------------------------------------
    "mesh.vertices_smooth": ("mesh_deform", "monkey_edit", {"factor": 0.5, "repeat": 5}, H),  # factor defaults to 0 = no-op
    "mesh.vertices_smooth_laplacian": ("mesh_deform", "monkey_edit", {"repeat": 3}, H),
    "mesh.face_make_planar": ("mesh_deform", "monkey_edit", {}, H),
    "transform.tosphere": ("mesh_deform", "monkey_edit", {"value": 1.0}, H),
    "transform.shrink_fatten": ("mesh_deform", "mesh_edit", {"value": 0.2}, H),
    "transform.push_pull": ("mesh_deform", "mesh_edit", {"value": 0.2}, H),
    "transform.vertex_random": ("mesh_deform", "mesh_edit", {"offset": 0.1}, H),
    "transform.edge_crease": ("mesh_deform", "mesh_edit", {"value": 1.0}, H),
    "transform.edge_bevelweight": ("mesh_deform", "mesh_edit", {"value": 1.0}, H),

    # --- UV --------------------------------------------------------------------------------------------------
    "uv.unwrap": ("uv", "mesh_edit", {}, H),
    "uv.smart_project": ("uv", "mesh_edit", {}, H),
    "uv.cube_project": ("uv", "mesh_edit", {}, H),
    "uv.cylinder_project": ("uv", "mesh_edit", {"direction": "ALIGN_TO_OBJECT"}, H),
    "uv.sphere_project": ("uv", "mesh_edit", {"direction": "ALIGN_TO_OBJECT"}, H),
    "uv.lightmap_pack": ("uv", "mesh_edit", {}, H),
    "uv.pack_islands": ("uv", "mesh_edit", {}, H),
    "uv.average_islands_scale": ("uv", "mesh_edit_uv_uneven", {}, H),
    "uv.reset": ("uv", "mesh_edit", {}, H),
    "uv.follow_active_quads": ("uv", "mesh_edit_all_active", {}, H),
    "uv.seams_from_islands": ("uv", "mesh_edit", {}, H),

    # --- animation (built-ins, since anim.keyframe_* need an editor) -------------------------------------------
    "bai.keyframe": ("animation", "object", {"object": "Cube", "property": "location"}, H),
    "bai.frame": ("animation", "object", {"frame": 50}, H),
    "screen.frame_jump": ("animation", "object", {"end": True}, H),
    "screen.frame_offset": ("animation", "object", {"delta": 10}, H),
    "screen.keyframe_jump": ("animation", "object_keyed", {"next": True}, H),

    # --- armature / pose / rig ------------------------------------------------------------------------------------
    "armature.extrude_move": ("rigging", "armature_edit_tip", {"TRANSFORM_OT_translate": {"value": [0.0, 0.0, 1.0]}}, H),
    "armature.subdivide": ("rigging", "armature_edit", {"number_cuts": 2}, H),
    "armature.duplicate_move": ("rigging", "armature_edit", {"TRANSFORM_OT_translate": {"value": [1.0, 0.0, 0.0]}}, H),
    "armature.delete": ("rigging", "armature_edit", {}, H),
    "armature.bone_primitive_add": ("rigging", "armature_edit", {"name": "Tail"}, H),
    "armature.switch_direction": ("rigging", "armature_edit", {}, H),
    "armature.select_all": ("rigging", "armature_edit", {"action": "DESELECT"}, H),
    "armature.calculate_roll": ("rigging", "armature_edit", {"type": "GLOBAL_POS_X"}, H),
    "pose.loc_clear": ("rigging", "pose", {}, H),
    "pose.rot_clear": ("rigging", "pose", {}, H),
    "pose.scale_clear": ("rigging", "pose", {}, H),
    "pose.transforms_clear": ("rigging", "pose", {}, H),
    "pose.select_all": ("rigging", "pose", {"action": "DESELECT"}, H),

    # --- curve edit -----------------------------------------------------------------------------------------------
    "curve.cyclic_toggle": ("curves", "curve_edit", {}, H),
    "curve.subdivide": ("curves", "curve_edit", {"number_cuts": 2}, H),
    "curve.switch_direction": ("curves", "curve_edit", {}, H),
    "curve.handle_type_set": ("curves", "curve_edit", {"type": "VECTOR"}, H),
    "curve.extrude_move": ("curves", "curve_edit_end", {"TRANSFORM_OT_translate": {"value": [1.0, 0.0, 0.0]}}, H),
    "curve.select_all": ("curves", "curve_edit", {"action": "DESELECT"}, H),
    "curve.smooth": ("curves", "curve_edit_dense", {}, H),
    "curve.spline_type_set": ("curves", "curve_edit", {"type": "POLY"}, H),
    "curve.delete": ("curves", "curve_edit_end", {"type": "VERT"}, H),

    # --- sculpt one-shots -------------------------------------------------------------------------------------------
    "sculpt.dynamic_topology_toggle": ("sculpt", "monkey_sculpt", {}, H),
    # These crash headless Blender (sculpt needs a window), so they are Way 1 only.
    "sculpt.symmetrize": ("sculpt", "monkey_sculpt_asym", {}, V),
    "paint.mask_flood_fill": ("sculpt", "monkey_sculpt", {"mode": "VALUE", "value": 1.0}, V),
    "sculpt.face_sets_init": ("sculpt", "monkey_sculpt", {"mode": "LOOSE_PARTS"}, V),

    # --- viewport-only (Way 1 only) -----------------------------------------------------------------------------------
    "view3d.view_axis": ("viewport", "object", {"type": "TOP"}, V),
    "view3d.view_all": ("viewport", "object", {}, V),
    "view3d.view_selected": ("viewport", "object", {}, V),
    "view3d.view_camera": ("viewport", "object", {}, V),
    "view3d.view_persportho": ("viewport", "object", {}, V),
    "view3d.toggle_xray": ("viewport", "object", {}, V),
    "view3d.toggle_shading": ("viewport", "object", {"type": "WIREFRAME"}, V),
    "view3d.localview": ("viewport", "object", {}, V),
    "view3d.snap_cursor_to_selected": ("viewport", "object", {}, V),
    "view3d.snap_cursor_to_center": ("viewport", "object", {}, V),
    "view3d.snap_cursor_to_active": ("viewport", "object", {}, V),
    "view3d.snap_selected_to_cursor": ("viewport", "object", {}, V),
    "view3d.snap_selected_to_grid": ("viewport", "object", {}, V),
    "view3d.snap_selected_to_active": ("viewport", "object_two", {}, V),
    "view3d.camera_to_view": ("viewport", "object", {}, V),
    "view3d.camera_to_view_selected": ("viewport", "object", {}, V),
    "view3d.object_as_camera": ("viewport", "object", {}, V),
    # Hiding one object is bai.set hidden (names its target, runs headless). The H-key operator was
    # removed 2026-10-02: teaching both made the 2B model pick it and hide whatever was selected.
    "object.hide_view_clear": ("viewport", "object", {}, V),  # "unhide everything" has no bai.set equivalent
    "ed.undo": ("viewport", "object", {}, V),
    "ed.redo": ("viewport", "object", {}, V),
    "screen.animation_play": ("viewport", "object", {}, V),
    "screen.animation_cancel": ("viewport", "object", {}, V),
    "screen.screen_full_area": ("viewport", "object", {}, V),
    "render.render": ("viewport", "object", {}, V),
    "render.opengl": ("viewport", "object", {}, V),
    "wm.save_mainfile": ("viewport", "object", {}, V),
}

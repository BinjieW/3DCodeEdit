from __future__ import annotations
from collections.abc import Iterable
from mathutils import Vector
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Low-level BPY primitives for the Antilia model."""





SCAD_UNIT = 25.4 / 90.0
MODEL_SCALE = 130.0 / 246.68

COLORS = {
    "building": (0.88, 0.63, 0.03, 1.0),
    "wall": (0.58, 0.42, 0.02, 1.0),
    "support": (1.0, 0.86, 0.06, 1.0),
    "shadow": (0.18, 0.15, 0.05, 1.0),
    "edge": (0.42, 0.32, 0.04, 1.0),
}


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def ensure_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
    material.diffuse_color = color
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled is None:
        nodes.clear()
        output = nodes.new(type="ShaderNodeOutputMaterial")
        principled = nodes.new(type="ShaderNodeBsdfPrincipled")
        material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    if "Base Color" in principled.inputs:
        principled.inputs["Base Color"].default_value = color
    if "Roughness" in principled.inputs:
        principled.inputs["Roughness"].default_value = 0.72
    return material


def assign_material(obj: bpy.types.Object, material_name: str) -> bpy.types.Object:
    material = ensure_material(material_name, COLORS[material_name])
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)
    return obj


def apply_boolean_difference(
    target: bpy.types.Object,
    cutter: bpy.types.Object,
    *,
    solver: str = "EXACT",
) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    cutter.select_set(True)
    modifier = target.modifiers.new(name="BoolDifference", type="BOOLEAN")
    modifier.operation = "DIFFERENCE"
    modifier.solver = solver
    modifier.object = cutter
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    cutter.select_set(False)
    bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def scad_to_bpy(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Map source SCAD coordinates after scale() and rotate([0,90,0])."""
    sx = x * SCAD_UNIT
    sy = -y * SCAD_UNIT
    sz = z
    return (sz * MODEL_SCALE, sy * MODEL_SCALE, -sx * MODEL_SCALE)


def bpy_to_scad(point: Vector) -> tuple[float, float, float]:
    """Inverse of scad_to_bpy() for mesh cleanup in source-coordinate windows."""
    return (
        -point.z / (SCAD_UNIT * MODEL_SCALE),
        -point.y / (SCAD_UNIT * MODEL_SCALE),
        point.x / MODEL_SCALE,
    )


def remove_faces_in_scad_window(
    obj: bpy.types.Object,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float] | None = None,
) -> bpy.types.Object:
    """Delete mesh faces whose centers fall inside a source-coordinate window."""
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    x1, x2 = sorted(x_range)
    y1, y2 = sorted(y_range)
    z1, z2 = (-math.inf, math.inf) if z_range is None else sorted(z_range)

    faces = []
    for face in bm.faces:
        scad_x, scad_y, scad_z = bpy_to_scad(face.calc_center_median())
        if x1 <= scad_x <= x2 and y1 <= scad_y <= y2 and z1 <= scad_z <= z2:
            faces.append(face)

    if faces:
        bmesh.ops.delete(bm, geom=faces, context="FACES")
        bm.to_mesh(obj.data)
        obj.data.update(calc_edges=True)
    bm.free()
    return obj


def antilia_to_bpy(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Map already-scaled local Antilia coordinates through the final orientation."""
    return (z * MODEL_SCALE, y * MODEL_SCALE, -x * MODEL_SCALE)


def polygon_prism(
    name: str,
    points: Iterable[tuple[float, float]],
    depth: float,
    z_offset: float = 0.0,
    material: str = "building",
) -> bpy.types.Object:
    """Create a linear_extrude-like prism using the source polygon points."""
    pts = list(points)
    verts = [scad_to_bpy(x, y, z_offset) for x, y in pts]
    verts.extend(scad_to_bpy(x, y, z_offset + depth) for x, y in pts)

    n = len(pts)
    faces: list[tuple[int, ...]] = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n, i + n))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return assign_material(obj, material)


def rect_prism(
    name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    depth: float,
    z_offset: float = 0.0,
    material: str = "building",
) -> bpy.types.Object:
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return polygon_prism(name, points, depth, z_offset=z_offset, material=material)


def cylinder_between(
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    material: str = "support",
    vertices: int = 32,
) -> bpy.types.Object:
    start_v = Vector(start)
    end_v = Vector(end)
    mid = (start_v + end_v) * 0.5
    axis = end_v - start_v
    length = axis.length
    if length <= 1e-6:
        raise ValueError(f"Cannot create zero-length cylinder {name}")

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=length,
        location=mid,
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = axis.to_track_quat("Z", "Y").to_euler()
    assign_material(obj, material)
    return obj


def scad_cylinder_between(
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    material: str = "support",
    vertices: int = 32,
) -> bpy.types.Object:
    mapped_start = scad_to_bpy(*start)
    mapped_end = scad_to_bpy(*end)
    return cylinder_between(
        name,
        mapped_start,
        mapped_end,
        radius * SCAD_UNIT * MODEL_SCALE,
        material=material,
        vertices=vertices,
    )


def local_cylinder_between(
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    material: str = "support",
    vertices: int = 32,
) -> bpy.types.Object:
    """Create a cylinder from local Antilia-space endpoints."""
    mapped_start = antilia_to_bpy(*start)
    mapped_end = antilia_to_bpy(*end)
    return cylinder_between(
        name,
        mapped_start,
        mapped_end,
        radius * MODEL_SCALE,
        material=material,
        vertices=vertices,
    )


def add_floor_edge(name: str, y: float, x1: float, x2: float, z: float, depth: float) -> bpy.types.Object:
    return scad_cylinder_between(
        name,
        (x1, y, z),
        (x2, y, z),
        radius=2.0,
        material="edge",
        vertices=18,
    )


def union_bounds(objs: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    mins = Vector((math.inf, math.inf, math.inf))
    maxs = Vector((-math.inf, -math.inf, -math.inf))
    bpy.context.view_layer.update()
    for obj in objs:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, world.x)
            mins.y = min(mins.y, world.y)
            mins.z = min(mins.z, world.z)
            maxs.x = max(maxs.x, world.x)
            maxs.y = max(maxs.y, world.y)
            maxs.z = max(maxs.z, world.z)
    return mins, maxs


# ============================================================
# Procedural assembly
# ============================================================

"""Procedural assembly for the Antilia building model."""






WALL_DEPTH = 70.0
BUILDING_DEPTH = WALL_DEPTH * 0.9
SIDE_OFFSET = (WALL_DEPTH - BUILDING_DEPTH) / 2.0
TOP_WIDTH = BUILDING_DEPTH / 2.0
WIDTH_2ND = BUILDING_DEPTH * 0.8
WIDTH_3RD = BUILDING_DEPTH * 0.75
WIDTH_BAND = BUILDING_DEPTH / 5.0


WALL_OUTLINE = [
    (-77.803585, -410.270195), (-77.803585, -400.145195), (110.602615, -400.145195),
    (110.602615, -318.832695), (-77.803585, -318.832695), (-77.803585, -308.738945),
    (-77.803585, -273.645195), (-77.803585, -269.332695), (110.602615, -269.332695),
    (110.602615, -132.957695), (-130.397335, -132.957695), (-130.397335, -129.176445),
    (-131.334835, -129.176445), (-131.334835, -89.801445), (-130.397335, -89.801445),
    (-130.397335, -84.488945), (110.602615, -84.488945), (110.602615, 3.917305),
    (-77.803585, 3.917305), (-77.803585, 9.698555), (-77.803585, 14.011055),
    (-77.803585, 102.386055), (-77.803585, 110.729805), (-77.803585, 112.511055),
    (110.602615, 112.511055), (110.602615, 202.354805), (110.602615, 224.636055),
    (-126.772335, 224.636055), (-126.772335, 234.729805), (-126.772335, 463.761055),
    (-126.772335, 463.792305), (110.602615, 463.792305), (115.165215, 463.792305),
    (120.696415, 463.792305), (120.696415, 213.167305), (120.696415, 202.354805),
    (120.696415, 108.198555), (120.696415, 102.386055), (-67.678585, 102.386055),
    (-67.678585, 14.011055), (120.696415, 14.011055), (120.696415, 3.917305),
    (120.696415, -88.770195), (120.696415, -94.582695), (-121.241085, -94.582695),
    (-121.241085, -122.863945), (120.696415, -122.863945), (120.696415, -132.957695),
    (120.696415, -274.145195), (120.696415, -279.457695), (-67.678585, -279.457695),
    (-67.678585, -308.738945), (110.602615, -308.738945), (120.696415, -308.738945),
    (120.696415, -318.832695), (120.696415, -405.207695), (120.696415, -410.270195),
]

WALL_OPENING = (-116.678585, 234.729805, 110.602615, 453.667305)
END_BAY_OPENING = (24.0, -411.0, 120.7, -317.0)
END_POCKET_OPEN_X = (31.5, 108.0)
END_FRAME_XS = (-116.678585, -95.0, 94.0, 110.602615)
END_FRAME_Z = (0.0, WALL_DEPTH)


PLATES = [
    ("far_plain_block", [(24.334895, -318.703485), (110.686999, -318.703485), (110.686999, -399.876443), (24.334895, -399.876443)], BUILDING_DEPTH, 0.0),
    ("far_plain_block_upper", [(4.334895, -318.703485), (90.686999, -318.703485), (90.686999, -399.876443), (4.334895, -399.876443)], TOP_WIDTH, TOP_WIDTH / 2.0),
    ("lower_step", [(-67.986515, -279.571515), (111.579875, -279.571515), (111.579875, -308.601613), (-67.986515, -308.601613)], BUILDING_DEPTH, 0.0),
    ("deep_bridge", [(-54.057955, -248.167765), (110.865585, -248.167765), (110.865585, -269.697863), (-54.057955, -269.697863)], WIDTH_2ND, -(WIDTH_2ND - BUILDING_DEPTH) / 2.0),
    ("center_block", [(-77.807975, -132.453495), (111.401285, -132.453495), (111.401285, -248.983595), (-77.807975, -248.983595)], BUILDING_DEPTH, 0.0),
    ("slot_bar_a", [(-121.379385, -94.239205), (111.758445, -94.239205), (111.758445, -122.912158), (-121.379385, -122.912158)], BUILDING_DEPTH, 0.0),
    ("slot_bar_b", [(-0.307955, -55.667795), (110.687025, -55.667795), (110.687025, -84.340748), (-0.307955, -84.340748)], WIDTH_2ND, -(WIDTH_2ND - BUILDING_DEPTH) / 2.0),
    ("middle_block", [(-28.879365, 4.153675), (112.115615, 4.153675), (112.115615, -55.947849), (-28.879365, -55.947849)], BUILDING_DEPTH, 0.0),
    ("middle_slab_a", [(45.684905, 44.510795), (112.108794, 44.510795), (112.108794, 14.052127), (45.684905, 14.052127)], WIDTH_2ND, -(WIDTH_2ND - BUILDING_DEPTH) / 2.0),
    ("middle_slab_b", [(4.934265, 75.225095), (112.279315, 75.225095), (112.279315, 44.766427), (4.934265, 44.766427)], BUILDING_DEPTH, 0.0),
    ("wide_step", [(-69.210325, 104.332205), (112.133825, 104.332205), (112.133825, 73.873537), (-69.210325, 73.873537)], BUILDING_DEPTH, 0.0),
    ("tall_rear_block", [(-6.736515, 225.225095), (111.044175, 225.225095), (111.044175, 111.909285), (-6.736515, 111.909285)], WIDTH_3RD, -(WIDTH_3RD - BUILDING_DEPTH) / 2.0),
    ("thin_intermediate_low", [(-66.200825, 188.796505), (112.294155, 188.796505), (112.294155, 176.19498), (-66.200825, 176.19498)], WIDTH_BAND, WIDTH_BAND),
    ("thin_intermediate_high", [(-66.200825, 188.796505), (112.294155, 188.796505), (112.294155, 176.19498), (-66.200825, 176.19498)], WIDTH_BAND, 3.0 * WIDTH_BAND),
]


LIP_DUPLICATES = {
    "far_plain_block": (0.0, -1.0, 1.0),
    "far_plain_block_upper": (0.0, -1.0, 1.0),
    "lower_step": (0.0, -1.0, 1.0),
    "deep_bridge": (0.0,),
    "center_block": (0.0, -1.0),
    "slot_bar_a": (0.0, -1.0, 1.0),
    "slot_bar_b": (0.0, -1.0, 1.0),
    "middle_block": (0.0, -1.0),
    "middle_slab_a": (0.0, -2.0, 1.0),
    "middle_slab_b": (0.0, -1.0),
    "wide_step": (0.0, -1.0),
    "tall_rear_block": (0.0,),
    "thin_intermediate_low": (0.0,),
    "thin_intermediate_high": (0.0,),
}


REAR_PLATE_BOUNDS = (-116.915105, 234.409275, 123.008445, 453.796515)
REAR_DEPTH_OFFSET = SIDE_OFFSET
REAR_SIDE_DEPTH = WIDTH_BAND * 1.10
REAR_CENTER_DEPTH = WIDTH_BAND * 1.35
REAR_CENTER_Z = (BUILDING_DEPTH - REAR_CENTER_DEPTH) / 2.0
REAR_BAY_FACE_Y = WALL_OPENING[1] + 10.0
REAR_BAY_BACK_Y = WALL_OPENING[1] + 76.0
END_BAY_FACE_Y = -400.2
END_BAY_BACK_Y = END_BAY_OPENING[1] + 70.0
SUPPORT_LOOP_DEPTHS = tuple((i, WALL_DEPTH / 6.0 * i) for i in (1, 3, 5))

STRAIGHT_COLUMN_MODULES = [
    ("far_left", -62.6676125, -401.047585, -318.603753),
    ("far_mid", -25.5247725, -400.511875, -318.068043),
    ("far_right", 14.1181075, -400.511875, -318.068043),
    ("slot_left", -65.5247525, -84.797585, 4.431961),
    ("mid_left", -28.7390325, 9.845265, 75.860524),
    ("mid_right", 13.7609675, 9.130975, 45.146234),
]


DIAGONAL_MODULES = [
    ("diag_a", (-73.432652, 99.475767, -32.718367, 226.158544, -22.616842, 226.158544), (0.0, 2.0, 0.0), -19.0),
    ("diag_b", (19.525955, 108.475767, -21.188331, 226.158544, -31.289856, 226.158544), (90.0, 53.0, 0.0), 19.0),
    ("diag_c", (5.495935, 98.475767, 46.210221, 226.158544, 56.311746, 226.158544), (4.0, 35.0, 0.0), -19.0),
    ("diag_d", (98.454605, 100.475767, 57.740319, 226.158544, 47.638794, 226.158544), (100.0, 31.0, 0.0), 19.0),
]


SUPPORT_CONNECTOR_MODULES = [
    ("far_to_lower_left", -62.6676125, -318.603753, -279.571515),
    ("far_to_lower_mid", -25.5247725, -318.068043, -279.571515),
    ("far_to_lower_right", 14.1181075, -318.068043, -279.571515),
    ("far_left_outer_lip", -57.616850, -318.603753, -279.571515),
    ("far_mid_outer_lip", -20.474010, -318.068043, -279.571515),
    ("far_right_outer_lip", 19.168870, -318.068043, -279.571515),
    ("slot_left_to_mid", -65.5247525, 4.431961, 14.011055),
    ("slot_left_outer_lip", -60.473990, 4.431961, 14.011055),
    ("mid_left_to_wide", -28.7390325, 75.860524, 108.475767),
    ("mid_left_outer_lip", -23.688270, 75.860524, 108.475767),
    ("mid_right_to_wide", 13.7609675, 45.146234, 108.475767),
    ("mid_right_outer_lip", 18.811730, 45.146234, 108.475767),
]


SUPPORT_EDGE_POSTS = [
    ("far_plain_edge", -77.807975, 24.334895, -318.603753),
    ("far_plain_back_edge", -77.807975, 24.334895, -400.511875),
    ("lower_step_edge", -67.986515, 24.334895, -279.571515),
    ("lower_step_inner_edge", -67.986515, 24.334895, -308.601613),
    ("slot_edge", -77.807975, -0.307955, -84.340748),
    ("slot_upper_edge", -121.379385, -0.307955, -94.239205),
    ("middle_low_edge", -33.789795, 18.811730, 9.130975),
    ("middle_high_edge", -33.789795, 18.811730, 75.225095),
    ("truss_lower_edge", -73.432652, 98.454605, 108.475767),
    ("truss_upper_edge", -32.718367, 57.740319, 226.158544),
]


SUPPORT_CROSS_ROWS = [
    ("far_base_back", -62.6676125, -25.5247725, -400.511875),
    ("far_base_mid", -25.5247725, 14.1181075, -400.511875),
    ("far_top_back", -62.6676125, -25.5247725, -318.603753),
    ("far_top_mid", -25.5247725, 14.1181075, -318.068043),
    ("lower_step_left", -67.986515, -25.5247725, -279.571515),
    ("lower_step_right", -25.5247725, 24.334895, -279.571515),
    ("slot_floor", -65.5247525, -0.307955, -84.340748),
    ("slot_roof", -65.5247525, -0.307955, 4.431961),
    ("middle_low", -28.7390325, 13.7609675, 9.845265),
    ("middle_mid", -28.7390325, 13.7609675, 45.146234),
    ("middle_high", -28.7390325, 13.7609675, 75.860524),
    ("wide_step_low", -67.986515, 24.334895, 104.332205),
    ("wide_step_high", -32.718367, 57.740319, 226.158544),
]


SHELF_REVEALS = [
    ("far_lower_shadow_lip", 24.334895, -318.703485, 110.686999, -317.2, BUILDING_DEPTH, 0.0),
    ("lower_step_top_lip", -67.986515, -279.571515, 111.579875, -277.8, BUILDING_DEPTH, 0.0),
    ("lower_step_back_lip", -67.986515, -309.4, 111.579875, -307.9, BUILDING_DEPTH, 0.0),
    ("deep_bridge_front_lip", -54.057955, -248.167765, 110.865585, -246.5, WIDTH_2ND, -(WIDTH_2ND - BUILDING_DEPTH) / 2.0),
    ("deep_bridge_back_lip", -54.057955, -270.4, 110.865585, -268.8, WIDTH_2ND, -(WIDTH_2ND - BUILDING_DEPTH) / 2.0),
    ("center_block_upper_lip", -77.807975, -132.453495, 111.401285, -130.7, BUILDING_DEPTH, 0.0),
    ("center_block_lower_lip", -77.807975, -249.7, 111.401285, -248.1, BUILDING_DEPTH, 0.0),
    ("slot_bar_a_lip", -121.379385, -94.239205, 111.758445, -92.5, BUILDING_DEPTH, 0.0),
    ("slot_bar_a_back_lip", -121.379385, -123.7, 111.758445, -122.1, BUILDING_DEPTH, 0.0),
    ("middle_block_lip", -28.879365, 4.153675, 112.115615, 5.9, BUILDING_DEPTH, 0.0),
    ("middle_slab_lip", 4.934265, 44.766427, 112.279315, 46.4, BUILDING_DEPTH, 0.0),
    ("wide_step_lip", -69.210325, 104.332205, 112.133825, 106.0, BUILDING_DEPTH, 0.0),
    ("rear_tall_block_lip", -6.736515, 225.225095, 111.044175, 227.0, WIDTH_3RD, -(WIDTH_3RD - BUILDING_DEPTH) / 2.0),
    ("rear_tall_block_back_lip", -6.736515, 111.909285, 111.044175, 113.6, WIDTH_3RD, -(WIDTH_3RD - BUILDING_DEPTH) / 2.0),
    ("rear_tall_block_side_gap_front", 111.3, 113.8, 112.6, 224.5, WIDTH_3RD, -(WIDTH_3RD - BUILDING_DEPTH) / 2.0),
    ("rear_tall_block_side_gap_back", -8.1, 113.8, -6.7, 224.5, WIDTH_3RD, -(WIDTH_3RD - BUILDING_DEPTH) / 2.0),
    ("rear_intermediate_low_lip", -66.200825, 176.19498, 112.294155, 177.8, WIDTH_BAND, WIDTH_BAND),
    ("rear_intermediate_high_lip", -66.200825, 188.796505, 112.294155, 190.3, WIDTH_BAND, 3.0 * WIDTH_BAND),
    ("rear_basis_front_gap", -116.915105, 233.3, 123.008445, 234.7, BUILDING_DEPTH, 0.0),
    ("rear_basis_back_gap", -116.915105, 453.1, 123.008445, 454.5, BUILDING_DEPTH, 0.0),
    ("rear_basis_right_depth_gap", 121.6, 236.0, 123.0, 450.8, WIDTH_BAND, WIDTH_BAND),
    ("rear_basis_left_depth_gap", -116.9, 236.0, -115.5, 450.8, WIDTH_BAND, 3.0 * WIDTH_BAND),
]


def _rot_z(point: tuple[float, float, float], degrees: float) -> tuple[float, float, float]:
    radians = math.radians(degrees)
    x, y, z = point
    return (
        x * math.cos(radians) - y * math.sin(radians),
        x * math.sin(radians) + y * math.cos(radians),
        z,
    )


def _rot_x(point: tuple[float, float, float], degrees: float) -> tuple[float, float, float]:
    radians = math.radians(degrees)
    x, y, z = point
    return (
        x,
        y * math.cos(radians) - z * math.sin(radians),
        y * math.sin(radians) + z * math.cos(radians),
    )


def _offset(point: tuple[float, float, float], delta: tuple[float, float, float]) -> tuple[float, float, float]:
    return (point[0] + delta[0], point[1] + delta[1], point[2] + delta[2])


def _cylrot_source_axis(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    height = abs(y1 - y2)
    center_x = (x1 + x3) / 2.0 - 50.0
    center_y = (y1 + y2) / 2.0 - 25.0
    return (center_x, center_y - height / 2.0, 0.0), (center_x, center_y + height / 2.0, 0.0)


def _scaled_cylinder_point(point: tuple[float, float, float], depth_offset: float) -> tuple[float, float, float]:
    x, y, z = point
    return (x * 25.4 / 90.0, -y * 25.4 / 90.0, depth_offset - z * 25.4 / 90.0)


def _add_wall_outline(parts: list[bpy.types.Object]) -> None:
    wall = polygon_prism("continuous_notched_wall", WALL_OUTLINE, WALL_DEPTH, material="wall")
    opening_x1, opening_y1, opening_x2, opening_y2 = WALL_OPENING
    near_cutter = rect_prism(
        "near_opening_cutter",
        opening_x1,
        opening_y1,
        opening_x2,
        opening_y2,
        WALL_DEPTH + 2.0,
        z_offset=-1.0,
        material="shadow",
    )
    wall = apply_boolean_difference(wall, near_cutter)
    end_x1, end_y1, end_x2, end_y2 = END_BAY_OPENING
    end_cutter = rect_prism(
        "end_bay_opening_cutter",
        end_x1 - 4.0,
        end_y1 - 8.0,
        end_x2 + 10.0,
        end_y2 + 8.0,
        WALL_DEPTH + 2.0,
        z_offset=-1.0,
        material="shadow",
    )
    wall = apply_boolean_difference(wall, end_cutter)
    remove_faces_in_scad_window(
        wall,
        x_range=(-70.0, 121.0),
        y_range=(-413.0, -317.0),
        z_range=(-1.0, WALL_DEPTH + 1.0),
    )
    parts.append(wall)


def _add_building_plates(parts: list[bpy.types.Object]) -> None:
    for name, points, depth, z_offset in PLATES:
        for dy in LIP_DUPLICATES.get(name, (0.0,)):
            suffix = "" if dy == 0.0 else f"_y{dy:+.0f}"
            shifted = [(x, y + dy) for x, y in points]
            if name in {"far_plain_block", "far_plain_block_upper"}:
                x_values = [x for x, _ in shifted]
                y_values = [y for _, y in shifted]
                x1, x2 = min(x_values), max(x_values)
                y1, y2 = min(y_values), max(y_values)
                open_x1, open_x2 = END_POCKET_OPEN_X
                if x1 < open_x1:
                    parts.append(rect_prism(
                        f"{name}{suffix}_top_header",
                        x1,
                        y1,
                        min(open_x1, x2),
                        y2,
                        depth,
                        z_offset=SIDE_OFFSET + z_offset,
                        material="building",
                    ))
                if x2 > open_x2:
                    parts.append(rect_prism(
                        f"{name}{suffix}_low_sill",
                        max(open_x2, x1),
                        y1,
                        x2,
                        y2,
                        depth,
                        z_offset=SIDE_OFFSET + z_offset,
                        material="building",
                    ))
                continue
            plate = polygon_prism(f"{name}{suffix}", shifted, depth, z_offset=SIDE_OFFSET + z_offset, material="building")
            parts.append(plate)


def _add_saeulengerade(parts: list[bpy.types.Object], depth_offset: float, suffix: str) -> None:
    for name, x, y1, y2 in STRAIGHT_COLUMN_MODULES:
        parts.append(scad_cylinder_between(
            f"saeulengerade_{suffix}_{name}",
            (x, y1, depth_offset),
            (x, y2, depth_offset),
            radius=10.1,
            material="support",
            vertices=36,
        ))


def _add_saeulenschraeg(parts: list[bpy.types.Object], depth_offset: float, suffix: str) -> None:
    for name, args, translate, z_rotation in DIAGONAL_MODULES:
        start, end = _cylrot_source_axis(*args)
        start = _offset(_rot_z(start, z_rotation), translate)
        end = _offset(_rot_z(end, z_rotation), translate)
        parts.append(local_cylinder_between(
            f"saeulenschraeg_{suffix}_{name}",
            _scaled_cylinder_point(start, depth_offset),
            _scaled_cylinder_point(end, depth_offset),
            radius=10.0 * 25.4 / 90.0,
            material="support",
            vertices=36,
        ))


def _add_support_connectors(parts: list[bpy.types.Object], depth_offset: float, suffix: str) -> None:
    for name, x, y1, y2 in SUPPORT_CONNECTOR_MODULES:
        parts.append(scad_cylinder_between(
            f"support_connector_{suffix}_{name}",
            (x, y1, depth_offset),
            (x, y2, depth_offset),
            radius=3.2,
            material="support",
            vertices=24,
        ))


def _add_support_edge_posts(parts: list[bpy.types.Object], depth_offset: float, suffix: str) -> None:
    for name, x1, x2, y in SUPPORT_EDGE_POSTS:
        parts.append(scad_cylinder_between(
            f"support_edge_post_{suffix}_{name}",
            (x1, y, depth_offset),
            (x2, y, depth_offset),
            radius=2.7,
            material="edge",
            vertices=20,
        ))


def _add_support_cross_rows(parts: list[bpy.types.Object], depth_offset: float, suffix: str) -> None:
    for name, x1, x2, y in SUPPORT_CROSS_ROWS:
        parts.append(scad_cylinder_between(
            f"support_cross_row_{suffix}_{name}",
            (x1, y, depth_offset),
            (x2, y, depth_offset),
            radius=2.35,
            material="edge",
            vertices=20,
        ))


def _add_rotated_saeule(parts: list[bpy.types.Object], depth_offset: float, suffix: str, *, variant: str, x_angle: float) -> None:
    translate = (50.0, 77.0, -10.0) if variant == "upper" else (50.0, 25.0, 0.0)
    base_start, base_end = _cylrot_source_axis(-33.075505, 88.845265, -33.075505, 226.574815, -22.973980, 226.574815)
    start = _offset(base_start, translate)
    end = _offset(base_end, translate)
    start = _rot_x(_scaled_cylinder_point(start, 0.0), x_angle)
    end = _rot_x(_scaled_cylinder_point(end, 0.0), x_angle)
    scene_translate = (0.0, 8.0, depth_offset) if variant == "upper" else (0.0, -6.0, depth_offset)
    parts.append(local_cylinder_between(
        f"saeulegeradeschraeg_{suffix}_{variant}",
        _offset(start, scene_translate),
        _offset(end, scene_translate),
        radius=10.0 * 25.4 / 90.0,
        material="support",
        vertices=36,
    ))


def _add_source_support_modules(parts: list[bpy.types.Object]) -> None:
    for i, depth in SUPPORT_LOOP_DEPTHS:
        suffix = f"d{i}"
        _add_saeulengerade(parts, depth, suffix)
        _add_support_connectors(parts, depth, suffix)
        _add_support_cross_rows(parts, depth, suffix)

    for i in (1, 6, 11):
        depth = WALL_DEPTH / 12.0 * i
        suffix = f"d{i}"
        _add_saeulenschraeg(parts, depth, suffix)
        _add_support_edge_posts(parts, depth, suffix)

    for i in (6, 11):
        _add_rotated_saeule(parts, WALL_DEPTH / 12.0 * i - 7.0, f"d{i}", variant="upper", x_angle=19.0)

    for i in (1, 6):
        _add_rotated_saeule(parts, WALL_DEPTH / 12.0 * i + 9.0, f"d{i}", variant="lower", x_angle=-19.0)


def _add_rear_portal_frame(parts: list[bpy.types.Object]) -> None:
    side_z_spans = (
        (REAR_DEPTH_OFFSET + 4.0, REAR_DEPTH_OFFSET + 22.0, "left"),
        (REAR_DEPTH_OFFSET + BUILDING_DEPTH - 22.0, REAR_DEPTH_OFFSET + BUILDING_DEPTH - 4.0, "right"),
    )
    side_rows = (72.0, 92.0, 112.0)

    for depth_idx, y in enumerate((REAR_BAY_FACE_Y, REAR_BAY_BACK_Y)):
        radius = 4.0 if depth_idx == 0 else 2.8
        material = "support" if depth_idx == 0 else "edge"
        for row_idx, x in enumerate(side_rows):
            for z1, z2, side in side_z_spans:
                parts.append(scad_cylinder_between(
                    f"rear_portal_side_stack_{depth_idx}_{side}_{row_idx}",
                    (x, y, z1),
                    (x, y, z2),
                    radius=radius,
                    material=material,
                    vertices=32,
                ))
    parts.append(scad_cylinder_between(
        "rear_portal_center_depth_rod",
        (92.0, REAR_BAY_FACE_Y, REAR_DEPTH_OFFSET + BUILDING_DEPTH / 2.0 - 7.0),
        (92.0, REAR_BAY_FACE_Y, REAR_DEPTH_OFFSET + BUILDING_DEPTH / 2.0 + 7.0),
        radius=4.0,
        material="support",
        vertices=32,
    ))
    parts.append(scad_cylinder_between(
        "rear_portal_center_back_rod",
        (92.0, REAR_BAY_BACK_Y, REAR_DEPTH_OFFSET + BUILDING_DEPTH / 2.0 - 7.0),
        (92.0, REAR_BAY_BACK_Y, REAR_DEPTH_OFFSET + BUILDING_DEPTH / 2.0 + 7.0),
        radius=2.8,
        material="edge",
        vertices=24,
    ))

    for idx, x in enumerate((-12.0, -33.0)):
        parts.append(scad_cylinder_between(
            f"rear_portal_upper_slot_rod_{idx}",
            (x, REAR_BAY_FACE_Y - 2.0, 20.0),
            (x, REAR_BAY_FACE_Y - 2.0, 34.0),
            radius=2.4,
            material="edge",
            vertices=20,
        ))
        parts.append(scad_cylinder_between(
            f"rear_portal_upper_slot_rod_back_{idx}",
            (x, REAR_BAY_FACE_Y - 2.0, 46.0),
            (x, REAR_BAY_FACE_Y - 2.0, 60.0),
            radius=2.4,
            material="edge",
            vertices=20,
        ))


def _add_rear_end_bay(parts: list[bpy.types.Object]) -> None:
    side_depth_back = REAR_DEPTH_OFFSET + BUILDING_DEPTH - REAR_SIDE_DEPTH
    center_z = REAR_DEPTH_OFFSET + REAR_CENTER_Z
    end_x1, end_y1, end_x2, end_y2 = END_BAY_OPENING

    block_specs = [
        ("end_bay_short_lower_wall", 88.0, end_y1 + 10.0, end_x2, end_y2, BUILDING_DEPTH, REAR_DEPTH_OFFSET, "wall"),
        ("end_bay_short_left_sill", 103.0, end_y1 + 82.0, end_x2, end_y2, REAR_SIDE_DEPTH, REAR_DEPTH_OFFSET, "building"),
        ("end_bay_short_right_sill", 103.0, end_y1 + 82.0, end_x2, end_y2, REAR_SIDE_DEPTH, side_depth_back, "building"),
        ("end_bay_left_pad", 72.0, end_y1 + 63.0, 99.0, end_y1 + 83.0, REAR_SIDE_DEPTH, REAR_DEPTH_OFFSET, "building"),
        ("end_bay_right_pad", 72.0, end_y1 + 63.0, 99.0, end_y1 + 83.0, REAR_SIDE_DEPTH, side_depth_back, "building"),
        ("end_bay_center_raised_block", 43.0, end_y1 + 48.0, 72.0, end_y1 + 74.0, REAR_CENTER_DEPTH, center_z, "building"),
        ("end_bay_back_plate", 53.0, end_y2 - 5.5, 111.5, end_y2, BUILDING_DEPTH, REAR_DEPTH_OFFSET, "edge"),
        ("end_bay_left_side_pad", 33.0, end_y1 + 8.0, 43.0, end_y2 - 3.0, REAR_SIDE_DEPTH * 0.58, REAR_DEPTH_OFFSET, "edge"),
        ("end_bay_right_side_pad", 33.0, end_y1 + 8.0, 43.0, end_y2 - 3.0, REAR_SIDE_DEPTH * 0.58, side_depth_back + REAR_SIDE_DEPTH * 0.42, "edge"),
    ]
    for name, bx1, by1, bx2, by2, depth, z_offset, material in block_specs:
        parts.append(rect_prism(name, bx1, by1, bx2, by2, depth, z_offset=z_offset, material=material))

    side_z_spans = (
        (REAR_DEPTH_OFFSET + 4.0, REAR_DEPTH_OFFSET + 22.0, "left"),
        (REAR_DEPTH_OFFSET + BUILDING_DEPTH - 22.0, REAR_DEPTH_OFFSET + BUILDING_DEPTH - 4.0, "right"),
    )
    for row_idx, x in enumerate((63.0, 82.0, 101.0)):
        for z1, z2, side in side_z_spans:
            parts.append(scad_cylinder_between(
                f"end_bay_side_cylinder_{side}_{row_idx}",
                (x, END_BAY_FACE_Y, z1),
                (x, END_BAY_FACE_Y, z2),
                radius=4.2,
                material="support",
                vertices=32,
            ))
            parts.append(scad_cylinder_between(
                f"end_bay_back_side_cylinder_{side}_{row_idx}",
                (x, END_BAY_BACK_Y, z1),
                (x, END_BAY_BACK_Y, z2),
                radius=2.7,
                material="edge",
                vertices=24,
            ))

    parts.append(scad_cylinder_between(
        "end_bay_center_cylinder",
        (84.0, END_BAY_FACE_Y, REAR_DEPTH_OFFSET + BUILDING_DEPTH / 2.0 - 7.0),
        (84.0, END_BAY_FACE_Y, REAR_DEPTH_OFFSET + BUILDING_DEPTH / 2.0 + 7.0),
        radius=4.2,
        material="support",
        vertices=32,
    ))


def _add_rear_inner_blocks(parts: list[bpy.types.Object]) -> None:
    x1, y1, x2, y2 = REAR_PLATE_BOUNDS
    side_depth_back = REAR_DEPTH_OFFSET + BUILDING_DEPTH - REAR_SIDE_DEPTH
    center_x1 = 68.0
    center_x2 = 102.0

    block_specs = [
        ("rear_lower_left_sill", 108.0, y1 + 86.0, x2, y2, REAR_SIDE_DEPTH, REAR_DEPTH_OFFSET, "building"),
        ("rear_lower_right_sill", 108.0, y1 + 86.0, x2, y2, REAR_SIDE_DEPTH, side_depth_back, "building"),
        ("rear_front_left_low_pad", 84.0, y1 + 58.0, 110.0, y1 + 84.0, REAR_SIDE_DEPTH, REAR_DEPTH_OFFSET, "building"),
        ("rear_front_right_low_pad", 84.0, y1 + 58.0, 110.0, y1 + 84.0, REAR_SIDE_DEPTH, side_depth_back, "building"),
        ("rear_back_left_low_pad", 84.0, y1 + 91.0, 110.0, y1 + 113.0, REAR_SIDE_DEPTH * 0.58, REAR_DEPTH_OFFSET, "edge"),
        ("rear_back_right_low_pad", 84.0, y1 + 91.0, 110.0, y1 + 113.0, REAR_SIDE_DEPTH * 0.58, side_depth_back, "edge"),
        ("rear_center_raised_block", center_x1, y1 + 43.0, center_x2, y1 + 78.0, REAR_CENTER_DEPTH, REAR_DEPTH_OFFSET + REAR_CENTER_Z, "building"),
        ("rear_center_back_step", center_x1 + 3.0, y1 + 89.0, center_x2 - 3.0, y1 + 116.0, REAR_CENTER_DEPTH * 0.62, REAR_DEPTH_OFFSET + REAR_CENTER_Z + REAR_CENTER_DEPTH * 0.19, "edge"),
        ("rear_back_header", x1, y1, -93.0, y2, BUILDING_DEPTH, REAR_DEPTH_OFFSET, "building"),
        ("rear_upper_inner_shelf", -63.0, y1 + 13.0, -55.0, y1 + 42.0, WIDTH_BAND, REAR_DEPTH_OFFSET + WIDTH_BAND * 1.05, "building"),
        ("rear_left_inner_jamb", -68.0, y1 + 48.0, 112.0, y1 + 50.0, REAR_SIDE_DEPTH * 0.44, REAR_DEPTH_OFFSET + 1.0, "edge"),
        ("rear_right_inner_jamb", -68.0, y1 + 48.0, 112.0, y1 + 50.0, REAR_SIDE_DEPTH * 0.44, side_depth_back + REAR_SIDE_DEPTH * 0.56, "edge"),
    ]
    for name, bx1, by1, bx2, by2, depth, z_offset, material in block_specs:
        parts.append(rect_prism(name, bx1, by1, bx2, by2, depth, z_offset=z_offset, material=material))


def _add_diagonal_supports(parts: list[bpy.types.Object]) -> None:
    _add_source_support_modules(parts)

def _add_depth_reveals(parts: list[bpy.types.Object]) -> None:
    for name, x1, y1, x2, y2, depth, z_offset in SHELF_REVEALS:
        parts.append(rect_prism(
            name,
            x1,
            y1,
            x2,
            y2,
            depth,
            z_offset=SIDE_OFFSET + z_offset - 0.35,
            material="edge",
        ))


def build_model() -> list[bpy.types.Object]:
    parts: list[bpy.types.Object] = []
    _add_building_plates(parts)
    _add_diagonal_supports(parts)
    _add_wall_outline(parts)
    _add_rear_end_bay(parts)
    _add_rear_inner_blocks(parts)
    _add_rear_portal_frame(parts)
    _add_depth_reveals(parts)

    return parts


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

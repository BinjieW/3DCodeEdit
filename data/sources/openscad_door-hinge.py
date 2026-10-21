from __future__ import annotations
from collections.abc import Iterable
from mathutils import Vector
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the door-hinge model."""





MaterialColor = tuple[float, float, float, float]


RED: MaterialColor = (0.862745, 0.078431, 0.235294, 1.0)
GREEN: MaterialColor = (0.0, 0.48, 0.04, 1.0)
PALE_GREY: MaterialColor = (0.85, 0.85, 0.80, 1.0)
TRANSLUCENT_GREY: MaterialColor = (0.76, 0.76, 0.70, 0.42)
MID_GREY: MaterialColor = (0.72, 0.72, 0.68, 1.0)
DARK_GREY: MaterialColor = (0.36, 0.36, 0.34, 1.0)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def material(name: str, color: MaterialColor) -> bpy.types.Material:
    existing = bpy.data.materials.get(name)
    if existing:
        return existing
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Alpha"].default_value = color[3]
        bsdf.inputs["Roughness"].default_value = 0.55
    if color[3] < 1.0:
        mat.blend_method = "BLEND"
        mat.use_screen_refraction = True
        mat.show_transparent_back = True
    return mat


def assign_material(obj: bpy.types.Object, mat: bpy.types.Material | str | None) -> bpy.types.Object:
    if mat is None:
        return obj
    if isinstance(mat, str):
        mat = bpy.data.materials[mat]
    obj.data.materials.append(mat)
    return obj


def shade_smooth(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.shade_smooth()
    finally:
        obj.select_set(False)
    return obj


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    mat: bpy.types.Material | str | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return assign_material(obj, mat)


def add_cylinder(
    name: str,
    diameter: float,
    height: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
    mat: bpy.types.Material | str | None = None,
    smooth: bool = True,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=diameter / 2.0,
        depth=height,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    assign_material(obj, mat)
    if smooth:
        shade_smooth(obj)
    return obj


def add_rounded_box(
    name: str,
    size: tuple[float, float, float],
    radius: float,
    location: tuple[float, float, float],
    mat: bpy.types.Material | str | None = None,
    segments: int = 10,
) -> bpy.types.Object:
    obj = add_box(name, size, location, mat)
    bevel = obj.modifiers.new(f"{name}_edge_radius", "BEVEL")
    bevel.width = radius
    bevel.segments = segments
    bevel.affect = "EDGES"
    obj.modifiers.new(f"{name}_weighted_normals", "WEIGHTED_NORMAL")
    return obj


def add_link_bar(
    name: str,
    length: float,
    width: float,
    thickness: float,
    location: tuple[float, float, float],
    mat: bpy.types.Material | str | None = None,
) -> bpy.types.Object:
    return add_rounded_box(name, (length, width, thickness), min(width, thickness) * 0.16, location, mat, segments=6)


def add_chamfered_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    mat: bpy.types.Material | str | None = None,
    bevel_width: float = 0.18,
) -> bpy.types.Object:
    return add_rounded_box(name, size, bevel_width, location, mat, segments=3)


def add_lightly_chamfered_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    mat: bpy.types.Material | str | None = None,
    bevel_width: float = 0.10,
) -> bpy.types.Object:
    return add_rounded_box(name, size, bevel_width, location, mat, segments=2)


def _extrude_polygon_y(
    name: str,
    points_xz: list[tuple[float, float]],
    thickness: float,
    center_y: float,
    mat: bpy.types.Material | str | None = None,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    half = thickness / 2.0
    front = [(x, center_y - half, z) for x, z in points_xz]
    back = [(x, center_y + half, z) for x, z in points_xz]
    vertices = front + back
    count = len(points_xz)
    faces: list[tuple[int, ...]] = [tuple(range(count - 1, -1, -1)), tuple(range(count, count * 2))]
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, count + next_index, count + index))
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, mat)
    obj.modifiers.new(f"{name}_weighted_normals", "WEIGHTED_NORMAL")
    return obj


def _arc_points(
    center_x: float,
    center_z: float,
    radius: float,
    start_degrees: float,
    end_degrees: float,
    segments: int = 18,
) -> list[tuple[float, float]]:
    return [
        (
            center_x + math.cos(math.radians(start_degrees + (end_degrees - start_degrees) * step / segments)) * radius,
            center_z + math.sin(math.radians(start_degrees + (end_degrees - start_degrees) * step / segments)) * radius,
        )
        for step in range(segments + 1)
    ]


def add_capsule_leaf_footprint(
    name: str,
    center: tuple[float, float, float],
    length: float,
    height: float,
    thickness: float,
    mat: bpy.types.Material | str,
    segments: int = 18,
) -> bpy.types.Object:
    cx, cy, cz = center
    radius = height / 2.0
    left_x = cx - length / 2.0 + radius
    right_x = cx + length / 2.0 - radius
    points = []
    points.extend(_arc_points(right_x, cz, radius, -90.0, 90.0, segments))
    points.extend(_arc_points(left_x, cz, radius, 90.0, 270.0, segments))
    return _extrude_polygon_y(name, points, thickness, cy, mat)


def add_rounded_tab_leaf_footprint(
    name: str,
    center: tuple[float, float, float],
    length: float,
    height: float,
    thickness: float,
    mat: bpy.types.Material | str,
    rounded_end: str = "right",
    square_extension: float = 0.52,
    corner_radius: float | None = None,
    segments: int = 18,
) -> bpy.types.Object:
    cx, cy, cz = center
    radius = min(height / 2.0, corner_radius if corner_radius is not None else height / 2.0)
    left = cx - length / 2.0
    right = cx + length / 2.0
    top = cz + height / 2.0
    bottom = cz - height / 2.0
    if rounded_end == "right":
        round_cx = right - square_extension - radius
        flat_left = left + square_extension
        points = [
            (flat_left, bottom),
            (right - radius, bottom),
            (right, bottom + radius),
            (right, top - radius),
            (right - radius, top),
            (flat_left, top),
            (left, top - radius * 0.62),
            (left, bottom + radius * 0.62),
        ]
        if corner_radius is None:
            points = [(flat_left, bottom), (round_cx, bottom)]
            points.extend(_arc_points(round_cx, cz, radius, -90.0, 90.0, segments)[1:])
            points.extend([(flat_left, top), (left, top * 0.86 + cz * 0.14), (left, bottom * 0.86 + cz * 0.14)])
    else:
        round_cx = left + square_extension + radius
        flat_right = right - square_extension
        points = [
            (flat_right, top),
            (left + radius, top),
            (left, top - radius),
            (left, bottom + radius),
            (left + radius, bottom),
            (flat_right, bottom),
            (right, bottom + radius * 0.62),
            (right, top - radius * 0.62),
        ]
        if corner_radius is None:
            points = [(flat_right, top), (round_cx, top)]
            points.extend(_arc_points(round_cx, cz, radius, 90.0, 270.0, segments)[1:])
            points.extend([(flat_right, bottom), (right, bottom * 0.86 + cz * 0.14), (right, top * 0.86 + cz * 0.14)])
    return _extrude_polygon_y(name, points, thickness, cy, mat)


def add_asymmetric_leaf_footprint(
    name: str,
    center: tuple[float, float, float],
    length: float,
    height: float,
    thickness: float,
    mat: bpy.types.Material | str,
    left_radius: float,
    right_radius: float,
    left_shoulder: float = 0.18,
    segments: int = 10,
) -> bpy.types.Object:
    cx, cy, cz = center
    left = cx - length / 2.0
    right = cx + length / 2.0
    bottom = cz - height / 2.0
    top = cz + height / 2.0
    left_radius = min(left_radius, height * 0.45, length * 0.22)
    right_radius = min(right_radius, height * 0.45, length * 0.22)
    points: list[tuple[float, float]] = []
    points.extend(_arc_points(right - right_radius, bottom + right_radius, right_radius, -90.0, 0.0, segments))
    points.extend(_arc_points(right - right_radius, top - right_radius, right_radius, 0.0, 90.0, segments)[1:])
    points.append((left + left_shoulder, top))
    points.append((left, top - left_radius))
    points.append((left, bottom + left_radius))
    points.append((left + left_shoulder, bottom))
    return _extrude_polygon_y(name, points, thickness, cy, mat)


def add_rectangular_leaf_footprint(
    name: str,
    center: tuple[float, float, float],
    length: float,
    height: float,
    thickness: float,
    mat: bpy.types.Material | str,
    corner_radius: float = 0.26,
    segments: int = 8,
) -> bpy.types.Object:
    cx, cy, cz = center
    half_length = length / 2.0
    half_height = height / 2.0
    radius = min(corner_radius, half_length * 0.45, half_height * 0.45)
    left = cx - half_length
    right = cx + half_length
    bottom = cz - half_height
    top = cz + half_height
    points: list[tuple[float, float]] = []
    points.extend(_arc_points(right - radius, bottom + radius, radius, -90.0, 0.0, segments))
    points.extend(_arc_points(right - radius, top - radius, radius, 0.0, 90.0, segments)[1:])
    points.extend(_arc_points(left + radius, top - radius, radius, 90.0, 180.0, segments)[1:])
    points.extend(_arc_points(left + radius, bottom + radius, radius, 180.0, 270.0, segments)[1:])
    return _extrude_polygon_y(name, points, thickness, cy, mat)


def add_leaf_block(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    mat: bpy.types.Material | str,
    bevel: float = 0.18,
) -> bpy.types.Object:
    return add_rounded_box(name, size, bevel, center, mat, segments=2)


def add_hinge_knuckle(
    name: str,
    axis_x: float,
    axis_y: float,
    z: float,
    height: float,
    diameter: float,
    mat: bpy.types.Material | str,
) -> bpy.types.Object:
    return add_cylinder(name, diameter=diameter, height=height, location=(axis_x, axis_y, z), vertices=72, mat=mat)


def add_leaf_with_knuckle(
    name: str,
    leaf_center: tuple[float, float, float],
    leaf_size: tuple[float, float, float],
    axis_x: float,
    axis_y: float,
    knuckle_z: float,
    knuckle_height: float,
    knuckle_diameter: float,
    mat: bpy.types.Material | str,
    bridge_width: float = 1.0,
    leaf_bevel: float = 0.18,
) -> list[bpy.types.Object]:
    leaf = add_leaf_block(f"{name}_broad_leaf", leaf_center, leaf_size, mat, bevel=leaf_bevel)
    leaf_left_edge = leaf_center[0] - leaf_size[0] / 2.0
    barrel_right_edge = axis_x + knuckle_diameter / 2.0
    bridge_left = min(leaf_left_edge, barrel_right_edge)
    bridge_right = max(leaf_left_edge, barrel_right_edge)
    bridge_length = bridge_right - bridge_left + 0.14
    bridge_x = (bridge_left + bridge_right) / 2.0
    bridge = add_lightly_chamfered_box(
        f"{name}_rectangular_leaf_to_barrel_web",
        (bridge_length, leaf_size[1] * 1.02, min(bridge_width, leaf_size[2] * 0.98)),
        (bridge_x, (leaf_center[1] + axis_y) / 2.0, knuckle_z),
        mat,
        bevel_width=min(leaf_bevel * 0.65, 0.08),
    )
    knuckle = add_hinge_knuckle(
        f"{name}_attached_barrel",
        axis_x=axis_x,
        axis_y=axis_y,
        z=knuckle_z,
        height=knuckle_height,
        diameter=knuckle_diameter,
        mat=mat,
    )
    return [leaf, bridge, knuckle]


def add_flat_front_screw(
    name: str,
    location: tuple[float, float, float],
    radius: float = 1.0,
    height: float = 0.36,
) -> list[bpy.types.Object]:
    return add_screw_head(name, location, radius=radius, height=height, axis="Y")


def add_compact_rear_nut(
    name: str,
    location: tuple[float, float, float],
    radius: float = 0.74,
    height: float = 0.48,
) -> list[bpy.types.Object]:
    base_y = location[1]
    inner_washer = add_cylinder(
        f"{name}_inner_rear_washer",
        diameter=radius * 2.96,
        height=0.46,
        location=(location[0], base_y - height * 0.98, location[2]),
        rotation=(math.pi / 2.0, 0.0, 0.0),
        vertices=40,
        mat=material("rear_washer_pale", PALE_GREY),
    )
    outer_washer = add_cylinder(
        f"{name}_outer_rear_washer",
        diameter=radius * 2.46,
        height=0.40,
        location=(location[0], base_y - height * 0.42, location[2]),
        rotation=(math.pi / 2.0, 0.0, 0.0),
        vertices=40,
        mat=material("screw_light_grey", (0.82, 0.82, 0.78, 1.0)),
    )
    spacer = add_cylinder(
        f"{name}_raised_rear_spacer",
        diameter=radius * 1.58,
        height=0.50,
        location=(location[0], base_y + height * 0.20, location[2]),
        rotation=(math.pi / 2.0, 0.0, 0.0),
        vertices=32,
        mat=material("bolt_shaft_grey", MID_GREY),
    )
    nut = add_cylinder(
        f"{name}_projecting_hex_nut",
        diameter=radius * 2.28,
        height=height * 1.12,
        location=(location[0], base_y + height * 0.92, location[2]),
        rotation=(math.pi / 2.0, 0.0, 0.0),
        vertices=6,
        mat=material("dark_nut_grey", DARK_GREY),
        smooth=False,
    )
    return [inner_washer, outer_washer, spacer, nut]


def add_source_fastener_y(
    name: str,
    x: float,
    z: float,
    y_front: float,
    y_back: float,
    head_radius: float,
    nut_radius: float,
    shaft_radius: float = 0.28,
    rear_height: float = 0.56,
) -> list[bpy.types.Object]:
    objects = add_flat_front_screw(f"{name}_front", (x, y_front, z), radius=head_radius, height=0.58)
    objects.append(
        add_cylinder(
            f"{name}_thick_visible_through_pin",
            diameter=shaft_radius * 2.0,
            height=abs(y_back - y_front),
            location=(x, (y_front + y_back) / 2.0, z),
            rotation=(math.pi / 2.0, 0.0, 0.0),
            vertices=24,
            mat=material("bolt_shaft_grey", MID_GREY),
        )
    )
    objects.extend(add_compact_rear_nut(f"{name}_rear", (x, y_back, z), radius=nut_radius, height=rear_height))
    return objects


def add_screw_head(
    name: str,
    location: tuple[float, float, float],
    radius: float = 1.75,
    height: float = 0.52,
    axis: str = "Y",
    mat: bpy.types.Material | str | None = None,
) -> list[bpy.types.Object]:
    rotation = (math.pi / 2.0, 0.0, 0.0) if axis == "Y" else (0.0, 0.0, 0.0)
    washer = add_cylinder(
        f"{name}_round_washer",
        diameter=radius * 2.0,
        height=height * 0.46,
        location=location,
        rotation=rotation,
        vertices=48,
        mat=material("screw_light_grey", (0.82, 0.82, 0.78, 1.0)) if mat is None else mat,
    )
    cap_offset = (0.0, -height * 0.28, 0.0) if axis == "Y" else (0.0, 0.0, height * 0.28)
    cap_location = tuple(location[i] + cap_offset[i] for i in range(3))
    cap = add_cylinder(
        f"{name}_small_recessed_hex_socket",
        diameter=radius * 0.72,
        height=height * 0.34,
        location=cap_location,
        rotation=rotation,
        vertices=6,
        mat=material("screw_mid_grey", MID_GREY),
        smooth=False,
    )
    return [washer, cap]


def add_bolt_stack_y(
    name: str,
    x: float,
    y_front: float,
    y_back: float,
    z: float,
    head_radius: float = 1.35,
    nut_radius: float = 1.05,
    shaft_radius: float = 0.48,
    mat: bpy.types.Material | str | None = None,
) -> list[bpy.types.Object]:
    objects = add_screw_head(f"{name}_front", (x, y_front, z), radius=head_radius, height=0.58, axis="Y", mat=mat)
    mid_y = (y_front + y_back) / 2.0
    objects.append(
        add_cylinder(
            f"{name}_shaft",
            diameter=shaft_radius * 2.0,
            height=abs(y_back - y_front),
            location=(x, mid_y, z),
            rotation=(math.pi / 2.0, 0.0, 0.0),
            vertices=24,
            mat=material("bolt_shaft_grey", MID_GREY),
        )
    )
    objects.append(
        add_cylinder(
            f"{name}_rear_washer",
            diameter=nut_radius * 2.15,
            height=0.28,
            location=(x, y_back - 0.16, z),
            rotation=(math.pi / 2.0, 0.0, 0.0),
            vertices=40,
            mat=material("rear_washer_pale", PALE_GREY),
        )
    )
    objects.append(
        add_cylinder(
            f"{name}_rear_hex_nut",
            diameter=nut_radius * 2.0,
            height=0.56,
            location=(x, y_back + 0.28, z),
            rotation=(math.pi / 2.0, 0.0, 0.0),
            vertices=6,
            mat=material("dark_nut_grey", DARK_GREY),
            smooth=False,
        )
    )
    return objects


def add_pin_end(
    name: str,
    location: tuple[float, float, float],
    axis: str = "Z",
    diameter: float = 2.45,
    height: float = 0.40,
) -> list[bpy.types.Object]:
    rotation = (0.0, 0.0, 0.0) if axis == "Z" else (math.pi / 2.0, 0.0, 0.0)
    washer = add_cylinder(
        f"{name}_washer",
        diameter=diameter,
        height=height,
        location=location,
        rotation=rotation,
        vertices=48,
        mat=material("pin_washer_pale", PALE_GREY),
    )
    pin_loc = (location[0], location[1], location[2] + height * 0.95) if axis == "Z" else location
    pin = add_cylinder(
        f"{name}_pin_stub",
        diameter=diameter * 0.49,
        height=height * 2.05,
        location=pin_loc,
        rotation=rotation,
        vertices=24,
        mat=material("pin_mid_grey", MID_GREY),
    )
    return [washer, pin]


def add_label_empty(name: str, location: tuple[float, float, float]) -> bpy.types.Object:
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = 0.5
    empty.location = Vector(location)
    bpy.context.collection.objects.link(empty)
    return empty


def set_origin_group(objects: Iterable[bpy.types.Object], name: str) -> bpy.types.Object:
    root = add_label_empty(name, (0.0, 0.0, 0.0))
    for obj in objects:
        obj.parent = root
    return root


# ============================================================
# Procedural assembly
# ============================================================

"""Readable high-level assembly for the reconstructed door hinge."""




AXIS_X = -7.10
AXIS_Y = -0.34
FRONT_Y = -1.02
REAR_LEAF_Y = 0.08
SLAB_Y = 1.22


def build_door_slab() -> list:
    mat = material("opaque_pale_rear_backing_slab", PALE_GREY)
    return [
        add_lightly_chamfered_box(
            "broad_square_right_rear_door_support_slab",
            size=(18.90, 2.12, 19.20),
            location=(4.82, SLAB_Y, -0.10),
            mat=mat,
            bevel_width=0.012,
        )
    ]


def add_source_like_leaf_with_knuckle(
    name: str,
    color_mat,
    leaf_kind: str,
    leaf_center: tuple[float, float, float],
    leaf_length: float,
    leaf_height: float,
    leaf_thickness: float,
    knuckle_z: float,
    knuckle_height: float,
    knuckle_diameter: float,
    tab_corner_radius: float | None = None,
) -> list:
    if leaf_kind == "green_tab":
        leaf = add_rounded_tab_leaf_footprint(
            f"{name}_compact_source_lobed_leaf_footprint",
            leaf_center,
            length=leaf_length,
            height=leaf_height,
            thickness=leaf_thickness,
            mat=color_mat,
            rounded_end="right",
            square_extension=1.26,
            corner_radius=0.24,
            segments=6,
        )
    elif leaf_kind == "rectangular":
        leaf = add_rectangular_leaf_footprint(
            f"{name}_flat_rectangular_leaf_footprint",
            leaf_center,
            length=leaf_length,
            height=leaf_height,
            thickness=leaf_thickness,
            mat=color_mat,
            corner_radius=tab_corner_radius or 0.24,
        )
    else:
        leaf = add_rounded_tab_leaf_footprint(
            f"{name}_rounded_outer_tab_leaf_footprint",
            leaf_center,
            length=leaf_length,
            height=leaf_height,
            thickness=leaf_thickness,
            mat=color_mat,
            rounded_end="right",
            square_extension=0.34,
            corner_radius=tab_corner_radius,
        )

    leaf_left_edge = leaf_center[0] - leaf_length / 2.0
    barrel_right_edge = AXIS_X + knuckle_diameter / 2.0
    wrap_overlap = -0.06 if leaf_kind == "green_tab" else 0.00
    bridge_left = leaf_left_edge + (0.36 if leaf_kind == "green_tab" else 0.18)
    bridge_right = barrel_right_edge + wrap_overlap
    bridge_depth = leaf_thickness * (0.18 if leaf_kind == "green_tab" else 0.22)
    bridge_height = min(
        leaf_height * (0.055 if leaf_kind == "green_tab" else 0.095),
        knuckle_height * (0.080 if leaf_kind == "green_tab" else 0.115),
    )
    bridge = add_lightly_chamfered_box(
        f"{name}_narrow_leaf_tab_to_axis_bridge",
        size=(bridge_right - bridge_left, bridge_depth, bridge_height),
        location=((bridge_left + bridge_right) / 2.0, (leaf_center[1] + AXIS_Y) / 2.0, knuckle_z),
        mat=color_mat,
        bevel_width=0.035,
    )
    collar_width = knuckle_diameter * (0.018 if leaf_kind == "green_tab" else 0.026)
    collar = add_lightly_chamfered_box(
        f"{name}_small_axis_side_collar",
        size=(collar_width, bridge_depth * 0.42, bridge_height * (0.20 if leaf_kind == "green_tab" else 0.24)),
        location=(AXIS_X + knuckle_diameter * 0.53, (leaf_center[1] + AXIS_Y) / 2.0, knuckle_z),
        mat=color_mat,
        bevel_width=0.025,
    )
    knuckle = add_hinge_knuckle(
        f"{name}_integrated_axis_barrel",
        axis_x=AXIS_X,
        axis_y=AXIS_Y,
        z=knuckle_z,
        height=knuckle_height,
        diameter=knuckle_diameter,
        mat=color_mat,
    )
    return [leaf, bridge, collar, knuckle]


def build_hinge_leaves_and_barrels() -> list:
    objects = []
    green = material("source_green_fixed_leaf", GREEN)
    red = material("source_red_moving_leaf", RED)

    objects.append(
        add_cylinder(
            "continuous_visible_hinge_pin_core",
            diameter=0.72,
            height=29.10,
            location=(AXIS_X, AXIS_Y, -0.02),
            vertices=40,
            mat=material("continuous_pin_grey", MID_GREY),
        )
    )

    objects.extend(
        add_source_like_leaf_with_knuckle(
            "top_green_leaf_assembly",
            green,
            "green_tab",
            leaf_center=(-1.02, FRONT_Y, 11.50),
            leaf_length=11.90,
            leaf_height=5.72,
            leaf_thickness=0.82,
            knuckle_z=11.50,
            knuckle_height=5.92,
            knuckle_diameter=2.98,
        )
    )
    objects.extend(
        add_source_like_leaf_with_knuckle(
            "bottom_green_leaf_assembly",
            green,
            "green_tab",
            leaf_center=(-1.02, FRONT_Y, -11.50),
            leaf_length=11.90,
            leaf_height=5.72,
            leaf_thickness=0.82,
            knuckle_z=-11.50,
            knuckle_height=5.92,
            knuckle_diameter=2.98,
        )
    )

    objects.extend(
        add_source_like_leaf_with_knuckle(
            "upper_red_middle_leaf_assembly",
            red,
            "rectangular",
            leaf_center=(-0.06, REAR_LEAF_Y, 4.48),
            leaf_length=11.55,
            leaf_height=4.28,
            leaf_thickness=1.08,
            knuckle_z=4.48,
            knuckle_height=4.58,
            knuckle_diameter=3.12,
            tab_corner_radius=0.010,
        )
    )
    objects.extend(
        add_source_like_leaf_with_knuckle(
            "lower_red_middle_leaf_assembly",
            red,
            "rectangular",
            leaf_center=(-0.06, REAR_LEAF_Y, -4.48),
            leaf_length=11.55,
            leaf_height=4.28,
            leaf_thickness=1.08,
            knuckle_z=-4.48,
            knuckle_height=4.58,
            knuckle_diameter=3.12,
            tab_corner_radius=0.010,
        )
    )

    objects.extend(add_pin_end("top_small_exposed_pin", (AXIS_X, AXIS_Y, 14.54), axis="Z", diameter=0.82, height=0.16))
    objects.extend(add_pin_end("upper_gap_exposed_pin", (AXIS_X, AXIS_Y, 8.36), axis="Z", diameter=0.68, height=0.14))
    objects.extend(add_pin_end("center_gap_exposed_pin", (AXIS_X, AXIS_Y, 0.00), axis="Z", diameter=0.68, height=0.14))
    objects.extend(add_pin_end("lower_gap_exposed_pin", (AXIS_X, AXIS_Y, -8.36), axis="Z", diameter=0.68, height=0.14))
    objects.extend(add_pin_end("bottom_small_exposed_pin", (AXIS_X, AXIS_Y, -14.54), axis="Z", diameter=0.82, height=0.16))
    return objects


def build_fasteners() -> list:
    objects = []

    for z in (11.50, -11.50):
        for x in (-4.68, 1.70):
            objects.extend(
                add_source_fastener_y(
                    f"green_leaf_large_screw_x{x:.1f}_z{z:.1f}",
                    x=x,
                    z=z,
                    y_front=FRONT_Y - 0.34,
                    y_back=2.72,
                    head_radius=0.82,
                    nut_radius=0.72,
                    shaft_radius=0.18,
                    rear_height=0.54,
                )
            )

    for leaf_name, z_base, dz_sign in (("upper", 4.48, 1.0), ("lower", -4.48, -1.0)):
        for index, (x, dz) in enumerate(((-2.38, 0.86 * dz_sign), (2.36, -0.86 * dz_sign))):
            objects.extend(
                add_source_fastener_y(
                    f"{leaf_name}_red_leaf_screw_{index}",
                    x=x,
                    z=z_base + dz,
                    y_front=REAR_LEAF_Y - 0.40,
                    y_back=2.62,
                    head_radius=0.78,
                    nut_radius=0.68,
                    shaft_radius=0.17,
                    rear_height=0.52,
                )
            )

    return objects


def build_model() -> list:
    objects = []
    objects.extend(build_door_slab())
    objects.extend(build_hinge_leaves_and_barrels())
    objects.extend(build_fasteners())
    root = set_origin_group(objects, "door_hinge_model_root")
    return [root, *objects]


# ============================================================
# Scene setup and rendering
# ============================================================

"""Minimal render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

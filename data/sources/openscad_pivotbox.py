from __future__ import annotations
from dataclasses import dataclass
from mathutils import Vector
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the pivot-box model."""





Color = tuple[float, float, float, float]


def get_material(name: str, color: Color) -> bpy.types.Material:
    """Return a simple diffuse material, creating it once per render."""
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = color
            bsdf.inputs["Roughness"].default_value = 0.72
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material | None) -> bpy.types.Object:
    if material is not None:
        obj.data.materials.append(material)
    return obj


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return assign_material(obj, material)


def add_cylinder(
    name: str,
    diameter: float,
    height: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
    material: bpy.types.Material | None = None,
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
    return assign_material(obj, material)


def _cube_vertices_faces(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    vertex_offset: int,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int, int]]]:
    cx, cy, cz = center
    sx, sy, sz = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    vertices = [
        (cx - sx, cy - sy, cz - sz),
        (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz),
        (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz),
        (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz),
        (cx - sx, cy + sy, cz + sz),
    ]
    faces = [
        (vertex_offset + 0, vertex_offset + 1, vertex_offset + 2, vertex_offset + 3),
        (vertex_offset + 4, vertex_offset + 7, vertex_offset + 6, vertex_offset + 5),
        (vertex_offset + 0, vertex_offset + 4, vertex_offset + 5, vertex_offset + 1),
        (vertex_offset + 1, vertex_offset + 5, vertex_offset + 6, vertex_offset + 2),
        (vertex_offset + 2, vertex_offset + 6, vertex_offset + 7, vertex_offset + 3),
        (vertex_offset + 3, vertex_offset + 7, vertex_offset + 4, vertex_offset + 0),
    ]
    return vertices, faces


def add_cut_box_mesh(
    name: str,
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> bpy.types.Object | None:
    """Create one temporary cutter object from many disjoint cuboids."""
    if not boxes:
        return None

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for center, size in boxes:
        box_vertices, box_faces = _cube_vertices_faces(center, size, len(vertices))
        vertices.extend(box_vertices)
        faces.extend(box_faces)

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    cutter = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(cutter)
    return cutter


def _remove_object(obj: bpy.types.Object) -> None:
    mesh = obj.data if hasattr(obj, "data") else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh and getattr(mesh, "users", 0) == 0:
        bpy.data.meshes.remove(mesh)


def apply_boolean_difference(
    target: bpy.types.Object,
    cutter: bpy.types.Object,
    name: str = "boolean_difference",
    delete_cutter: bool = True,
) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    modifier = target.modifiers.new(name=name, type="BOOLEAN")
    modifier.operation = "DIFFERENCE"
    modifier.operand_type = "OBJECT"
    modifier.object = cutter
    if hasattr(modifier, "solver"):
        modifier.solver = "EXACT"
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    if delete_cutter:
        _remove_object(cutter)
    return target


def apply_boolean_union(
    target: bpy.types.Object,
    operand: bpy.types.Object,
    name: str = "boolean_union",
    delete_operand: bool = True,
) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    modifier = target.modifiers.new(name=name, type="BOOLEAN")
    modifier.operation = "UNION"
    modifier.operand_type = "OBJECT"
    modifier.object = operand
    if hasattr(modifier, "solver"):
        modifier.solver = "EXACT"
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    if delete_operand:
        _remove_object(operand)
    return target


def apply_boolean_cut_boxes(
    target: bpy.types.Object,
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    name: str,
) -> bpy.types.Object:
    cutter = add_cut_box_mesh(f"{name}_cutter", boxes)
    if cutter is None:
        return target
    return apply_boolean_difference(target, cutter, name=f"{name}_cuts")


def finger_segments(
    length: float,
    thickness: float,
    finger_width: float | None = None,
    finger_margin: float | None = None,
    left_inset: float = 0.0,
    right_inset: float = 0.0,
) -> list[tuple[float, float]]:
    """Match the spacing logic from box.scad's `cuts()` helper."""
    fw = finger_width if finger_width is not None else thickness * 2.0
    fm = finger_margin if finger_margin is not None else thickness * 2.0
    usable = length - left_inset - right_inset
    inner_width = usable - thickness * 2.0 - 2.0 * fm
    count_index = math.floor((inner_width / fw - 1.0) / 2.0)
    count_index = max(count_index, 0)
    steps = count_index * 2 + 1
    if length < 3.0 * thickness or steps <= 0:
        return []

    fitting_width = inner_width / steps if inner_width >= fw else fw
    if length < fw * 3.0:
        fitting_width = length / 3.0

    start = left_inset + (usable - steps * fitting_width) / 2.0
    return [(start + i * fitting_width * 2.0, fitting_width) for i in range(count_index + 1)]


def add_label(
    name: str,
    text: str,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float],
    size: float,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Add an extruded text label and convert it to mesh for scene normalization."""
    bpy.ops.object.text_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = size
    obj.data.extrude = 0.18
    assign_material(obj, material)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target="MESH")
    converted = bpy.context.object
    converted.name = name
    return converted


def add_rotation_label(
    name: str,
    text: str,
    location: Vector,
    rotation: tuple[float, float, float],
    size: float,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    return add_label(name, text, tuple(location), rotation, size, material)


# ============================================================
# Procedural assembly
# ============================================================

"""Assembly logic for the pivot-box model."""






@dataclass(frozen=True)
class Materials:
    front: bpy.types.Material
    back: bpy.types.Material
    bottom: bpy.types.Material
    top: bpy.types.Material
    side: bpy.types.Material
    w_divider: bpy.types.Material
    h_divider: bpy.types.Material
    cut_shadow: bpy.types.Material
    label: bpy.types.Material


def make_materials() -> Materials:
    return Materials(
        front=get_material("front_royal_blue", (0.05, 0.18, 0.70, 1.0)),
        back=get_material("back_royal_blue", (0.03, 0.11, 0.42, 1.0)),
        bottom=get_material("bottom_slate_blue", (0.34, 0.29, 0.80, 1.0)),
        top=get_material("top_slate_blue", (0.33, 0.28, 0.76, 1.0)),
        side=get_material("side_medium_aquamarine", (0.21, 0.62, 0.48, 1.0)),
        w_divider=get_material("w_divider_cadet_blue", (0.25, 0.49, 0.60, 1.0)),
        h_divider=get_material("h_divider_cornflower_blue", (0.30, 0.55, 0.92, 1.0)),
        cut_shadow=get_material("recessed_finger_shadow", (0.07, 0.05, 0.22, 1.0)),
        label=get_material("engraved_yellow_labels", (1.0, 0.88, 0.0, 1.0)),
    )


def add_panel(
    objects: list[bpy.types.Object],
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: bpy.types.Material,
) -> bpy.types.Object:
    obj = add_box(name, size, location, material)
    objects.append(obj)
    return obj


def subtract_panel_cuts(
    panel: bpy.types.Object,
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    name: str,
) -> bpy.types.Object:
    return apply_boolean_cut_boxes(panel, boxes, name)


def front_back_panel_cut_boxes(
    x_min: float,
    y_center: float,
    z_base: float,
    width: float,
    height: float,
    thickness: float,
    keep_top: bool,
    h_dividers: int = 0,
    ears: bool = False,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    over_y = thickness * 1.55
    eps = 0.08

    for offset, finger_width in finger_segments(width, thickness):
        boxes.append(
            (
                (x_min + offset + finger_width / 2.0, y_center, z_base + thickness / 2.0),
                (finger_width, over_y, thickness + eps),
            )
        )
    if keep_top and not ears:
        for offset, finger_width in finger_segments(width, thickness):
            boxes.append(
                (
                    (x_min + width - offset - finger_width / 2.0, y_center, z_base + height - thickness / 2.0),
                    (finger_width, over_y, thickness + eps),
                )
            )

    for z_offset, finger_width in finger_segments(height, thickness):
        z_center = z_base + z_offset + finger_width / 2.0
        boxes.append(((x_min + thickness / 2.0, y_center, z_center), (thickness + eps, over_y, finger_width)))
        boxes.append(((x_min + width - thickness / 2.0, y_center, z_center), (thickness + eps, over_y, finger_width)))

    for index in range(h_dividers):
        x = x_min + width / (h_dividers + 1) * (index + 1) - thickness / 2.0
        for z_offset, finger_width in finger_segments(height, thickness, left_inset=thickness * 2.0):
            boxes.append(((x + thickness / 2.0, y_center, z_base + z_offset + finger_width / 2.0), (thickness + eps, over_y, finger_width)))

    return boxes


def side_panel_cut_boxes(
    x_center: float,
    y_min: float,
    z_base: float,
    depth: float,
    height: float,
    thickness: float,
    keep_top: bool,
    w_dividers: int = 0,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    over_x = thickness * 1.55
    eps = 0.08

    for offset, finger_width in finger_segments(depth, thickness, left_inset=thickness, right_inset=thickness):
        boxes.append(
            (
                (x_center, y_min + offset + finger_width / 2.0, z_base + thickness / 2.0),
                (over_x, finger_width, thickness + eps),
            )
        )
    if keep_top:
        for offset, finger_width in finger_segments(depth, thickness, left_inset=thickness, right_inset=thickness):
            boxes.append(
                (
                    (x_center, y_min + depth - offset - finger_width / 2.0, z_base + height - thickness / 2.0),
                    (over_x, finger_width, thickness + eps),
                )
            )

    for z_offset, finger_width in finger_segments(height, thickness):
        z_center = z_base + z_offset + finger_width / 2.0
        boxes.append(((x_center, y_min + thickness / 2.0, z_center), (over_x, thickness + eps, finger_width)))
        boxes.append(((x_center, y_min + depth - thickness / 2.0, z_center), (over_x, thickness + eps, finger_width)))

    for index in range(w_dividers):
        y = y_min + depth / (w_dividers + 1) * (index + 1) - thickness / 2.0
        for z_offset, finger_width in finger_segments(height, thickness, left_inset=thickness * 2.0):
            boxes.append(((x_center, y + thickness / 2.0, z_base + z_offset + finger_width / 2.0), (over_x, thickness + eps, finger_width)))

    return boxes


def top_bottom_panel_cut_boxes(
    x_min: float,
    y_min: float,
    z_center: float,
    width: float,
    depth: float,
    thickness: float,
    inverse: bool,
    hinge_reliefs: bool = False,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    over_z = thickness * 1.7
    eps = 0.08

    if hinge_reliefs and not inverse:
        # Ear-bearing lids in box.scad skip the normal cut_top() rhythm and only
        # remove two rectangular clearances near the hinge side.
        pass
    elif inverse:
        # box.scad's invcuts() subtracts the complement of cuts(): the panel keeps raised tabs
        # and loses the long strips between them.
        for offset, finger_width in complementary_finger_segments(width, thickness):
            boxes.append(
                (
                    (x_min + offset + finger_width / 2.0, y_min + thickness / 2.0, z_center),
                    (finger_width, thickness + eps, over_z),
                )
            )
            boxes.append(
                (
                    (x_min + offset + finger_width / 2.0, y_min + depth - thickness / 2.0, z_center),
                    (finger_width, thickness + eps, over_z),
                )
            )
        for offset, finger_width in complementary_finger_segments(depth - thickness * 2.0, thickness):
            y_center = y_min + thickness + offset + finger_width / 2.0
            boxes.append(((x_min + thickness / 2.0, y_center, z_center), (thickness + eps, finger_width, over_z)))
            boxes.append(((x_min + width - thickness / 2.0, y_center, z_center), (thickness + eps, finger_width, over_z)))
    else:
        for offset, finger_width in finger_segments(width, thickness):
            boxes.append(
                (
                    (x_min + offset + finger_width / 2.0, y_min + thickness / 2.0, z_center),
                    (finger_width, thickness + eps, over_z),
                )
            )
            boxes.append(
                (
                    (x_min + offset + finger_width / 2.0, y_min + depth - thickness / 2.0, z_center),
                    (finger_width, thickness + eps, over_z),
                )
            )
        for offset, finger_width in finger_segments(depth - thickness * 2.0, thickness):
            y_center = y_min + thickness + offset + finger_width / 2.0
            boxes.append(((x_min + thickness / 2.0, y_center, z_center), (thickness + eps, finger_width, over_z)))
            boxes.append(((x_min + width - thickness / 2.0, y_center, z_center), (thickness + eps, finger_width, over_z)))

    if hinge_reliefs:
        for y_center in (y_min + thickness / 2.0, y_min + depth - thickness / 2.0):
            boxes.append(((x_min + thickness * 2.0, y_center, z_center), (thickness * 2.0, thickness + eps, over_z)))

    return boxes


def complementary_finger_segments(length: float, thickness: float) -> list[tuple[float, float]]:
    source = finger_segments(length, thickness)
    if not source:
        return []

    intervals: list[tuple[float, float]] = []
    cursor = 0.0
    for start, width in source:
        if start > cursor:
            intervals.append((cursor, start - cursor))
        cursor = max(cursor, start + width)
    if cursor < length:
        intervals.append((cursor, length - cursor))
    return [(start, width) for start, width in intervals if width > 0.12]


def add_notched_front_back_panel(
    objects: list[bpy.types.Object],
    name: str,
    x_min: float,
    y_center: float,
    z_base: float,
    width: float,
    height: float,
    thickness: float,
    material: bpy.types.Material,
    keep_top: bool,
    h_dividers: int = 0,
    ears: bool = False,
    ear_side: str = "left",
) -> bpy.types.Object:
    panel = add_panel(
        objects,
        name,
        (width, thickness, height),
        (x_min + width / 2.0, y_center, z_base + height / 2.0),
        material,
    )
    subtract_panel_cuts(
        panel,
        front_back_panel_cut_boxes(x_min, y_center, z_base, width, height, thickness, keep_top, h_dividers, ears),
        name,
    )
    if ears:
        add_integrated_ear(panel, name, x_min, x_min + width, y_center, z_base, height, thickness, material, ear_side)
    return panel


def add_notched_side_panel(
    objects: list[bpy.types.Object],
    name: str,
    x_center: float,
    y_min: float,
    z_base: float,
    depth: float,
    height: float,
    thickness: float,
    material: bpy.types.Material,
    keep_top: bool,
    w_dividers: int = 0,
) -> bpy.types.Object:
    panel = add_panel(
        objects,
        name,
        (thickness, depth, height),
        (x_center, y_min + depth / 2.0, z_base + height / 2.0),
        material,
    )
    subtract_panel_cuts(panel, side_panel_cut_boxes(x_center, y_min, z_base, depth, height, thickness, keep_top, w_dividers), name)
    return panel


def add_notched_top_bottom_panel(
    objects: list[bpy.types.Object],
    name: str,
    x_min: float,
    y_min: float,
    z_center: float,
    width: float,
    depth: float,
    thickness: float,
    material: bpy.types.Material,
    inverse: bool,
    hinge_reliefs: bool = False,
) -> bpy.types.Object:
    panel = add_panel(
        objects,
        name,
        (width, depth, thickness),
        (x_min + width / 2.0, y_min + depth / 2.0, z_center),
        material,
    )
    subtract_panel_cuts(
        panel,
        top_bottom_panel_cut_boxes(x_min, y_min, z_center, width, depth, thickness, inverse, hinge_reliefs),
        name,
    )
    return panel


def add_integrated_ear(
    panel: bpy.types.Object,
    prefix: str,
    x_left: float,
    x_right: float,
    y_center: float,
    z_base: float,
    height: float,
    thickness: float,
    material: bpy.types.Material,
    side: str,
) -> None:
    radius = 10.0
    ring_width = 3.0
    x_center = x_left if side == "left" else x_right
    z_center = z_base + height
    ear = add_cylinder(
        f"{prefix}_integrated_outer_ear",
        diameter=radius * 2.0,
        height=thickness,
        location=(x_center, y_center, z_center),
        rotation=(math.radians(90), 0.0, 0.0),
        vertices=80,
        material=material,
    )
    apply_boolean_union(panel, ear, name=f"{prefix}_ear_union")

    inner = add_cylinder(
        f"{prefix}_inner_ear_void",
        diameter=(radius - ring_width) * 2.0,
        height=thickness * 1.8,
        location=(x_center, y_center, z_center),
        rotation=(math.radians(90), 0.0, 0.0),
        vertices=72,
        material=None,
    )
    apply_boolean_difference(panel, inner, name=f"{prefix}_ear_inner_cut")
    direction = 1.0 if side == "left" else -1.0
    root = add_box(
        f"{prefix}_ear_inner_square_root",
        (thickness * 1.15, thickness * 1.85, thickness * 1.15),
        (x_center + direction * thickness * 0.5, y_center, z_center + thickness * 0.5),
        material,
    )
    apply_boolean_union(panel, root, name=f"{prefix}_ear_square_root_union")


def add_front_back_interlock_tabs(
    objects: list[bpy.types.Object],
    prefix: str,
    x_min: float,
    y_center: float,
    z_base: float,
    width: float,
    height: float,
    thickness: float,
    material: bpy.types.Material,
) -> None:
    for index, (offset, finger_width) in enumerate(finger_segments(width, thickness)):
        add_panel(
            objects,
            f"{prefix}_bottom_interlock_tab_{index:02d}",
            (finger_width * 0.94, thickness * 1.08, thickness * 0.92),
            (x_min + offset + finger_width / 2.0, y_center, z_base + thickness / 2.0),
            material,
        )
    for side_x, side_name in ((x_min + thickness / 2.0, "left"), (x_min + width - thickness / 2.0, "right")):
        for index, (offset, finger_width) in enumerate(finger_segments(height, thickness)):
            add_panel(
                objects,
                f"{prefix}_{side_name}_vertical_interlock_tab_{index:02d}",
                (thickness * 0.92, thickness * 1.08, finger_width * 0.94),
                (side_x, y_center, z_base + offset + finger_width / 2.0),
                material,
            )


def add_side_interlock_tabs(
    objects: list[bpy.types.Object],
    prefix: str,
    x_center: float,
    y_min: float,
    z_base: float,
    depth: float,
    height: float,
    thickness: float,
    bottom_material: bpy.types.Material,
    wall_material: bpy.types.Material,
) -> None:
    for index, (offset, finger_width) in enumerate(finger_segments(depth, thickness, left_inset=thickness, right_inset=thickness)):
        add_panel(
            objects,
            f"{prefix}_bottom_interlock_tab_{index:02d}",
            (thickness * 1.08, finger_width * 0.94, thickness * 0.92),
            (x_center, y_min + offset + finger_width / 2.0, z_base + thickness / 2.0),
            bottom_material,
        )
    for side_y, side_name in ((y_min + thickness / 2.0, "front"), (y_min + depth - thickness / 2.0, "back")):
        for index, (offset, finger_width) in enumerate(finger_segments(height, thickness)):
            add_panel(
                objects,
                f"{prefix}_{side_name}_vertical_interlock_tab_{index:02d}",
                (thickness * 1.08, thickness * 0.92, finger_width * 0.94),
                (x_center, side_y, z_base + offset + finger_width / 2.0),
                wall_material,
            )


def add_box_panels(
    objects: list[bpy.types.Object],
    prefix: str,
    x_min: float,
    y_min: float,
    z_base: float,
    width: float,
    depth: float,
    height: float,
    thickness: float,
    open_top: bool,
    dividers: tuple[int, int],
    materials: Materials,
    ears: bool = False,
) -> None:
    """Build one assembled laser-cut box from panel-like solids."""
    x_center = x_min + width / 2.0
    y_center = y_min + depth / 2.0
    top_z = z_base + height

    add_notched_top_bottom_panel(
        objects,
        f"{prefix}_bottom_panel",
        x_min,
        y_min,
        z_base + thickness / 2.0,
        width,
        depth,
        thickness,
        materials.bottom,
        inverse=True,
    )
    add_notched_front_back_panel(
        objects,
        f"{prefix}_front_wall",
        x_min,
        y_min + thickness / 2.0,
        z_base,
        width,
        height,
        thickness,
        materials.front,
        keep_top=not open_top,
        h_dividers=dividers[1],
        ears=ears,
    )
    add_notched_front_back_panel(
        objects,
        f"{prefix}_back_wall",
        x_min,
        y_min + depth - thickness / 2.0,
        z_base,
        width,
        height,
        thickness,
        materials.back,
        keep_top=not open_top,
        h_dividers=dividers[1],
        ears=ears,
        ear_side="right",
    )
    add_notched_side_panel(
        objects,
        f"{prefix}_left_wall",
        x_min + thickness / 2.0,
        y_min,
        z_base,
        depth,
        height,
        thickness,
        materials.side,
        keep_top=not open_top,
        w_dividers=dividers[0],
    )
    add_notched_side_panel(
        objects,
        f"{prefix}_right_wall",
        x_min + width - thickness / 2.0,
        y_min,
        z_base,
        depth,
        height,
        thickness,
        materials.side,
        keep_top=not open_top,
        w_dividers=dividers[0],
    )
    add_front_back_interlock_tabs(
        objects,
        f"{prefix}_front_wall",
        x_min,
        y_min + thickness / 2.0,
        z_base,
        width,
        height,
        thickness,
        materials.bottom,
    )
    add_front_back_interlock_tabs(
        objects,
        f"{prefix}_back_wall",
        x_min,
        y_min + depth - thickness / 2.0,
        z_base,
        width,
        height,
        thickness,
        materials.bottom,
    )
    add_side_interlock_tabs(
        objects,
        f"{prefix}_left_wall",
        x_min + thickness / 2.0,
        y_min,
        z_base,
        depth,
        height,
        thickness,
        materials.bottom,
        materials.front,
    )
    add_side_interlock_tabs(
        objects,
        f"{prefix}_right_wall",
        x_min + width - thickness / 2.0,
        y_min,
        z_base,
        depth,
        height,
        thickness,
        materials.bottom,
        materials.front,
    )

    if not open_top:
        add_notched_top_bottom_panel(
            objects,
            f"{prefix}_top_panel",
            x_min,
            y_min,
            top_z + thickness / 2.0,
            width,
            depth,
            thickness,
            materials.top,
            inverse=not ears,
            hinge_reliefs=ears,
        )

    w_divs, h_divs = dividers
    for index in range(w_divs):
        y = y_min + depth / (w_divs + 1) * (index + 1)
        divider = add_panel(
            objects,
            f"{prefix}_width_divider_{index + 1}",
            (width, thickness, height - thickness),
            (x_center, y, z_base + thickness + (height - thickness) / 2.0),
            materials.w_divider,
        )
        subtract_panel_cuts(
            divider,
            front_back_panel_cut_boxes(x_min, y, z_base + thickness, width, height - thickness, thickness, keep_top=False),
            f"{prefix}_width_divider_{index + 1}",
        )
    for index in range(h_divs):
        x = x_min + width / (h_divs + 1) * (index + 1)
        divider = add_panel(
            objects,
            f"{prefix}_height_divider_{index + 1}",
            (thickness, depth, height - thickness),
            (x, y_center, z_base + thickness + (height - thickness) / 2.0),
            materials.h_divider,
        )
        subtract_panel_cuts(
            divider,
            side_panel_cut_boxes(x, y_min, z_base + thickness, depth, height - thickness, thickness, keep_top=False),
            f"{prefix}_height_divider_{index + 1}",
        )


def add_source_labels(objects: list[bpy.types.Object], materials: Materials) -> None:
    # The OpenSCAD source was rendered with panel labels on; these thin yellow meshes help compare orientation.
    label_specs = [
        ("front_left_label", "Front", (-82.0, -75.15, 27.0), (math.radians(90), 0.0, 0.0), 15.0),
        ("front_right_label", "Front", (10.0, -75.15, 27.0), (math.radians(90), 0.0, 0.0), 15.0),
        ("back_left_label", "Back", (-36.0, 75.15, 27.0), (math.radians(90), 0.0, math.radians(180)), 14.0),
        ("back_right_label", "Back", (82.0, 75.15, 27.0), (math.radians(90), 0.0, math.radians(180)), 14.0),
        ("right_label", "Right", (150.15, 0.0, 25.0), (math.radians(90), 0.0, math.radians(90)), 14.0),
        ("top_label", "Top", (-75.0, 0.0, 54.6), (0.0, 0.0, math.radians(-20)), 10.0),
        ("open_divider_label", "Divider", (0.05, 0.0, 30.5), (math.radians(90), 0.0, math.radians(90)), 11.5),
        ("middle_divider_label", "Divider", (-75.0, 25.0, 45.5), (math.radians(90), 0.0, 0.0), 11.0),
    ]
    for name, text, location, rotation, size in label_specs:
        objects.append(add_label(name, text, location, rotation, size, materials.label))

    # Small raised hints that read as labels when the top view compresses the text.
    for x in (-150.0, 0.0):
        objects.append(
            add_rotation_label(
                f"hinge_axis_tick_{x:g}",
                "|",
                Vector((x, -75.4, 51.0)),
                (math.radians(90), 0.0, 0.0),
                12.0,
                materials.label,
            )
        )


def build_model() -> list[bpy.types.Object]:
    """Build the assembled laser-cut pivot box."""
    mats = make_materials()
    objects: list[bpy.types.Object] = []

    thickness = 4.0
    width = 150.0
    height = 50.0
    depth = 150.0

    # Center the 300 x 150 lower tray around the origin before the shared renderer normalizes it.
    add_box_panels(
        objects,
        "lower_tray",
        x_min=-150.0,
        y_min=-75.0,
        z_base=0.0,
        width=width * 2.0,
        depth=depth,
        height=height,
        thickness=thickness,
        open_top=True,
        dividers=(0, 1),
        materials=mats,
    )

    # The smaller middle and closed top boxes occupy the same left 150 mm footprint.
    # Their side walls coincide with the tray walls; the visible contribution is the
    # left lid sheet, hinge ears, and the internal divider panel evidence under it.
    for index, y in enumerate((-25.0, 25.0), start=1):
        divider = add_panel(
            objects,
            f"middle_open_box_hidden_width_divider_{index}",
            (width, thickness, height - thickness),
            (-75.0, y, thickness + (height - thickness) / 2.0),
            mats.w_divider,
        )
        subtract_panel_cuts(
            divider,
            front_back_panel_cut_boxes(-150.0, y, thickness, width, height - thickness, thickness, keep_top=False),
            f"middle_open_box_hidden_width_divider_{index}",
        )
    divider = add_panel(
        objects,
        "top_closed_box_internal_width_divider",
        (width, thickness, height - thickness),
        (-75.0, 0.0, thickness + (height - thickness) / 2.0),
        mats.w_divider,
    )
    subtract_panel_cuts(
        divider,
        front_back_panel_cut_boxes(-150.0, 0.0, thickness, width, height - thickness, thickness, keep_top=False),
        "top_closed_box_internal_width_divider",
    )
    add_notched_top_bottom_panel(
        objects,
        "top_closed_box_top_panel",
        -150.0,
        -75.0,
        height + thickness / 2.0,
        width,
        depth,
        thickness,
        mats.top,
        inverse=False,
        hinge_reliefs=True,
    )
    add_notched_front_back_panel(
        objects,
        "top_closed_box_front_ear_sheet",
        -150.0,
        -75.0 + thickness / 2.0,
        0.0,
        width,
        height,
        thickness,
        mats.front,
        keep_top=True,
        ears=True,
        ear_side="left",
    )
    add_notched_front_back_panel(
        objects,
        "top_closed_box_back_ear_sheet",
        -150.0,
        75.0 - thickness / 2.0,
        0.0,
        width,
        height,
        thickness,
        mats.back,
        keep_top=True,
        ears=True,
        ear_side="left",
    )

    # Emphasize the visible seam between the covered half and the open tray.
    add_panel(
        objects,
        "center_lap_dark_gap",
        (1.1, depth + thickness * 0.6, thickness * 0.8),
        (0.0, 0.0, height + 0.9),
        mats.cut_shadow,
    )
    add_panel(
        objects,
        "front_panel_split_seam",
        (0.75, thickness * 1.35, height * 0.72),
        (0.0, -75.3, height * 0.47),
        mats.cut_shadow,
    )
    add_panel(
        objects,
        "back_panel_split_seam",
        (0.75, thickness * 1.35, height * 0.72),
        (0.0, 75.3, height * 0.47),
        mats.cut_shadow,
    )

    add_source_labels(objects, mats)
    return objects


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

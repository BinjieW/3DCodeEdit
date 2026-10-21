from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the tool maker's vise model."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    metallic: float = 0.0
    roughness: float = 0.48


BODY_YELLOW = MaterialSpec("openscad_warm_body", (0.96, 0.76, 0.04, 1.0), 0.0, 0.42)
ROD_GREEN = MaterialSpec("source_threaded_rod_green", (0.12, 0.55, 0.10, 1.0), 0.15, 0.34)
NUT_RED = MaterialSpec("source_hex_nut_red", (0.86, 0.04, 0.03, 1.0), 0.05, 0.38)
KNOB_BLUE = MaterialSpec("source_knob_blue", (0.05, 0.20, 0.86, 1.0), 0.0, 0.36)
CLAMP_CYAN = MaterialSpec("source_movable_clamp_cyan", (0.0, 0.75, 0.82, 1.0), 0.0, 0.42)
RECESS_SHADOW = MaterialSpec("base_recess_shadow", (0.42, 0.46, 0.08, 1.0), 0.0, 0.55)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def get_material(spec: MaterialSpec) -> bpy.types.Material:
    material = bpy.data.materials.get(spec.name) or bpy.data.materials.new(spec.name)
    material.diffuse_color = spec.color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = spec.color
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = spec.metallic
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = spec.roughness
    return material


def assign_material(obj: bpy.types.Object, spec: MaterialSpec) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(get_material(spec))
    return obj


def _activate(obj: bpy.types.Object) -> None:
    for selected in bpy.context.selected_objects:
        selected.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def shade_smooth(obj: bpy.types.Object) -> bpy.types.Object:
    _activate(obj)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)
    return obj


def shade_flat(obj: bpy.types.Object) -> bpy.types.Object:
    _activate(obj)
    bpy.ops.object.shade_flat()
    obj.select_set(False)
    return obj


def axis_rotation(axis: str) -> tuple[float, float, float]:
    normalized = axis.upper()
    if normalized == "X":
        return (0.0, math.pi / 2.0, 0.0)
    if normalized == "Y":
        return (math.pi / 2.0, 0.0, 0.0)
    if normalized == "Z":
        return (0.0, 0.0, 0.0)
    raise ValueError(f"Unsupported cylinder axis: {axis!r}")


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: MaterialSpec | None = None,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    _activate(obj)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.select_set(False)
    if material is not None:
        assign_material(obj, material)
    return obj


def add_cylinder(
    name: str,
    diameter: float,
    height: float,
    location: tuple[float, float, float],
    axis: str = "Z",
    vertices: int = 64,
    material: MaterialSpec | None = None,
    rotation_offset: float = 0.0,
) -> bpy.types.Object:
    rotation = axis_rotation(axis)
    if axis.upper() == "Z":
        rotation = (rotation[0], rotation[1], rotation[2] + rotation_offset)
    else:
        rotation = (rotation[0], rotation[1], rotation[2] + rotation_offset)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=diameter / 2.0,
        depth=height,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    if material is not None:
        assign_material(obj, material)
    if vertices > 16:
        shade_smooth(obj)
    else:
        shade_flat(obj)
    return obj


def apply_boolean(target: bpy.types.Object, cutter: bpy.types.Object, operation: str) -> bpy.types.Object:
    modifier = target.modifiers.new(name=f"{operation.lower()}_{cutter.name}", type="BOOLEAN")
    modifier.operation = operation
    modifier.object = cutter
    if hasattr(modifier, "solver"):
        modifier.solver = "EXACT"
    _activate(target)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    target.select_set(False)
    bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def boolean_union(name: str, objects: Iterable[bpy.types.Object]) -> bpy.types.Object:
    parts = list(objects)
    if not parts:
        raise ValueError("boolean_union requires at least one object")
    target = parts[0]
    for part in parts[1:]:
        apply_boolean(target, part, "UNION")
    target.name = name
    return target


def subtract_box(
    target: bpy.types.Object,
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    cutter = add_box(f"cutter_{name}", size, location, rotation=rotation)
    return apply_boolean(target, cutter, "DIFFERENCE")


def subtract_cylinder(
    target: bpy.types.Object,
    name: str,
    diameter: float,
    height: float,
    location: tuple[float, float, float],
    axis: str = "Z",
    vertices: int = 64,
    rotation_offset: float = 0.0,
) -> bpy.types.Object:
    cutter = add_cylinder(
        f"cutter_{name}",
        diameter=diameter,
        height=height,
        location=location,
        axis=axis,
        vertices=vertices,
        rotation_offset=rotation_offset,
    )
    return apply_boolean(target, cutter, "DIFFERENCE")


def add_hex_nut(
    name: str,
    corner_width: float,
    thickness: float,
    hole_diameter: float,
    location: tuple[float, float, float],
    axis: str,
    material: MaterialSpec,
) -> bpy.types.Object:
    nut = add_cylinder(
        name=name,
        diameter=corner_width,
        height=thickness,
        location=location,
        axis=axis,
        vertices=6,
        material=material,
        rotation_offset=math.radians(30.0),
    )
    subtract_cylinder(
        nut,
        f"{name}_center_bore",
        diameter=hole_diameter,
        height=thickness + 0.5,
        location=location,
        axis=axis,
        vertices=48,
    )
    shade_flat(nut)
    return nut


def _angle_delta(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def add_knurled_y_cylinder(
    name: str,
    center: tuple[float, float, float],
    diameter: float,
    length: float,
    knurl_count: int,
    knurl_depth: float,
    knurl_width: float,
    twist: float,
    material: MaterialSpec,
    radial_segments: int = 180,
    length_segments: int = 90,
    plain_end_fraction: float = 0.0,
) -> bpy.types.Object:
    """Create a Y-axis cylinder with crisp recessed crossing helical knurl grooves."""
    x0, y0, z0 = center
    radius = diameter / 2.0
    half_width_angle = knurl_width / max(radius * 2.0, 1e-6)

    vertices: list[tuple[float, float, float]] = []
    vertex_depressions: list[float] = []
    faces: list[tuple[int, ...]] = []

    for y_index in range(length_segments + 1):
        t = y_index / length_segments
        y = y0 - length / 2.0 + length * t
        if plain_end_fraction > 0.0:
            end_gate = min(1.0, min(t, 1.0 - t) / plain_end_fraction)
        else:
            end_gate = 1.0
        for angle_index in range(radial_segments):
            theta = 2.0 * math.pi * angle_index / radial_segments
            depression = 0.0
            for knurl_index in range(knurl_count):
                phase = 2.0 * math.pi * knurl_index / knurl_count
                for handedness in (-1.0, 1.0):
                    helix_theta = phase + handedness * twist * (t - 0.5)
                    distance = abs(_angle_delta(theta, helix_theta))
                    if distance <= half_width_angle:
                        depression = max(depression, knurl_depth * end_gate)
            local_radius = radius - depression
            vertices.append((x0 + local_radius * math.cos(theta), y, z0 + local_radius * math.sin(theta)))
            vertex_depressions.append(depression)

    for y_index in range(length_segments):
        row = y_index * radial_segments
        next_row = (y_index + 1) * radial_segments
        for angle_index in range(radial_segments):
            next_angle = (angle_index + 1) % radial_segments
            faces.append(
                (
                    row + angle_index,
                    next_row + angle_index,
                    next_row + next_angle,
                    row + next_angle,
                )
            )

    side_face_count = len(faces)
    start_center = len(vertices)
    vertices.append((x0, y0 - length / 2.0, z0))
    end_center = len(vertices)
    vertices.append((x0, y0 + length / 2.0, z0))

    end_row = length_segments * radial_segments
    for angle_index in range(radial_segments):
        next_angle = (angle_index + 1) % radial_segments
        faces.append((start_center, next_angle, angle_index))
        faces.append((end_center, end_row + angle_index, end_row + next_angle))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for index, polygon in enumerate(mesh.polygons):
        if index < side_face_count:
            polygon.use_smooth = all(vertex_depressions[vertex_index] == 0.0 for vertex_index in polygon.vertices)
        else:
            polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def set_origin_center(objects: Iterable[bpy.types.Object]) -> None:
    for obj in objects:
        _activate(obj)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.select_set(False)


# ============================================================
# Procedural assembly
# ============================================================

"""Readable assembly logic for the tool maker's vise model."""





VISE_WIDTH = 63.5
VISE_LENGTH = 101.0
BASE_HEIGHT = 20.0

GROOVE_WIDTH = 10.0
GROOVE_DEPTH = 10.0
GROOVE_TOLERANCE = 0.1
MOVABLE_JAW_GUIDE = 20.0
TOOL_GROOVE = 5.0

JAW_HEIGHT = 40.0
FIXED_JAW_THICKNESS = 20.0
MOVABLE_JAW_THICKNESS = 20.0

MOVABLE_JAW_BOLT_DIAMETER = 3.3
MOVABLE_JAW_BOLT_LENGTH = 25.0
MOVABLE_JAW_BOLT_HEAD_DIAMETER = 5.5
MOVABLE_JAW_BOLT_HEAD_LENGTH = 3.0
MOVABLE_JAW_NUT_WIDTH = 6.0
MOVABLE_JAW_NUT_THICKNESS = 2.75
MOVABLE_JAW_CAPTIVE_NUT_SCALE = 1.2

NUT_WIDTH = 14.75
NUT_THICKNESS = 7.0
BOLT_DIAMETER = 8.25
BOLT_LENGTH = VISE_LENGTH - 2.0 * FIXED_JAW_THICKNESS + MOVABLE_JAW_THICKNESS
BOLT_KNOB_DIAMETER = 20.0
BOLT_KNOB_LENGTH = 30.0
BOLT_KNOB_KNURL_NUM = 15
BOLT_KNOB_KNURL_ANGLE = math.radians(100.0)
BOLT_KNOB_KNURL_DEPTH = 1.0
BOLT_KNOB_KNURL_WIDTH = 1.0

BASE_TOP_Z = BASE_HEIGHT / 2.0
JAW_CENTER_Z = BASE_HEIGHT / 2.0 + JAW_HEIGHT / 2.0
FIXED_JAW_Y = VISE_LENGTH / 2.0 - FIXED_JAW_THICKNESS / 2.0
GUIDE_WALL_Y = -FIXED_JAW_Y
GUIDE_NUT_Y = -VISE_LENGTH / 2.0 + FIXED_JAW_THICKNESS - NUT_THICKNESS / 2.0
ROD_CENTER_Y = -BOLT_LENGTH / 2.0
OUTER_NUT_Y = -BOLT_LENGTH + NUT_THICKNESS / 2.0
KNOB_CENTER_Y = -BOLT_LENGTH - BOLT_KNOB_LENGTH / 2.0 + NUT_THICKNESS

BORE_CLEARANCE = 0.25
SOCKETED_NUT_WIDTH = NUT_WIDTH * 0.78
SOCKETED_NUT_THICKNESS = NUT_THICKNESS * 0.48


def build_slotted_base() -> bpy.types.Object:
    base = add_box(
        "base_with_center_and_side_slots",
        size=(VISE_WIDTH, VISE_LENGTH, BASE_HEIGHT),
        location=(0.0, 0.0, 0.0),
        material=BODY_YELLOW,
    )

    side_x = VISE_WIDTH * 0.5 - GROOVE_DEPTH * 0.5 - GROOVE_TOLERANCE * 0.5
    side_slot_size = (GROOVE_WIDTH + GROOVE_TOLERANCE, VISE_LENGTH + 0.8, GROOVE_DEPTH + GROOVE_TOLERANCE)
    for sign, label in ((1.0, "right"), (-1.0, "left")):
        subtract_box(base, f"{label}_full_length_side_groove", side_slot_size, (sign * side_x, 0.0, 0.0))

    travel_slot_length = VISE_LENGTH - 2.0 * FIXED_JAW_THICKNESS
    subtract_box(
        base,
        "through_center_movable_jaw_slot",
        size=(MOVABLE_JAW_GUIDE + GROOVE_TOLERANCE, travel_slot_length + 0.2, BASE_HEIGHT + 1.0),
        location=(0.0, 0.0, 0.0),
    )
    lower_channel_size = (GROOVE_DEPTH + GROOVE_TOLERANCE, travel_slot_length + 0.6, GROOVE_WIDTH + GROOVE_TOLERANCE)
    for sign, label in ((1.0, "right"), (-1.0, "left")):
        subtract_box(base, f"{label}_underside_t_rail_relief", lower_channel_size, (sign * GROOVE_DEPTH, 0.0, -GROOVE_WIDTH))

    return base


def add_base_recess_witness_surfaces() -> list[bpy.types.Object]:
    """Add dark inset faces so the boolean side groove reads like the OpenSCAD preview."""
    surfaces: list[bpy.types.Object] = []
    travel_slot_length = VISE_LENGTH - 2.0 * FIXED_JAW_THICKNESS
    side_inner_x = VISE_WIDTH * 0.5 - GROOVE_DEPTH
    for sign, label in ((1.0, "right"), (-1.0, "left")):
        surfaces.append(
            add_box(
                f"{label}_side_groove_recess_face",
                size=(0.05, VISE_LENGTH + 0.05, GROOVE_DEPTH * 0.82),
                location=(sign * side_inner_x, 0.0, 0.0),
                material=RECESS_SHADOW,
            )
        )
        surfaces.append(
            add_box(
                f"{label}_underside_t_slot_back_face",
                size=(0.05, travel_slot_length + 0.05, GROOVE_WIDTH * 0.90),
                location=(sign * (GROOVE_DEPTH - GROOVE_DEPTH * 0.5), 0.0, -GROOVE_WIDTH),
                material=RECESS_SHADOW,
            )
        )
    return surfaces


def build_fixed_jaw() -> bpy.types.Object:
    return add_box(
        "plain_fixed_jaw_block",
        size=(VISE_WIDTH, FIXED_JAW_THICKNESS, JAW_HEIGHT),
        location=(0.0, FIXED_JAW_Y, JAW_CENTER_Z),
        material=BODY_YELLOW,
    )


def build_lead_screw_guide_wall() -> bpy.types.Object:
    wall = add_box(
        "lead_screw_guide_wall_with_hex_recess",
        size=(VISE_WIDTH, FIXED_JAW_THICKNESS, JAW_HEIGHT),
        location=(0.0, GUIDE_WALL_Y, JAW_CENTER_Z),
        material=BODY_YELLOW,
    )
    subtract_cylinder(
        wall,
        "guide_wall_round_bore",
        diameter=BOLT_DIAMETER + BORE_CLEARANCE,
        height=FIXED_JAW_THICKNESS + 2.0,
        location=(0.0, GUIDE_WALL_Y, JAW_CENTER_Z),
        axis="Y",
        vertices=64,
    )
    subtract_cylinder(
        wall,
        "guide_wall_hex_nut_socket",
        diameter=NUT_WIDTH,
        height=NUT_THICKNESS + 0.8,
        location=(0.0, GUIDE_NUT_Y, JAW_CENTER_Z),
        axis="Y",
        vertices=6,
        rotation_offset=math.radians(30.0),
    )
    return wall


def build_movable_jaw() -> bpy.types.Object:
    jaw_body = add_box(
        "movable_jaw_main_block",
        size=(VISE_WIDTH, MOVABLE_JAW_THICKNESS, JAW_HEIGHT),
        location=(0.0, 0.0, JAW_CENTER_Z),
        material=BODY_YELLOW,
    )
    guide_foot = add_box(
        "movable_jaw_lower_guide_foot",
        size=(MOVABLE_JAW_GUIDE, MOVABLE_JAW_THICKNESS, BASE_HEIGHT),
        location=(0.0, 0.0, 5.0),
        material=BODY_YELLOW,
    )
    jaw = boolean_union("movable_jaw_with_integral_guide_foot", [jaw_body, guide_foot])

    for sign, label in ((1.0, "right"), (-1.0, "left")):
        x = sign * MOVABLE_JAW_GUIDE * 0.25
        subtract_cylinder(
            jaw,
            f"{label}_vertical_clamp_bolt_bore",
            diameter=MOVABLE_JAW_BOLT_DIAMETER,
            height=MOVABLE_JAW_BOLT_LENGTH + 1.0,
            location=(x, 0.0, 2.5),
            axis="Z",
            vertices=32,
        )
        subtract_box(
            jaw,
            f"{label}_small_captive_nut_side_slot",
            size=(MOVABLE_JAW_NUT_WIDTH, MOVABLE_JAW_THICKNESS + 0.5, MOVABLE_JAW_NUT_THICKNESS),
            location=(x, MOVABLE_JAW_THICKNESS * 0.25, 3.625),
        )

    face_y = MOVABLE_JAW_THICKNESS * 0.5
    subtract_box(
        jaw,
        "movable_jaw_horizontal_v_groove_cut",
        size=(VISE_WIDTH + 1.0, TOOL_GROOVE, TOOL_GROOVE),
        location=(0.0, face_y, JAW_CENTER_Z),
        rotation=(math.radians(45.0), 0.0, 0.0),
    )
    subtract_box(
        jaw,
        "movable_jaw_vertical_v_groove_cut",
        size=(TOOL_GROOVE, TOOL_GROOVE, JAW_HEIGHT + 1.0),
        location=(0.0, face_y, JAW_CENTER_Z),
        rotation=(0.0, 0.0, math.radians(45.0)),
    )
    subtract_cylinder(
        jaw,
        "movable_jaw_lead_screw_bore",
        diameter=BOLT_DIAMETER + BORE_CLEARANCE,
        height=MOVABLE_JAW_THICKNESS + 3.0,
        location=(0.0, 0.0, JAW_CENTER_Z),
        axis="Y",
        vertices=64,
    )

    pocket_width = NUT_WIDTH * MOVABLE_JAW_CAPTIVE_NUT_SCALE
    pocket_depth = NUT_THICKNESS * 1.1
    pocket_height = JAW_HEIGHT * 0.75 + 0.5
    subtract_box(
        jaw,
        "movable_jaw_tall_captive_nut_pocket",
        size=(pocket_width, pocket_depth, pocket_height),
        location=(0.0, -NUT_THICKNESS * 0.5 + pocket_depth * 0.5, 20.0 + pocket_height * 0.5),
    )
    return jaw


def build_movable_jaw_clamp() -> bpy.types.Object:
    clamp = add_box(
        "cyan_movable_jaw_under_clamp",
        size=(MOVABLE_JAW_GUIDE + GROOVE_DEPTH, MOVABLE_JAW_THICKNESS - GROOVE_TOLERANCE, GROOVE_DEPTH * 0.5 - GROOVE_TOLERANCE),
        location=(0.0, 0.0, -BASE_HEIGHT * 0.5 + GROOVE_DEPTH * 0.25),
        material=CLAMP_CYAN,
    )
    for sign, label in ((1.0, "right"), (-1.0, "left")):
        x = sign * MOVABLE_JAW_GUIDE * 0.25
        subtract_cylinder(
            clamp,
            f"{label}_clamp_bolt_through_hole",
            diameter=MOVABLE_JAW_BOLT_DIAMETER,
            height=MOVABLE_JAW_BOLT_LENGTH,
            location=(x, 0.0, -BASE_HEIGHT * 0.5 + GROOVE_DEPTH * 0.25),
            axis="Z",
            vertices=32,
        )
        subtract_cylinder(
            clamp,
            f"{label}_clamp_counterbore",
            diameter=MOVABLE_JAW_BOLT_HEAD_DIAMETER,
            height=MOVABLE_JAW_BOLT_HEAD_LENGTH + 0.25,
            location=(x, 0.0, -8.45),
            axis="Z",
            vertices=48,
        )
    return clamp


def build_lead_screw_hardware() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    objects.append(
        add_cylinder(
            "green_lead_screw_rod",
            diameter=BOLT_DIAMETER,
            height=BOLT_LENGTH,
            location=(0.0, ROD_CENTER_Y, JAW_CENTER_Z),
            axis="Y",
            vertices=72,
            material=ROD_GREEN,
        )
    )
    objects.append(
        add_hex_nut(
            "red_captive_nut_in_guide_wall",
            corner_width=SOCKETED_NUT_WIDTH,
            thickness=SOCKETED_NUT_THICKNESS,
            hole_diameter=BOLT_DIAMETER + BORE_CLEARANCE,
            location=(0.0, GUIDE_NUT_Y - 1.05, JAW_CENTER_Z),
            axis="Y",
            material=NUT_RED,
        )
    )
    objects.append(
        add_hex_nut(
            "red_rear_lock_nut_at_knob",
            corner_width=SOCKETED_NUT_WIDTH,
            thickness=SOCKETED_NUT_THICKNESS,
            hole_diameter=BOLT_DIAMETER + BORE_CLEARANCE,
            location=(0.0, OUTER_NUT_Y - 1.0, JAW_CENTER_Z),
            axis="Y",
            material=NUT_RED,
        )
    )

    knob = add_knurled_y_cylinder(
        "blue_knurled_adjustment_knob",
        diameter=BOLT_KNOB_DIAMETER,
        length=BOLT_KNOB_LENGTH,
        center=(0.0, KNOB_CENTER_Y, JAW_CENTER_Z),
        knurl_count=BOLT_KNOB_KNURL_NUM,
        knurl_depth=BOLT_KNOB_KNURL_DEPTH,
        knurl_width=BOLT_KNOB_KNURL_WIDTH,
        twist=BOLT_KNOB_KNURL_ANGLE,
        material=KNOB_BLUE,
    )
    subtract_cylinder(
        knob,
        "knob_hex_nut_socket",
        diameter=NUT_WIDTH,
        height=NUT_THICKNESS + 0.8,
        location=(0.0, OUTER_NUT_Y, JAW_CENTER_Z),
        axis="Y",
        vertices=6,
        rotation_offset=math.radians(30.0),
    )
    objects.append(knob)
    return objects


def build_model() -> list[bpy.types.Object]:
    """Build the assembled vise using the source OpenSCAD dimensions and placements."""
    objects: list[bpy.types.Object] = [
        build_slotted_base(),
        build_fixed_jaw(),
        build_lead_screw_guide_wall(),
        build_movable_jaw(),
        build_movable_jaw_clamp(),
    ]
    objects.extend(add_base_recess_witness_surfaces())
    objects.extend(build_lead_screw_hardware())
    set_origin_center(objects)
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

from __future__ import annotations
from dataclasses import dataclass
from math import cos, pi, sin
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
from typing import Iterable
import bpy


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the child safety latch model."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.56


OPENSCAD_YELLOW = MaterialSpec("openscad_preview_yellow", (0.95, 0.74, 0.04, 1.0))
CUTTER_PREFIX = "boolean_cutter_"


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def get_material(spec: MaterialSpec) -> bpy.types.Material:
    material = bpy.data.materials.get(spec.name) or bpy.data.materials.new(spec.name)
    material.diffuse_color = spec.color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = spec.color
        principled.inputs["Roughness"].default_value = spec.roughness
    return material


def assign_material(obj: bpy.types.Object, spec: MaterialSpec = OPENSCAD_YELLOW) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(get_material(spec))
    return obj


def _new_mesh_object(
    name: str,
    vertices: list[tuple[float, float, float]],
    faces: list[list[int]],
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material is not None:
        assign_material(obj, material)
    return obj


def circle_points(
    radius: float,
    center: tuple[float, float] = (0.0, 0.0),
    segments: int = 96,
    start_angle: float = 0.0,
    end_angle: float = 2.0 * pi,
) -> list[tuple[float, float]]:
    return [
        (
            center[0] + radius * cos(start_angle + (end_angle - start_angle) * index / max(segments, 1)),
            center[1] + radius * sin(start_angle + (end_angle - start_angle) * index / max(segments, 1)),
        )
        for index in range(segments)
    ]


def add_cylinder(
    name: str,
    radius: float,
    height: float,
    location: tuple[float, float, float],
    vertices: int = 96,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=height,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    if material is not None:
        assign_material(obj, material)
    return obj


def add_extruded_xy_profile(
    name: str,
    profile: list[tuple[float, float]],
    z_min: float,
    z_max: float,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    vertices = [(x, y, z_min) for x, y in profile] + [(x, y, z_max) for x, y in profile]
    count = len(profile)
    faces: list[list[int]] = [list(reversed(range(count))), list(range(count, 2 * count))]
    for index in range(count):
        nxt = (index + 1) % count
        faces.append([index, nxt, nxt + count, index + count])
    return _new_mesh_object(name, vertices, faces, material)


def add_extruded_xy_profile_with_holes(
    name: str,
    outer_profile: list[tuple[float, float]],
    hole_profiles: list[list[tuple[float, float]]],
    z_min: float,
    z_max: float,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    loops = [outer_profile] + hole_profiles
    flat_points = [point for loop in loops for point in loop]
    vertices = [(x, y, z_min) for x, y in flat_points] + [(x, y, z_max) for x, y in flat_points]
    count = len(flat_points)

    triangles = tessellate_polygon([[Vector((x, y, 0.0)) for x, y in loop] for loop in loops])
    faces: list[list[int]] = []
    for triangle in triangles:
        faces.append([triangle[2], triangle[1], triangle[0]])
        faces.append([count + triangle[0], count + triangle[1], count + triangle[2]])

    offset = 0
    for loop_index, loop in enumerate(loops):
        loop_count = len(loop)
        for index in range(loop_count):
            a = offset + index
            b = offset + (index + 1) % loop_count
            if loop_index == 0:
                faces.append([a, b, b + count, a + count])
            else:
                faces.append([a, a + count, b + count, b])
        offset += loop_count

    return _new_mesh_object(name, vertices, faces, material)


def add_boolean_modifier(
    target: bpy.types.Object,
    cutter: bpy.types.Object,
    operation: str,
    name: str,
) -> None:
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    modifier = target.modifiers.new(name=name, type="BOOLEAN")
    modifier.operation = operation
    modifier.object = cutter
    try:
        modifier.solver = "EXACT"
    except TypeError:
        pass
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    target.select_set(False)


def boolean_difference(
    target: bpy.types.Object,
    cutters: Iterable[bpy.types.Object],
    modifier_prefix: str = "subtract",
) -> bpy.types.Object:
    for index, cutter in enumerate(cutters, start=1):
        add_boolean_modifier(target, cutter, "DIFFERENCE", f"{modifier_prefix}_{index}_{cutter.name}")
        bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def boolean_union(name: str, objects: Iterable[bpy.types.Object]) -> bpy.types.Object:
    parts = list(objects)
    if not parts:
        raise ValueError("boolean_union requires at least one object")

    target = parts[0]
    for part in parts[1:]:
        add_boolean_modifier(target, part, "UNION", f"join_{part.name}")
        bpy.data.objects.remove(part, do_unlink=True)
    target.name = name
    target.data.name = f"{name}_mesh"
    return target


def convex_hull_2d(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(set(points))
    if len(ordered) <= 1:
        return ordered

    def cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def scad_frame_polygon(length: float, knob_diameter: float, knob_radius_outer: float) -> list[tuple[float, float]]:
    return [
        (-length / 2.0 - knob_radius_outer, -knob_diameter),
        (-length / 2.0 - knob_radius_outer, knob_diameter - 1.0),
        (length / 2.0 + knob_radius_outer, knob_diameter + 1.0),
        (length / 2.0 + knob_radius_outer, -(knob_diameter + 2.0)),
    ]


def scad_center_hull_slot(
    length: float,
    knob_diameter: float,
    corner_radius: float = 2.0,
    segments: int = 48,
) -> list[tuple[float, float]]:
    centers = [
        (-(length / 2.0 - knob_diameter), -(knob_diameter - 6.0)),
        (-(length / 2.0 - knob_diameter / 1.58), knob_diameter - 6.0),
        (length / 2.0 - knob_diameter / 1.465, knob_diameter - 6.0),
        (length / 2.0 - knob_diameter, -(knob_diameter - 6.0)),
    ]
    samples: list[tuple[float, float]] = []
    for center in centers:
        samples.extend(circle_points(corner_radius, center, segments=segments))
    return convex_hull_2d(samples)


def scad_key_slot_polygon(
    center_x: float,
    knob_radius: float,
    top_divisor: float,
    top_extra: float,
) -> list[tuple[float, float]]:
    local = [
        (-knob_radius, -5.0),
        (knob_radius, -5.0),
        (knob_radius / top_divisor, knob_radius + top_extra),
        (-knob_radius / top_divisor, knob_radius + top_extra),
    ]
    return [(x + center_x, y + 5.0) for x, y in local]


def add_scad_child_latch(
    name: str,
    length: float,
    knob_diameter: float,
    thickness: float,
    material: MaterialSpec = OPENSCAD_YELLOW,
    segments: int = 96,
) -> bpy.types.Object:
    """Build the latch from the same polygon, cylinder, hull, and slot parameters as the SCAD."""
    knob_radius = (knob_diameter + 1.0) / 2.0
    left_center_x = -length / 2.0 - knob_radius
    right_center_x = length / 2.0 + knob_radius
    z_mid = thickness / 2.0

    frame = add_extruded_xy_profile(
        name=f"{name}_tapered_frame_blank",
        profile=scad_frame_polygon(length, knob_diameter, knob_radius),
        z_min=0.0,
        z_max=thickness,
        material=material,
    )
    center_slot = add_extruded_xy_profile(
        name=f"{CUTTER_PREFIX}{name}_long_hull_slot",
        profile=scad_center_hull_slot(length, knob_diameter, segments=max(24, segments // 2)),
        z_min=-1.0,
        z_max=thickness + 1.0,
    )
    boolean_difference(frame, [center_slot], modifier_prefix=f"{name}_frame_cut")

    left_end = add_cylinder(
        name=f"{name}_left_round_latching_end",
        radius=knob_diameter,
        height=thickness,
        location=(left_center_x, 0.0, z_mid),
        vertices=segments,
        material=material,
    )
    right_end = add_cylinder(
        name=f"{name}_right_larger_pivot_end",
        radius=knob_diameter + 2.0,
        height=thickness,
        location=(right_center_x, 0.0, z_mid),
        vertices=segments,
        material=material,
    )

    latch = boolean_union(name, [frame, left_end, right_end])
    cutters = [
        add_cylinder(
            name=f"{CUTTER_PREFIX}{name}_left_round_hole",
            radius=knob_radius,
            height=thickness + 2.0,
            location=(left_center_x, 0.0, z_mid),
            vertices=segments,
        ),
        add_extruded_xy_profile(
            name=f"{CUTTER_PREFIX}{name}_left_tapered_key_opening",
            profile=scad_key_slot_polygon(left_center_x, knob_radius, top_divisor=1.58, top_extra=3.0),
            z_min=-2.0,
            z_max=thickness + 2.0,
        ),
        add_cylinder(
            name=f"{CUTTER_PREFIX}{name}_right_round_hole",
            radius=knob_radius,
            height=thickness + 2.0,
            location=(right_center_x, 0.0, z_mid),
            vertices=segments,
        ),
        add_extruded_xy_profile(
            name=f"{CUTTER_PREFIX}{name}_right_tapered_key_opening",
            profile=scad_key_slot_polygon(right_center_x, knob_radius, top_divisor=1.465, top_extra=5.0),
            z_min=-2.0,
            z_max=thickness + 2.0,
        ),
    ]
    boolean_difference(latch, cutters, modifier_prefix=f"{name}_end_opening")
    assign_material(latch, material)
    return latch


# ============================================================
# Procedural assembly
# ============================================================

"""High-level assembly for the child safety latch model."""




SOURCE_KNOB_DIAMETER = 10.0
SOURCE_KNOB_DISTANCE = 80.0
SOURCE_THICKNESS = 5.0


def build_model() -> list:
    """Compose the single printable latch with source OpenSCAD parameters."""
    latch = add_scad_child_latch(
        name="child_safety_latch",
        length=SOURCE_KNOB_DISTANCE,
        knob_diameter=SOURCE_KNOB_DIAMETER,
        thickness=SOURCE_THICKNESS,
        material=OPENSCAD_YELLOW,
        segments=120,
    )
    return [latch]


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

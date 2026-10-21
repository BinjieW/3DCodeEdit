from __future__ import annotations
from dataclasses import dataclass
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the shelf/drawer bracket model."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.55


OPENSCAD_YELLOW = MaterialSpec("openscad_preview_yellow", (1.0, 0.70, 0.02, 1.0))


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def get_material(spec: MaterialSpec) -> bpy.types.Material:
    material = bpy.data.materials.get(spec.name) or bpy.data.materials.new(spec.name)
    material.diffuse_color = spec.color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = spec.color
        bsdf.inputs["Roughness"].default_value = spec.roughness
    return material


def assign_material(obj: bpy.types.Object, spec: MaterialSpec) -> bpy.types.Object:
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


def rounded_end_footprint(
    length: float,
    width: float,
    fillet_radius: float,
    segments: int = 12,
) -> list[tuple[float, float]]:
    """Footprint of the SCAD hull: square back, rounded far end corners."""
    half_width = width / 2.0
    center_x = length - fillet_radius
    top_center_y = half_width - fillet_radius
    bottom_center_y = -top_center_y

    points: list[tuple[float, float]] = [(0.0, -half_width), (center_x, -half_width)]

    for step in range(1, segments + 1):
        angle = math.radians(-90.0 + 90.0 * step / segments)
        points.append(
            (
                center_x + fillet_radius * math.cos(angle),
                bottom_center_y + fillet_radius * math.sin(angle),
            )
        )

    points.append((length, top_center_y))

    for step in range(1, segments + 1):
        angle = math.radians(90.0 * step / segments)
        points.append(
            (
                center_x + fillet_radius * math.cos(angle),
                top_center_y + fillet_radius * math.sin(angle),
            )
        )

    points.append((0.0, half_width))
    return points


def capsule_footprint(
    radius: float,
    length: float,
    long_axis: str,
    segments: int = 16,
) -> list[tuple[float, float]]:
    """Return a CCW stadium footprint centered on the origin."""
    straight = max(0.0, length - 2.0 * radius)
    half_straight = straight / 2.0
    points: list[tuple[float, float]] = []

    if long_axis == "Y":
        top_center = (0.0, half_straight)
        bottom_center = (0.0, -half_straight)
        for step in range(segments + 1):
            angle = math.radians(180.0 - 180.0 * step / segments)
            points.append((top_center[0] + radius * math.cos(angle), top_center[1] + radius * math.sin(angle)))
        for step in range(segments + 1):
            angle = math.radians(0.0 - 180.0 * step / segments)
            points.append((bottom_center[0] + radius * math.cos(angle), bottom_center[1] + radius * math.sin(angle)))
    elif long_axis == "X":
        right_center = (half_straight, 0.0)
        left_center = (-half_straight, 0.0)
        for step in range(segments + 1):
            angle = math.radians(90.0 - 180.0 * step / segments)
            points.append((right_center[0] + radius * math.cos(angle), right_center[1] + radius * math.sin(angle)))
        for step in range(segments + 1):
            angle = math.radians(270.0 - 180.0 * step / segments)
            points.append((left_center[0] + radius * math.cos(angle), left_center[1] + radius * math.sin(angle)))
    else:
        raise ValueError(f"Unsupported capsule long axis: {long_axis}")

    return points


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


def add_extruded_yz_profile(
    name: str,
    profile: list[tuple[float, float]],
    x_min: float,
    x_max: float,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    vertices = [(x_min, y, z) for z, y in profile] + [(x_max, y, z) for z, y in profile]
    count = len(profile)
    faces: list[list[int]] = [list(range(count)), list(reversed(range(count, 2 * count)))]
    for index in range(count):
        nxt = (index + 1) % count
        faces.append([index, index + count, nxt + count, nxt])
    return _new_mesh_object(name, vertices, faces, material)


def add_extruded_yz_profile_with_holes(
    name: str,
    outer_profile: list[tuple[float, float]],
    hole_profiles: list[list[tuple[float, float]]],
    x_min: float,
    x_max: float,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    loops = [outer_profile] + hole_profiles
    flat_points = [point for loop in loops for point in loop]
    vertices = [(x_min, y, z) for z, y in flat_points] + [(x_max, y, z) for z, y in flat_points]
    count = len(flat_points)

    triangles = tessellate_polygon([[Vector((z, y, 0.0)) for z, y in loop] for loop in loops])
    faces: list[list[int]] = []
    for triangle in triangles:
        faces.append([triangle[0], triangle[1], triangle[2]])
        faces.append([count + triangle[2], count + triangle[1], count + triangle[0]])

    offset = 0
    for loop_index, loop in enumerate(loops):
        loop_count = len(loop)
        for index in range(loop_count):
            a = offset + index
            b = offset + (index + 1) % loop_count
            if loop_index == 0:
                faces.append([a, a + count, b + count, b])
            else:
                faces.append([a, b, b + count, a + count])
        offset += loop_count

    return _new_mesh_object(name, vertices, faces, material)


def shifted_capsule_profile(
    center: tuple[float, float],
    radius: float,
    length: float,
    long_axis: str,
    segments: int = 24,
) -> list[tuple[float, float]]:
    footprint = capsule_footprint(radius=radius, length=length, long_axis=long_axis, segments=segments)
    return [(x + center[0], y + center[1]) for x, y in footprint]


def add_rounded_horizontal_plate(
    name: str,
    length: float,
    width: float,
    thickness: float,
    fillet_radius: float,
    material: MaterialSpec,
    holes: list[list[tuple[float, float]]] | None = None,
) -> bpy.types.Object:
    profile = rounded_end_footprint(length, width, fillet_radius)
    if holes:
        return add_extruded_xy_profile_with_holes(
            name,
            outer_profile=profile,
            hole_profiles=holes,
            z_min=0.0,
            z_max=thickness,
            material=material,
        )
    return add_extruded_xy_profile(name, profile, z_min=0.0, z_max=thickness, material=material)


def add_rounded_upright_plate(
    name: str,
    height: float,
    width: float,
    thickness: float,
    fillet_radius: float,
    material: MaterialSpec,
    holes: list[list[tuple[float, float]]] | None = None,
) -> bpy.types.Object:
    profile = rounded_end_footprint(height, width, fillet_radius)
    if holes:
        return add_extruded_yz_profile_with_holes(
            name,
            outer_profile=profile,
            hole_profiles=holes,
            x_min=0.0,
            x_max=thickness,
            material=material,
        )
    return add_extruded_yz_profile(name, profile, x_min=0.0, x_max=thickness, material=material)


def add_triangular_rib(
    name: str,
    x_start: float,
    x_end: float,
    y_min: float,
    y_max: float,
    z_base: float,
    z_peak: float,
    material: MaterialSpec,
) -> bpy.types.Object:
    profile = [(x_start, z_base), (x_end, z_base), (x_start, z_peak)]
    vertices = [(x, y_min, z) for x, z in profile] + [(x, y_max, z) for x, z in profile]
    faces = [
        [0, 1, 2],
        [5, 4, 3],
        [0, 3, 4, 1],
        [1, 4, 5, 2],
        [2, 5, 3, 0],
    ]
    return _new_mesh_object(name, vertices, faces, material)


# ============================================================
# Procedural assembly
# ============================================================

"""Procedural assembly for the shelf/drawer bracket model."""



WIDTH = 15.0
SIDE_LENGTH = 27.5
SIDE_HEIGHT = 14.5
BASE_LENGTH = 30.0
UPRIGHT_HEIGHT = 17.75
THICKNESS = 2.0
SIDE_THICKNESS = 1.75
FILLET_RADIUS = 3.25
BASE_HOLE_GAP = 4.5
SIDE_HOLE_GAP = 5.25
HOLE_DIAMETER = 6.75
HOLE_LENGTH = 9.75
DT = 0.1


def _add_base_plate() -> object:
    hole_center_x = BASE_LENGTH - BASE_HOLE_GAP - HOLE_DIAMETER / 2.0
    horizontal_slot = shifted_capsule_profile(
        center=(hole_center_x, 0.0),
        radius=HOLE_DIAMETER / 2.0,
        length=HOLE_LENGTH,
        long_axis="Y",
    )
    base = add_rounded_horizontal_plate(
        name="horizontal_base_plate_rounded_front",
        length=BASE_LENGTH,
        width=WIDTH,
        thickness=THICKNESS,
        fillet_radius=FILLET_RADIUS,
        material=OPENSCAD_YELLOW,
        holes=[horizontal_slot],
    )
    return base


def _add_upright_plate() -> object:
    hole_center_z = SIDE_HOLE_GAP + HOLE_LENGTH / 2.0
    upright_slot = shifted_capsule_profile(
        center=(hole_center_z, 0.0),
        radius=HOLE_DIAMETER / 2.0,
        length=HOLE_LENGTH,
        long_axis="X",
    )
    upright = add_rounded_upright_plate(
        name="vertical_upright_plate_rounded_top",
        height=UPRIGHT_HEIGHT,
        width=WIDTH,
        thickness=THICKNESS,
        fillet_radius=FILLET_RADIUS,
        material=OPENSCAD_YELLOW,
        holes=[upright_slot],
    )
    return upright


def _add_side_ribs() -> list[object]:
    ribs: list[object] = []
    x_start = THICKNESS - 3.0 * DT
    x_end = x_start + SIDE_LENGTH - THICKNESS + DT
    y_positions = (-WIDTH / 2.0 + DT, WIDTH / 2.0 - SIDE_THICKNESS - DT)

    for index, y_min in enumerate(y_positions, start=1):
        ribs.append(
            add_triangular_rib(
                name=f"triangular_side_reinforcement_{index}",
                x_start=x_start,
                x_end=x_end,
                y_min=y_min,
                y_max=y_min + SIDE_THICKNESS,
                z_base=THICKNESS,
                z_peak=SIDE_HEIGHT,
                material=OPENSCAD_YELLOW,
            )
        )
    return ribs


def build_model() -> list[object]:
    """Build the SCAD bracket: L plates, two obround slots, and paired side ribs."""
    objects: list[object] = [_add_base_plate(), _add_upright_plate()]
    objects.extend(_add_side_ribs())
    return objects


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current shelf/drawer bracket iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

from __future__ import annotations
from dataclasses import dataclass
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the cabinet latch catch model."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.55


OPENSCAD_YELLOW = MaterialSpec("openscad_preview_yellow", (1.0, 0.72, 0.02, 1.0))


@dataclass(frozen=True)
class CatchCurveSpec:
    curve_height: float
    curve_width: float
    radius: float
    center_y: float
    center_z: float
    local_center_y: float
    local_center_z: float


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


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material is not None:
        assign_material(obj, material)
    return obj


def capsule_profile_y_axis(radius: float, length: float, segments: int = 16) -> list[tuple[float, float]]:
    """Return a CCW obround footprint centered at the origin with its long axis on Y."""
    straight = max(0.0, length - 2.0 * radius)
    half_straight = straight / 2.0
    top_center = (0.0, half_straight)
    bottom_center = (0.0, -half_straight)
    points: list[tuple[float, float]] = []

    for step in range(segments + 1):
        angle = math.radians(180.0 - 180.0 * step / segments)
        points.append((top_center[0] + radius * math.cos(angle), top_center[1] + radius * math.sin(angle)))
    for step in range(segments + 1):
        angle = math.radians(0.0 - 180.0 * step / segments)
        points.append(
            (bottom_center[0] + radius * math.cos(angle), bottom_center[1] + radius * math.sin(angle))
        )
    return points


def shifted_profile(
    profile: list[tuple[float, float]],
    offset: tuple[float, float],
) -> list[tuple[float, float]]:
    return [(x + offset[0], y + offset[1]) for x, y in profile]


def rounded_slot_profile(
    center: tuple[float, float],
    length: float,
    width: float,
    segments: int = 16,
) -> list[tuple[float, float]]:
    """OpenSCAD screwhole() equivalent: hull of two cylinders 5.25 mm apart."""
    return shifted_profile(
        capsule_profile_y_axis(radius=width / 2.0, length=length, segments=segments),
        center,
    )


def _extrude_xy_loops(
    name: str,
    outer_loop: list[tuple[float, float]],
    hole_loops: list[list[tuple[float, float]]],
    z_min: float,
    z_max: float,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    loops = [outer_loop] + hole_loops
    flat_points = [point for loop in loops for point in loop]
    vertices = [(x, y, z_min) for x, y in flat_points] + [(x, y, z_max) for x, y in flat_points]
    point_count = len(flat_points)
    triangles = tessellate_polygon([[Vector((x, y, 0.0)) for x, y in loop] for loop in loops])

    faces: list[list[int]] = []
    for triangle in triangles:
        faces.append([triangle[2], triangle[1], triangle[0]])
        faces.append([point_count + triangle[0], point_count + triangle[1], point_count + triangle[2]])

    offset = 0
    for loop_index, loop in enumerate(loops):
        loop_count = len(loop)
        for index in range(loop_count):
            a = offset + index
            b = offset + (index + 1) % loop_count
            if loop_index == 0:
                faces.append([a, b, b + point_count, a + point_count])
            else:
                faces.append([a, a + point_count, b + point_count, b])
        offset += loop_count

    return _new_mesh_object(name, vertices, faces, material)


def add_slotted_base_plate(
    name: str,
    length: float,
    width: float,
    thickness: float,
    slot_centers: list[tuple[float, float]],
    slot_length: float,
    slot_width: float,
    material: MaterialSpec,
) -> bpy.types.Object:
    outer = [(0.0, 0.0), (length, 0.0), (length, width), (0.0, width)]
    slot = capsule_profile_y_axis(radius=slot_width / 2.0, length=slot_length, segments=16)
    holes = [shifted_profile(slot, center) for center in slot_centers]
    return _extrude_xy_loops(name, outer, holes, z_min=0.0, z_max=thickness, material=material)


def scad_catch_curve_spec(
    width: float,
    height: float,
    base_thick: float,
    top_width: float,
    sag: float,
) -> CatchCurveSpec:
    """Reproduce the OpenSCAD radius and cylinder-center calculations."""
    curve_height = height - base_thick
    curve_width = width - top_width
    chord_length = math.hypot(curve_height, curve_width)
    radius = ((chord_length / 2.0) ** 2 + sag**2) / (2.0 * sag)
    ratio = curve_height / curve_width
    a = 1.0 + ratio**2
    b = curve_height
    c = curve_width**2 * a / 4.0 - radius**2 / a
    local_center_z = (-b + math.sqrt(b**2 - 4.0 * c)) / 2.0
    local_center_y = curve_width * (ratio**2 - 1.0) / 2.0 + ratio * local_center_z
    return CatchCurveSpec(
        curve_height=curve_height,
        curve_width=curve_width,
        radius=radius,
        center_y=width + local_center_y,
        center_z=height + local_center_z,
        local_center_y=local_center_y,
        local_center_z=local_center_z,
    )


def circular_cut_boundary_y(
    z: float,
    center_y: float,
    center_z: float,
    radius: float,
) -> float:
    return center_y - math.sqrt(max(0.0, radius**2 - (z - center_z) ** 2))


def catch_ramp_yz_profile(
    width: float,
    base_thick: float,
    height: float,
    top_width: float,
    sag: float,
    segments: int = 36,
) -> tuple[list[tuple[float, float]], CatchCurveSpec]:
    """Return the YZ cross-section left by cube-minus-X-cylinder in the SCAD catch."""
    curve = scad_catch_curve_spec(
        width=width,
        height=height,
        base_thick=base_thick,
        top_width=top_width,
        sag=sag,
    )
    arc: list[tuple[float, float]] = []
    for step in range(segments + 1):
        z = height - curve.curve_height * step / segments
        y = circular_cut_boundary_y(
            z=z,
            center_y=curve.center_y,
            center_z=curve.center_z,
            radius=curve.radius,
        )
        arc.append((y, z))

    profile = [
        (0.0, base_thick),
        (0.0, height),
        (top_width, height),
        *arc[1:],
        (width, base_thick),
    ]
    return profile, curve


def add_x_extruded_yz_profile(
    name: str,
    profile: list[tuple[float, float]],
    x_min: float,
    x_max: float,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    vertices = [(x_min, y, z) for y, z in profile] + [(x_max, y, z) for y, z in profile]
    count = len(profile)
    faces: list[list[int]] = [list(reversed(range(count))), list(range(count, 2 * count))]
    for index in range(count):
        nxt = (index + 1) % count
        faces.append([index, index + count, nxt + count, nxt])
    return _new_mesh_object(name, vertices, faces, material)


def add_curved_catch_body(
    name: str,
    length: float,
    width: float,
    base_thick: float,
    height: float,
    top_width: float,
    sag: float,
    x_min: float,
    material: MaterialSpec,
    segments: int = 36,
) -> bpy.types.Object:
    profile, _curve = catch_ramp_yz_profile(
        width=width,
        base_thick=base_thick,
        height=height,
        top_width=top_width,
        sag=sag,
        segments=segments,
    )
    return add_x_extruded_yz_profile(
        name=name,
        profile=profile,
        x_min=x_min,
        x_max=x_min + length,
        material=material,
    )


# ============================================================
# Procedural assembly
# ============================================================

"""Assembly logic for the cabinet latch catch model."""



LENGTH = 40.0
WIDTH = 11.0
HEIGHT = 11.0
BASE_THICK = 2.0
TOP_WIDTH = 2.0
CURVE_LENGTH = 19.0
SAG = 1.5
HOLE_SPACING = 32.0
HOLE_LENGTH = 9.0
HOLE_WIDTH = 3.75


def _slot_centers() -> list[tuple[float, float]]:
    lower_cylinder_center_y = (WIDTH - HOLE_LENGTH + HOLE_WIDTH) / 2.0
    y_center = lower_cylinder_center_y + (HOLE_LENGTH - HOLE_WIDTH) / 2.0
    return [
        (LENGTH / 2.0 - HOLE_SPACING / 2.0, y_center),
        (LENGTH / 2.0 + HOLE_SPACING / 2.0, y_center),
    ]


def _add_base() -> object:
    return add_slotted_base_plate(
        name="thin_base_plate_with_two_rounded_slots",
        length=LENGTH,
        width=WIDTH,
        thickness=BASE_THICK,
        slot_centers=_slot_centers(),
        slot_length=HOLE_LENGTH,
        slot_width=HOLE_WIDTH,
        material=OPENSCAD_YELLOW,
    )


def _add_catch() -> object:
    return add_curved_catch_body(
        name="raised_curved_latch_catch",
        length=CURVE_LENGTH,
        width=WIDTH,
        base_thick=BASE_THICK,
        height=HEIGHT,
        top_width=TOP_WIDTH,
        sag=SAG,
        x_min=(LENGTH - CURVE_LENGTH) / 2.0,
        material=OPENSCAD_YELLOW,
        segments=48,
    )


def build_model() -> list[object]:
    """Build the SCAD union: slotted base plus cylindrical-cut raised catch."""
    return [_add_base(), _add_catch()]


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

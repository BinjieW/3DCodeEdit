from __future__ import annotations
from dataclasses import dataclass
from mathutils import Vector
from typing import Iterable, Sequence
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitives for the Futaba-style servo arm model."""





Point2D = tuple[float, float]


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.55


@dataclass(frozen=True)
class ServoArmSpec:
    head_diameter: float = 5.92
    head_height: float = 4.0
    head_thickness: float = 1.1
    head_screw_diameter: float = 2.5
    tooth_count: int = 25
    tooth_height: float = 0.3
    tooth_length: float = 0.7
    tooth_width: float = 0.1
    servo_clearance: float = 0.2
    arm_length: float = 20.0
    arm_count: int = 4
    arm_screw_diameter: float = 2.0
    radial_segments: int = 40

    @property
    def hub_outer_radius(self) -> float:
        return self.head_diameter / 2.0 + self.head_thickness

    @property
    def spline_root_radius(self) -> float:
        return self.head_diameter / 2.0 - self.tooth_height + 0.03 + self.servo_clearance

    @property
    def spline_outer_radius(self) -> float:
        return self.spline_root_radius + self.tooth_height

    @property
    def arm_width(self) -> float:
        return self.head_diameter + self.head_thickness * 2.0

    @property
    def arm_tip_radius(self) -> float:
        return self.arm_width / 3.0

    @property
    def base_plate_height(self) -> float:
        return 2.0

    @property
    def raised_socket_height(self) -> float:
        return self.head_height + 1.0

    @property
    def arm_height(self) -> float:
        return 2.0

    @property
    def socket_top_z(self) -> float:
        return 6.0

    @property
    def arm_z_min(self) -> float:
        return 0.0

    @property
    def arm_z_max(self) -> float:
        return self.arm_height

    @property
    def hub_z_min(self) -> float:
        return 0.9

    @property
    def hub_z_max(self) -> float:
        return self.hub_z_min + self.raised_socket_height

    @property
    def socket_z_min(self) -> float:
        return self.arm_z_max

    @property
    def rib_height(self) -> float:
        return self.head_height


SERVO_YELLOW = MaterialSpec("openscad_servo_yellow", (1.0, 0.78, 0.02, 1.0))
SOCKET_SHADOW = MaterialSpec("socket_shadow_green", (0.15, 0.30, 0.11, 1.0))


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
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.22
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.22
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = spec.color
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 0.12
    return material


def assign_material(obj: bpy.types.Object, spec: MaterialSpec = SERVO_YELLOW) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(get_material(spec))
    return obj


def recalculate_normals(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return obj


def shade_flat(obj: bpy.types.Object) -> bpy.types.Object:
    for polygon in obj.data.polygons:
        polygon.use_smooth = False
    return obj


def shade_smooth(obj: bpy.types.Object) -> bpy.types.Object:
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def add_cylinder(
    name: str,
    radius: float,
    height: float,
    location: tuple[float, float, float],
    *,
    vertices: int = 64,
    material: MaterialSpec = SERVO_YELLOW,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    end_fill_type: str = "NGON",
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=height,
        end_fill_type=end_fill_type,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    assign_material(obj, material)
    return shade_flat(obj)


def add_box_from_bounds(
    name: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    *,
    material: MaterialSpec = SERVO_YELLOW,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(
        location=((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = ((xmax - xmin) / 2.0, (ymax - ymin) / 2.0, (zmax - zmin) / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign_material(obj, material)
    return shade_flat(obj)


def _signed_area(points: Sequence[Point2D]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area * 0.5


def add_extruded_polygon(
    name: str,
    points: Sequence[Point2D],
    z_min: float,
    z_max: float,
    *,
    material: MaterialSpec = SERVO_YELLOW,
) -> bpy.types.Object:
    """Create a straight prism from a simple XY polygon."""
    ordered = list(points)
    if _signed_area(ordered) < 0.0:
        ordered.reverse()

    count = len(ordered)
    vertices = [(x, y, z_min) for x, y in ordered] + [(x, y, z_max) for x, y in ordered]
    faces: list[tuple[int, ...]] = [tuple(reversed(range(count))), tuple(range(count, count * 2))]
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, next_index + count, index + count))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    return recalculate_normals(shade_flat(obj))


def _circle_points(radius: float, segments: int, *, start_angle: float = 0.0) -> list[Point2D]:
    return [
        (
            math.cos(start_angle + math.tau * index / segments) * radius,
            math.sin(start_angle + math.tau * index / segments) * radius,
        )
        for index in range(segments)
    ]


def _toothed_loop(spec: ServoArmSpec) -> list[Point2D]:
    """Approximate the OpenSCAD union of the root spline cylinder and 25 tooth prisms."""
    points: list[Point2D] = []
    step = math.tau / spec.tooth_count
    half_angle = math.atan((spec.tooth_length / 2.0) / spec.spline_root_radius)
    flat_half_angle = math.atan((spec.tooth_width / 2.0) / spec.spline_outer_radius)

    for index in range(spec.tooth_count):
        angle = index * step
        radial = Vector((-math.sin(angle), math.cos(angle)))
        tangent = Vector((math.cos(angle), math.sin(angle)))
        base = radial * spec.spline_root_radius
        outer = radial * spec.spline_outer_radius
        points.extend(
            [
                tuple(base - tangent * (math.tan(step * 0.38) * spec.spline_root_radius)),
                tuple(base - tangent * (math.tan(half_angle) * spec.spline_root_radius)),
                tuple(outer - tangent * (math.tan(flat_half_angle) * spec.spline_outer_radius)),
                tuple(outer + tangent * (math.tan(flat_half_angle) * spec.spline_outer_radius)),
                tuple(base + tangent * (math.tan(half_angle) * spec.spline_root_radius)),
                tuple(base + tangent * (math.tan(step * 0.38) * spec.spline_root_radius)),
            ]
        )
    return points


def add_annular_prism_with_inner_loop(
    name: str,
    outer_radius: float,
    inner_loop: Sequence[Point2D],
    z_min: float,
    z_max: float,
    *,
    outer_segments: int = 80,
    material: MaterialSpec = SERVO_YELLOW,
) -> bpy.types.Object:
    """Build a ring mesh with a detailed inner opening."""
    outer = _circle_points(outer_radius, outer_segments)
    inner = list(inner_loop)
    if _signed_area(inner) < 0.0:
        inner.reverse()

    vertices: list[tuple[float, float, float]] = []
    vertices.extend((x, y, z_min) for x, y in outer)
    vertices.extend((x, y, z_max) for x, y in outer)
    vertices.extend((x, y, z_min) for x, y in inner)
    vertices.extend((x, y, z_max) for x, y in inner)

    outer_count = len(outer)
    inner_count = len(inner)
    outer_bottom = 0
    outer_top = outer_count
    inner_bottom = outer_count * 2
    inner_top = outer_count * 2 + inner_count
    faces: list[tuple[int, ...]] = []

    for index in range(outer_count):
        next_index = (index + 1) % outer_count
        faces.append(
            (
                outer_bottom + index,
                outer_bottom + next_index,
                outer_top + next_index,
                outer_top + index,
            )
        )

    for index in range(inner_count):
        next_index = (index + 1) % inner_count
        faces.append(
            (
                inner_bottom + next_index,
                inner_bottom + index,
                inner_top + index,
                inner_top + next_index,
            )
        )

    for index in range(outer_count):
        next_index = (index + 1) % outer_count
        angle = (index + 0.5) * math.tau / outer_count
        inner_index = min(range(inner_count), key=lambda idx: abs(math.atan2(inner[idx][1], inner[idx][0]) - angle))
        next_inner_index = min(
            range(inner_count),
            key=lambda idx: abs(math.atan2(inner[idx][1], inner[idx][0]) - ((next_index + 0.5) * math.tau / outer_count)),
        )
        faces.append(
            (
                outer_top + index,
                outer_top + next_index,
                inner_top + next_inner_index,
                inner_top + inner_index,
            )
        )
        faces.append(
            (
                outer_bottom + next_index,
                outer_bottom + index,
                inner_bottom + inner_index,
                inner_bottom + next_inner_index,
            )
        )

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    return recalculate_normals(shade_flat(obj))


def add_socket_wall_liner(spec: ServoArmSpec) -> bpy.types.Object:
    """Add a recessed visual liner on the already-cut spline wall."""
    loop = _toothed_loop(spec)
    inset_loop: list[Point2D] = []
    for x, y in loop:
        radius = math.hypot(x, y)
        if radius == 0.0:
            inset_loop.append((x, y))
            continue
        inset = 0.035
        inset_loop.append((x * (radius - inset) / radius, y * (radius - inset) / radius))

    count = len(inset_loop)
    z_min = spec.socket_z_min + 0.04
    z_max = spec.hub_z_max - 0.08
    vertices = [(x, y, z_min) for x, y in inset_loop] + [(x, y, z_max) for x, y in inset_loop]
    faces = []
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, next_index + count, index + count))

    mesh = bpy.data.meshes.new("recessed_toothed_socket_liner_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()

    obj = bpy.data.objects.new("recessed_toothed_socket_liner", mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, SOCKET_SHADOW)
    return recalculate_normals(shade_flat(obj))


def add_spline_ring(spec: ServoArmSpec) -> bpy.types.Object:
    return add_annular_prism_with_inner_loop(
        "raised_central_spline_ring",
        outer_radius=spec.hub_outer_radius,
        inner_loop=_toothed_loop(spec),
        z_min=spec.hub_z_min,
        z_max=spec.hub_z_max,
        outer_segments=spec.radial_segments,
    )


def add_hub_blank(spec: ServoArmSpec) -> bpy.types.Object:
    return add_cylinder(
        "raised_cylindrical_hub_blank",
        radius=spec.hub_outer_radius,
        height=spec.raised_socket_height,
        location=(0.0, 0.0, (spec.hub_z_min + spec.hub_z_max) / 2.0),
        vertices=spec.radial_segments,
    )


def _arm_footprint(spec: ServoArmSpec, *, segments: int = 18) -> list[Point2D]:
    width = spec.arm_width
    length = spec.arm_length
    tip_radius = spec.arm_tip_radius
    half_root = width / 2.0
    half_neck = width / 3.0

    left_tip_angle = math.pi
    right_tip_angle = 0.0
    tip: list[Point2D] = [
        (
            tip_radius * math.cos(left_tip_angle + (right_tip_angle - left_tip_angle) * index / segments),
            length + tip_radius * math.sin(left_tip_angle + (right_tip_angle - left_tip_angle) * index / segments),
        )
        for index in range(segments + 1)
    ]

    return [(-half_root, 0.0), (-half_neck, length), *tip, (half_neck, length), (half_root, 0.0)]


def add_arm_blank(spec: ServoArmSpec, angle: float, *, name: str) -> bpy.types.Object:
    obj = add_extruded_polygon(name, _arm_footprint(spec), spec.arm_z_min, spec.arm_z_max)
    obj.rotation_euler[2] = angle
    return obj


def add_arm_root_pad(spec: ServoArmSpec, angle: float, *, name: str) -> bpy.types.Object:
    obj = add_cylinder(
        name,
        radius=spec.arm_width / 2.0,
        height=spec.arm_height,
        location=(0.0, 0.0, (spec.arm_z_min + spec.arm_z_max) / 2.0),
        vertices=spec.radial_segments,
    )
    obj.rotation_euler[2] = angle
    return obj


def add_arm_tip_pad(spec: ServoArmSpec, angle: float, *, name: str) -> bpy.types.Object:
    x = -math.sin(angle) * spec.arm_length
    y = math.cos(angle) * spec.arm_length
    return add_cylinder(
        name,
        radius=spec.arm_tip_radius,
        height=spec.arm_height,
        location=(x, y, (spec.arm_z_min + spec.arm_z_max) / 2.0),
        vertices=spec.radial_segments,
    )


def add_arm_hole_cutter(spec: ServoArmSpec, angle: float, *, name: str) -> bpy.types.Object:
    x = -math.sin(angle) * spec.arm_length
    y = math.cos(angle) * spec.arm_length
    return add_cylinder(
        name,
        radius=spec.arm_screw_diameter / 2.0,
        height=spec.raised_socket_height + 4.0,
        location=(x, y, spec.raised_socket_height / 2.0),
        vertices=32,
    )


def add_center_hole_cutter(spec: ServoArmSpec, *, name: str = "center_screw_bore_cutter") -> bpy.types.Object:
    return add_cylinder(
        name,
        radius=spec.head_screw_diameter / 2.0,
        height=spec.socket_top_z + 4.0,
        location=(0.0, 0.0, spec.socket_top_z / 2.0),
        vertices=40,
    )


def add_spline_cutter(spec: ServoArmSpec, *, name: str = "toothed_socket_cutter") -> bpy.types.Object:
    return add_extruded_polygon(
        name,
        _toothed_loop(spec),
        spec.socket_z_min,
        spec.socket_top_z + 0.25,
        material=SOCKET_SHADOW,
    )


def add_spline_cutters(spec: ServoArmSpec) -> list[bpy.types.Object]:
    """Create the same root cylinder plus 25 trapezoid tooth cutters used by SCAD."""
    cutters = [
        add_cylinder(
            "spline_root_cylinder_cutter",
            radius=spec.spline_root_radius,
            height=spec.head_height + 0.35,
            location=(0.0, 0.0, (spec.socket_z_min + spec.socket_top_z) / 2.0),
            vertices=spec.radial_segments,
            material=SOCKET_SHADOW,
        )
    ]
    tooth_profile = [
        (-spec.tooth_length / 2.0, 0.0),
        (-spec.tooth_width / 2.0, spec.tooth_height),
        (spec.tooth_width / 2.0, spec.tooth_height),
        (spec.tooth_length / 2.0, 0.0),
    ]
    tooth_base_radius = spec.head_diameter / 2.0 - spec.tooth_height + spec.servo_clearance
    for index in range(spec.tooth_count):
        angle = index * math.tau / spec.tooth_count
        tooth = add_extruded_polygon(
            f"spline_tooth_cutter_{index:02d}",
            tooth_profile,
            spec.socket_z_min,
            spec.socket_top_z + 0.25,
            material=SOCKET_SHADOW,
        )
        tooth.location.x = -math.sin(angle) * tooth_base_radius
        tooth.location.y = math.cos(angle) * tooth_base_radius
        tooth.rotation_euler[2] = angle
        cutters.append(tooth)
    return cutters


def add_top_rib(spec: ServoArmSpec, angle: float, *, name: str) -> bpy.types.Object:
    return add_side_reinforcement_rib(spec, angle, name=name)


def add_side_reinforcement_rib(
    spec: ServoArmSpec,
    angle: float,
    *,
    name: str,
) -> bpy.types.Object:
    """Create the transformed SCAD reinforcement web as one narrow prism per arm."""
    local_y_near = spec.head_diameter / 2.0 + spec.head_thickness - 0.5
    half_width = spec.arm_height / 2.0
    side_profile = [
        (spec.arm_length / 1.7 + local_y_near, spec.arm_z_max),
        (local_y_near, spec.arm_z_max),
        (local_y_near, spec.socket_top_z - 1.0),
        (local_y_near + 2.0, spec.socket_top_z - 1.0),
    ]
    x_min = -half_width
    x_max = half_width

    verts_local = [
        (x_min, side_profile[0][0], side_profile[0][1]),
        (x_max, side_profile[0][0], side_profile[0][1]),
        (x_min, side_profile[1][0], side_profile[1][1]),
        (x_max, side_profile[1][0], side_profile[1][1]),
        (x_min, side_profile[2][0], side_profile[2][1]),
        (x_max, side_profile[2][0], side_profile[2][1]),
        (x_min, side_profile[3][0], side_profile[3][1]),
        (x_max, side_profile[3][0], side_profile[3][1]),
    ]
    faces = [
        (0, 1, 3, 2),
        (2, 3, 5, 4),
        (4, 5, 7, 6),
        (6, 7, 1, 0),
        (0, 2, 4, 6),
        (1, 7, 5, 3),
    ]

    sin_a = math.sin(angle)
    cos_a = math.cos(angle)
    verts = [
        (
            x * cos_a - y * sin_a,
            x * sin_a + y * cos_a,
            z,
        )
        for x, y, z in verts_local
    ]
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj)
    return recalculate_normals(shade_flat(obj))


def boolean_difference(
    target: bpy.types.Object,
    cutters: Iterable[bpy.types.Object],
    *,
    keep_cutters: bool = False,
) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    for cutter in cutters:
        modifier = target.modifiers.new(name=f"cut_{cutter.name}", type="BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        if hasattr(modifier, "solver"):
            modifier.solver = "EXACT"
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        if keep_cutters:
            cutter.hide_render = True
            cutter.hide_set(True)
        else:
            bpy.data.objects.remove(cutter, do_unlink=True)
    target.select_set(False)
    return recalculate_normals(target)


def boolean_union(name: str, objects: Sequence[bpy.types.Object]) -> bpy.types.Object:
    if not objects:
        raise ValueError("boolean_union requires at least one object")
    target = objects[0]
    target.name = name
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    for obj in objects[1:]:
        modifier = target.modifiers.new(name=f"join_{obj.name}", type="BOOLEAN")
        modifier.operation = "UNION"
        modifier.object = obj
        if hasattr(modifier, "solver"):
            modifier.solver = "EXACT"
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(obj, do_unlink=True)
    target.select_set(False)
    assign_material(target)
    return recalculate_normals(shade_flat(target))


# ============================================================
# Procedural assembly
# ============================================================

"""High-level assembly for the four-arm servo horn model."""






def _arm_angles(spec: ServoArmSpec) -> list[float]:
    return [index * math.tau / spec.arm_count for index in range(spec.arm_count)]


def _build_main_body(spec: ServoArmSpec) -> bpy.types.Object:
    parts = [add_hub_blank(spec)]
    for index, angle in enumerate(_arm_angles(spec)):
        parts.extend(
            [
                add_arm_root_pad(spec, angle, name=f"arm_{index}_root_round"),
                add_arm_blank(spec, angle, name=f"arm_{index}_tapered_web"),
                add_arm_tip_pad(spec, angle, name=f"arm_{index}_rounded_tip"),
            ]
        )
        parts.append(add_side_reinforcement_rib(spec, angle, name=f"arm_{index}_triangular_side_web"))

    body = boolean_union("single_piece_servo_horn_body", parts)

    cutters = [add_center_hole_cutter(spec), *add_spline_cutters(spec)]
    cutters.extend(
        add_arm_hole_cutter(spec, angle, name=f"arm_{index}_screw_hole_cutter")
        for index, angle in enumerate(_arm_angles(spec))
    )
    return boolean_difference(body, cutters)


def build_model() -> list[bpy.types.Object]:
    """Build the Blender model in the same final orientation as the source demo."""
    spec = ServoArmSpec()
    objects: list[bpy.types.Object] = []
    objects.append(_build_main_body(spec))
    objects.append(add_socket_wall_liner(spec))
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

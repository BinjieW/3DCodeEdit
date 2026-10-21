from __future__ import annotations
from dataclasses import dataclass
from mathutils import Matrix, Vector
from pathlib import Path
from typing import Iterable, Sequence
import bpy
import math
import sys


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the Shapeoko belt clamp model."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.55


OPENSCAD_YELLOW = MaterialSpec("openscad_preview_yellow", (1.0, 0.78, 0.02, 1.0), 0.58)


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


def shade_smooth(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)
    return obj


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: MaterialSpec | None = OPENSCAD_YELLOW,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material is not None:
        assign_material(obj, material)
    return obj


def add_cylinder(
    name: str,
    diameter: float,
    height: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
    material: MaterialSpec | None = OPENSCAD_YELLOW,
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
    if material is not None:
        assign_material(obj, material)
    if smooth and vertices > 12:
        shade_smooth(obj)
    return obj


def make_box_cutter(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
) -> bpy.types.Object:
    return add_box(name=name, size=size, location=location, material=None)


def scad_rotation_matrix(rotation_degrees: tuple[float, float, float]) -> Matrix:
    """OpenSCAD vector rotate order: rotate X, then Y, then Z."""

    rx, ry, rz = (math.radians(value) for value in rotation_degrees)
    return (
        Matrix.Rotation(rz, 4, "Z")
        @ Matrix.Rotation(ry, 4, "Y")
        @ Matrix.Rotation(rx, 4, "X")
    )


def scad_transform_matrix(
    translation: tuple[float, float, float],
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Matrix:
    return Matrix.Translation(Vector(translation)) @ scad_rotation_matrix(rotation_degrees)


def boolean_difference(base: bpy.types.Object, cutters: Iterable[bpy.types.Object]) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = base
    base.select_set(True)
    for cutter in cutters:
        modifier = base.modifiers.new(f"cut_{cutter.name}", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
    base.select_set(False)
    return base


def boolean_union(name: str, parts: Sequence[bpy.types.Object]) -> bpy.types.Object:
    if not parts:
        raise ValueError("boolean_union needs at least one mesh object")
    base = parts[0]
    bpy.context.view_layer.objects.active = base
    base.select_set(True)
    for part in parts[1:]:
        modifier = base.modifiers.new(f"join_{part.name}", "BOOLEAN")
        modifier.operation = "UNION"
        modifier.object = part
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(part, do_unlink=True)
    base.name = name
    base.select_set(False)
    return base


def apply_weighted_normals(obj: bpy.types.Object) -> bpy.types.Object:
    modifier = obj.modifiers.new("weighted_preview_normals", "WEIGHTED_NORMAL")
    modifier.keep_sharp = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    obj.select_set(False)
    return obj


def add_triangular_prism(
    name: str,
    width: float,
    length: float,
    height: float,
    location: tuple[float, float, float],
    material: MaterialSpec | None = OPENSCAD_YELLOW,
) -> bpy.types.Object:
    """Create the source Rib(): a right-triangle prism extruded across X."""

    half_width = width / 2.0
    verts = [
        (-half_width, 0.0, 0.0),
        (-half_width, length, 0.0),
        (-half_width, 0.0, height),
        (half_width, 0.0, 0.0),
        (half_width, length, 0.0),
        (half_width, 0.0, height),
    ]
    faces = [
        (0, 2, 1),
        (3, 4, 5),
        (0, 1, 4, 3),
        (0, 3, 5, 2),
        (1, 2, 5, 4),
    ]
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    if material is not None:
        assign_material(obj, material)
    return obj


def add_inset_hole_cutters(
    name: str,
    center: tuple[float, float, float],
    clearance_diameter: float,
    socket_diameter: float,
    socket_depth: float = 10.0,
) -> list[bpy.types.Object]:
    """Cutters matching SCAD InsetHole(): full through shaft plus top counterbore."""

    x, y, z = center
    return [
        add_cylinder(
            f"{name}_shaft",
            clearance_diameter,
            120.0,
            (x, y, z),
            vertices=72,
            material=None,
        ),
        add_cylinder(
            f"{name}_socket_counterbore",
            socket_diameter,
            socket_depth,
            (x, y, z + socket_depth / 2.0),
            vertices=72,
            material=None,
        ),
    ]


def add_scad_nut_hole_cutters(
    name: str,
    transform: Matrix,
    clearance_diameter: float,
    nut_diameter: float,
) -> list[bpy.types.Object]:
    """Cutters matching SCAD NutHole() after the caller's translate/rotate stack."""

    shaft = add_cylinder(
        f"{name}_screw_clearance",
        clearance_diameter,
        50.0,
        (0.0, 0.0, 26.0),
        vertices=48,
        material=None,
    )
    hex_trap = add_cylinder(
        f"{name}_hex_nut_trap",
        nut_diameter,
        6.0,
        (0.0, 0.0, 3.0),
        vertices=6,
        material=None,
        smooth=False,
    )
    return [transform_mesh_vertices(obj, transform) for obj in (shaft, hex_trap)]


def _recalculate_mesh_normals(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)


def add_scad_roundover_cutter(
    name: str,
    radius: float,
    transform: Matrix,
    length: float = 100.0,
    epsilon: float = 0.1,
    cylinder_fn: int = 20,
) -> bpy.types.Object:
    """Create the bounded Roundover(r) cube-minus-cylinder cutter from the SCAD source."""

    quarter_steps = max(2, cylinder_fn // 4)
    cross_section = [(epsilon + radius, epsilon + radius)]
    for index in range(quarter_steps + 1):
        theta = (math.pi / 2.0) * index / quarter_steps
        cross_section.append((epsilon + radius * math.cos(theta), epsilon + radius * math.sin(theta)))

    half_length = length / 2.0
    verts: list[tuple[float, float, float]] = [
        (x, y, -half_length) for x, y in cross_section
    ] + [
        (x, y, half_length) for x, y in cross_section
    ]

    count = len(cross_section)
    faces: list[tuple[int, ...]] = []
    for index in range(1, count - 1):
        faces.append((0, index + 1, index))
        faces.append((count, count + index, count + index + 1))
    for index in range(count):
        nxt = (index + 1) % count
        faces.append((index, nxt, count + nxt, count + index))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    transform_mesh_vertices(obj, transform)
    _recalculate_mesh_normals(obj)
    return obj


def transform_mesh_vertices(obj: bpy.types.Object, matrix: Matrix) -> bpy.types.Object:
    combined = matrix @ obj.matrix_world
    for vertex in obj.data.vertices:
        vertex.co = combined @ vertex.co
    obj.data.update()
    obj.matrix_world = Matrix.Identity(4)
    return obj


# ============================================================
# Procedural assembly
# ============================================================

"""Assembly logic for the Shapeoko belt clamp model."""






@dataclass(frozen=True)
class ClampParams:
    thick: float = 8.0
    rib_height: float = 6.0
    top_roundover_radius: float = 4.0
    side_roundover_radius: float = 4.0
    main_width: float = 16.0
    top_width: float = 30.0
    rib_width: float = 3.4
    hole_spacing: float = 20.0
    belt_width: float = 14.0
    belt_pitch: float = 2.032
    belt_thick: float = 1.5
    m3_clearance: float = 3.5
    m3_nut: float = 6.4
    m5_clearance: float = 5.5
    m5_socket: float = 9.4

    @property
    def tall_height(self) -> float:
        return self.thick + self.rib_height


def _cut_upright_roundovers(body: bpy.types.Object, params: ClampParams) -> None:
    r = params.top_roundover_radius
    cutters: list[bpy.types.Object] = [
        add_scad_roundover_cutter(
            "upright_top_corner_roundover_right",
            radius=r,
            transform=scad_transform_matrix(
                (params.top_width / 2.0 - r, 0.0, params.tall_height - r),
                (90.0, 0.0, 0.0),
            ),
        ),
        add_scad_roundover_cutter(
            "upright_top_corner_roundover_left",
            radius=r,
            transform=scad_transform_matrix(
                (-(params.top_width / 2.0 - r), 0.0, params.tall_height - r),
                (90.0, 0.0, 180.0),
            ),
        ),
    ]

    r = params.side_roundover_radius
    cutters.extend(
        [
            add_scad_roundover_cutter(
                "upright_front_mount_roundover_right",
                radius=r,
                transform=scad_transform_matrix(
                    (params.main_width / 2.0 - r, 26.0, 0.0),
                ),
            ),
            add_scad_roundover_cutter(
                "upright_front_mount_roundover_left",
                radius=r,
                transform=scad_transform_matrix(
                    (-(params.main_width / 2.0 - r), 26.0, 0.0),
                    (0.0, 0.0, 90.0),
                ),
            ),
        ]
    )

    boolean_difference(body, cutters)


def _cut_top_roundovers(body: bpy.types.Object, params: ClampParams) -> None:
    r = params.top_roundover_radius
    top_transform = scad_transform_matrix((0.0, -50.0, 0.0), (-90.0, 0.0, 0.0))
    cutters: list[bpy.types.Object] = [
        add_scad_roundover_cutter(
            "top_plate_front_corner_roundover_right",
            radius=r,
            transform=top_transform
            @ scad_transform_matrix(
                (params.top_width / 2.0 - r, 0.0, params.tall_height - r),
                (90.0, 0.0, 0.0),
            ),
        ),
        add_scad_roundover_cutter(
            "top_plate_front_corner_roundover_left",
            radius=r,
            transform=top_transform
            @ scad_transform_matrix(
                (-(params.top_width / 2.0 - r), 0.0, params.tall_height - r),
                (90.0, 0.0, 180.0),
            ),
        ),
    ]
    boolean_difference(body, cutters)


def build_upright(params: ClampParams) -> list[bpy.types.Object]:
    long_base = add_box(
        "upright_long_mounting_bar",
        (params.main_width, 60.0, params.thick),
        (0.0, 0.0, params.thick / 2.0),
    )
    rear_block = add_box(
        "upright_raised_rear_cross_block",
        (params.top_width, 10.0, params.tall_height),
        (0.0, -25.0, params.tall_height / 2.0),
    )
    ribs = [
        add_triangular_prism(
            "upright_side_rib_right",
            params.rib_width,
            40.0,
            params.rib_height,
            ((params.main_width - params.rib_width) / 2.0, -20.0, params.thick - 0.1),
        ),
        add_triangular_prism(
            "upright_side_rib_left",
            params.rib_width,
            40.0,
            params.rib_height,
            (-(params.main_width - params.rib_width) / 2.0, -20.0, params.thick - 0.1),
        ),
    ]

    body = boolean_union("upright_body_boolean_union", [long_base, rear_block, *ribs])
    _cut_upright_roundovers(body, params)

    cutters: list[bpy.types.Object] = []
    cutters.extend(
        add_inset_hole_cutters(
            "upright_m5_rear_mount",
            (0.0, 0.0, params.thick - 2.0),
            params.m5_clearance,
            params.m5_socket,
        )
    )
    cutters.extend(
        add_inset_hole_cutters(
            "upright_m5_front_mount",
            (0.0, 20.0, params.thick - 2.0),
            params.m5_clearance,
            params.m5_socket,
        )
    )
    for sign in (-1.0, 1.0):
        cutters.extend(
            add_scad_nut_hole_cutters(
                f"upright_m3_side_trap_{'left' if sign < 0 else 'right'}",
                scad_transform_matrix(
                    (sign * params.hole_spacing / 2.0, -16.0, params.tall_height / 2.0),
                    (90.0, -90.0, 0.0),
                ),
                params.m3_clearance,
                params.m3_nut,
            )
        )
    boolean_difference(body, cutters)
    apply_weighted_normals(body)
    return [body]


def build_top_clamp(params: ClampParams) -> list[bpy.types.Object]:
    plate = add_box(
        "top_clamp_plate_body",
        (params.top_width, params.tall_height, 10.0),
        (0.0, -43.0, 5.0),
    )
    _cut_top_roundovers(plate, params)

    belt_cutters: list[bpy.types.Object] = [
        make_box_cutter(
            "top_plate_belt_flat_channel",
            (params.belt_width, 30.0, params.belt_thick),
            (0.0, -50.0, 10.35),
        )
    ]
    tooth_count = int(30.0 / params.belt_pitch) + 1
    for index in range(tooth_count):
        belt_cutters.append(
            make_box_cutter(
                f"top_plate_belt_tooth_slot_{index:02d}",
                (params.belt_width, params.belt_pitch / 2.0, params.belt_thick + 0.05),
                (0.0, -50.0 + index * params.belt_pitch, 8.87),
            )
        )

    hole_cutters: list[bpy.types.Object] = []
    top_transform = scad_transform_matrix((0.0, -50.0, 0.0), (-90.0, 0.0, 0.0))
    for sign in (-1.0, 1.0):
        hole_cutters.extend(
            add_scad_nut_hole_cutters(
                f"top_plate_m3_nut_trap_{'left' if sign < 0 else 'right'}",
                top_transform
                @ scad_transform_matrix(
                    (sign * params.hole_spacing / 2.0, 20.0, params.tall_height / 2.0),
                    (90.0, -90.0, 0.0),
                ),
                params.m3_clearance,
                params.m3_nut,
            )
        )
    boolean_difference(plate, [*belt_cutters, *hole_cutters])
    apply_weighted_normals(plate)
    return [plate]

def build_model() -> list[bpy.types.Object]:
    params = ClampParams()
    objects = [*build_upright(params), *build_top_clamp(params)]
    for obj in objects:
        if obj.type == "MESH" and not obj.data.materials:
            assign_material(obj, OPENSCAD_YELLOW)
    bpy.context.view_layer.update()
    return objects


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""


sys.path.insert(0, str(Path(__file__).resolve().parent))



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

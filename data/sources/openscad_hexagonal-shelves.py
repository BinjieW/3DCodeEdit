from __future__ import annotations
from dataclasses import dataclass
from mathutils import Matrix, Vector
from mathutils.geometry import tessellate_polygon
from typing import Iterable
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the hexagonal shelf bracket model."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.45


BRACKET_YELLOW = MaterialSpec("openscad_preview_yellow", (0.94, 0.74, 0.03, 1.0), 0.48)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def get_material(spec: MaterialSpec) -> bpy.types.Material:
    material = bpy.data.materials.get(spec.name) or bpy.data.materials.new(spec.name)
    material.diffuse_color = spec.color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = spec.color
        bsdf.inputs["Roughness"].default_value = spec.roughness
    return material


def assign_material(obj: bpy.types.Object, spec: MaterialSpec) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(get_material(spec))
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
    if material:
        assign_material(obj, material)
    return obj


def add_cylinder(
    name: str,
    diameter: float,
    height: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
    material: MaterialSpec | None = None,
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
    if material:
        assign_material(obj, material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = abs(polygon.normal.z) < 0.5
    return obj


def add_polygon_prism(
    name: str,
    points: list[tuple[float, float]],
    height: float,
    z_center: float,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Build a vertical prism from a counter-clockwise 2D outline."""
    bottom_z = z_center - height / 2.0
    top_z = z_center + height / 2.0
    vertices = [(x, y, bottom_z) for x, y in points] + [(x, y, top_z) for x, y in points]
    count = len(points)
    triangles = tessellate_polygon([[Vector((x, y, 0.0)) for x, y in points]])
    faces = [tuple(reversed(triangle)) for triangle in triangles]
    faces.extend(tuple(index + count for index in triangle) for triangle in triangles)
    for index in range(count):
        faces.append((index, (index + 1) % count, count + (index + 1) % count, count + index))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material:
        assign_material(obj, material)
    return obj


def boolean_union(name: str, objects: Iterable[bpy.types.Object]) -> bpy.types.Object:
    objects = [obj for obj in objects if obj is not None]
    if not objects:
        raise ValueError("boolean_union requires at least one object")
    active = objects[0]
    bpy.context.view_layer.objects.active = active
    active.select_set(True)
    for obj in objects[1:]:
        modifier = active.modifiers.new(f"union_{obj.name}", "BOOLEAN")
        modifier.operation = "UNION"
        modifier.object = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(obj, do_unlink=True)
    active.name = name
    active.select_set(False)
    return active


def subtract_cutters(target: bpy.types.Object, cutters: Iterable[bpy.types.Object]) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    for cutter in cutters:
        modifier = target.modifiers.new(f"cut_{cutter.name}", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
    target.select_set(False)
    return target


def rotate_object_z(obj: bpy.types.Object, degrees: float) -> bpy.types.Object:
    """Bake a SCAD-style rotate([0, 0, degrees]) around world origin."""
    obj.data.transform(obj.matrix_world)
    obj.matrix_world = Matrix.Identity(4)
    obj.data.transform(Matrix.Rotation(math.radians(degrees), 4, "Z"))
    obj.data.update()
    return obj


def add_rounded_rectangle_prism(
    name: str,
    width: float,
    length: float,
    height: float,
    radius: float,
    location: tuple[float, float, float],
    material: MaterialSpec,
    vertices: int = 24,
) -> bpy.types.Object:
    """SCAD roundedBox(..., sidesonly=true) equivalent for vertical extrusions."""
    parts = [
        add_box(f"{name}_center_x", (max(width - 2.0 * radius, 0.01), length, height), location, material),
        add_box(f"{name}_center_y", (width, max(length - 2.0 * radius, 0.01), height), location, material),
    ]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            parts.append(
                add_cylinder(
                    f"{name}_corner_{sx:g}_{sy:g}",
                    diameter=2.0 * radius,
                    height=height,
                    location=(
                        location[0] + sx * (width / 2.0 - radius),
                        location[1] + sy * (length / 2.0 - radius),
                        location[2],
                    ),
                    vertices=vertices,
                    material=material,
                )
            )
    return boolean_union(name, parts)


def add_wall(
    name: str,
    x_center: float,
    y_start: float,
    wall_thick: float,
    wall_length: float,
    depth: float,
    corner_radius: float,
    screw_diameter: float,
    material: MaterialSpec,
) -> bpy.types.Object:
    """One bracket side wall with the SCAD screw bore cut through its width."""
    body = add_rounded_rectangle_prism(
        f"{name}_rounded_body",
        wall_thick,
        wall_length,
        depth,
        corner_radius,
        (x_center, y_start + wall_length / 2.0, depth / 2.0),
        material,
        vertices=20,
    )
    root_block = add_box(
        f"{name}_square_root",
        (wall_thick, 2.0, depth),
        (x_center, y_start + 1.0, depth / 2.0),
        material,
    )
    wall = boolean_union(name, [body, root_block])
    cutter = add_cylinder(
        f"{name}_screw_cutter",
        screw_diameter,
        wall_thick + 1.4,
        (x_center, y_start + wall_length / 2.0, depth / 2.0),
        rotation=(0.0, math.pi / 2.0, 0.0),
        vertices=32,
    )
    return subtract_cutters(wall, [cutter])


def add_open_channel_pair(
    name: str,
    y_start: float,
    shelf_gap: float,
    wall_thick: float,
    wall_length: float,
    depth: float,
    corner_radius: float,
    screw_diameter: float,
    material: MaterialSpec,
    rotation_degrees: float,
) -> bpy.types.Object:
    wall_offset = shelf_gap / 2.0 + wall_thick / 2.0
    left = add_wall(
        f"{name}_left_wall",
        -wall_offset,
        y_start,
        wall_thick,
        wall_length,
        depth,
        corner_radius,
        screw_diameter,
        material,
    )
    right = add_wall(
        f"{name}_right_wall",
        wall_offset,
        y_start,
        wall_thick,
        wall_length,
        depth,
        corner_radius,
        screw_diameter,
        material,
    )
    channel = boolean_union(name, [left, right])
    return rotate_object_z(channel, rotation_degrees)


def add_hub(
    name: str,
    bracket_thick: float,
    tri_height: float,
    centroid: float,
    depth: float,
    center_hole_diameter: float,
    material: MaterialSpec,
) -> bpy.types.Object:
    hub = add_polygon_prism(
        name,
        [(-bracket_thick / 2.0, centroid), (0.0, centroid - tri_height), (bracket_thick / 2.0, centroid)],
        depth,
        depth / 2.0,
        material,
    )
    cutter = add_cylinder(
        f"{name}_center_hole_cutter",
        center_hole_diameter,
        depth + 1.4,
        (0.0, 0.0, depth / 2.0),
        vertices=32,
    )
    return subtract_cutters(hub, [cutter])


def add_hanger_tab(
    name: str,
    center: tuple[float, float],
    bracket_thick: float,
    thickness: float,
    screw_diameter: float,
    material: MaterialSpec,
) -> bpy.types.Object:
    body = add_cylinder(
        f"{name}_rounded_eye",
        bracket_thick,
        thickness,
        (center[0], center[1], thickness / 2.0),
        vertices=32,
        material=material,
    )
    neck = add_box(
        f"{name}_neck_fill",
        (bracket_thick, bracket_thick / 2.0, thickness),
        (center[0], center[1] + bracket_thick / 4.0, thickness / 2.0),
        material,
    )
    tab = boolean_union(name, [body, neck])
    cutter = add_cylinder(
        f"{name}_through_hole_cutter",
        screw_diameter,
        thickness + 1.0,
        (center[0], center[1], thickness / 2.0),
        vertices=40,
    )
    return subtract_cutters(tab, [cutter])


# ============================================================
# Procedural assembly
# ============================================================

"""Assembly logic for the two-way hexagonal shelf tensioner bracket."""





SHELF_THICK = 5.0
WALL_THICK = 3.0
WALL_LENGTH = 25.0
DEPTH = 25.0
SCREW_DIA = 4.75
BRACKET_THICK = SHELF_THICK + WALL_THICK * 2.0
TRI_HEIGHT = math.sqrt(BRACKET_THICK**2 - (BRACKET_THICK / 2.0) ** 2)
CENTROID = math.tan(math.radians(30.0)) * (BRACKET_THICK / 2.0)
CORNER_RAD = 1.0
FLOOR_THICK = 3.0


def build_model() -> list:
    """Build the active SCAD call: twowaybracket_tensioner()."""
    objects = [
        add_hub(
            "triangular_center_hub",
            BRACKET_THICK,
            TRI_HEIGHT,
            CENTROID,
            DEPTH,
            center_hole_diameter=SHELF_THICK / 2.0,
            material=BRACKET_YELLOW,
        )
    ]

    for angle in (120.0, 240.0):
        objects.append(
            add_open_channel_pair(
                f"open_shelf_channel_{int(angle)}deg",
                y_start=CENTROID - 0.1,
                shelf_gap=SHELF_THICK,
                wall_thick=WALL_THICK,
                wall_length=WALL_LENGTH,
                depth=DEPTH,
                corner_radius=CORNER_RAD,
                screw_diameter=SCREW_DIA,
                material=BRACKET_YELLOW,
                rotation_degrees=angle,
            )
        )

    # In the source, hanger() is translated by [0, centroid, 0] and then rotated 180 degrees.
    hanger_center_y = -(2.0 * CENTROID - 0.1 + BRACKET_THICK / 2.0)
    objects.append(
        add_hanger_tab(
            "front_tensioner_hanger_tab",
            center=(0.0, hanger_center_y),
            bracket_thick=BRACKET_THICK,
            thickness=FLOOR_THICK,
            screw_diameter=SCREW_DIA,
            material=BRACKET_YELLOW,
        )
    )

    return [boolean_union("two_way_hex_shelf_tensioner", objects)]


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the customizable column model."""





@dataclass(frozen=True)
class ColumnDimensions:
    """Default OpenSCAD parameters resolved to concrete dimensions."""

    height: float = 20.0
    width: float = 4.0
    flute_count: int = 24
    shaft_segments_per_flute: int = 10

    @property
    def radius(self) -> float:
        return self.width / 2.0

    @property
    def base_size(self) -> float:
        return self.width * 1.25

    @property
    def base_height(self) -> float:
        return self.width * 0.25

    @property
    def capital_height(self) -> float:
        return self.width * 0.25

    @property
    def flute_radius(self) -> float:
        return math.pi * self.width / 72.0


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.55,
    emission_strength: float = 0.22,
) -> bpy.types.Material:
    existing = bpy.data.materials.get(name)
    if existing:
        return existing
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = roughness
        if "Emission Color" in principled.inputs:
            principled.inputs["Emission Color"].default_value = color
        elif "Emission" in principled.inputs:
            principled.inputs["Emission"].default_value = color
        if "Emission Strength" in principled.inputs:
            principled.inputs["Emission Strength"].default_value = emission_strength
        if "Specular" in principled.inputs:
            principled.inputs["Specular"].default_value = 0.25
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
    return obj


def recalculate_normals(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return obj


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
    if material:
        assign_material(obj, material)
    return obj


def _fluted_radius(theta: float, dimensions: ColumnDimensions) -> float:
    """Return the shaft boundary after subtracting the nearest roman flute cutter."""
    main_radius = dimensions.radius
    cutter_radius = dimensions.flute_radius
    pitch = math.tau / dimensions.flute_count
    nearest_index = round(theta / pitch)
    center_angle = nearest_index * pitch
    delta = (theta - center_angle + math.pi) % math.tau - math.pi

    sin_delta = math.sin(delta)
    if abs(main_radius * sin_delta) >= cutter_radius:
        return main_radius

    # Ray/circle intersection for a cutter centered on the shaft circumference.
    inner = main_radius * math.cos(delta) - math.sqrt(
        max(cutter_radius * cutter_radius - main_radius * main_radius * sin_delta * sin_delta, 0.0)
    )
    return min(main_radius, inner)


def add_fluted_shaft(
    name: str,
    dimensions: ColumnDimensions,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Create a mesh shaft matching the SCAD cylinder minus 24 vertical roman flutes."""
    samples = dimensions.flute_count * dimensions.shaft_segments_per_flute
    verts: list[tuple[float, float, float]] = []
    for z in (0.0, dimensions.height):
        for i in range(samples):
            theta = math.tau * i / samples
            radius = _fluted_radius(theta, dimensions)
            verts.append((radius * math.sin(theta), radius * math.cos(theta), z))

    bottom_center = len(verts)
    top_center = bottom_center + 1
    verts.extend([(0.0, 0.0, 0.0), (0.0, 0.0, dimensions.height)])

    faces: list[tuple[int, ...]] = []
    for i in range(samples):
        j = (i + 1) % samples
        faces.append((i, j, samples + j, samples + i))
        faces.append((bottom_center, j, i))
        faces.append((top_center, samples + i, samples + j))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material:
        assign_material(obj, material)
    return recalculate_normals(obj)


def set_origin_to_geometry(objects: Iterable[bpy.types.Object]) -> None:
    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.select_set(False)


# ============================================================
# Procedural assembly
# ============================================================

"""High-level assembly for the default customizable column source model."""






def _materials() -> dict[str, bpy.types.Material]:
    return {
        "gold": make_material("source_default_gold", (0.98, 0.73, 0.05, 1.0)),
        "shaft": make_material("fluted_shaft_gold_olive", (0.74, 0.62, 0.16, 1.0)),
    }


def build_base(dimensions: ColumnDimensions, material: bpy.types.Material) -> bpy.types.Object:
    return add_box(
        name="square_bottom_plinth",
        size=(dimensions.base_size, dimensions.base_size, dimensions.base_height),
        location=(0.0, 0.0, dimensions.base_height / 2.0),
        material=material,
    )


def build_ionic_capital(dimensions: ColumnDimensions, material: bpy.types.Material) -> list[bpy.types.Object]:
    cap_height = dimensions.capital_height
    slab_width = dimensions.width + 2.0 * cap_height
    slab_depth = dimensions.base_size
    z_top_translate = dimensions.height - 0.001
    scroll_y = 0.0

    slab = add_box(
        name="ionic_rectangular_abacus_slab",
        size=(slab_width, slab_depth, cap_height),
        location=(0.0, 0.0, z_top_translate),
        material=material,
    )

    scroll_x = dimensions.width * 0.5 + cap_height
    scroll_z = z_top_translate - 0.5 * cap_height
    scrolls = [
        add_cylinder(
            name="ionic_left_horizontal_scroll_cylinder",
            diameter=cap_height * 2.0,
            height=slab_depth,
            location=(-scroll_x, scroll_y, scroll_z),
            rotation=(math.radians(90.0), 0.0, 0.0),
            vertices=12,
            material=material,
        ),
        add_cylinder(
            name="ionic_right_horizontal_scroll_cylinder",
            diameter=cap_height * 2.0,
            height=slab_depth,
            location=(scroll_x, scroll_y, scroll_z),
            rotation=(math.radians(90.0), 0.0, 0.0),
            vertices=12,
            material=material,
        ),
    ]
    return [slab, *scrolls]


def build_model() -> list[bpy.types.Object]:
    dimensions = ColumnDimensions()
    materials = _materials()

    objects: list[bpy.types.Object] = [
        build_base(dimensions, materials["gold"]),
        add_fluted_shaft("roman_fluted_cylindrical_shaft", dimensions, materials["shaft"]),
        *build_ionic_capital(dimensions, materials["gold"]),
    ]
    set_origin_to_geometry(objects)
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

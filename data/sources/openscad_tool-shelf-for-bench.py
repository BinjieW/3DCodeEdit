from __future__ import annotations
from dataclasses import dataclass
from mathutils import Matrix, Vector
from typing import Iterable
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the bench tool shelf model."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.55


OPENSCAD_YELLOW = MaterialSpec("openscad_preview_yellow", (1.0, 0.78, 0.02, 1.0))


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
    """Create an axis-aligned rectangular prism using OpenSCAD cube dimensions."""

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
    shade_smooth(obj)
    return obj


def boolean_difference(
    name: str,
    base: bpy.types.Object,
    cutters: Iterable[bpy.types.Object],
) -> bpy.types.Object:
    """Apply boolean subtraction and remove the temporary cutter objects."""

    bpy.context.view_layer.objects.active = base
    base.select_set(True)
    for cutter in cutters:
        modifier = base.modifiers.new(f"cut_{cutter.name}", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
    base.name = name
    base.select_set(False)
    return base


def transform_mesh_vertices(obj: bpy.types.Object, matrix: Matrix) -> bpy.types.Object:
    """Bake a transform into mesh coordinates while leaving object transforms clean."""

    combined = matrix @ obj.matrix_world
    for vertex in obj.data.vertices:
        vertex.co = combined @ vertex.co
    obj.data.update()
    obj.matrix_world = Matrix.Identity(4)
    return obj


def source_to_visible_matrix(
    carrier_width: float,
    rear_height: float,
    total_depth: float,
) -> Matrix:
    """Return the final translate/rotate transform used by the source SCAD.

    The SCAD finishes with:
    translate([-carrier_width/2, -rear_height/2, total_depth]) rotate([-90,0,0])
    union() {...}
    """

    translation = Matrix.Translation(Vector((-carrier_width / 2.0, -rear_height / 2.0, total_depth)))
    rotation = Matrix.Rotation(math.radians(-90.0), 4, "X")
    return translation @ rotation


def add_scad_box(
    name: str,
    corner: tuple[float, float, float],
    size: tuple[float, float, float],
    transform: Matrix,
    material: MaterialSpec | None = OPENSCAD_YELLOW,
) -> bpy.types.Object:
    """Create a cube([x,y,z]) from an OpenSCAD lower-left-bottom corner."""

    center = (corner[0] + size[0] / 2.0, corner[1] + size[1] / 2.0, corner[2] + size[2] / 2.0)
    obj = add_box(name=name, size=size, location=center, material=material)
    return transform_mesh_vertices(obj, transform)


def add_shelf_plate_with_holes(
    name: str,
    corner: tuple[float, float, float],
    carrier_width: float,
    shelf_depth: float,
    wall_thickness: float,
    hole_diameter: float,
    hole_count: int,
    hole_spacing: float,
    transform: Matrix,
    material: MaterialSpec | None = OPENSCAD_YELLOW,
) -> bpy.types.Object:
    """Create one perforated shelf carrier from the source `carrier()` module."""

    plate = add_box(
        name=f"{name}_solid_plate",
        size=(carrier_width, shelf_depth, wall_thickness),
        location=(
            corner[0] + carrier_width / 2.0,
            corner[1] + shelf_depth / 2.0,
            corner[2] + wall_thickness / 2.0,
        ),
        material=material,
    )
    cutters: list[bpy.types.Object] = []
    for index in range(hole_count):
        hole_x = corner[0] + hole_spacing / 2.0 + index * hole_spacing
        cutters.append(
            add_cylinder(
                name=f"{name}_hole_cutter_{index + 1:02d}",
                diameter=hole_diameter,
                height=wall_thickness * 5.0,
                location=(hole_x, corner[1] + shelf_depth / 2.0, corner[2] - 1.0 + wall_thickness * 2.5),
                vertices=50,
                material=None,
            )
        )

    boolean_difference(name, plate, cutters)
    return transform_mesh_vertices(plate, transform)


# ============================================================
# Procedural assembly
# ============================================================

"""Procedural assembly for the parametric bench tool shelf."""






@dataclass(frozen=True)
class ShelfParameters:
    number_of_holes: int = 12
    spacing_of_holes: float = 15.0
    rear_hole: float = 9.0
    mid_hole: float = 8.0
    front_hole: float = 8.0
    rear_height: float = 100.0
    mid_height: float = 55.0
    front_height: float = 40.0
    shelf_depth_rear: float = 18.0
    shelf_depth_mid: float = 18.0
    shelf_depth_front: float = 25.0
    wall_thickness: float = 3.0

    @property
    def carrier_width(self) -> float:
        return self.number_of_holes * self.spacing_of_holes

    @property
    def rear_slat_height(self) -> float:
        return (self.rear_height + self.wall_thickness) / 7.0

    @property
    def middle_wall_position(self) -> float:
        return (self.number_of_holes // 2) * self.spacing_of_holes + self.wall_thickness / 2.0

    @property
    def total_depth(self) -> float:
        return self.shelf_depth_rear + self.shelf_depth_mid + self.shelf_depth_front


@dataclass(frozen=True)
class ShelfLevel:
    name: str
    y_start: float
    height: float
    depth: float
    hole_diameter: float


def shelf_levels(params: ShelfParameters) -> tuple[ShelfLevel, ShelfLevel, ShelfLevel]:
    return (
        ShelfLevel(
            name="rear",
            y_start=params.shelf_depth_front + params.shelf_depth_mid,
            height=params.rear_height,
            depth=params.shelf_depth_rear,
            hole_diameter=params.rear_hole,
        ),
        ShelfLevel(
            name="mid",
            y_start=params.shelf_depth_front,
            height=params.mid_height,
            depth=params.shelf_depth_mid,
            hole_diameter=params.mid_hole,
        ),
        ShelfLevel(
            name="front",
            y_start=0.0,
            height=params.front_height,
            depth=params.shelf_depth_front,
            hole_diameter=params.front_hole,
        ),
    )


def add_perforated_shelf(level: ShelfLevel, params: ShelfParameters, transform) -> bpy.types.Object:
    return add_shelf_plate_with_holes(
        name=f"{level.name}_perforated_shelf",
        corner=(params.wall_thickness, level.y_start, level.height),
        carrier_width=params.carrier_width,
        shelf_depth=level.depth,
        wall_thickness=params.wall_thickness,
        hole_diameter=level.hole_diameter,
        hole_count=params.number_of_holes,
        hole_spacing=params.spacing_of_holes,
        transform=transform,
        material=OPENSCAD_YELLOW,
    )


def add_side_walls(level: ShelfLevel, params: ShelfParameters, transform) -> list[bpy.types.Object]:
    wall_depth = level.depth + (params.wall_thickness if level.name == "rear" else 0.0)
    wall_height = level.height + params.wall_thickness
    objects: list[bpy.types.Object] = []
    for side_name, x in (("left", 0.0), ("right", params.carrier_width + params.wall_thickness)):
        objects.append(
            add_scad_box(
                name=f"{level.name}_{side_name}_side_wall",
                corner=(x, level.y_start, 0.0),
                size=(params.wall_thickness, wall_depth, wall_height),
                transform=transform,
                material=OPENSCAD_YELLOW,
            )
        )
    return objects


def add_center_supports(params: ShelfParameters, transform) -> list[bpy.types.Object]:
    if params.carrier_width <= 100.0:
        return []

    objects: list[bpy.types.Object] = []
    for level in shelf_levels(params):
        wall_depth = level.depth + (params.wall_thickness if level.name == "rear" else 0.0)
        objects.append(
            add_scad_box(
                name=f"{level.name}_center_support_wall",
                corner=(params.middle_wall_position, level.y_start, 0.0),
                size=(params.wall_thickness, wall_depth, level.height + params.wall_thickness),
                transform=transform,
                material=OPENSCAD_YELLOW,
            )
        )
    return objects


def add_rear_slats(params: ShelfParameters, transform) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    for source_i in range(0, 7, 2):
        objects.append(
            add_scad_box(
                name=f"rear_back_slat_{source_i:02d}",
                corner=(0.0, params.total_depth, source_i * params.rear_slat_height),
                size=(params.carrier_width + params.wall_thickness, params.wall_thickness, params.rear_slat_height),
                transform=transform,
                material=OPENSCAD_YELLOW,
            )
        )
    return objects


def build_tool_shelf(params: ShelfParameters = ShelfParameters()) -> list[bpy.types.Object]:
    transform = source_to_visible_matrix(
        carrier_width=params.carrier_width,
        rear_height=params.rear_height,
        total_depth=params.total_depth,
    )

    objects: list[bpy.types.Object] = []
    for level in shelf_levels(params):
        objects.append(add_perforated_shelf(level, params, transform))
        objects.extend(add_side_walls(level, params, transform))

    objects.extend(add_center_supports(params, transform))
    objects.extend(add_rear_slats(params, transform))
    return objects


def build_model() -> list[bpy.types.Object]:
    return build_tool_shelf()


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

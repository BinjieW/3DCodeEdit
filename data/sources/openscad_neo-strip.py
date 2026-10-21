from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from mathutils import Vector
import bpy


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the NeoPixel strip channel model."""





def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.55,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = color
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 0.08
    return material


def assign_material(
    obj: bpy.types.Object,
    material: bpy.types.Material | None,
) -> bpy.types.Object:
    if material is not None:
        obj.data.materials.clear()
        obj.data.materials.append(material)
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
    return assign_material(obj, material)


def add_hexagonal_prism(
    name: str,
    radius: float,
    height: float,
    location: tuple[float, float, float],
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Create an OpenSCAD-style six-sided vertical prism."""
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=6,
        radius=radius,
        depth=height,
        location=location,
        rotation=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.object
    obj.name = name
    return assign_material(obj, material)


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
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        if keep_cutters:
            cutter.hide_set(True)
            cutter.hide_render = True
        else:
            bpy.data.objects.remove(cutter, do_unlink=True)
    target.select_set(False)
    return target


def shade_flat(obj: bpy.types.Object) -> bpy.types.Object:
    for polygon in obj.data.polygons:
        polygon.use_smooth = False
    return obj


def center_objects_on_origin(objects: Iterable[bpy.types.Object]) -> None:
    mesh_objects = [obj for obj in objects if obj.type == "MESH"]
    if not mesh_objects:
        return

    bpy.context.view_layer.update()
    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in mesh_objects
        for corner in obj.bound_box
    ]
    min_corner = Vector(
        (
            min(corner.x for corner in corners),
            min(corner.y for corner in corners),
            min(corner.z for corner in corners),
        )
    )
    max_corner = Vector(
        (
            max(corner.x for corner in corners),
            max(corner.y for corner in corners),
            max(corner.z for corner in corners),
        )
    )
    center = (min_corner + max_corner) / 2.0
    for obj in mesh_objects:
        obj.location -= center


# ============================================================
# Procedural assembly
# ============================================================

"""Assembly logic for the NeoPixel strip channel model."""






@dataclass(frozen=True)
class NeoStripParams:
    pixel_count: int = 8
    include_interior_hexes: bool = False
    include_start_half_slot: bool = False
    include_end_half_slot: bool = False
    hex_radius: float = 5.0
    pixel_spacing: float = 16.2
    strip_channel_height: float = 4.1
    strip_channel_width: float = 12.3
    body_width: float = 20.0
    hex_plane_height: float = 6.0
    body_height: float = 10.0
    cover_height: float = 1.0
    divider_thickness: float = 1.0
    slot_grow: float = 1.2

    @property
    def end_margin(self) -> float:
        return self.pixel_spacing / 2.0

    @property
    def length(self) -> float:
        return self.end_margin + (self.pixel_count - 1) * self.pixel_spacing + self.end_margin

    @property
    def divider_slot_height(self) -> float:
        return self.hex_plane_height - self.strip_channel_height

    @property
    def cover_y_offset(self) -> float:
        return 2.0 * self.body_width


def create_materials() -> dict[str, bpy.types.Material]:
    return {
        "body": make_material("openscad_default_yellow", (0.95, 0.78, 0.04, 1.0)),
        "red_divider": make_material("source_red_divider", (0.85, 0.05, 0.03, 1.0)),
    }


def pixel_centers(params: NeoStripParams, y_offset: float = 0.0) -> list[tuple[float, float]]:
    return [
        (params.end_margin + n * params.pixel_spacing, y_offset + params.body_width / 2.0)
        for n in range(params.pixel_count)
    ]


def divider_centers(params: NeoStripParams, y_offset: float = 0.0) -> list[tuple[float, float]]:
    start_n = -1 if params.include_start_half_slot else 0
    end_n = params.pixel_count - 1 if params.include_end_half_slot else params.pixel_count - 2
    return [
        (
            params.end_margin + n * params.pixel_spacing + params.pixel_spacing / 2.0,
            y_offset + params.body_width / 2.0,
        )
        for n in range(start_n, end_n + 1)
    ]


def build_strip_channel(
    params: NeoStripParams,
    material: bpy.types.Material,
) -> bpy.types.Object:
    body = add_box(
        name="main_strip_channel_body",
        size=(params.length, params.body_width, params.body_height),
        location=(params.length / 2.0, params.body_width / 2.0, params.body_height / 2.0),
        material=material,
    )

    channel_void = add_box(
        name="neopixel_strip_rectangular_channel_void",
        size=(params.length + 0.2, params.strip_channel_width, params.strip_channel_height + 0.05),
        location=(
            params.length / 2.0,
            params.body_width / 2.0,
            params.strip_channel_height / 2.0,
        ),
    )

    cutters: list[bpy.types.Object] = [channel_void]

    if params.include_interior_hexes:
        for index, (x, y) in enumerate(pixel_centers(params)):
            cutters.append(
                add_hexagonal_prism(
                    name=f"optional_body_hex_void_{index}",
                    radius=params.hex_radius,
                    height=params.divider_slot_height + 0.2,
                    location=(
                        x,
                        y,
                        params.strip_channel_height + params.divider_slot_height / 2.0,
                    ),
                )
            )

    for index, (x, y) in enumerate(divider_centers(params)):
        cutters.append(
            add_box(
                name=f"divider_slot_void_{index}",
                size=(
                    params.divider_thickness * params.slot_grow,
                    (2.0 * params.hex_radius) * params.slot_grow,
                    4.0 * params.divider_slot_height,
                ),
                location=(x, y, params.body_height),
            )
        )

    boolean_difference(body, cutters)
    return shade_flat(body)


def build_cover_plate(
    params: NeoStripParams,
    body_material: bpy.types.Material,
    divider_material: bpy.types.Material,
) -> list[bpy.types.Object]:
    y_offset = params.cover_y_offset
    cover = add_box(
        name="separate_cover_plate_with_hex_holes",
        size=(params.length, params.body_width, params.cover_height),
        location=(params.length / 2.0, y_offset + params.body_width / 2.0, params.cover_height / 2.0),
        material=body_material,
    )

    hex_cutters = [
        add_hexagonal_prism(
            name=f"cover_hexagonal_pixel_void_{index}",
            radius=params.hex_radius,
            height=params.cover_height + 0.2,
            location=(x, y, params.cover_height / 2.0),
        )
        for index, (x, y) in enumerate(pixel_centers(params, y_offset))
    ]
    boolean_difference(cover, hex_cutters)
    shade_flat(cover)

    objects: list[bpy.types.Object] = [cover]
    divider_height = 2.0 * params.divider_slot_height
    for index, (x, y) in enumerate(divider_centers(params, y_offset)):
        divider = add_box(
            name=f"raised_cover_light_blocking_divider_{index}",
            size=(params.divider_thickness, 2.0 * params.hex_radius, divider_height),
            location=(x, y, params.cover_height + params.divider_slot_height / 2.0),
            material=divider_material,
        )
        objects.append(shade_flat(divider))

    return objects


def build_model() -> list[bpy.types.Object]:
    """Build the two-part model from the parameters in the source SCAD file."""
    params = NeoStripParams()
    materials = create_materials()

    objects: list[bpy.types.Object] = [
        build_strip_channel(params, materials["body"]),
        *build_cover_plate(params, materials["body"], materials["red_divider"]),
    ]
    center_objects_on_origin(objects)
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

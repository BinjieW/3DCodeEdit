from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Low-level builders for the curved corner bracket model."""





Point2D = tuple[float, float]
Axis = Literal["X", "Y"]


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.5


BRACKET_YELLOW = MaterialSpec("openscad_preview_yellow", (0.96, 0.78, 0.04, 1.0))


@dataclass(frozen=True)
class BracketSpec:
    source_length: float = 70.0
    width: float = 10.0
    thickness: float = 2.5
    hole_count: int = 3
    hole_diameter: float = 2.5
    include_outer_holes: bool = True
    circular_segments: int = 35

    @property
    def bracket_length(self) -> float:
        return self.source_length - self.thickness

    @property
    def arc_center(self) -> Point2D:
        center = self.bracket_length - self.thickness
        return (center, center)

    @property
    def inner_arc_radius(self) -> float:
        return self.bracket_length - self.thickness

    @property
    def outer_arc_radius(self) -> float:
        return self.bracket_length

    @property
    def small_hole_radius(self) -> float:
        return self.hole_diameter / 2.0

    @property
    def large_hole_radius(self) -> float:
        return self.hole_diameter / 1.4

    @property
    def hole_spacing(self) -> float:
        return self.source_length / (self.hole_count + 1)

    @property
    def hole_offsets(self) -> list[float]:
        return [
            self.hole_spacing - self.thickness + index * self.hole_spacing
            for index in range(self.hole_count)
        ]


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
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.25
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.25
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = spec.color
        elif "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = spec.color
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 0.25
    return material


def assign_material(obj: bpy.types.Object, spec: MaterialSpec = BRACKET_YELLOW) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(get_material(spec))
    return obj


def recalculate_normals(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def add_box_from_bounds(
    name: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    material: MaterialSpec = BRACKET_YELLOW,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(
        location=((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = ((xmax - xmin) / 2.0, (ymax - ymin) / 2.0, (zmax - zmin) / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign_material(obj, material)
    return obj


def arc_points(center: Point2D, radius: float, start: float, end: float, segments: int) -> list[Point2D]:
    return [
        (
            center[0] + math.cos(start + (end - start) * index / segments) * radius,
            center[1] + math.sin(start + (end - start) * index / segments) * radius,
        )
        for index in range(segments + 1)
    ]


def add_annular_arc_prism(
    name: str,
    center: Point2D,
    inner_radius: float,
    outer_radius: float,
    height: float,
    start_angle: float,
    end_angle: float,
    segments: int,
    material: MaterialSpec = BRACKET_YELLOW,
) -> bpy.types.Object:
    """Build the quarter hoop from the source annulus, extruded along Z."""
    outer = arc_points(center, outer_radius, start_angle, end_angle, segments)
    inner = arc_points(center, inner_radius, start_angle, end_angle, segments)
    outline = outer + list(reversed(inner))
    count = len(outline)

    vertices = [(x, y, 0.0) for x, y in outline] + [(x, y, height) for x, y in outline]
    bottom = tuple(reversed(range(count)))
    top = tuple(range(count, count * 2))
    faces: list[tuple[int, ...]] = [bottom, top]
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
    for polygon in obj.data.polygons:
        polygon.use_smooth = False
    return recalculate_normals(obj)


def add_cylinder_cutter(
    name: str,
    axis: Axis,
    center: tuple[float, float, float],
    radius: float,
    depth: float,
    vertices: int,
) -> bpy.types.Object:
    rotation = (0.0, math.pi / 2.0, 0.0) if axis == "X" else (math.pi / 2.0, 0.0, 0.0)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        end_fill_type="NGON",
        location=center,
        rotation=rotation,
    )
    cutter = bpy.context.object
    cutter.name = name
    return cutter


def apply_difference(obj: bpy.types.Object, cutter: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    modifier = obj.modifiers.new(f"subtract_{cutter.name}", "BOOLEAN")
    modifier.operation = "DIFFERENCE"
    modifier.object = cutter
    if hasattr(modifier, "solver"):
        modifier.solver = "EXACT"
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    obj.select_set(False)
    assign_material(obj)
    return recalculate_normals(obj)


def apply_cutters(objects: Iterable[bpy.types.Object], cutters: Iterable[bpy.types.Object]) -> list[bpy.types.Object]:
    result = list(objects)
    for cutter in cutters:
        for obj in result:
            apply_difference(obj, cutter)
        bpy.data.objects.remove(cutter, do_unlink=True)
    return result


def straight_body_parts(spec: BracketSpec) -> list[bpy.types.Object]:
    t = spec.thickness
    length = spec.bracket_length
    width = spec.width
    return [
        add_box_from_bounds("front_straight_leg", 0.0, length, -t, 0.0, 0.0, width),
        add_box_from_bounds("left_straight_leg", -t, 0.0, 0.0, length, 0.0, width),
        add_box_from_bounds("source_corner_fill_block", -t, 0.0, -t, -t + 1.0, 0.0, width),
    ]


def curved_body_part(spec: BracketSpec) -> bpy.types.Object:
    # OpenSCAD $fn=35 gives about nine facets over this quarter turn.
    arc_segments = max(6, math.ceil(spec.circular_segments / 4))
    return add_annular_arc_prism(
        name="curved_outer_hoop",
        center=spec.arc_center,
        inner_radius=spec.inner_arc_radius,
        outer_radius=spec.outer_arc_radius,
        height=spec.width,
        start_angle=math.pi,
        end_angle=1.5 * math.pi,
        segments=arc_segments,
    )


def small_hole_cutters(spec: BracketSpec) -> list[bpy.types.Object]:
    cutters: list[bpy.types.Object] = []
    t = spec.thickness
    radius = spec.small_hole_radius
    z = spec.width / 2.0
    for index, offset in enumerate(spec.hole_offsets, start=1):
        cutters.append(
            add_cylinder_cutter(
                name=f"front_small_hole_{index}",
                axis="Y",
                center=(offset, -(t + 1.0) / 2.0, z),
                radius=radius,
                depth=t + 1.0,
                vertices=spec.circular_segments,
            )
        )
        cutters.append(
            add_cylinder_cutter(
                name=f"left_small_hole_{index}",
                axis="X",
                center=(-t / 2.0, offset, z),
                radius=radius,
                depth=t,
                vertices=spec.circular_segments,
            )
        )
    return cutters


def large_hole_cutters(spec: BracketSpec) -> list[bpy.types.Object]:
    if not spec.include_outer_holes:
        return []
    cutters: list[bpy.types.Object] = []
    radius = spec.large_hole_radius
    z = spec.width / 2.0
    span = spec.source_length
    for index, offset in enumerate(spec.hole_offsets, start=1):
        cutters.append(
            add_cylinder_cutter(
                name=f"curved_y_axis_large_hole_{index}",
                axis="Y",
                center=(offset, span / 2.0, z),
                radius=radius,
                depth=span,
                vertices=spec.circular_segments,
            )
        )
        cutters.append(
            add_cylinder_cutter(
                name=f"curved_x_axis_large_hole_{index}",
                axis="X",
                center=(span / 2.0, offset, z),
                radius=radius,
                depth=span,
                vertices=spec.circular_segments,
            )
        )
    return cutters


def add_curved_corner_bracket(spec: BracketSpec) -> list[bpy.types.Object]:
    parts = straight_body_parts(spec)
    parts.append(curved_body_part(spec))
    cutters = small_hole_cutters(spec) + large_hole_cutters(spec)
    return apply_cutters(parts, cutters)


# ============================================================
# Procedural assembly
# ============================================================

"""Readable assembly for the curved corner bracket model."""




def build_model() -> list:
    """Compose the bracket using the same public parameters as the source SCAD."""
    spec = BracketSpec(
        source_length=70.0,
        width=10.0,
        thickness=2.5,
        hole_count=3,
        hole_diameter=2.5,
        include_outer_holes=True,
        circular_segments=35,
    )
    return add_curved_corner_bracket(spec)


# ============================================================
# Scene setup and rendering
# ============================================================

"""Minimal render entrypoint for current_version."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

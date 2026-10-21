from __future__ import annotations
from dataclasses import dataclass
from math import cos, pi, sin
from mathutils import Vector
from typing import Iterable
import bpy


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the spice-rack model."""





DEFAULT_BODY_COLOR = (0.90, 0.74, 0.08, 1.0)
CUTTER_PREFIX = "boolean_cutter_"


def clear_scene() -> None:
    """Remove all objects from the current Blender scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def get_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        material.diffuse_color = color
        material.use_nodes = True
        principled = material.node_tree.nodes.get("Principled BSDF")
        if principled:
            principled.inputs["Base Color"].default_value = color
            principled.inputs["Roughness"].default_value = 0.58
    return material


def assign_material(obj: bpy.types.Object, material_name: str = "source_yellow", color: tuple[float, float, float, float] = DEFAULT_BODY_COLOR) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(get_material(material_name, color))
    return obj


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material_name: str | None = None,
    color: tuple[float, float, float, float] = DEFAULT_BODY_COLOR,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material_name:
        assign_material(obj, material_name, color)
    return obj


def add_cylinder(
    name: str,
    diameter: float,
    height: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 96,
    material_name: str | None = None,
    color: tuple[float, float, float, float] = DEFAULT_BODY_COLOR,
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
    if material_name:
        assign_material(obj, material_name, color)
    return obj


def add_cone(
    name: str,
    radius1: float,
    radius2: float,
    height: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 96,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cone_add(
        vertices=vertices,
        radius1=radius1,
        radius2=radius2,
        depth=height,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    return obj


def hide_object(obj: bpy.types.Object) -> None:
    obj.hide_set(True)
    obj.hide_render = True


def boolean_difference(target: bpy.types.Object, cutters: Iterable[bpy.types.Object]) -> bpy.types.Object:
    """Apply exact boolean subtraction and remove temporary cutter objects."""
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    for cutter in cutters:
        modifier = target.modifiers.new(name=f"subtract_{cutter.name}", type="BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        modifier.solver = "EXACT"
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def _cross_2d(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def convex_hull_2d(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return a counter-clockwise monotonic-chain convex hull."""
    ordered = sorted(set(points))
    if len(ordered) <= 1:
        return ordered

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def cylinder_points(radius: float, segments: int = 128) -> list[tuple[float, float]]:
    return [
        (radius * cos(2.0 * pi * index / segments), radius * sin(2.0 * pi * index / segments))
        for index in range(segments)
    ]


def box_points(
    center: tuple[float, float],
    size: tuple[float, float],
) -> list[tuple[float, float]]:
    half_x = size[0] / 2.0
    half_y = size[1] / 2.0
    return [
        (center[0] - half_x, center[1] - half_y),
        (center[0] + half_x, center[1] - half_y),
        (center[0] + half_x, center[1] + half_y),
        (center[0] - half_x, center[1] + half_y),
    ]


def extrude_polygon(
    name: str,
    points: list[tuple[float, float]],
    height: float,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    material_name: str = "source_yellow",
    color: tuple[float, float, float, float] = DEFAULT_BODY_COLOR,
) -> bpy.types.Object:
    vertices: list[tuple[float, float, float]] = []
    for z in (0.0, height):
        vertices.extend((x + location[0], y + location[1], z + location[2]) for x, y in points)

    count = len(points)
    faces: list[tuple[int, ...]] = [tuple(range(count - 1, -1, -1)), tuple(range(count, count * 2))]
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, next_index + count, index + count))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material_name, color)
    return obj


def add_hulled_support_blank(
    name: str,
    container_diameter: float,
    wall_thickness: float,
    back_height: float,
    large_back: bool = True,
    material_name: str = "source_yellow",
) -> bpy.types.Object:
    """Approximate OpenSCAD hull(cylinder, rear cube) as one extruded convex footprint."""
    outer_radius = container_diameter / 2.0 + wall_thickness
    back_factor = 2.0 if large_back else 1.0
    hull_points = cylinder_points(outer_radius, 160)
    hull_points.extend(
        box_points(
            center=(0.0, outer_radius - wall_thickness),
            size=(outer_radius * back_factor, wall_thickness * 2.0),
        )
    )
    return extrude_polygon(
        name=name,
        points=convex_hull_2d(hull_points),
        height=back_height,
        material_name=material_name,
    )


def make_inner_bore_cutter(
    name: str,
    container_diameter: float,
    wall_thickness: float,
    back_height: float,
    location_xy: tuple[float, float],
) -> bpy.types.Object:
    cutter = add_cylinder(
        name=f"{CUTTER_PREFIX}{name}",
        diameter=container_diameter,
        height=back_height + wall_thickness * 2.0,
        location=(location_xy[0], location_xy[1], wall_thickness + (back_height + wall_thickness * 2.0) / 2.0),
        vertices=128,
    )
    return cutter


def make_bottom_drain_cutter(
    name: str,
    container_diameter: float,
    back_height: float,
    location_xy: tuple[float, float],
) -> bpy.types.Object:
    cutter = add_cylinder(
        name=f"{CUTTER_PREFIX}{name}",
        diameter=container_diameter * 2.0 / 3.0,
        height=back_height * 2.4,
        location=(location_xy[0], location_xy[1], back_height / 2.0),
        vertices=96,
    )
    return cutter


def make_front_opening_cutter(
    name: str,
    container_diameter: float,
    front_height: float,
    back_height: float,
    wall_thickness: float,
    location_xy: tuple[float, float],
) -> bpy.types.Object:
    """OpenSCAD cuts the upper front with a square prism beginning at y=-diameter/2."""
    cube_height = back_height + wall_thickness * 3.0
    cutter = add_box(
        name=f"{CUTTER_PREFIX}{name}",
        size=(container_diameter, container_diameter, cube_height),
        location=(
            location_xy[0],
            location_xy[1] - container_diameter / 2.0,
            front_height + cube_height / 2.0,
        ),
    )
    return cutter


def make_screw_cutter(
    name: str,
    container_diameter: float,
    wall_thickness: float,
    back_height: float,
    head_diameter: float = 6.5,
    shaft_diameter_start: float = 2.7,
    shaft_diameter_end: float = 2.0,
    head_height: float = 2.75,
    length: float = 10.0,
    location_xy: tuple[float, float] = (0.0, 0.0),
) -> bpy.types.Object:
    """Build the mapped SCAD screw subtractor as one continuous solid mesh.

    SCAD places the screw at x=head_h-thickness-cont_dia/2 and then rotates it
    around Y by 270 degrees. After the final rack Z rotation used by this BPY
    model, that becomes a cutter running from the cup interior toward +Y.
    """
    x = location_xy[0]
    z = back_height * 2.0 / 3.0
    head_start_y = location_xy[1] + container_diameter / 2.0 + wall_thickness - head_height
    head_end_y = head_start_y + head_height
    shaft_end_y = head_end_y + length
    sections = [
        (head_start_y, head_diameter / 2.0),
        (head_end_y, shaft_diameter_start / 2.0),
        (shaft_end_y, shaft_diameter_end / 2.0),
    ]
    segments = 96

    vertices: list[tuple[float, float, float]] = []
    for y, radius in sections:
        for index in range(segments):
            angle = 2.0 * pi * index / segments
            vertices.append((x + radius * cos(angle), y, z + radius * sin(angle)))

    faces: list[tuple[int, ...]] = []
    for section_index in range(len(sections) - 1):
        ring_start = section_index * segments
        next_ring_start = (section_index + 1) * segments
        for index in range(segments):
            next_index = (index + 1) % segments
            faces.append(
                (
                    ring_start + index,
                    next_ring_start + index,
                    next_ring_start + next_index,
                    ring_start + next_index,
                )
            )
    faces.append(tuple(range(segments)))
    faces.append(tuple(range(len(sections) * segments - 1, (len(sections) - 1) * segments - 1, -1)))

    mesh = bpy.data.meshes.new(f"{CUTTER_PREFIX}{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    cutter = bpy.data.objects.new(f"{CUTTER_PREFIX}{name}", mesh)
    bpy.context.collection.objects.link(cutter)
    return cutter


def shade_flat(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_flat()
    obj.select_set(False)
    return obj


def recenter_objects(objects: Iterable[bpy.types.Object], offset: Vector) -> None:
    for obj in objects:
        obj.location -= offset


# ============================================================
# Procedural assembly
# ============================================================

"""High-level assembly for the five-cup spice rack."""






@dataclass(frozen=True)
class SpiceRackSpec:
    container_diameter: float = 40.5
    back_height: float = 25.0
    containers: int = 5
    front_height: float = 12.5
    wall_thickness: float = 1.5
    bottom_opening: bool = True
    front_opening: bool = True
    large_back: bool = True

    @property
    def pitch(self) -> float:
        return self.container_diameter + self.wall_thickness

    @property
    def rack_length(self) -> float:
        return self.pitch * (self.containers - 1)


def build_holder(spec: SpiceRackSpec, index: int, x_position: float) -> bpy.types.Object:
    """Create one OpenSCAD support() instance, including the nested difference cuts."""
    holder = add_hulled_support_blank(
        name=f"holder_{index + 1:02d}_hulled_shell",
        container_diameter=spec.container_diameter,
        wall_thickness=spec.wall_thickness,
        back_height=spec.back_height,
        large_back=spec.large_back,
    )

    cutters = [
        make_inner_bore_cutter(
            name=f"holder_{index + 1:02d}_inner_bore",
            container_diameter=spec.container_diameter,
            wall_thickness=spec.wall_thickness,
            back_height=spec.back_height,
            location_xy=(0.0, 0.0),
        )
    ]
    if spec.front_opening:
        cutters.append(
            make_front_opening_cutter(
                name=f"holder_{index + 1:02d}_front_opening",
                container_diameter=spec.container_diameter,
                front_height=spec.front_height,
                back_height=spec.back_height,
                wall_thickness=spec.wall_thickness,
                location_xy=(0.0, 0.0),
            )
        )
    if spec.bottom_opening:
        cutters.append(
            make_bottom_drain_cutter(
                name=f"holder_{index + 1:02d}_bottom_drain",
                container_diameter=spec.container_diameter,
                back_height=spec.back_height,
                location_xy=(0.0, 0.0),
            )
        )

    cutters.append(
        make_screw_cutter(
            name=f"holder_{index + 1:02d}_countersunk_screw",
            container_diameter=spec.container_diameter,
            wall_thickness=spec.wall_thickness,
            back_height=spec.back_height,
        )
    )

    boolean_difference(holder, cutters)
    holder.location.x = x_position
    shade_flat(holder)
    return holder


def centerline_offset(spec: SpiceRackSpec) -> Vector:
    """Match OpenSCAD's final rotate([0,0,270]) while keeping the rack centered."""
    return Vector((spec.rack_length / 2.0, 0.0, 0.0))


def build_model() -> list[bpy.types.Object]:
    """Compose the final spice rack from five reusable holder modules."""
    spec = SpiceRackSpec()
    holders = [build_holder(spec, index, spec.pitch * index) for index in range(spec.containers)]
    recenter_objects(holders, centerline_offset(spec))
    return holders


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

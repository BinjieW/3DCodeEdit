from __future__ import annotations
from dataclasses import dataclass
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
from typing import Iterable, Sequence
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the V-slot carriage base model."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    metallic: float = 0.0
    roughness: float = 0.55


@dataclass(frozen=True)
class CarriageBaseSpec:
    """SCAD defaults from ParametricCarriageBase_Rev18.scad and its dependency."""

    w_x_offset: float = 39.9
    w_y_offset: float = 16.0
    ledge_offset_top: float = 11.3
    ledge_offset_bottom: float = 11.3
    base_height: float = 6.0
    m5_diameter: float = 5.5
    m5_eccentric_diameter: float = 7.5
    corner_od: float = 30.0
    up_chamfer_od: float = 10.0
    side_chamfer_od: float = 12.0
    carriage_height: float = 11.71
    m3_diameter: float = 3.4
    m3_length: float = 14.5
    m3_nut_diameter: float = 7.0

    @property
    def corner_radius(self) -> float:
        return self.corner_od / 2.0

    @property
    def up_chamfer_radius(self) -> float:
        return self.up_chamfer_od / 2.0

    @property
    def side_chamfer_radius(self) -> float:
        return self.side_chamfer_od / 2.0

    @property
    def side_chamfer_x_offset(self) -> float:
        return self.w_x_offset + math.sin(math.radians(45.0)) * (self.corner_radius + self.side_chamfer_radius)

    @property
    def side_chamfer_y_offset(self) -> float:
        return self.w_y_offset - math.sin(math.radians(45.0)) * (self.corner_radius + self.side_chamfer_radius)

    @property
    def side_cube_len(self) -> float:
        return math.sin(math.radians(45.0)) * self.corner_radius

    @property
    def side_cube_width(self) -> float:
        return self.corner_radius + self.side_chamfer_radius

    @property
    def top_cube_len(self) -> float:
        return self.corner_radius + self.up_chamfer_radius

    @property
    def top_cube_width(self) -> float:
        return self.ledge_offset_top + self.up_chamfer_radius

    @property
    def bottom_cube_width(self) -> float:
        return self.ledge_offset_bottom + self.up_chamfer_radius

    @property
    def up_chamfer_x_offset(self) -> float:
        return self.w_x_offset - self.top_cube_len

    @property
    def side_bump_od(self) -> float:
        return self.side_chamfer_od + 12.0

    @property
    def upper_bump_od(self) -> float:
        return self.up_chamfer_od + 12.0

    @property
    def side_bump_cylinder_x_offset(self) -> float:
        return self.side_chamfer_x_offset - (self.side_chamfer_radius + 3.0)

    @property
    def upper_bump_cylinder_x_offset(self) -> float:
        return self.w_x_offset - (self.corner_radius - 3.0)

    @property
    def ledge_span(self) -> float:
        return self.up_chamfer_x_offset * 2.0

    @property
    def upper_bump_cylinder_height(self) -> float:
        return self.w_y_offset - self.top_cube_width

    @property
    def m3_nut_height(self) -> float:
        return 0.9 * self.m3_diameter


BASE_GOLD = MaterialSpec("openscad_gold_yellow", (0.95, 0.76, 0.05, 1.0), 0.0, 0.48)
CARRIAGE_GREEN = MaterialSpec("openscad_carriage_green", (0.58, 0.78, 0.25, 1.0), 0.0, 0.55)
CUT_WALL_GREEN = MaterialSpec("openscad_cut_wall_green", (0.20, 0.34, 0.08, 1.0), 0.0, 0.78)
DEEP_CUT_GREEN = MaterialSpec("openscad_deep_cut_green", (0.055, 0.095, 0.02, 1.0), 0.0, 0.82)
HOLE_DARK = MaterialSpec("open_hole_dark", (0.035, 0.055, 0.025, 1.0), 0.0, 0.7)


TRAXXAS_PROFILE: tuple[tuple[float, float], ...] = (
    (-19.16, -15.95),
    (19.16, -15.95),
    (27.5, -7.61),
    (27.5, -0.39),
    (21.41, 5.71),
    (-21.41, 5.71),
    (-27.5, -0.39),
    (-27.5, -7.61),
)


BELT_GROOVE_PROFILE: tuple[tuple[float, float], ...] = (
    (9.14, -14.78),
    (8.85, -14.78),
    (8.59, -14.66),
    (8.4, -14.46),
    (8.3, -14.19),
    (8.3, -13.91),
    (8.42, -13.65),
    (8.62, -13.46),
    (8.88, -13.36),
    (8.79, -12.81),
    (8.51, -12.81),
    (8.25, -12.69),
    (8.05, -12.49),
    (7.95, -12.22),
    (7.96, -11.94),
    (8.07, -11.68),
    (8.27, -11.49),
    (8.54, -11.39),
    (8.44, -10.84),
    (8.16, -10.84),
    (7.9, -10.72),
    (7.7, -10.52),
    (7.6, -10.26),
    (7.61, -9.97),
    (7.72, -9.71),
    (7.93, -9.52),
    (8.19, -9.42),
    (8.09, -8.87),
    (7.81, -8.87),
    (7.55, -8.75),
    (7.36, -8.55),
    (7.25, -8.29),
    (7.26, -8.0),
    (7.37, -7.74),
    (7.58, -7.55),
    (7.84, -7.45),
    (7.75, -6.9),
    (7.46, -6.9),
    (7.21, -6.78),
    (7.01, -6.58),
    (6.91, -6.32),
    (6.91, -6.03),
    (7.03, -5.77),
    (7.23, -5.58),
    (7.49, -5.48),
    (7.4, -4.93),
    (7.12, -4.93),
    (6.86, -4.81),
    (6.66, -4.61),
    (6.56, -4.35),
    (6.57, -4.06),
    (6.68, -3.81),
    (6.88, -3.61),
    (7.15, -3.51),
    (7.05, -2.96),
    (6.77, -2.96),
    (6.51, -2.84),
    (6.32, -2.64),
    (6.21, -2.38),
    (6.22, -2.09),
    (6.33, -1.84),
    (6.54, -1.64),
    (6.8, -1.54),
    (6.7, -0.99),
    (6.42, -0.99),
    (6.16, -0.87),
    (5.97, -0.67),
    (5.87, -0.41),
    (5.87, -0.12),
    (5.99, 0.13),
    (6.19, 0.33),
    (6.45, 0.43),
    (6.36, 0.97),
    (6.07, 0.98),
    (5.82, 1.09),
    (5.62, 1.3),
    (5.52, 1.56),
    (5.52, 1.84),
    (5.64, 2.1),
    (5.84, 2.3),
    (6.11, 2.4),
    (6.01, 2.94),
    (5.73, 2.95),
    (5.47, 3.06),
    (5.27, 3.27),
    (5.17, 3.53),
    (5.18, 3.81),
    (5.29, 4.07),
    (5.5, 4.27),
    (5.76, 4.37),
    (5.66, 4.91),
    (5.38, 4.92),
    (5.12, 5.03),
    (4.93, 5.24),
    (4.69, 5.61),
    (4.47, 6.2),
    (6.55, 6.2),
    (10.54, -16.45),
    (8.55, -16.45),
    (8.65, -15.88),
    (8.76, -15.62),
    (8.97, -15.43),
    (9.23, -15.32),
)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def get_material(spec: MaterialSpec) -> bpy.types.Material:
    material = bpy.data.materials.get(spec.name) or bpy.data.materials.new(spec.name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = spec.color
        bsdf.inputs["Metallic"].default_value = spec.metallic
        bsdf.inputs["Roughness"].default_value = spec.roughness
    material.diffuse_color = spec.color
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


def mark_sides_smooth(obj: bpy.types.Object) -> bpy.types.Object:
    for poly in obj.data.polygons:
        poly.use_smooth = abs(poly.normal.z) < 0.5
    return obj


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: MaterialSpec | None = None,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material:
        assign_material(obj, material)
    return obj


def add_box_from_extents(
    name: str,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z_min: float,
    height: float,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    min_x, max_x = sorted((x0, x1))
    min_y, max_y = sorted((y0, y1))
    return add_box(
        name,
        size=(max_x - min_x, max_y - min_y, height),
        location=((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, z_min + height / 2.0),
        material=material,
    )


def add_cylinder(
    name: str,
    diameter: float,
    height: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
    material: MaterialSpec | None = None,
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
    if material:
        assign_material(obj, material)
    if smooth:
        shade_smooth(obj)
    return obj


def add_cone(
    name: str,
    diameter1: float,
    diameter2: float,
    height: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
    material: MaterialSpec | None = None,
    smooth: bool = True,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cone_add(
        vertices=vertices,
        radius1=diameter1 / 2.0,
        radius2=diameter2 / 2.0,
        depth=height,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    if material:
        assign_material(obj, material)
    if smooth:
        shade_smooth(obj)
    return obj


def apply_boolean(
    target: bpy.types.Object,
    cutter: bpy.types.Object,
    operation: str = "DIFFERENCE",
    keep_cutter: bool = False,
) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    modifier = target.modifiers.new(f"{operation.lower()}_{cutter.name}", "BOOLEAN")
    modifier.operation = operation
    modifier.object = cutter
    if hasattr(modifier, "solver"):
        modifier.solver = "EXACT"
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    target.select_set(False)
    if not keep_cutter:
        bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def boolean_union(name: str, objects: Iterable[bpy.types.Object]) -> bpy.types.Object:
    objects = list(objects)
    if not objects:
        raise ValueError("boolean_union requires at least one object")
    active = objects[0]
    bpy.context.view_layer.objects.active = active
    active.select_set(True)
    for obj in objects[1:]:
        modifier = active.modifiers.new(f"union_{obj.name}", "BOOLEAN")
        modifier.operation = "UNION"
        modifier.object = obj
        if hasattr(modifier, "solver"):
            modifier.solver = "EXACT"
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(obj, do_unlink=True)
    active.name = name
    active.data.name = f"{name}_mesh"
    active.select_set(False)
    return active


def join_objects(name: str, objects: Iterable[bpy.types.Object]) -> bpy.types.Object:
    objects = list(objects)
    if not objects:
        raise ValueError("join_objects requires at least one object")
    if len(objects) == 1:
        objects[0].name = name
        objects[0].data.name = f"{name}_mesh"
        return objects[0]

    bpy.ops.object.select_all(action="DESELECT")
    active = objects[0]
    bpy.context.view_layer.objects.active = active
    for obj in objects:
        obj.select_set(True)
    bpy.ops.object.join()
    active.name = name
    active.data.name = f"{name}_mesh"
    active.select_set(False)
    return active


def add_polygon_prism(
    name: str,
    points: Sequence[tuple[float, float]],
    height: float,
    z_min: float,
    material: MaterialSpec | None = None,
    smooth_sides: bool = False,
) -> bpy.types.Object:
    if len(points) < 3:
        raise ValueError("A polygon prism needs at least three points")

    vertices = [(x, y, z_min) for x, y in points] + [(x, y, z_min + height) for x, y in points]
    n = len(points)
    faces: list[tuple[int, ...]] = []
    triangles = tessellate_polygon([[Vector((x, y, 0.0)) for x, y in points]])
    for tri in triangles:
        faces.append(tuple(reversed(tri)))
        faces.append(tuple(i + n for i in tri))
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n, i + n))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if material:
        assign_material(obj, material)
    if smooth_sides:
        mark_sides_smooth(obj)
    return obj


def add_polygon_prism_with_holes(
    name: str,
    outer_points: Sequence[tuple[float, float]],
    hole_loops: Sequence[Sequence[tuple[float, float]]],
    height: float,
    z_min: float,
    material: MaterialSpec = CARRIAGE_GREEN,
    cut_wall_material: MaterialSpec = CUT_WALL_GREEN,
) -> bpy.types.Object:
    """Build a vertical prism whose top face contains real source-profile voids."""
    loops = [list(outer_points)] + [list(loop) for loop in hole_loops if len(loop) >= 3]
    flat_points = [point for loop in loops for point in loop]
    if len(flat_points) < 3:
        raise ValueError("A prism with holes needs at least one valid loop")

    vertices = [(x, y, z_min) for x, y in flat_points] + [(x, y, z_min + height) for x, y in flat_points]
    total = len(flat_points)
    faces: list[tuple[int, ...]] = []
    material_indices: list[int] = []

    triangles = tessellate_polygon([[Vector((x, y, 0.0)) for x, y in loop] for loop in loops])
    for tri in triangles:
        faces.append(tuple(reversed(tri)))
        material_indices.append(0)
        faces.append(tuple(i + total for i in tri))
        material_indices.append(0)

    offset = 0
    for loop_index, loop in enumerate(loops):
        loop_len = len(loop)
        for i in range(loop_len):
            j = (i + 1) % loop_len
            if loop_index == 0:
                faces.append((offset + i, offset + j, offset + j + total, offset + i + total))
                material_indices.append(0)
            else:
                faces.append((offset + j, offset + i, offset + i + total, offset + j + total))
                material_indices.append(1)
        offset += loop_len

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(get_material(material))
    obj.data.materials.append(get_material(cut_wall_material))
    for polygon, material_index in zip(obj.data.polygons, material_indices):
        polygon.material_index = material_index
    return obj


def add_polygon_prism_with_top_inlays(
    name: str,
    outer_points: Sequence[tuple[float, float]],
    inlay_loops: Sequence[Sequence[tuple[float, float]]],
    height: float,
    z_min: float,
    material: MaterialSpec = CARRIAGE_GREEN,
    inlay_material: MaterialSpec = CUT_WALL_GREEN,
) -> bpy.types.Object:
    """Build a full lower prism whose top face has source-profile groove floors."""
    outer = list(outer_points)
    inlays = [list(loop) for loop in inlay_loops if len(loop) >= 3]
    if len(outer) < 3:
        raise ValueError("A top-inlay prism needs a valid outer loop")

    loops = [outer] + inlays
    flat_top = [point for loop in loops for point in loop]
    bottom_count = len(outer)
    top_offset = bottom_count
    vertices = [(x, y, z_min) for x, y in outer] + [(x, y, z_min + height) for x, y in flat_top]
    faces: list[tuple[int, ...]] = []
    material_indices: list[int] = []

    for tri in tessellate_polygon([[Vector((x, y, 0.0)) for x, y in outer]]):
        faces.append(tuple(reversed(tri)))
        material_indices.append(0)

    top_triangles = tessellate_polygon([[Vector((x, y, 0.0)) for x, y in loop] for loop in loops])
    for tri in top_triangles:
        faces.append(tuple(top_offset + i for i in tri))
        material_indices.append(0)

    for i in range(bottom_count):
        j = (i + 1) % bottom_count
        faces.append((i, j, top_offset + j, top_offset + i))
        material_indices.append(0)

    loop_offset = bottom_count
    for loop in inlays:
        for tri in tessellate_polygon([[Vector((x, y, 0.0)) for x, y in loop]]):
            faces.append(tuple(top_offset + loop_offset + i for i in tri))
            material_indices.append(1)
        loop_offset += len(loop)

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(get_material(material))
    obj.data.materials.append(get_material(inlay_material))
    for polygon, material_index in zip(obj.data.polygons, material_indices):
        polygon.material_index = material_index
    return obj


def add_loop_wall_sleeve(
    name: str,
    loop: Sequence[tuple[float, float]],
    height: float,
    z_min: float,
    material: MaterialSpec = DEEP_CUT_GREEN,
) -> bpy.types.Object:
    """Add explicit dark vertical wall faces for a source-profile cut loop."""
    if len(loop) < 3:
        raise ValueError("A wall sleeve needs at least three loop points")
    vertices = [(x, y, z_min) for x, y in loop] + [(x, y, z_min + height) for x, y in loop]
    n = len(loop)
    faces = [(i, (i + 1) % n, ((i + 1) % n) + n, i + n) for i in range(n)]

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def add_polygon_floor(
    name: str,
    points: Sequence[tuple[float, float]],
    z: float,
    material: MaterialSpec = DEEP_CUT_GREEN,
) -> bpy.types.Object:
    """Add a flat dark floor to make a deep source-profile channel readable."""
    if len(points) < 3:
        raise ValueError("A polygon floor needs at least three points")
    vertices = [(x, y, z) for x, y in points]
    faces = [tuple(tri) for tri in tessellate_polygon([[Vector((x, y, 0.0)) for x, y in points]])]
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def arc_points(
    center: tuple[float, float],
    radius: float,
    start_deg: float,
    end_deg: float,
    steps: int,
) -> list[tuple[float, float]]:
    if steps < 1:
        steps = 1
    return [
        (
            center[0] + radius * math.cos(math.radians(start_deg + (end_deg - start_deg) * i / steps)),
            center[1] + radius * math.sin(math.radians(start_deg + (end_deg - start_deg) * i / steps)),
        )
        for i in range(steps + 1)
    ]


def _clip_polygon_axis(
    points: Sequence[tuple[float, float]],
    axis: str,
    value: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip against one vertical 2D half-plane."""
    if not points:
        return []

    coord_index = 0 if axis == "x" else 1

    def inside(point: tuple[float, float]) -> bool:
        coord = point[coord_index]
        return coord >= value - 1e-8 if keep_greater else coord <= value + 1e-8

    def intersect(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
        a_coord = a[coord_index]
        b_coord = b[coord_index]
        if abs(b_coord - a_coord) < 1e-9:
            return a
        t = (value - a_coord) / (b_coord - a_coord)
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    output: list[tuple[float, float]] = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def cut_vertical_cylinder(
    target: bpy.types.Object,
    name: str,
    center: tuple[float, float],
    diameter: float,
    z_min: float,
    height: float,
    vertices: int = 48,
) -> bpy.types.Object:
    cutter = add_cylinder(
        name,
        diameter=diameter,
        height=height,
        location=(center[0], center[1], z_min + height / 2.0),
        vertices=vertices,
        smooth=False,
    )
    return apply_boolean(target, cutter, "DIFFERENCE")


def cut_box(
    target: bpy.types.Object,
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    cutter = add_box(name, size=size, location=location, rotation=rotation)
    return apply_boolean(target, cutter, "DIFFERENCE")


def add_hole_shadow(
    name: str,
    center: tuple[float, float],
    diameter: float,
    z: float,
    material: MaterialSpec = HOLE_DARK,
    vertices: int = 48,
) -> bpy.types.Object:
    return add_cylinder(
        name,
        diameter=diameter * 0.96,
        height=0.04,
        location=(center[0], center[1], z),
        vertices=vertices,
        material=material,
    )


def add_hex_ledge_bar(
    name: str,
    diameter: float,
    length: float,
    location: tuple[float, float, float],
    axis: str,
    material: MaterialSpec = BASE_GOLD,
) -> bpy.types.Object:
    if axis.upper() == "X":
        rotation = (0.0, math.pi / 2.0, 0.0)
    elif axis.upper() == "Y":
        rotation = (math.pi / 2.0, 0.0, 0.0)
    else:
        rotation = (0.0, 0.0, 0.0)
    return add_cylinder(
        name,
        diameter=diameter,
        height=length,
        location=location,
        rotation=rotation,
        vertices=6,
        material=material,
        smooth=False,
    )


def make_bump_ledge(
    name: str,
    center: tuple[float, float],
    outer_diameter: float,
    z_min: float,
    material: MaterialSpec = BASE_GOLD,
    vertices: int = 108,
) -> bpy.types.Object:
    """Recreate the tapered annular BumpLedge() CSG helper from the source."""
    inner_diameter = outer_diameter - 12.0
    inner_bevel_top = outer_diameter - 9.0 + 0.0577
    body = add_cone(
        f"{name}_outer_taper",
        diameter1=outer_diameter,
        diameter2=outer_diameter - 3.0,
        height=2.6,
        location=(center[0], center[1], z_min + 1.3),
        vertices=vertices,
        material=material,
        smooth=False,
    )
    cut_vertical_cylinder(
        body,
        f"{name}_inner_straight_cut",
        center=center,
        diameter=inner_diameter,
        z_min=z_min - 1.0,
        height=6.0,
        vertices=vertices,
    )
    inner_bevel = add_cone(
        f"{name}_inner_bevel_cut",
        diameter1=inner_diameter,
        diameter2=inner_bevel_top,
        height=2.7,
        location=(center[0], center[1], z_min + 1.35),
        vertices=vertices,
        smooth=False,
    )
    apply_boolean(body, inner_bevel, "DIFFERENCE")
    return body


def _mirror_xy(point: tuple[float, float], sx: int, sy: int) -> tuple[float, float]:
    return (point[0] * (-sx), point[1] * sy)


def _mirror_angle_z(angle: float, sx: int, sy: int) -> float:
    return math.atan2(sy * math.sin(angle), (-sx) * math.cos(angle))


def _rotate_xy(point: tuple[float, float], angle: float) -> tuple[float, float]:
    c = math.cos(angle)
    s = math.sin(angle)
    return (point[0] * c - point[1] * s, point[0] * s + point[1] * c)


def _add_mirrored_box_from_extents(
    name: str,
    sx: int,
    sy: int,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z_min: float,
    height: float,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    ax, ay = _mirror_xy((x0, y0), sx, sy)
    bx, by = _mirror_xy((x1, y1), sx, sy)
    return add_box_from_extents(name, ax, bx, ay, by, z_min, height, material=material)


def _cut_mirrored_box_from_extents(
    target: bpy.types.Object,
    name: str,
    sx: int,
    sy: int,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z_min: float,
    height: float,
) -> bpy.types.Object:
    cutter = _add_mirrored_box_from_extents(name, sx, sy, x0, x1, y0, y1, z_min, height)
    return apply_boolean(target, cutter, "DIFFERENCE")


def _cut_mirrored_vertical_cylinder(
    target: bpy.types.Object,
    name: str,
    sx: int,
    sy: int,
    center: tuple[float, float],
    diameter: float,
    z_min: float,
    height: float,
    vertices: int = 48,
) -> bpy.types.Object:
    return cut_vertical_cylinder(
        target,
        name,
        center=_mirror_xy(center, sx, sy),
        diameter=diameter,
        z_min=z_min,
        height=height,
        vertices=vertices,
    )


def _cut_mirrored_oriented_box(
    target: bpy.types.Object,
    name: str,
    sx: int,
    sy: int,
    center: tuple[float, float],
    size: tuple[float, float, float],
    z_min: float,
    angle_degrees: float,
) -> bpy.types.Object:
    mx, my = _mirror_xy(center, sx, sy)
    angle = _mirror_angle_z(math.radians(angle_degrees), sx, sy)
    cutter = add_box(
        name,
        size=size,
        location=(mx, my, z_min + size[2] / 2.0),
        rotation=(0.0, 0.0, angle),
    )
    return apply_boolean(target, cutter, "DIFFERENCE")


def make_corner_module(
    name: str,
    sx: int,
    sy: int,
    spec: CarriageBaseSpec,
    material: MaterialSpec = BASE_GOLD,
) -> bpy.types.Object:
    """Build one mirrored CornerTop/CornerBot body, including its clipped wheel ledge."""
    wheel_center = _mirror_xy((-spec.w_x_offset, spec.w_y_offset), sx, sy)
    parts = [
        add_cylinder(
            f"{name}_corner_disc",
            diameter=spec.corner_od,
            height=spec.base_height,
            location=(wheel_center[0], wheel_center[1], spec.base_height / 2.0),
            vertices=96,
            material=material,
            smooth=False,
        ),
        _add_mirrored_box_from_extents(
            f"{name}_inner_fill",
            sx,
            sy,
            -spec.w_x_offset,
            -spec.w_x_offset + spec.corner_radius,
            0.0,
            spec.w_y_offset,
            0.0,
            spec.base_height,
            material,
        ),
        _add_mirrored_box_from_extents(
            f"{name}_outer_fill",
            sx,
            sy,
            -spec.w_x_offset - spec.side_cube_len,
            -spec.w_x_offset,
            0.0,
            spec.w_y_offset,
            0.0,
            spec.base_height,
            material,
        ),
        _add_mirrored_box_from_extents(
            f"{name}_center_web",
            sx,
            sy,
            -spec.w_x_offset,
            0.01,
            0.0,
            spec.ledge_offset_top,
            0.0,
            spec.base_height,
            material,
        ),
        _add_mirrored_box_from_extents(
            f"{name}_wide_inner_web",
            sx,
            sy,
            -spec.w_x_offset,
            -spec.w_x_offset + spec.top_cube_len,
            0.0,
            spec.top_cube_width,
            0.0,
            spec.base_height,
            material,
        ),
        make_bump_ledge(f"{name}_wheel_bump_ledge", wheel_center, spec.corner_od, spec.base_height, material),
    ]
    body = boolean_union(f"{name}_corner_module", parts)

    _cut_mirrored_vertical_cylinder(
        body,
        f"{name}_side_curve_cut",
        sx,
        sy,
        (-spec.side_chamfer_x_offset, spec.side_chamfer_y_offset),
        spec.side_chamfer_od,
        -1.0,
        spec.base_height + 2.0,
        vertices=36,
    )
    _cut_mirrored_vertical_cylinder(
        body,
        f"{name}_inner_neck_curve_cut",
        sx,
        sy,
        (-spec.up_chamfer_x_offset, spec.top_cube_width),
        spec.up_chamfer_od,
        -1.0,
        spec.base_height + 2.0,
        vertices=32,
    )
    _cut_mirrored_box_from_extents(
        body,
        f"{name}_side_curve_flat_cut",
        sx,
        sy,
        -spec.side_chamfer_x_offset,
        -spec.side_chamfer_x_offset + spec.side_chamfer_radius,
        -0.1,
        spec.side_chamfer_y_offset + 0.1,
        -1.0,
        spec.base_height + 2.0,
    )

    angle = math.radians(-45.0)
    corner = (-spec.side_chamfer_x_offset, spec.side_chamfer_y_offset)
    rotated_center_offset = _rotate_xy((spec.side_cube_width / 2.0, spec.side_cube_width / 2.0), angle)
    rotated_center = (corner[0] + rotated_center_offset[0], corner[1] + rotated_center_offset[1])
    _cut_mirrored_oriented_box(
        body,
        f"{name}_upper_z_diagonal_ledge_cut",
        sx,
        sy,
        rotated_center,
        (spec.side_cube_width, spec.side_cube_width, spec.base_height),
        spec.base_height,
        -45.0,
    )
    _cut_mirrored_box_from_extents(
        body,
        f"{name}_upper_z_quadrant_ledge_cut",
        sx,
        sy,
        -spec.w_x_offset,
        -spec.w_x_offset + spec.corner_od,
        0.0,
        spec.w_y_offset,
        spec.base_height,
        spec.base_height + 0.2,
    )
    return body


def make_side_bump(
    name: str,
    sx: int,
    sy: int,
    spec: CarriageBaseSpec,
    material: MaterialSpec = BASE_GOLD,
) -> bpy.types.Object:
    center = (-spec.side_chamfer_x_offset, spec.side_chamfer_y_offset)
    obj = make_bump_ledge(name, _mirror_xy(center, sx, sy), spec.side_bump_od, spec.base_height, material, vertices=108)
    _cut_mirrored_box_from_extents(
        obj,
        f"{name}_bottom_half_cut",
        sx,
        sy,
        center[0] - spec.side_bump_od,
        center[0] + spec.side_bump_od,
        center[1] - spec.side_bump_od,
        center[1],
        spec.base_height - 1.0,
        spec.base_height + 2.0,
    )
    rel_center = _rotate_xy((-spec.side_bump_od / 2.0, 1.0), math.radians(-45.0))
    _cut_mirrored_oriented_box(
        obj,
        f"{name}_diagonal_match_cut",
        sx,
        sy,
        (center[0] + rel_center[0], center[1] + rel_center[1]),
        (spec.side_bump_od, spec.side_bump_od, spec.base_height + 2.0),
        spec.base_height - 1.0,
        -45.0,
    )
    return obj


def make_up_bump(
    name: str,
    sx: int,
    sy: int,
    spec: CarriageBaseSpec,
    material: MaterialSpec = BASE_GOLD,
) -> bpy.types.Object:
    center = (-spec.up_chamfer_x_offset, spec.top_cube_width)
    obj = make_bump_ledge(name, _mirror_xy(center, sx, sy), spec.upper_bump_od, spec.base_height, material, vertices=108)
    _cut_mirrored_box_from_extents(
        obj,
        f"{name}_top_half_cut",
        sx,
        sy,
        center[0] - spec.upper_bump_od,
        center[0] + spec.upper_bump_od,
        center[1],
        center[1] + spec.upper_bump_od,
        spec.base_height - 1.0,
        spec.base_height + 2.0,
    )
    _cut_mirrored_box_from_extents(
        obj,
        f"{name}_inner_half_cut",
        sx,
        sy,
        center[0],
        center[0] + spec.upper_bump_od,
        center[1] - spec.upper_bump_od,
        center[1] + 1.0,
        spec.base_height - 1.0,
        spec.base_height + 2.0,
    )
    return obj


def apply_base_ledge_sharpening(target: bpy.types.Object, spec: CarriageBaseSpec) -> bpy.types.Object:
    """Add small top-ledge-only interruption cuts at the visible SCAD overlap seams."""
    z_min = spec.base_height + 0.05
    z_size = 3.85
    z_center = z_min + z_size / 2.0

    for sx in (-1, 1):
        for sy in (-1, 1):
            cut_box(
                target,
                f"ledge_inner_stop_notch_{sx}_{sy}",
                size=(0.72, 4.4, z_size),
                location=(sx * (spec.up_chamfer_x_offset + 0.08), sy * (spec.ledge_offset_top - 3.0), z_center),
            )
            cut_box(
                target,
                f"ledge_outer_side_scallop_break_{sx}_{sy}",
                size=(3.4, 0.7, z_size),
                location=(sx * spec.side_bump_cylinder_x_offset, sy * spec.side_chamfer_y_offset, z_center),
            )
            cut_box(
                target,
                f"inner_bump_chord_stop_{sx}_{sy}",
                size=(3.4, 0.62, z_size),
                location=(sx * spec.up_chamfer_x_offset, sy * (spec.top_cube_width - 1.05), z_center),
            )
            cut_box(
                target,
                f"wheel_ledge_neck_undercut_{sx}_{sy}",
                size=(0.72, 5.2, z_size),
                location=(sx * (spec.w_x_offset - spec.corner_radius + 1.45), sy * (spec.w_y_offset - 3.2), z_center),
            )
            _cut_mirrored_oriented_box(
                target,
                f"diagonal_side_bump_overlap_clip_{sx}_{sy}",
                sx,
                sy,
                (-spec.side_chamfer_x_offset + 2.55, spec.side_chamfer_y_offset + 1.05),
                (4.8, 0.72, z_size),
                z_min,
                -45.0,
            )

    for sx in (-1, 1):
        cut_box(
            target,
            f"side_waist_ledge_break_{sx}",
            size=(3.8, 0.64, z_size),
            location=(sx * spec.side_bump_cylinder_x_offset, 0.0, z_center),
        )
    return target


def build_scad_base_and_ledges(
    name: str,
    spec: CarriageBaseSpec,
    material: MaterialSpec = BASE_GOLD,
) -> bpy.types.Object:
    """Compose the source DrawTopSide()/DrawBottomSide() base and ledges."""
    parts: list[bpy.types.Object] = []
    for sx in (-1, 1):
        for sy in (1, -1):
            parts.append(make_corner_module(f"{name}_corner_{sx}_{sy}", sx, sy, spec, material))
            parts.append(make_side_bump(f"{name}_side_bump_{sx}_{sy}", sx, sy, spec, material))
            parts.append(make_up_bump(f"{name}_inner_bump_{sx}_{sy}", sx, sy, spec, material))

    for sx in (-1, 1):
        x, y = _mirror_xy((-spec.side_bump_cylinder_x_offset, 0.0), sx, 1)
        parts.append(
            add_hex_ledge_bar(
                f"{name}_short_side_ledge_{sx}",
                diameter=6.0,
                length=abs(spec.side_chamfer_y_offset) * 2.0,
                location=(x, y, spec.base_height),
                axis="Y",
                material=material,
            )
        )

    parts.append(
        add_hex_ledge_bar(
            f"{name}_top_straight_ledge",
            diameter=6.0,
            length=spec.ledge_span,
            location=(0.0, spec.ledge_offset_top - 3.0, spec.base_height),
            axis="X",
            material=material,
        )
    )
    parts.append(
        add_hex_ledge_bar(
            f"{name}_bottom_straight_ledge",
            diameter=6.0,
            length=spec.ledge_span,
            location=(0.0, -(spec.ledge_offset_bottom - 3.0), spec.base_height),
            axis="X",
            material=material,
        )
    )

    if spec.upper_bump_cylinder_height > 0:
        for sx in (-1, 1):
            for sy in (1, -1):
                x, y = _mirror_xy((-spec.upper_bump_cylinder_x_offset, spec.w_y_offset), sx, sy)
                parts.append(
                    add_hex_ledge_bar(
                        f"{name}_up_down_ledge_{sx}_{sy}",
                        diameter=6.0,
                        length=spec.upper_bump_cylinder_height,
                        location=(x, y, spec.base_height),
                        axis="Y",
                        material=material,
                    )
                )

    body = boolean_union(name, parts)
    apply_base_ledge_sharpening(body, spec)
    return body


def apply_final_base_cuts(
    target: bpy.types.Object,
    spec: CarriageBaseSpec,
    *,
    wheel_holes: bool,
    belt_clearance: bool,
) -> bpy.types.Object:
    """Apply the DrawFinal() cuts that affect the assembled union."""
    if wheel_holes:
        for sx, diameter in [(-1, spec.m5_diameter), (1, spec.m5_eccentric_diameter)]:
            for sy in (-1, 1):
                cut_vertical_cylinder(
                    target,
                    f"wheel_hole_{sx}_{sy}",
                    center=(sx * spec.w_x_offset, sy * spec.w_y_offset),
                    diameter=diameter,
                    z_min=-1.0,
                    height=spec.base_height + 2.0,
                    vertices=32,
                )

    if belt_clearance:
        cut_box(
            target,
            "final_belt_clearance_slot",
            size=(10.0, spec.w_y_offset * 6.0, 10.0),
            location=(-7.0, -spec.ledge_offset_bottom * 4.0 + (spec.w_y_offset * 6.0) / 2.0, spec.base_height + 5.0),
        )
    return target


def make_traxxas_carriage_body(
    name: str,
    spec: CarriageBaseSpec,
    material: MaterialSpec = CARRIAGE_GREEN,
) -> bpy.types.Object:
    y_sliced = _clip_polygon_axis(TRAXXAS_PROFILE, "y", -15.3, keep_greater=True)
    y_sliced = _clip_polygon_axis(y_sliced, "y", 4.9, keep_greater=False)
    shifted_profile = [(x, y + 4.0) for x, y in y_sliced]
    groove_loops = _shifted_belt_groove_loops()

    lower = add_polygon_prism_with_top_inlays(
        f"{name}_lower_source_body_with_real_groove_floors",
        shifted_profile,
        groove_loops,
        4.0,
        spec.base_height,
        material,
        DEEP_CUT_GREEN,
    )
    upper = add_polygon_prism_with_holes(
        f"{name}_upper_source_body_with_real_groove_voids",
        shifted_profile,
        groove_loops,
        spec.carriage_height - 4.0,
        spec.base_height + 4.0,
        material,
        DEEP_CUT_GREEN,
    )
    return join_objects(name, [lower, upper])


def _scale_loop(points: Sequence[tuple[float, float]], scale: float) -> list[tuple[float, float]]:
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    return [(cx + (x - cx) * scale, cy + (y - cy) * scale) for x, y in points]


def _shifted_belt_groove_loops() -> list[list[tuple[float, float]]]:
    groove_profile_sliced = _clip_polygon_axis(BELT_GROOVE_PROFILE, "y", -15.3, keep_greater=True)
    groove_profile_sliced = _clip_polygon_axis(groove_profile_sliced, "y", 4.9, keep_greater=False)
    return [[(x + x_offset, y + 4.0) for x, y in groove_profile_sliced] for x_offset in (0.0, -3.85)]


def add_traxxas_cut_readbacks(target: bpy.types.Object, spec: CarriageBaseSpec) -> bpy.types.Object:
    """Shade the real clearance-bay interior; groove walls are actual mesh holes."""
    parts: list[bpy.types.Object] = [target]
    clear_y_min = -15.3 + 4.0
    clear_y_max = 4.9 + 4.0
    clear_height = spec.carriage_height
    for x in (-12.0, -2.0):
        parts.append(
            add_box(
                f"{target.name}_source_clearance_wall_x{x}",
                size=(0.045, clear_y_max - clear_y_min, clear_height),
                location=(x, (clear_y_min + clear_y_max) / 2.0, spec.base_height + clear_height / 2.0),
                material=DEEP_CUT_GREEN,
            )
        )
    parts.append(
        add_box(
            f"{target.name}_source_clearance_bay_dark_bottom",
            size=(9.86, clear_y_max - clear_y_min, 0.035),
            location=(-7.0, (clear_y_min + clear_y_max) / 2.0, spec.base_height + 0.035),
            material=HOLE_DARK,
        )
    )
    return join_objects(target.name, parts)


def _carriage_point(x: float, y: float, z: float, spec: CarriageBaseSpec) -> tuple[float, float, float]:
    return (x, y + 4.0, z + spec.base_height)


def _add_axis_x_cylinder_cut(
    target: bpy.types.Object,
    name: str,
    start: tuple[float, float, float],
    diameter: float,
    length: float,
    spec: CarriageBaseSpec,
    vertices: int,
    smooth: bool = False,
) -> bpy.types.Object:
    x, y, z = _carriage_point(start[0] + length / 2.0, start[1], start[2], spec)
    cutter = add_cylinder(
        name,
        diameter=diameter,
        height=length,
        location=(x, y, z),
        rotation=(0.0, math.pi / 2.0, 0.0),
        vertices=vertices,
        smooth=smooth,
    )
    return apply_boolean(target, cutter, "DIFFERENCE")


def _add_axis_x_cone_cut(
    target: bpy.types.Object,
    name: str,
    start: tuple[float, float, float],
    diameter1: float,
    diameter2: float,
    length: float,
    spec: CarriageBaseSpec,
    vertices: int,
) -> bpy.types.Object:
    x, y, z = _carriage_point(start[0] + length / 2.0, start[1], start[2], spec)
    cutter = add_cone(
        name,
        diameter1=diameter1,
        diameter2=diameter2,
        height=length,
        location=(x, y, z),
        rotation=(0.0, math.pi / 2.0, 0.0),
        vertices=vertices,
        smooth=False,
    )
    return apply_boolean(target, cutter, "DIFFERENCE")


def _add_belt_groove_diamond_relief_cuts(target: bpy.types.Object, spec: CarriageBaseSpec) -> bpy.types.Object:
    for x_offset in (0.0, -3.85):
        local_x = 6.55 + x_offset
        local_y = 6.2
        length = 23.0
        relief_angle = math.radians(10.0)
        x, y, z = _carriage_point(
            local_x + math.sin(relief_angle) * length / 2.0,
            local_y - math.cos(relief_angle) * length / 2.0,
            11.7,
            spec,
        )
        cutter = add_cylinder(
            f"belt_groove_upper_diamond_relief_{x_offset}",
            diameter=2.2,
            height=length,
            location=(x, y, z),
            rotation=(math.pi / 2.0, 0.0, relief_angle),
            vertices=4,
            smooth=False,
        )
        apply_boolean(target, cutter, "DIFFERENCE")
    return target


def apply_traxxas_carriage_cuts(target: bpy.types.Object, spec: CarriageBaseSpec) -> bpy.types.Object:
    """Recreate DrawCarriage() difference cuts, including the +4 y translation."""
    cut_box(
        target,
        "source_drawcarriage_left_clearance_cube",
        size=(10.0, 23.65, 20.0),
        location=_carriage_point(-7.0, -5.125, 11.0, spec),
    )
    _add_belt_groove_diamond_relief_cuts(target, spec)

    for x in (-28.5, 14.0):
        _add_axis_x_cylinder_cut(
            target,
            f"m3_side_bolt_cut_{x}",
            (x, -4.0, 7.0),
            spec.m3_diameter,
            spec.m3_length,
            spec,
            vertices=24,
        )

    for x in (-20.0, 17.2):
        for z in (7.0, 10.0):
            _add_axis_x_cylinder_cut(
                target,
                f"m3_hex_nut_pocket_{x}_{z}",
                (x, -4.0, z),
                spec.m3_nut_diameter,
                spec.m3_nut_height,
                spec,
                vertices=6,
            )

    _add_axis_x_cylinder_cut(target, "rear_left_large_diamond_chamfer", (-28.0, 5.71, spec.carriage_height), 16.0, 17.0, spec, vertices=4)
    _add_axis_x_cone_cut(target, "rear_right_tapered_diamond_chamfer", (11.2, 5.71, spec.carriage_height), 0.0, 16.0, 6.0, spec, vertices=4)
    _add_axis_x_cylinder_cut(target, "rear_right_large_diamond_chamfer", (17.2, 5.71, spec.carriage_height), 16.0, 17.0, spec, vertices=4)
    _add_axis_x_cylinder_cut(target, "front_long_diamond_chamfer", (-27.5, -15.95, spec.carriage_height), 6.0, 55.0, spec, vertices=4)
    _add_axis_x_cylinder_cut(target, "front_left_large_diamond_chamfer", (-27.5, -15.95, spec.carriage_height), 16.0, 16.0, spec, vertices=4)
    _add_axis_x_cylinder_cut(target, "rear_long_diamond_chamfer", (-27.5, 5.71, spec.carriage_height), 6.0, 55.0, spec, vertices=4)

    cut_box(
        target,
        "front_dimension_slice",
        size=(60.0, 10.0, 20.0),
        location=_carriage_point(0.0, -25.3 + 5.0, -1.0 + 10.0, spec),
    )
    cut_box(
        target,
        "rear_dimension_slice",
        size=(60.0, 10.0, 20.0),
        location=_carriage_point(0.0, 4.9 + 5.0, -1.0 + 10.0, spec),
    )
    return target


def make_collection(name: str, objects: Iterable[bpy.types.Object]) -> bpy.types.Collection:
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    for obj in objects:
        for owner in list(obj.users_collection):
            owner.objects.unlink(obj)
        collection.objects.link(obj)
    return collection


# ============================================================
# Procedural assembly
# ============================================================

"""Procedural assembly for the ParametricCarriageBase Rev18 model."""





SPEC = CarriageBaseSpec()


def _base_plate_and_ledges() -> bpy.types.Object:
    base = build_scad_base_and_ledges("scad_transcribed_base_and_ledges", SPEC)
    apply_final_base_cuts(base, SPEC, wheel_holes=True, belt_clearance=True)
    return base


def _traxxas_carriage() -> bpy.types.Object:
    carriage = make_traxxas_carriage_body("traxxas_carriage_drawcarriage_body", SPEC)
    apply_traxxas_carriage_cuts(carriage, SPEC)
    apply_final_base_cuts(carriage, SPEC, wheel_holes=False, belt_clearance=True)
    carriage = add_traxxas_cut_readbacks(carriage, SPEC)
    return carriage


def build_model() -> list[bpy.types.Object]:
    """Build the current model as named model assemblies."""
    objects = [_base_plate_and_ledges(), _traxxas_carriage()]
    make_collection("parametric_carriage_base_rev18", objects)
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

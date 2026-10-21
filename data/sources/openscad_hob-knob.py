from __future__ import annotations
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
import bmesh
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the hob knob model."""




SOURCE_YELLOW = (0.86, 0.73, 0.08, 1.0)
SCALLOP_GREEN = (0.34, 0.56, 0.08, 1.0)


@dataclass(frozen=True)
class HorizontalCylinder:
    """Finite horizontal cylinder matching the rotated OpenSCAD scallop cutters."""

    name: str
    angle_deg: float
    center_radius: float
    z_center: float
    radius: float
    length: float

    def frame(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float, float]]:
        angle = math.radians(self.angle_deg)
        axis = (math.cos(angle), math.sin(angle))
        lateral = (-math.sin(angle), math.cos(angle))
        center = (lateral[0] * self.center_radius, lateral[1] * self.center_radius, self.z_center)
        return axis, lateral, center

    def contains(self, point: tuple[float, float, float], radius: float | None = None, pad: float = 0.0) -> bool:
        axis, lateral, center = self.frame()
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        dz = point[2] - center[2]
        axial = dx * axis[0] + dy * axis[1]
        if abs(axial) > self.length / 2.0 + pad:
            return False
        side = dx * lateral[0] + dy * lateral[1]
        test_radius = self.radius if radius is None else radius
        return side * side + dz * dz <= (test_radius + pad) ** 2

    def surface_point(self, axial: float, theta: float, radius: float | None = None) -> tuple[float, float, float]:
        axis, lateral, center = self.frame()
        surface_radius = self.radius if radius is None else radius
        return (
            center[0] + axis[0] * axial + lateral[0] * surface_radius * math.cos(theta),
            center[1] + axis[1] * axial + lateral[1] * surface_radius * math.cos(theta),
            center[2] + surface_radius * math.sin(theta),
        )


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.55
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.3
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.3
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
    return obj


def recalculate_normals(obj: bpy.types.Object) -> bpy.types.Object:
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def weld_vertices(obj: bpy.types.Object, distance: float = 1e-5) -> bpy.types.Object:
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=distance)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def shade_flat(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_flat()
    obj.select_set(False)
    return obj


def shade_smooth(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)
    return obj


def create_mesh_object(
    name: str,
    verts: Iterable[tuple[float, float, float]],
    faces: Iterable[tuple[int, ...]],
    material: bpy.types.Material | None = None,
    recalc: bool = True,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(list(verts), [], list(faces))
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if recalc and mesh.polygons:
        recalculate_normals(obj)
    if material is not None:
        assign_material(obj, material)
    return obj


def polygon_boundary_point(radius: float, sides: int, angle: float) -> tuple[float, float]:
    step = 2.0 * math.pi / sides
    apothem = radius * math.cos(step / 2.0)
    delta = (angle % step) - step / 2.0
    edge_radius = apothem / max(math.cos(delta), 1e-6)
    return (edge_radius * math.cos(angle), edge_radius * math.sin(angle))


def polygon_radius_at_angle(radius: float, sides: int, angle: float) -> float:
    x, y = polygon_boundary_point(radius, sides, angle)
    return math.hypot(x, y)


def inside_regular_polygon(
    x: float,
    y: float,
    radius: float,
    sides: int,
    margin: float = 0.0,
) -> bool:
    angle = math.atan2(y, x)
    return math.hypot(x, y) <= polygon_radius_at_angle(radius, sides, angle) + margin


def d_shape_radius(angle: float, radius: float, flat_offset: float) -> float:
    sin_value = math.sin(angle)
    if sin_value < -flat_offset / radius:
        return -flat_offset / sin_value
    return radius


def create_cylinder_wall_mesh(
    name: str,
    radius: float,
    z_min: float,
    z_max: float,
    vertices: int = 96,
    inward: bool = False,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    verts: list[tuple[float, float, float]] = []
    for z in (z_min, z_max):
        for index in range(vertices):
            angle = 2.0 * math.pi * index / vertices
            verts.append((radius * math.cos(angle), radius * math.sin(angle), z))

    def idx(z_side: int, index: int) -> int:
        return z_side * vertices + index % vertices

    faces: list[tuple[int, int, int, int]] = []
    for index in range(vertices):
        if inward:
            faces.append((idx(0, index + 1), idx(0, index), idx(1, index), idx(1, index + 1)))
        else:
            faces.append((idx(0, index), idx(0, index + 1), idx(1, index + 1), idx(1, index)))
    return create_mesh_object(name, verts, faces, material)


def create_profile_wall_mesh(
    name: str,
    radius_fn: Callable[[float], float],
    z_min: float,
    z_max: float,
    segments: int = 128,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    verts: list[tuple[float, float, float]] = []
    for z in (z_min, z_max):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            radius = radius_fn(angle)
            verts.append((radius * math.cos(angle), radius * math.sin(angle), z))

    def idx(z_side: int, index: int) -> int:
        return z_side * segments + index % segments

    faces = [(idx(0, index), idx(0, index + 1), idx(1, index + 1), idx(1, index)) for index in range(segments)]
    return create_mesh_object(name, verts, faces, material)


def create_radial_ring_mesh(
    name: str,
    outer_radius_fn: Callable[[float], float],
    inner_radius_fn: Callable[[float], float],
    z_min: float,
    z_max: float,
    segments: int = 128,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    verts: list[tuple[float, float, float]] = []
    for z in (z_min, z_max):
        for radius_fn in (outer_radius_fn, inner_radius_fn):
            for index in range(segments):
                angle = 2.0 * math.pi * index / segments
                radius = radius_fn(angle)
                verts.append((radius * math.cos(angle), radius * math.sin(angle), z))

    def idx(z_side: int, loop: int, index: int) -> int:
        return z_side * segments * 2 + loop * segments + index % segments

    faces: list[tuple[int, int, int, int]] = []
    for index in range(segments):
        faces.append((idx(0, 0, index), idx(0, 0, index + 1), idx(1, 0, index + 1), idx(1, 0, index)))
        faces.append((idx(0, 1, index + 1), idx(0, 1, index), idx(1, 1, index), idx(1, 1, index + 1)))
        faces.append((idx(1, 0, index), idx(1, 0, index + 1), idx(1, 1, index + 1), idx(1, 1, index)))
        faces.append((idx(0, 0, index + 1), idx(0, 0, index), idx(0, 1, index), idx(0, 1, index + 1)))
    return create_mesh_object(name, verts, faces, material)


def create_cone_frustum_mesh(
    name: str,
    radius_bottom: float,
    radius_top: float,
    z_min: float,
    z_max: float,
    vertices: int = 96,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    verts: list[tuple[float, float, float]] = []
    for z, radius in ((z_min, radius_bottom), (z_max, radius_top)):
        for index in range(vertices):
            angle = 2.0 * math.pi * index / vertices
            verts.append((radius * math.cos(angle), radius * math.sin(angle), z))
    bottom_center = len(verts)
    verts.append((0.0, 0.0, z_min))
    top_center = len(verts)
    verts.append((0.0, 0.0, z_max))

    def idx(z_side: int, index: int) -> int:
        return z_side * vertices + index % vertices

    faces: list[tuple[int, ...]] = []
    for index in range(vertices):
        faces.append((idx(0, index), idx(0, index + 1), idx(1, index + 1), idx(1, index)))
        faces.append((bottom_center, idx(0, index + 1), idx(0, index)))
        faces.append((top_center, idx(1, index), idx(1, index + 1)))
    return create_mesh_object(name, verts, faces, material)


def create_horizontal_cylinder_mesh(
    name: str,
    cylinder: HorizontalCylinder,
    vertices: int = 96,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    verts: list[tuple[float, float, float]] = []
    axial_values = (-cylinder.length / 2.0, cylinder.length / 2.0)
    for axial in axial_values:
        for index in range(vertices):
            theta = 2.0 * math.pi * index / vertices
            verts.append(cylinder.surface_point(axial, theta))
    start_center = len(verts)
    verts.append(cylinder.surface_point(axial_values[0], 0.0, radius=0.0))
    end_center = len(verts)
    verts.append(cylinder.surface_point(axial_values[1], 0.0, radius=0.0))

    def idx(side: int, index: int) -> int:
        return side * vertices + index % vertices

    faces: list[tuple[int, ...]] = []
    for index in range(vertices):
        faces.append((idx(0, index), idx(1, index), idx(1, index + 1), idx(0, index + 1)))
        faces.append((start_center, idx(0, index + 1), idx(0, index)))
        faces.append((end_center, idx(1, index), idx(1, index + 1)))
    return create_mesh_object(name, verts, faces, material)


def create_horizontal_tube_mesh(
    name: str,
    cylinder: HorizontalCylinder,
    inner_radius: float,
    outer_radius: float,
    vertices: int = 128,
    axial_segments: int = 8,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Create a finite annular horizontal cylinder without coaxial boolean cuts."""

    verts: list[tuple[float, float, float]] = []
    for axial_index in range(axial_segments + 1):
        axial = -cylinder.length / 2.0 + cylinder.length * axial_index / axial_segments
        for radius in (outer_radius, inner_radius):
            for index in range(vertices):
                theta = 2.0 * math.pi * index / vertices
                verts.append(cylinder.surface_point(axial, theta, radius=radius))

    def idx(axial_index: int, radius_loop: int, index: int) -> int:
        return axial_index * vertices * 2 + radius_loop * vertices + index % vertices

    faces: list[tuple[int, ...]] = []
    for axial_index in range(axial_segments):
        for index in range(vertices):
            faces.append((
                idx(axial_index, 0, index),
                idx(axial_index + 1, 0, index),
                idx(axial_index + 1, 0, index + 1),
                idx(axial_index, 0, index + 1),
            ))
            faces.append((
                idx(axial_index, 1, index + 1),
                idx(axial_index + 1, 1, index + 1),
                idx(axial_index + 1, 1, index),
                idx(axial_index, 1, index),
            ))

    for axial_index, reverse in ((0, True), (axial_segments, False)):
        for index in range(vertices):
            face = (
                idx(axial_index, 0, index),
                idx(axial_index, 0, index + 1),
                idx(axial_index, 1, index + 1),
                idx(axial_index, 1, index),
            )
            faces.append(tuple(reversed(face)) if reverse else face)

    return create_mesh_object(name, verts, faces, material)


def create_profile_prism_mesh(
    name: str,
    radius_fn: Callable[[float], float],
    z_min: float,
    z_max: float,
    segments: int = 128,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    verts: list[tuple[float, float, float]] = []
    for z in (z_min, z_max):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            radius = radius_fn(angle)
            verts.append((radius * math.cos(angle), radius * math.sin(angle), z))
    bottom_center = len(verts)
    verts.append((0.0, 0.0, z_min))
    top_center = len(verts)
    verts.append((0.0, 0.0, z_max))

    def idx(side: int, index: int) -> int:
        return side * segments + index % segments

    faces: list[tuple[int, ...]] = []
    for index in range(segments):
        faces.append((idx(0, index), idx(0, index + 1), idx(1, index + 1), idx(1, index)))
        faces.append((bottom_center, idx(0, index + 1), idx(0, index)))
        faces.append((top_center, idx(1, index), idx(1, index + 1)))
    return create_mesh_object(name, verts, faces, material)


def create_extruded_polygon_mesh(
    name: str,
    points: Iterable[tuple[float, float]],
    z_min: float,
    z_max: float,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    point_list = list(points)
    verts: list[tuple[float, float, float]] = [(x, y, z_min) for x, y in point_list]
    verts.extend((x, y, z_max) for x, y in point_list)
    count = len(point_list)
    faces: list[tuple[int, ...]] = []

    for index in range(count):
        faces.append((index, (index + 1) % count, count + (index + 1) % count, count + index))

    polygon = [[Vector((x, y, 0.0)) for x, y in point_list]]
    for tri in tessellate_polygon(polygon):
        faces.append(tuple(reversed(tri)))
        faces.append(tuple(count + vertex_index for vertex_index in tri))
    return create_mesh_object(name, verts, faces, material)


def apply_boolean(
    target: bpy.types.Object,
    tool: bpy.types.Object,
    operation: str,
    solver: str = "EXACT",
) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    modifier = target.modifiers.new(f"{operation.lower()}_{tool.name}", "BOOLEAN")
    modifier.operation = operation
    modifier.object = tool
    if hasattr(modifier, "solver"):
        modifier.solver = solver
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    target.select_set(False)
    target.data.update()
    return target


def delete_objects(objects: Iterable[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        if obj.name in bpy.data.objects:
            obj.select_set(True)
    bpy.ops.object.delete()


def create_polygon_ring_cap(
    name: str,
    outer_radius: float,
    inner_radius: float,
    z: float,
    sides: int,
    segments: int = 144,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    verts: list[tuple[float, float, float]] = []
    for radius_kind in ("outer", "inner"):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            if radius_kind == "outer":
                x, y = polygon_boundary_point(outer_radius, sides, angle)
            else:
                x, y = (inner_radius * math.cos(angle), inner_radius * math.sin(angle))
            verts.append((x, y, z))

    def outer(index: int) -> int:
        return index % segments

    def inner(index: int) -> int:
        return segments + index % segments

    faces = [(outer(index), outer(index + 1), inner(index + 1), inner(index)) for index in range(segments)]
    return create_mesh_object(name, verts, faces, material)


def create_faceted_wall_with_cylindrical_cuts(
    name: str,
    radius: float,
    z_min: float,
    z_max: float,
    sides: int,
    cutters: Iterable[HorizontalCylinder],
    u_steps: int = 34,
    z_steps: int = 80,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    cutter_list = list(cutters)
    panel_points = [
        (radius * math.cos(2.0 * math.pi * index / sides), radius * math.sin(2.0 * math.pi * index / sides))
        for index in range(sides)
    ]
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []

    def panel_point(p0: tuple[float, float], p1: tuple[float, float], u: float, z: float) -> tuple[float, float, float]:
        return (p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u, z)

    for panel_index in range(sides):
        p0 = panel_points[panel_index]
        p1 = panel_points[(panel_index + 1) % sides]
        for u_index in range(u_steps):
            u0 = u_index / u_steps
            u1 = (u_index + 1) / u_steps
            um = (u0 + u1) / 2.0
            for z_index in range(z_steps):
                z0 = z_min + (z_max - z_min) * z_index / z_steps
                z1 = z_min + (z_max - z_min) * (z_index + 1) / z_steps
                center = panel_point(p0, p1, um, (z0 + z1) / 2.0)
                if any(cutter.contains(center, pad=0.025) for cutter in cutter_list):
                    continue
                start = len(verts)
                verts.extend((
                    panel_point(p0, p1, u0, z0),
                    panel_point(p0, p1, u1, z0),
                    panel_point(p0, p1, u1, z1),
                    panel_point(p0, p1, u0, z1),
                ))
                faces.append((start, start + 1, start + 2, start + 3))
    return create_mesh_object(name, verts, faces, material)


def create_cylinder_wall_with_cylindrical_cuts(
    name: str,
    radius: float,
    z_min: float,
    z_max: float,
    cutters: Iterable[HorizontalCylinder],
    segments: int = 168,
    z_steps: int = 80,
    inward: bool = False,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    cutter_list = list(cutters)
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []

    def point(angle: float, z: float) -> tuple[float, float, float]:
        return (radius * math.cos(angle), radius * math.sin(angle), z)

    for angle_index in range(segments):
        a0 = 2.0 * math.pi * angle_index / segments
        a1 = 2.0 * math.pi * (angle_index + 1) / segments
        am = (a0 + a1) / 2.0
        for z_index in range(z_steps):
            z0 = z_min + (z_max - z_min) * z_index / z_steps
            z1 = z_min + (z_max - z_min) * (z_index + 1) / z_steps
            center = point(am, (z0 + z1) / 2.0)
            if any(cutter.contains(center, pad=0.025) for cutter in cutter_list):
                continue
            start = len(verts)
            verts.extend((point(a0, z0), point(a1, z0), point(a1, z1), point(a0, z1)))
            faces.append((start, start + 3, start + 2, start + 1) if inward else (start, start + 1, start + 2, start + 3))
    return create_mesh_object(name, verts, faces, material)


def create_cylinder_surface_patch(
    name: str,
    cutter: HorizontalCylinder,
    surface_radius: float,
    clip_fn: Callable[[tuple[float, float, float]], bool],
    axial_steps: int = 120,
    theta_steps: int = 54,
    material: bpy.types.Material | None = None,
    reverse: bool = False,
) -> bpy.types.Object:
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for axial_index in range(axial_steps):
        s0 = -cutter.length / 2.0 + cutter.length * axial_index / axial_steps
        s1 = -cutter.length / 2.0 + cutter.length * (axial_index + 1) / axial_steps
        sm = (s0 + s1) / 2.0
        for theta_index in range(theta_steps):
            t0 = 2.0 * math.pi * theta_index / theta_steps
            t1 = 2.0 * math.pi * (theta_index + 1) / theta_steps
            tm = (t0 + t1) / 2.0
            center = cutter.surface_point(sm, tm, surface_radius)
            if not clip_fn(center):
                continue
            start = len(verts)
            verts.extend((
                cutter.surface_point(s0, t0, surface_radius),
                cutter.surface_point(s1, t0, surface_radius),
                cutter.surface_point(s1, t1, surface_radius),
                cutter.surface_point(s0, t1, surface_radius),
            ))
            faces.append((start, start + 3, start + 2, start + 1) if reverse else (start, start + 1, start + 2, start + 3))
    return create_mesh_object(name, verts, faces, material)


def create_heightfield_surface_mesh(
    name: str,
    bounds: tuple[float, float, float, float],
    steps: tuple[int, int],
    z_fn: Callable[[float, float], float | None],
    material: bpy.types.Material | None = None,
    reverse: bool = False,
) -> bpy.types.Object:
    x_min, x_max, y_min, y_max = bounds
    x_steps, y_steps = steps
    verts: list[tuple[float, float, float]] = []
    vertex_grid: list[list[int | None]] = []

    for y_index in range(y_steps + 1):
        row: list[int | None] = []
        y = y_min + (y_max - y_min) * y_index / y_steps
        for x_index in range(x_steps + 1):
            x = x_min + (x_max - x_min) * x_index / x_steps
            z = z_fn(x, y)
            if z is None:
                row.append(None)
                continue
            row.append(len(verts))
            verts.append((x, y, z))
        vertex_grid.append(row)

    faces: list[tuple[int, int, int, int]] = []
    for y_index in range(y_steps):
        for x_index in range(x_steps):
            corners = (
                vertex_grid[y_index][x_index],
                vertex_grid[y_index][x_index + 1],
                vertex_grid[y_index + 1][x_index + 1],
                vertex_grid[y_index + 1][x_index],
            )
            if any(corner is None for corner in corners):
                continue
            face = tuple(corner for corner in corners if corner is not None)
            faces.append(tuple(reversed(face)) if reverse else face)

    return create_mesh_object(name, verts, faces, material)


def create_polar_heightfield_surface_mesh(
    name: str,
    radius_min: float,
    radius_max: float,
    radial_steps: int,
    angle_steps: int,
    z_fn: Callable[[float, float, float, float], float | None],
    material: bpy.types.Material | None = None,
    reverse: bool = False,
) -> bpy.types.Object:
    """Create a polar-grid visibility surface with optional masked cells."""

    verts: list[tuple[float, float, float]] = []
    vertex_grid: list[list[int | None]] = []

    for radial_index in range(radial_steps + 1):
        radius = radius_min + (radius_max - radius_min) * radial_index / radial_steps
        row: list[int | None] = []
        for angle_index in range(angle_steps):
            angle = 2.0 * math.pi * angle_index / angle_steps
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = z_fn(radius, angle, x, y)
            if z is None:
                row.append(None)
                continue
            row.append(len(verts))
            verts.append((x, y, z))
        vertex_grid.append(row)

    faces: list[tuple[int, int, int, int]] = []
    for radial_index in range(radial_steps):
        for angle_index in range(angle_steps):
            corners = (
                vertex_grid[radial_index][angle_index],
                vertex_grid[radial_index + 1][angle_index],
                vertex_grid[radial_index + 1][(angle_index + 1) % angle_steps],
                vertex_grid[radial_index][(angle_index + 1) % angle_steps],
            )
            if any(corner is None for corner in corners):
                continue
            face = tuple(corner for corner in corners if corner is not None)
            faces.append(tuple(reversed(face)) if reverse else face)

    return create_mesh_object(name, verts, faces, material)


def create_heightfield_solid_mesh(
    name: str,
    bounds: tuple[float, float, float, float],
    steps: tuple[int, int],
    z_top_fn: Callable[[float, float], float | None],
    z_bottom_fn: Callable[[float, float], float],
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    x_min, x_max, y_min, y_max = bounds
    x_steps, y_steps = steps
    verts: list[tuple[float, float, float]] = []
    top_grid: list[list[int | None]] = []
    bottom_grid: list[list[int | None]] = []

    for y_index in range(y_steps + 1):
        top_row: list[int | None] = []
        bottom_row: list[int | None] = []
        y = y_min + (y_max - y_min) * y_index / y_steps
        for x_index in range(x_steps + 1):
            x = x_min + (x_max - x_min) * x_index / x_steps
            z_top = z_top_fn(x, y)
            z_bottom = z_bottom_fn(x, y)
            if z_top is None or z_top <= z_bottom + 0.015:
                top_row.append(None)
                bottom_row.append(None)
                continue
            bottom_row.append(len(verts))
            verts.append((x, y, z_bottom))
            top_row.append(len(verts))
            verts.append((x, y, z_top))
        bottom_grid.append(bottom_row)
        top_grid.append(top_row)

    faces: list[tuple[int, ...]] = []

    def valid(y_index: int, x_index: int) -> bool:
        return top_grid[y_index][x_index] is not None and bottom_grid[y_index][x_index] is not None

    for y_index in range(y_steps):
        for x_index in range(x_steps):
            corners = (
                (y_index, x_index),
                (y_index, x_index + 1),
                (y_index + 1, x_index + 1),
                (y_index + 1, x_index),
            )
            if not all(valid(y, x) for y, x in corners):
                continue
            top_face = tuple(top_grid[y][x] for y, x in corners)
            bottom_face = tuple(bottom_grid[y][x] for y, x in reversed(corners))
            faces.append(tuple(index for index in top_face if index is not None))
            faces.append(tuple(index for index in bottom_face if index is not None))

    for y_index in range(y_steps + 1):
        for x_index in range(x_steps):
            if not (valid(y_index, x_index) and valid(y_index, x_index + 1)):
                continue
            below_open = y_index == 0 or not (valid(y_index - 1, x_index) and valid(y_index - 1, x_index + 1))
            above_open = y_index == y_steps or not (valid(y_index + 1, x_index) and valid(y_index + 1, x_index + 1))
            if below_open:
                faces.append((
                    bottom_grid[y_index][x_index],
                    bottom_grid[y_index][x_index + 1],
                    top_grid[y_index][x_index + 1],
                    top_grid[y_index][x_index],
                ))
            if above_open:
                faces.append((
                    bottom_grid[y_index][x_index + 1],
                    bottom_grid[y_index][x_index],
                    top_grid[y_index][x_index],
                    top_grid[y_index][x_index + 1],
                ))

    for y_index in range(y_steps):
        for x_index in range(x_steps + 1):
            if not (valid(y_index, x_index) and valid(y_index + 1, x_index)):
                continue
            left_open = x_index == 0 or not (valid(y_index, x_index - 1) and valid(y_index + 1, x_index - 1))
            right_open = x_index == x_steps or not (valid(y_index, x_index + 1) and valid(y_index + 1, x_index + 1))
            if left_open:
                faces.append((
                    bottom_grid[y_index + 1][x_index],
                    bottom_grid[y_index][x_index],
                    top_grid[y_index][x_index],
                    top_grid[y_index + 1][x_index],
                ))
            if right_open:
                faces.append((
                    bottom_grid[y_index][x_index],
                    bottom_grid[y_index + 1][x_index],
                    top_grid[y_index + 1][x_index],
                    top_grid[y_index][x_index],
                ))

    clean_faces = [tuple(index for index in face if index is not None) for face in faces]
    return create_mesh_object(name, verts, clean_faces, material)


def create_voxel_mask_mesh(
    name: str,
    bounds: tuple[float, float, float, float, float, float],
    steps: tuple[int, int, int],
    solid_fn: Callable[[float, float, float], bool],
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    nx, ny, nz = steps
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    solid = [False] * (nx * ny * nz)

    def cell_index(i: int, j: int, k: int) -> int:
        return (k * ny + j) * nx + i

    def is_solid(i: int, j: int, k: int) -> bool:
        if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
            return False
        return solid[cell_index(i, j, k)]

    for k in range(nz):
        z = z_min + (k + 0.5) * dz
        for j in range(ny):
            y = y_min + (j + 0.5) * dy
            for i in range(nx):
                x = x_min + (i + 0.5) * dx
                solid[cell_index(i, j, k)] = solid_fn(x, y, z)

    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    vertex_lookup: dict[tuple[int, int, int], int] = {}

    def vertex(i: int, j: int, k: int) -> int:
        key = (i, j, k)
        if key not in vertex_lookup:
            vertex_lookup[key] = len(verts)
            verts.append((x_min + i * dx, y_min + j * dy, z_min + k * dz))
        return vertex_lookup[key]

    face_specs = (
        ((-1, 0, 0), lambda i, j, k: (vertex(i, j, k), vertex(i, j + 1, k), vertex(i, j + 1, k + 1), vertex(i, j, k + 1))),
        ((1, 0, 0), lambda i, j, k: (vertex(i + 1, j, k), vertex(i + 1, j, k + 1), vertex(i + 1, j + 1, k + 1), vertex(i + 1, j + 1, k))),
        ((0, -1, 0), lambda i, j, k: (vertex(i, j, k), vertex(i, j, k + 1), vertex(i + 1, j, k + 1), vertex(i + 1, j, k))),
        ((0, 1, 0), lambda i, j, k: (vertex(i, j + 1, k), vertex(i + 1, j + 1, k), vertex(i + 1, j + 1, k + 1), vertex(i, j + 1, k + 1))),
        ((0, 0, -1), lambda i, j, k: (vertex(i, j, k), vertex(i + 1, j, k), vertex(i + 1, j + 1, k), vertex(i, j + 1, k))),
        ((0, 0, 1), lambda i, j, k: (vertex(i, j, k + 1), vertex(i, j + 1, k + 1), vertex(i + 1, j + 1, k + 1), vertex(i + 1, j, k + 1))),
    )

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if not is_solid(i, j, k):
                    continue
                for (di, dj, dk), corners_fn in face_specs:
                    if not is_solid(i + di, j + dj, k + dk):
                        faces.append(corners_fn(i, j, k))
    return create_mesh_object(name, verts, faces, material)


def create_tetrahedron(
    name: str,
    points: Iterable[tuple[float, float, float]],
    faces: Iterable[tuple[int, int, int]],
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    return create_mesh_object(name, points, faces, material)


# ============================================================
# Procedural assembly
# ============================================================

"""Readable assembly of the faceted hob knob source model."""





HEIGHT = 20.0
SHAFT_DIAMETER = 6.1
D_WIDTH = 4.7
OUTER_DIAMETER = 36.0
WALL_THICKNESS = 3.0
INNER_DIAMETER = 14.0
SHAFT_LENGTH = 11.0
PLUNGE_DEPTH = 3.0
CLEARANCE_HEIGHT = 8.0
NO_SIDES = 9
TOP_CHAMFER = 1.5
POINT_SIZE = 5.0
POINT_ANGLE = 90.0
NUMBER_OF_FEATURES = 3
OFFSET_ANGLE = -30.0
CLEARANCE = 0.1
STIFFENER_THICKNESS = 2.0

OUTER_RADIUS = OUTER_DIAMETER / 2.0
INNER_RADIUS = INNER_DIAMETER / 2.0
HOLLOW_RADIUS = OUTER_RADIUS - WALL_THICKNESS
BORE_RADIUS = (SHAFT_DIAMETER + CLEARANCE) / 2.0
D_FLAT_OFFSET = (D_WIDTH + CLEARANCE) - SHAFT_DIAMETER / 2.0
SCALLOP_RADIUS = HEIGHT / 8.0
SCALLOP_WALL_RADIUS = SCALLOP_RADIUS + WALL_THICKNESS
GREEN_BODY_RADIUS = SCALLOP_WALL_RADIUS
SCALLOP_CENTER_RADIUS = OUTER_RADIUS + SCALLOP_RADIUS - WALL_THICKNESS
SCALLOP_Z = HEIGHT / 3.0
GREEN_TOP_COVER_Z = HEIGHT - CLEARANCE_HEIGHT + 0.10
BOOLEAN_EPSILON = 0.04
PROJECTED_CRESCENT_SIDE_MIN = -SCALLOP_WALL_RADIUS - 1.10
PROJECTED_CRESCENT_SIDE_MAX = -SCALLOP_RADIUS + 0.75


def d_bore_profile_radius(angle: float) -> float:
    """Keep the D-hole orientation from the earlier matching iterations."""
    return d_shape_radius(angle + math.pi, BORE_RADIUS, D_FLAT_OFFSET)


def d_bore_cut_radius(angle: float) -> float:
    return d_bore_profile_radius(angle) + 0.08


def scallop_cutters(radius: float) -> list[HorizontalCylinder]:
    return [
        HorizontalCylinder(
            name=f"scallop_{index}",
            angle_deg=OFFSET_ANGLE + (360.0 / NUMBER_OF_FEATURES) * index,
            center_radius=SCALLOP_CENTER_RADIUS,
            z_center=SCALLOP_Z,
            radius=radius,
            length=OUTER_DIAMETER,
        )
        for index in range(NUMBER_OF_FEATURES)
    ]


def cutter_offsets(cutter: HorizontalCylinder, x: float, y: float) -> tuple[float, float]:
    axis, lateral, center = cutter.frame()
    dx = x - center[0]
    dy = y - center[1]
    return dx * axis[0] + dy * axis[1], dx * lateral[0] + dy * lateral[1]


def upper_bowl_point_is_visible(x: float, y: float, radial_pad: float = 0.0) -> bool:
    radius = math.hypot(x, y)
    return (
        INNER_RADIUS + 0.16 <= radius <= HOLLOW_RADIUS + radial_pad
        and inside_regular_polygon(x, y, OUTER_RADIUS, NO_SIDES, margin=-0.08)
    )


def support_lane_blocks_upper_overlay(x: float, y: float, radius: float) -> bool:
    """Keep the cross supports yellow while letting the outer tabs be curved over."""
    lane_half_width = STIFFENER_THICKNESS / 2.0 + 0.10
    return (abs(x) <= lane_half_width or abs(y) <= lane_half_width) and radius < HOLLOW_RADIUS - 0.85


def large_cylinder_envelope_z(cutters: list[HorizontalCylinder], x: float, y: float) -> float | None:
    best_z: float | None = None
    for cutter in cutters:
        axial, side = cutter_offsets(cutter, x, y)
        if abs(axial) > cutter.length / 2.0 - 0.10:
            continue
        remainder = cutter.radius * cutter.radius - side * side
        if remainder <= 0.0:
            continue
        z = cutter.z_center + math.sqrt(remainder)
        if best_z is None or z > best_z:
            best_z = z
    return best_z


def projected_crescent_width(side: float) -> float:
    span = PROJECTED_CRESCENT_SIDE_MAX - PROJECTED_CRESCENT_SIDE_MIN
    u = min(max((side - PROJECTED_CRESCENT_SIDE_MIN) / span, 0.0), 1.0)
    return 4.15 + 3.05 * math.sin(math.pi * u)


def projected_crescent_hit(
    cutters: list[HorizontalCylinder],
    x: float,
    y: float,
    radial_pad: float = 0.18,
) -> tuple[HorizontalCylinder, float, float, float] | None:
    radius = math.hypot(x, y)
    if radius < HOLLOW_RADIUS - 5.00 or radius > HOLLOW_RADIUS + radial_pad:
        return None
    if not inside_regular_polygon(x, y, OUTER_RADIUS, NO_SIDES, margin=-0.08):
        return None

    best: tuple[float, HorizontalCylinder, float, float, float] | None = None
    for cutter in cutters:
        axial, side = cutter_offsets(cutter, x, y)
        if not (PROJECTED_CRESCENT_SIDE_MIN <= side <= PROJECTED_CRESCENT_SIDE_MAX):
            continue
        width = projected_crescent_width(side)
        if abs(axial) > width:
            continue
        score = abs(axial) / width + 0.08 * abs(side - (PROJECTED_CRESCENT_SIDE_MIN + PROJECTED_CRESCENT_SIDE_MAX) / 2.0)
        if best is None or score < best[0]:
            best = (score, cutter, axial, side, width)
    if best is None:
        return None
    _score, cutter, axial, side, width = best
    return cutter, axial, side, width


def projected_crescent_floor_z(axial: float, side: float, width: float) -> float:
    span = PROJECTED_CRESCENT_SIDE_MAX - PROJECTED_CRESCENT_SIDE_MIN
    u = min(max((side - PROJECTED_CRESCENT_SIDE_MIN) / span, 0.0), 1.0)
    effective_side = -SCALLOP_RADIUS + 0.18 + u * (2.0 * (SCALLOP_RADIUS - 0.18))
    cylinder_rise = math.sqrt(max(SCALLOP_RADIUS * SCALLOP_RADIUS - effective_side * effective_side, 0.0))
    edge_falloff = 0.38 * (abs(axial) / max(width, 0.01)) ** 2
    return SCALLOP_Z + cylinder_rise - edge_falloff - 0.28


def projected_green_upper_field_z(
    large_cutters: list[HorizontalCylinder],
    small_cutters: list[HorizontalCylinder],
    radius: float,
    x: float,
    y: float,
) -> float | None:
    if not upper_bowl_point_is_visible(x, y, radial_pad=0.08):
        return None
    if support_lane_blocks_upper_overlay(x, y, radius):
        return None
    if projected_crescent_hit(small_cutters, x, y, radial_pad=0.10) is not None:
        return None

    z = large_cylinder_envelope_z(large_cutters, x, y)
    if z is None:
        if radius < INNER_RADIUS + 0.24:
            return None
        # From the top-oblique camera the low large-cylinder wall projects
        # across the cleared bowl. This overlay keeps that source-order green
        # body visually dominant without reopening the accepted exterior halos.
        t = min(max((radius - (INNER_RADIUS + 0.24)) / (HOLLOW_RADIUS - INNER_RADIUS), 0.0), 1.0)
        z = GREEN_TOP_COVER_Z - 0.04 + 0.11 * t
    return min(max(z, GREEN_TOP_COVER_Z - 0.04), GREEN_TOP_COVER_Z + 0.30) + 0.075


def apply_scallop_difference(
    obj: bpy.types.Object,
    prefix: str,
    radius: float = SCALLOP_RADIUS,
    vertices: int = 144,
) -> None:
    tools = [
        create_horizontal_cylinder_mesh(f"{prefix}_small_scallop_cut_{index}", cutter, vertices=vertices)
        for index, cutter in enumerate(scallop_cutters(radius))
    ]
    for tool in tools:
        apply_boolean(obj, tool, "DIFFERENCE")
    delete_objects(tools)


def build_faceted_outer_wall(material: bpy.types.Material) -> list[bpy.types.Object]:
    """Legacy decomposed shell builder kept as a readable fallback."""
    chamfer_outer = lambda angle: polygon_boundary_point(OUTER_RADIUS - TOP_CHAMFER, NO_SIDES, angle)
    wall_outer = lambda angle: polygon_boundary_point(OUTER_RADIUS, NO_SIDES, angle)
    chamfer = create_radial_ring_mesh(
        "bottom_nine_sided_chamfer_ring",
        outer_radius_fn=lambda angle: math.hypot(*chamfer_outer(angle)),
        inner_radius_fn=lambda _angle: HOLLOW_RADIUS,
        z_min=0.0,
        z_max=TOP_CHAMFER,
        segments=144,
        material=material,
    )
    for vertex in chamfer.data.vertices:
        angle = math.atan2(vertex.co.y, vertex.co.x)
        bottom_radius = math.hypot(*chamfer_outer(angle))
        top_radius = math.hypot(*wall_outer(angle))
        if math.hypot(vertex.co.x, vertex.co.y) > HOLLOW_RADIUS + 0.01:
            t = vertex.co.z / TOP_CHAMFER
            target_radius = bottom_radius + (top_radius - bottom_radius) * t
            vertex.co.x = target_radius * math.cos(angle)
            vertex.co.y = target_radius * math.sin(angle)
    chamfer.data.update()

    shell = create_radial_ring_mesh(
        "closed_nine_sided_shell_with_boolean_scallop_cuts",
        outer_radius_fn=lambda angle: math.hypot(*wall_outer(angle)),
        inner_radius_fn=lambda _angle: HOLLOW_RADIUS,
        z_min=TOP_CHAMFER,
        z_max=HEIGHT,
        segments=216,
        material=material,
    )
    apply_scallop_difference(shell, "boolean_tool_shell", radius=SCALLOP_RADIUS + 0.05, vertices=160)
    assign_material(shell, material)
    shade_flat(shell)
    return [chamfer, shell]


def build_center_column_and_collar(material: bpy.types.Material) -> list[bpy.types.Object]:
    """Legacy decomposed collar builder kept as a readable fallback."""
    central_core = create_profile_wall_mesh(
        "central_core_outer_wall_below_raised_collar",
        radius_fn=lambda _angle: INNER_RADIUS,
        z_min=WALL_THICKNESS,
        z_max=HEIGHT,
        segments=192,
        material=material,
    )
    collar = create_radial_ring_mesh(
        "raised_collar_cut_by_shared_crisp_d_profile",
        outer_radius_fn=lambda _angle: INNER_RADIUS,
        inner_radius_fn=d_bore_cut_radius,
        z_min=HEIGHT,
        z_max=HEIGHT + PLUNGE_DEPTH,
        segments=224,
        material=material,
    )
    bore_wall = create_profile_wall_mesh(
        "shared_yellow_d_profile_wall_from_shaft_depth_through_collar",
        radius_fn=d_bore_cut_radius,
        z_min=HEIGHT - SHAFT_LENGTH,
        z_max=HEIGHT + PLUNGE_DEPTH,
        segments=224,
        material=material,
    )
    for obj in (central_core, collar, bore_wall):
        shade_flat(obj)
    return [central_core, collar, bore_wall]


def plus_support_profile() -> list[tuple[float, float]]:
    support_extent = OUTER_DIAMETER - WALL_THICKNESS
    half_extent = support_extent / 2.0
    half_thickness = STIFFENER_THICKNESS / 2.0
    return [
        (-half_extent, -half_thickness),
        (-half_thickness, -half_thickness),
        (-half_thickness, -half_extent),
        (half_thickness, -half_extent),
        (half_thickness, -half_thickness),
        (half_extent, -half_thickness),
        (half_extent, half_thickness),
        (half_thickness, half_thickness),
        (half_thickness, half_extent),
        (-half_thickness, half_extent),
        (-half_thickness, half_thickness),
        (-half_extent, half_thickness),
    ]


def build_internal_supports(material: bpy.types.Material) -> list[bpy.types.Object]:
    """Legacy support-only builder kept as a readable fallback."""

    support = create_extruded_polygon_mesh(
        "two_centered_support_cubes_cut_by_shared_d_bore_and_clearance",
        plus_support_profile(),
        z_min=TOP_CHAMFER,
        z_max=HEIGHT,
        material=material,
    )
    d_bore = create_profile_prism_mesh(
        "boolean_tool_support_d_shaft_hole",
        radius_fn=lambda angle: d_bore_profile_radius(angle) + 0.08,
        z_min=TOP_CHAMFER - BOOLEAN_EPSILON,
        z_max=HEIGHT + PLUNGE_DEPTH + 2.0 + BOOLEAN_EPSILON,
        segments=192,
    )
    clearance = create_radial_ring_mesh(
        "boolean_tool_support_upper_internal_clearance",
        outer_radius_fn=lambda _angle: HOLLOW_RADIUS + BOOLEAN_EPSILON,
        inner_radius_fn=lambda _angle: INNER_RADIUS - BOOLEAN_EPSILON,
        z_min=HEIGHT - CLEARANCE_HEIGHT - BOOLEAN_EPSILON,
        z_max=HEIGHT * 2.0 + BOOLEAN_EPSILON,
        segments=160,
    )
    apply_boolean(support, d_bore, "DIFFERENCE")
    apply_boolean(support, clearance, "DIFFERENCE")
    delete_objects([d_bore, clearance])
    apply_scallop_difference(support, "boolean_tool_support", radius=SCALLOP_RADIUS + 1.35, vertices=160)
    assign_material(support, material)
    shade_flat(support)
    return [support]


def build_d_bore_cut_wall(material: bpy.types.Material) -> bpy.types.Object:
    """Same-material side wall for the shared D-shaped shaft subtraction."""
    cut_wall = create_profile_wall_mesh(
        "same_material_d_profile_shaft_cut_wall",
        radius_fn=lambda angle: d_bore_profile_radius(angle) + 0.02,
        z_min=WALL_THICKNESS + 0.02,
        z_max=HEIGHT + PLUNGE_DEPTH - 0.02,
        segments=224,
        material=material,
    )
    shade_smooth(cut_wall)
    return cut_wall


def build_closed_scallop_wall_volumes(material: bpy.types.Material) -> list[bpy.types.Object]:
    """Build visible surfaces of the independent `scollop_wall() - scollop()` body."""

    upper_surfaces = build_visible_green_upper_scollop_wall_surfaces(material)
    inner_wall = build_green_inner_shell_wall(material)
    slot_surfaces = build_clean_green_slot_surfaces(material)
    return [*upper_surfaces, inner_wall, *slot_surfaces]


def build_visible_green_upper_scollop_wall_surfaces(material: bpy.types.Material) -> list[bpy.types.Object]:
    """Visible upper portion of the source `scollop_wall() - scollop()` body."""

    large_cutters = scallop_cutters(SCALLOP_WALL_RADIUS)
    small_cutters = scallop_cutters(SCALLOP_RADIUS)

    # Build the green source-order body as the actual OpenSCAD sequence:
    # union the large horizontal wall cylinders, clip them to the nine-sided
    # shell, then subtract the smaller coaxial scollop() cylinders.
    source_body = create_horizontal_cylinder_mesh(
        "source_order_green_upper_scollop_wall_large_cylinder_body",
        large_cutters[0],
        vertices=192,
        material=material,
    )
    large_tools = [
        create_horizontal_cylinder_mesh(
            f"boolean_tool_green_upper_large_scollop_union_{index}",
            cutter,
            vertices=192,
        )
        for index, cutter in enumerate(large_cutters[1:], start=1)
    ]
    for tool in large_tools:
        apply_boolean(source_body, tool, "UNION")

    shell_clip = create_cone_frustum_mesh(
        "boolean_tool_green_upper_nine_sided_shell_clip",
        radius_bottom=OUTER_RADIUS,
        radius_top=OUTER_RADIUS,
        z_min=TOP_CHAMFER,
        z_max=HEIGHT,
        vertices=NO_SIDES,
    )
    apply_boolean(source_body, shell_clip, "INTERSECT")
    top_visibility_clip = create_radial_ring_mesh(
        "boolean_tool_green_upper_top_bowl_visibility_clip",
        outer_radius_fn=lambda _angle: HOLLOW_RADIUS + 0.16,
        inner_radius_fn=lambda _angle: INNER_RADIUS + 0.02,
        z_min=SCALLOP_Z - SCALLOP_RADIUS - 0.12,
        z_max=HEIGHT + 0.08,
        segments=224,
    )
    apply_boolean(source_body, top_visibility_clip, "INTERSECT")

    small_tools = [
        create_horizontal_cylinder_mesh(f"boolean_tool_green_upper_small_scollop_removal_{index}", cutter, vertices=192)
        for index, cutter in enumerate(small_cutters)
    ]
    for tool in small_tools:
        apply_boolean(source_body, tool, "DIFFERENCE")

    delete_objects([*large_tools, shell_clip, top_visibility_clip, *small_tools])
    assign_material(source_body, material)
    shade_flat(source_body)

    def in_large_body(point: tuple[float, float, float], pad: float = 0.0) -> bool:
        return any(cutter.contains(point, pad=pad) for cutter in large_cutters)

    upper_green_field = create_polar_heightfield_surface_mesh(
        "dominant_projected_green_large_cylinder_upper_field",
        radius_min=INNER_RADIUS + 0.22,
        radius_max=HOLLOW_RADIUS + 0.08,
        radial_steps=180,
        angle_steps=432,
        z_fn=lambda radius, _angle, x, y: projected_green_upper_field_z(large_cutters, small_cutters, radius, x, y),
        material=material,
    )
    weld_vertices(upper_green_field, distance=0.0002)
    assign_material(upper_green_field, material)
    shade_smooth(upper_green_field)

    cut_wall_material = make_material("source_preview_darker_green_cylinder_cut_walls", (0.15, 0.28, 0.025, 1.0))
    visible_surfaces: list[bpy.types.Object] = []
    for index, cutter in enumerate(large_cutters):
        other_large_cutters = [other for other in large_cutters if other is not cutter]

        def large_surface_clip(point: tuple[float, float, float]) -> bool:
            x, y, z = point
            radius = math.hypot(x, y)
            return (
                SCALLOP_Z + SCALLOP_RADIUS - 0.18 <= z <= SCALLOP_Z + SCALLOP_WALL_RADIUS + 0.05
                and HOLLOW_RADIUS - 2.90 <= radius <= HOLLOW_RADIUS + 0.16
                and inside_regular_polygon(x, y, OUTER_RADIUS, NO_SIDES, margin=-0.06)
                and not any(other.contains(point, pad=-0.05) for other in other_large_cutters)
            )

        large_surface = create_cylinder_surface_patch(
            f"true_large_cylinder_green_upper_surface_{index}",
            cutter,
            surface_radius=SCALLOP_WALL_RADIUS + 0.018,
            clip_fn=large_surface_clip,
            axial_steps=320,
            theta_steps=128,
            material=material,
        )
        weld_vertices(large_surface, distance=0.0002)
        assign_material(large_surface, material)
        shade_smooth(large_surface)
        visible_surfaces.append(large_surface)

    for index, cutter in enumerate(small_cutters):

        def cavity_clip(point: tuple[float, float, float]) -> bool:
            x, y, z = point
            hit = projected_crescent_hit(small_cutters, x, y, radial_pad=0.16)
            return (
                hit is not None
                and hit[0] is cutter
                and SCALLOP_Z - SCALLOP_RADIUS + 0.02 <= z <= GREEN_TOP_COVER_Z + 0.24
                and upper_bowl_point_is_visible(x, y, radial_pad=0.14)
                and in_large_body(point, pad=0.22)
            )

        wall_surface = create_cylinder_surface_patch(
            f"deep_projected_small_cylinder_green_cut_wall_{index}",
            cutter,
            surface_radius=SCALLOP_RADIUS - 0.06,
            clip_fn=cavity_clip,
            axial_steps=320,
            theta_steps=128,
            material=cut_wall_material,
            reverse=True,
        )
        weld_vertices(wall_surface, distance=0.0002)
        assign_material(wall_surface, cut_wall_material)
        shade_smooth(wall_surface)
        visible_surfaces.append(wall_surface)

    vertical_edges = build_projected_crescent_boundary_walls(cut_wall_material)

    return [source_body, upper_green_field, *visible_surfaces, *vertical_edges]


def build_projected_crescent_boundary_walls(material: bpy.types.Material) -> list[bpy.types.Object]:
    """Dark curved side walls that make the small-cylinder interruptions read as hard cuts."""

    small_cutters = scallop_cutters(SCALLOP_RADIUS)
    large_cutters = scallop_cutters(SCALLOP_WALL_RADIUS)
    walls: list[bpy.types.Object] = []

    for index, cutter in enumerate(small_cutters):
        axis, lateral, center = cutter.frame()
        for side_sign, side_label in ((-1.0, "left"), (1.0, "right")):
            verts: list[tuple[float, float, float]] = []
            faces: list[tuple[int, int, int, int]] = []
            side_steps = 44
            for side_index in range(side_steps + 1):
                side_t = side_index / side_steps
                side = PROJECTED_CRESCENT_SIDE_MIN + (
                    PROJECTED_CRESCENT_SIDE_MAX - PROJECTED_CRESCENT_SIDE_MIN
                ) * side_t
                width = projected_crescent_width(side)
                axial = side_sign * width
                x = center[0] + axis[0] * axial + lateral[0] * side
                y = center[1] + axis[1] * axial + lateral[1] * side
                radius = math.hypot(x, y)
                if not upper_bowl_point_is_visible(x, y, radial_pad=0.12):
                    verts.extend(((x, y, SCALLOP_Z), (x, y, SCALLOP_Z)))
                    continue
                floor_z = projected_crescent_floor_z(axial, side, width)
                top_z = projected_green_upper_field_z(large_cutters, small_cutters, radius, x, y)
                if top_z is None:
                    top_z = GREEN_TOP_COVER_Z + 0.05
                top_z = max(top_z, floor_z + 0.32)
                verts.extend(((x, y, floor_z), (x, y, top_z)))

            for side_index in range(side_steps):
                start = side_index * 2
                faces.append((start, start + 2, start + 3, start + 1))

            wall = create_mesh_object(
                f"hard_dark_projected_crescent_{index}_{side_label}_cut_boundary",
                verts,
                faces,
                material=material,
            )
            weld_vertices(wall, distance=0.0002)
            assign_material(wall, material)
            shade_smooth(wall)
            walls.append(wall)

    return walls


def build_yellow_upper_crescent_interruption_surfaces(material: bpy.types.Material) -> list[bpy.types.Object]:
    """Lower yellow crescent floors visible through the upper scallop-wall interruptions."""

    small_cutters = scallop_cutters(SCALLOP_RADIUS)

    floors: list[bpy.types.Object] = []
    for index, cutter in enumerate(small_cutters):
        axis, lateral, center = cutter.frame()
        side_steps = 68
        axial_steps = 112
        verts: list[tuple[float, float, float]] = []
        vertex_grid: list[list[int | None]] = []

        for side_index in range(side_steps + 1):
            side = PROJECTED_CRESCENT_SIDE_MIN + (
                PROJECTED_CRESCENT_SIDE_MAX - PROJECTED_CRESCENT_SIDE_MIN
            ) * side_index / side_steps
            width = projected_crescent_width(side)
            row: list[int | None] = []
            for axial_index in range(axial_steps + 1):
                axial = -width + 2.0 * width * axial_index / axial_steps
                x = center[0] + axis[0] * axial + lateral[0] * side
                y = center[1] + axis[1] * axial + lateral[1] * side
                radius = math.hypot(x, y)
                hit = projected_crescent_hit(small_cutters, x, y, radial_pad=0.12)
                if (
                    hit is None
                    or hit[0] is not cutter
                    or not upper_bowl_point_is_visible(x, y, radial_pad=0.12)
                    or support_lane_blocks_upper_overlay(x, y, radius)
                ):
                    row.append(None)
                    continue
                z = projected_crescent_floor_z(axial, side, width)
                row.append(len(verts))
                verts.append((x, y, z))
            vertex_grid.append(row)

        faces: list[tuple[int, int, int, int]] = []
        for side_index in range(side_steps):
            for axial_index in range(axial_steps):
                corners = (
                    vertex_grid[side_index][axial_index],
                    vertex_grid[side_index][axial_index + 1],
                    vertex_grid[side_index + 1][axial_index + 1],
                    vertex_grid[side_index + 1][axial_index],
                )
                if any(corner is None for corner in corners):
                    continue
                faces.append(tuple(corner for corner in corners if corner is not None))

        floor = create_mesh_object(
            f"lower_yellow_projected_crescent_floor_{index}",
            verts,
            faces,
            material=material,
        )
        weld_vertices(floor, distance=0.0002)
        assign_material(floor, material)
        shade_smooth(floor)
        floors.append(floor)

    return floors


def build_green_inner_shell_wall(material: bpy.types.Material) -> bpy.types.Object:
    """Continuous green wall visible behind the upper yellow rim."""

    wall = create_profile_wall_mesh(
        "continuous_green_inner_shell_wall_at_upper_clearance",
        radius_fn=lambda _angle: HOLLOW_RADIUS + 0.025,
        z_min=GREEN_TOP_COVER_Z - 0.10,
        z_max=HEIGHT - 0.04,
        segments=216,
        material=material,
    )
    shade_smooth(wall)
    return wall


def build_clean_green_slot_surfaces(material: bpy.types.Material) -> list[bpy.types.Object]:
    """Overlay the source small-cylinder cut surfaces so side openings stay round."""

    large_cutters = scallop_cutters(SCALLOP_WALL_RADIUS)

    def inside_large_wall(point: tuple[float, float, float]) -> bool:
        return any(cutter.contains(point, pad=0.10) for cutter in large_cutters)

    surfaces: list[bpy.types.Object] = []
    for index, cutter in enumerate(scallop_cutters(SCALLOP_RADIUS)):
        axis, lateral, center = cutter.frame()

        def slot_clip(point: tuple[float, float, float]) -> bool:
            x, y, z = point
            radial = math.hypot(x, y)
            angle = math.atan2(y, x)
            outer_x, outer_y = polygon_boundary_point(OUTER_RADIUS, NO_SIDES, angle)
            outer_limit = math.hypot(outer_x, outer_y)
            dx = x - center[0]
            dy = y - center[1]
            axial = dx * axis[0] + dy * axis[1]
            side = dx * lateral[0] + dy * lateral[1]
            return (
                SCALLOP_Z - SCALLOP_RADIUS - 0.02 <= z <= SCALLOP_Z + SCALLOP_RADIUS + 0.02
                and HOLLOW_RADIUS - 0.42 <= radial <= outer_limit - 0.04
                and inside_regular_polygon(x, y, OUTER_RADIUS, NO_SIDES, margin=-0.04)
                and abs(axial) <= cutter.length / 2.0 - 0.15
                and -SCALLOP_RADIUS - 0.03 <= side <= 0.62
                and inside_large_wall(point)
            )

        surface = create_cylinder_surface_patch(
            f"clean_green_horizontal_scollop_slot_surface_{index}",
            cutter,
            surface_radius=SCALLOP_RADIUS - 0.10,
            clip_fn=slot_clip,
            axial_steps=260,
            theta_steps=96,
            material=material,
            reverse=True,
        )
        weld_vertices(surface, distance=0.0002)
        assign_material(surface, material)
        shade_smooth(surface)
        surfaces.append(surface)
    return surfaces


def build_source_order_yellow_knob_body(material: bpy.types.Material) -> bpy.types.Object:
    """Follow the SCAD `knob()` boolean order as one shared yellow body."""

    body = create_cone_frustum_mesh(
        "source_order_yellow_knob_union_cut_by_shared_booleans",
        radius_bottom=OUTER_RADIUS,
        radius_top=OUTER_RADIUS,
        z_min=TOP_CHAMFER,
        z_max=HEIGHT,
        vertices=NO_SIDES,
        material=material,
    )
    chamfer = create_cone_frustum_mesh(
        "boolean_tool_bottom_chamfer_frustum",
        radius_bottom=OUTER_RADIUS - TOP_CHAMFER,
        radius_top=OUTER_RADIUS,
        z_min=0.0,
        z_max=TOP_CHAMFER,
        vertices=NO_SIDES,
    )
    collar = create_cone_frustum_mesh(
        "boolean_tool_raised_center_collar",
        radius_bottom=INNER_RADIUS,
        radius_top=INNER_RADIUS,
        z_min=HEIGHT,
        z_max=HEIGHT + PLUNGE_DEPTH,
        vertices=192,
    )
    for tool in (chamfer, collar):
        apply_boolean(body, tool, "UNION")

    hollow = create_radial_ring_mesh(
        "boolean_tool_hollow_torus_annulus",
        outer_radius_fn=lambda _angle: HOLLOW_RADIUS,
        inner_radius_fn=lambda _angle: INNER_RADIUS,
        z_min=WALL_THICKNESS,
        z_max=HEIGHT + WALL_THICKNESS,
        segments=192,
    )
    apply_boolean(body, hollow, "DIFFERENCE")

    supports = create_extruded_polygon_mesh(
        "boolean_tool_two_centered_support_cubes",
        plus_support_profile(),
        z_min=TOP_CHAMFER,
        z_max=HEIGHT,
    )
    apply_boolean(body, supports, "UNION")

    d_bore = create_profile_prism_mesh(
        "boolean_tool_shared_d_shaft_hole_through_union",
        radius_fn=d_bore_cut_radius,
        z_min=HEIGHT - SHAFT_LENGTH - BOOLEAN_EPSILON,
        z_max=HEIGHT + PLUNGE_DEPTH + 2.0 + BOOLEAN_EPSILON,
        segments=224,
    )
    clearance = create_radial_ring_mesh(
        "boolean_tool_upper_internal_clearance_annulus",
        outer_radius_fn=lambda _angle: HOLLOW_RADIUS + BOOLEAN_EPSILON,
        inner_radius_fn=lambda _angle: INNER_RADIUS - BOOLEAN_EPSILON,
        z_min=HEIGHT - CLEARANCE_HEIGHT - BOOLEAN_EPSILON,
        z_max=HEIGHT * 2.0 + BOOLEAN_EPSILON,
        segments=192,
    )
    small_scallops = [
        create_horizontal_cylinder_mesh(f"boolean_tool_yellow_small_scollop_cut_{index}", cutter, vertices=176)
        for index, cutter in enumerate(scallop_cutters(SCALLOP_RADIUS))
    ]
    for tool in (d_bore, clearance, *small_scallops):
        apply_boolean(body, tool, "DIFFERENCE")

    delete_objects([chamfer, collar, hollow, supports, d_bore, clearance, *small_scallops])
    assign_material(body, material)
    shade_flat(body)
    return body


def build_zero_pointer(material: bpy.types.Material) -> bpy.types.Object:
    base_x = OUTER_RADIUS
    points = [
        (base_x + POINT_SIZE, 0.0, HEIGHT),
        (base_x - WALL_THICKNESS / 3.0, -POINT_SIZE / 2.0, HEIGHT),
        (base_x - WALL_THICKNESS / 3.0, POINT_SIZE / 2.0, HEIGHT),
        (base_x, 0.0, TOP_CHAMFER),
    ]
    faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (2, 1, 3)]
    pointer = create_tetrahedron("triangular_zero_pointer", points, faces, material=material)
    shade_flat(pointer)
    return pointer


def build_model() -> list[bpy.types.Object]:
    yellow = make_material("source_preview_yellow", SOURCE_YELLOW)
    green = make_material("source_preview_green_closed_scallop_wall_volumes", (0.24, 0.46, 0.055, 1.0))

    objects: list[bpy.types.Object] = []
    objects.append(build_source_order_yellow_knob_body(yellow))
    objects.extend(build_closed_scallop_wall_volumes(green))
    objects.extend(build_yellow_upper_crescent_interruption_surfaces(yellow))
    objects.append(build_d_bore_cut_wall(yellow))
    objects.append(build_zero_pointer(yellow))

    for obj in objects:
        if obj.type == "MESH" and not obj.data.materials:
            assign_material(obj, yellow)
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

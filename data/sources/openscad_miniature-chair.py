from __future__ import annotations
from dataclasses import dataclass
from mathutils import Vector
from typing import Iterable
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the miniature chair model."""





@dataclass(frozen=True)
class ChairDimensions:
    """Source OpenSCAD dimensions in millimeters."""

    back_height: float = 30.0
    back_top_radius: float = 3.0
    back_bottom_radius: float = 2.0
    back_bottom_width: float = 35.0
    back_handhold_half_width: float = 6.0
    back_handhold_height: float = 1.0
    back_angle_degrees: float = 10.0

    seat_thickness: float = 3.0
    seat_depth: float = 40.0
    seat_front_width: float = 50.0

    leg_height: float = 40.0
    leg_edge_offset: float = 3.0
    leg_top_radius: float = 2.5
    leg_middle_radius: float = 2.5
    leg_bottom_radius: float = 1.5


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.55,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = roughness
    material.diffuse_color = color
    return material


def shade_smooth(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)
    return obj


def add_bevel_modifier(
    obj: bpy.types.Object,
    amount: float,
    segments: int,
    name: str = "soft rounded source edges",
) -> bpy.types.Modifier:
    bevel = obj.modifiers.new(name=name, type="BEVEL")
    bevel.width = amount
    bevel.segments = segments
    bevel.profile = 0.5
    bevel.affect = "EDGES"
    bevel.harden_normals = True
    return bevel


def add_weighted_normals(obj: bpy.types.Object) -> bpy.types.Modifier:
    modifier = obj.modifiers.new(name="weighted normals", type="WEIGHTED_NORMAL")
    modifier.keep_sharp = True
    return modifier


def assign_material(obj: bpy.types.Object, material: bpy.types.Material | None) -> None:
    if material is not None:
        obj.data.materials.append(material)


def create_trapezoid_seat(
    name: str,
    rear_width: float,
    front_width: float,
    depth: float,
    thickness: float,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Build the hull-like rounded trapezoid seat from the SCAD source."""

    z_half = thickness / 2.0
    verts = [
        (-rear_width / 2.0, 0.0, -z_half),
        (rear_width / 2.0, 0.0, -z_half),
        (front_width / 2.0, depth, -z_half),
        (-front_width / 2.0, depth, -z_half),
        (-rear_width / 2.0, 0.0, z_half),
        (rear_width / 2.0, 0.0, z_half),
        (front_width / 2.0, depth, z_half),
        (-front_width / 2.0, depth, z_half),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    add_bevel_modifier(obj, amount=thickness, segments=18)
    add_weighted_normals(obj)
    return obj


def _ring_vertices(
    center: Vector,
    radius: float,
    radial_segments: int,
) -> list[tuple[float, float, float]]:
    return [
        (
            center.x + radius * math.cos(2.0 * math.pi * i / radial_segments),
            center.y + radius * math.sin(2.0 * math.pi * i / radial_segments),
            center.z,
        )
        for i in range(radial_segments)
    ]


def create_tapered_leg(
    name: str,
    top_radius: float,
    middle_radius: float,
    bottom_radius: float,
    height: float,
    location: tuple[float, float, float],
    material: bpy.types.Material | None = None,
    radial_segments: int = 40,
) -> bpy.types.Object:
    """Approximate the SCAD hull of top cylinder, middle sphere, and bottom cylinder."""

    rings = [
        (Vector((0.0, 0.0, 0.5)), top_radius),
        (Vector((0.0, 0.0, 0.0)), top_radius),
        (Vector((0.0, 0.0, -height / 2.0)), middle_radius),
        (Vector((0.0, 0.0, -height + 1.0)), bottom_radius),
        (Vector((0.0, 0.0, -height)), bottom_radius),
    ]

    verts: list[tuple[float, float, float]] = []
    for center, radius in rings:
        verts.extend(_ring_vertices(center, radius, radial_segments))
    verts.append((0.0, 0.0, 0.5))
    top_center = len(verts) - 1
    verts.append((0.0, 0.0, -height))
    bottom_center = len(verts) - 1

    faces: list[tuple[int, ...]] = []
    for ring_index in range(len(rings) - 1):
        start = ring_index * radial_segments
        next_start = (ring_index + 1) * radial_segments
        for i in range(radial_segments):
            faces.append((start + i, start + (i + 1) % radial_segments, next_start + (i + 1) % radial_segments, next_start + i))

    for i in range(radial_segments):
        faces.append((top_center, (i + 1) % radial_segments, i))
        bottom_start = (len(rings) - 1) * radial_segments
        faces.append((bottom_center, bottom_start + i, bottom_start + (i + 1) % radial_segments))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    shade_smooth(obj)
    add_weighted_normals(obj)
    return obj


def create_horizontal_cylinder(
    name: str,
    length: float,
    radius: float,
    material: bpy.types.Material | None = None,
    radial_segments: int = 48,
) -> bpy.types.Object:
    """Create a cylinder along the local X axis for rails and rounded bars."""

    verts: list[tuple[float, float, float]] = []
    for x in (-length / 2.0, length / 2.0):
        for i in range(radial_segments):
            angle = 2.0 * math.pi * i / radial_segments
            verts.append((x, radius * math.cos(angle), radius * math.sin(angle)))
    verts.append((-length / 2.0, 0.0, 0.0))
    left_center = len(verts) - 1
    verts.append((length / 2.0, 0.0, 0.0))
    right_center = len(verts) - 1

    faces: list[tuple[int, ...]] = []
    for i in range(radial_segments):
        faces.append((i, (i + 1) % radial_segments, radial_segments + (i + 1) % radial_segments, radial_segments + i))
        faces.append((left_center, i, (i + 1) % radial_segments))
        faces.append((right_center, radial_segments + (i + 1) % radial_segments, radial_segments + i))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    shade_smooth(obj)
    add_weighted_normals(obj)
    return obj


def _scad_circle_polygon(radius: float, sides: int, rotation_degrees: float, translation: tuple[float, float]) -> list[tuple[float, float]]:
    rotation = math.radians(rotation_degrees)
    tx, ty = translation
    return [
        (
            tx + radius * math.cos(rotation + 2.0 * math.pi * i / sides),
            ty + radius * math.sin(rotation + 2.0 * math.pi * i / sides),
        )
        for i in range(sides)
    ]


def _point_in_convex_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    orientation: bool | None = None
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) < 1.0e-6:
            continue
        current = cross > 0.0
        if orientation is None:
            orientation = current
        elif orientation != current:
            return False
    return True


def _to_back_local_from_scad(point: Vector, back_height: float) -> Vector:
    """Apply the source's inner rotate([90,0,0]) and translate([0,0,height])."""

    return Vector((point.x, -point.z, point.y + back_height))


def _deduped_points(points: Iterable[Vector], precision: int = 5) -> list[Vector]:
    deduped: list[Vector] = []
    seen: set[tuple[float, float, float]] = set()
    for point in points:
        key = (round(point.x, precision), round(point.y, precision), round(point.z, precision))
        if key not in seen:
            seen.add(key)
            deduped.append(point)
    return deduped


def _convex_hull_mesh_from_points(name: str, points: Iterable[Vector]) -> bpy.types.Mesh:
    """Create a watertight convex hull mesh using Blender's bmesh operator."""

    import bmesh

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    for point in _deduped_points(points):
        bm.verts.new(point)
    bm.verts.ensure_lookup_table()

    result = bmesh.ops.convex_hull(bm, input=list(bm.verts))
    discard = []
    seen_ids = set()
    for result_key in ("geom_interior", "geom_unused", "geom_holes"):
        for element in result.get(result_key, []):
            if id(element) not in seen_ids:
                seen_ids.add(id(element))
                discard.append(element)
    if discard:
        bmesh.ops.delete(bm, geom=discard, context="VERTS")

    bm.to_mesh(mesh)
    bm.free()
    mesh.validate(clean_customdata=False)
    mesh.update()
    return mesh


def _source_bottom_cylinder_points(dim: ChairDimensions, radial_segments: int) -> list[Vector]:
    points: list[Vector] = []
    for x in (-dim.back_bottom_width / 2.0, dim.back_bottom_width / 2.0):
        for index in range(radial_segments):
            angle = 2.0 * math.pi * index / radial_segments
            scad_point = Vector(
                (
                    x,
                    -dim.back_height + dim.back_bottom_radius * math.cos(angle),
                    dim.back_bottom_radius * math.sin(angle),
                )
            )
            points.append(_to_back_local_from_scad(scad_point, dim.back_height))
    return points


def _source_clipped_torus_points(
    dim: ChairDimensions,
    ring_segments: int,
    tube_segments: int,
) -> list[Vector]:
    """Sample the clipped rotate_extrude top profile from the OpenSCAD back hull."""

    clip_triangle = _scad_circle_polygon(
        radius=50.0,
        sides=3,
        rotation_degrees=30.0,
        translation=(0.0, 40.0),
    )
    points: list[Vector] = []
    major_radius = 20.0
    x_scale = 1.5
    for ring_index in range(ring_segments):
        theta = 2.0 * math.pi * ring_index / ring_segments
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        for tube_index in range(tube_segments):
            phi = 2.0 * math.pi * tube_index / tube_segments
            swept_radius = major_radius + dim.back_top_radius * math.cos(phi)
            sx = swept_radius * cos_theta
            sy = swept_radius * sin_theta
            sz = dim.back_top_radius * math.sin(phi)
            if _point_in_convex_polygon((sx, sy), clip_triangle):
                scad_point = Vector((sx * x_scale, sy, sz))
                points.append(_to_back_local_from_scad(scad_point, dim.back_height))
    return points


def _sphere_points(center: Vector, radius: float, latitude_segments: int, longitude_segments: int) -> list[Vector]:
    points: list[Vector] = []
    for lat_index in range(latitude_segments + 1):
        polar = math.pi * lat_index / latitude_segments
        z = radius * math.cos(polar)
        ring_radius = radius * math.sin(polar)
        for lon_index in range(longitude_segments):
            azimuth = 2.0 * math.pi * lon_index / longitude_segments
            points.append(
                Vector(
                    (
                        center.x + ring_radius * math.cos(azimuth),
                        center.y + ring_radius * math.sin(azimuth),
                        center.z + z,
                    )
                )
            )
    return points


def _source_handhold_cutter_points(dim: ChairDimensions) -> list[Vector]:
    """Sample the SCAD hull of two capsule/sphere levels used as the handhold cutter."""

    points: list[Vector] = []
    capsule_levels = [
        (-10.0, dim.back_handhold_half_width, dim.back_handhold_height),
        (10.0, 3.0, 3.0),
    ]
    for scad_z, half_width, radius in capsule_levels:
        for x in (-half_width, half_width):
            center = _to_back_local_from_scad(Vector((x, 12.0, scad_z)), dim.back_height)
            points.extend(_sphere_points(center, radius, latitude_segments=10, longitude_segments=20))
    return points


def _make_convex_hull_object(
    name: str,
    points: Iterable[Vector],
    material: bpy.types.Material | None = None,
    smooth: bool = True,
) -> bpy.types.Object:
    mesh = _convex_hull_mesh_from_points(name, points)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    if smooth:
        shade_smooth(obj)
    return obj


def _subtract_and_remove(target: bpy.types.Object, cutter: bpy.types.Object, modifier_name: str) -> None:
    boolean = target.modifiers.new(name=modifier_name, type="BOOLEAN")
    boolean.operation = "DIFFERENCE"
    boolean.object = cutter
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    try:
        bpy.ops.object.modifier_apply(modifier=boolean.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
    except RuntimeError:
        cutter.hide_render = True
        cutter.hide_viewport = True
    target.select_set(False)


def create_source_hull_chair_back(
    name: str,
    dimensions: ChairDimensions,
    material: bpy.types.Material | None = None,
    ring_segments: int = 120,
    tube_segments: int = 24,
    bottom_radial_segments: int = 48,
) -> bpy.types.Object:
    """Build the chair back from the source hull inputs instead of a flat slab approximation."""

    hull_points = []
    hull_points.extend(_source_bottom_cylinder_points(dimensions, bottom_radial_segments))
    hull_points.extend(_source_clipped_torus_points(dimensions, ring_segments, tube_segments))

    obj = _make_convex_hull_object(name, hull_points, material)
    cutter = _make_convex_hull_object(f"{name}_handhold_cutter", _source_handhold_cutter_points(dimensions), smooth=False)
    _subtract_and_remove(obj, cutter, modifier_name="source two-level handhold void")

    # The SCAD object is a hull of rounded inputs, so only a small normal cleanup is needed here.
    add_weighted_normals(obj)
    return obj


def add_collection_instance_members(
    collection: list[bpy.types.Object],
    objects: Iterable[bpy.types.Object],
) -> list[bpy.types.Object]:
    collection.extend(objects)
    return collection


# ============================================================
# Procedural assembly
# ============================================================

"""High-level assembly for the miniature chair model."""





def _leg_locations(dim: ChairDimensions) -> list[tuple[float, float, float]]:
    return [
        (dim.back_bottom_width / 2.0 - dim.leg_edge_offset, dim.leg_edge_offset, 0.0),
        (-dim.back_bottom_width / 2.0 + dim.leg_edge_offset, dim.leg_edge_offset, 0.0),
        (dim.seat_front_width / 2.0 - dim.leg_edge_offset, dim.seat_depth - dim.leg_edge_offset, 0.0),
        (-dim.seat_front_width / 2.0 + dim.leg_edge_offset, dim.seat_depth - dim.leg_edge_offset, 0.0),
    ]


def build_model() -> list:
    """Build the complete chair as separate readable subassemblies."""

    dim = ChairDimensions()
    yellow = make_material("source warm yellow plastic", (1.0, 0.82, 0.02, 1.0))
    objects = []

    seat = create_trapezoid_seat(
        name="rounded_trapezoid_seat",
        rear_width=dim.back_bottom_width + dim.seat_thickness,
        front_width=dim.seat_front_width + dim.seat_thickness * 2.0,
        depth=dim.seat_depth + dim.seat_thickness * 2.0,
        thickness=dim.seat_thickness * 2.0,
        material=yellow,
    )
    seat.location.y = -dim.seat_thickness
    objects.append(seat)

    legs = [
        create_tapered_leg(
            name=f"tapered_hull_leg_{index + 1}",
            top_radius=dim.leg_top_radius,
            middle_radius=dim.leg_middle_radius,
            bottom_radius=dim.leg_bottom_radius,
            height=dim.leg_height,
            location=location,
            material=yellow,
        )
        for index, location in enumerate(_leg_locations(dim))
    ]
    add_collection_instance_members(objects, legs)

    back = create_source_hull_chair_back(
        name="source_hull_tilted_back_with_integrated_lower_rail",
        dimensions=dim,
        material=yellow,
    )
    back.rotation_euler[0] = math.radians(dim.back_angle_degrees)
    objects.append(back)

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

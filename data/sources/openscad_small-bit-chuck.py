from __future__ import annotations
from dataclasses import dataclass
from mathutils import Vector
from typing import Iterable
import bmesh
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the small bit chuck collet."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.58


SCAD_YELLOW = MaterialSpec("openscad_default_yellow", (0.96, 0.76, 0.04, 1.0), 0.54)
CUT_GREEN = MaterialSpec("openscad_cut_face_green", (0.44, 0.62, 0.18, 1.0), 0.66)
DARK_CAVITY = MaterialSpec("deep_cavity_shadow", (0.05, 0.055, 0.02, 1.0), 0.75)


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
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = 0.0
    return material


def assign_material(obj: bpy.types.Object, material: MaterialSpec) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(get_material(material))
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


def add_cylinder(
    name: str,
    diameter: float,
    height: float,
    z_min: float,
    vertices: int = 72,
    material: MaterialSpec | None = None,
    smooth: bool = True,
    rotation_z: float = 0.0,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=diameter / 2.0,
        depth=height,
        location=(0.0, 0.0, z_min + height / 2.0),
        rotation=(0.0, 0.0, rotation_z),
    )
    obj = bpy.context.object
    obj.name = name
    if material:
        assign_material(obj, material)
    if smooth and vertices > 12:
        shade_smooth(obj)
    return obj


def add_cone(
    name: str,
    diameter1: float,
    diameter2: float,
    height: float,
    z_min: float,
    vertices: int = 72,
    material: MaterialSpec | None = None,
    smooth: bool = True,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cone_add(
        vertices=vertices,
        radius1=diameter1 / 2.0,
        radius2=diameter2 / 2.0,
        depth=height,
        location=(0.0, 0.0, z_min + height / 2.0),
    )
    obj = bpy.context.object
    obj.name = name
    if material:
        assign_material(obj, material)
    if smooth:
        shade_smooth(obj)
    return obj


def scad_thread_center(
    step: float,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
) -> Vector:
    local_z = step * spread / 36.0
    theta = math.radians(step * 10.0)
    center_radius = radius - step * taper
    return Vector(
        (
            math.sin(theta) * center_radius,
            -math.cos(theta) * center_radius,
            z_offset + local_z,
        )
    )


def collet_thread_envelope_radius(local_z: float) -> float:
    """Outer radius of the SCAD intersection envelope around the thread."""
    if local_z < 4.0:
        return 6.0 + 0.5 * local_z
    if local_z <= 6.0:
        return 8.0
    return 8.0 - 0.5 * (local_z - 6.0)


def polygon_radius_at_angle(theta: float, circumradius: float, sides: int, rotation_z: float) -> float:
    """Radius of a regular polygon boundary at a polar angle."""
    sector = math.tau / sides
    normal_angle = rotation_z + sector / 2.0
    delta = math.atan2(math.sin(theta - normal_angle), math.cos(theta - normal_angle))
    delta = (delta + sector / 2.0) % sector - sector / 2.0
    apothem = circumradius * math.cos(math.pi / sides)
    return apothem / max(math.cos(delta), 0.001)


def add_supportless_thread_transition_mesh(
    name: str,
    material: MaterialSpec,
    z_base: float,
    angular_segments: int = 48,
    rotation_z: float = math.radians(30.0),
) -> bpy.types.Object:
    """Build the source supportless hull between the stem and first bead."""
    del angular_segments
    round_segments = 6
    hull_points: list[Vector] = []

    # Source lines 70-76 hull a d=10 six-sided cylinder from z=stem-3
    # through z=stem+7 with a d=12 round cylinder from z=stem through
    # z=stem+1. Keep the render mesh as six broad panels because the jaw cuts
    # expose this underside directly and small facets read as rejected pockets.
    for z_abs in (z_base - 3.0, z_base + 7.0):
        for index in range(6):
            theta = rotation_z + math.tau * index / 6
            hull_points.append(Vector((5.0 * math.cos(theta), 5.0 * math.sin(theta), z_abs)))

    for z_abs in (z_base, z_base + 1.0):
        for index in range(round_segments):
            theta = rotation_z + math.tau * index / round_segments
            hull_points.append(Vector((6.0 * math.cos(theta), 6.0 * math.sin(theta), z_abs)))

    bm = bmesh.new()
    for point in hull_points:
        bm.verts.new(point)
    bm.verts.ensure_lookup_table()
    bmesh.ops.convex_hull(bm, input=list(bm.verts), use_existing_faces=False)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    used_verts = sorted(
        {vert for face in bm.faces for vert in face.verts},
        key=lambda vert: vert.index,
    )
    vert_lookup = {vert: index for index, vert in enumerate(used_verts)}
    verts = [tuple(vert.co) for vert in used_verts]
    faces = [
        tuple(vert_lookup[vert] for vert in face.verts)
        for face in bm.faces
        if len(face.verts) >= 3
    ]
    bm.free()

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _support_panel_fill_mesh(
    name: str,
    material: MaterialSpec,
    z_base: float,
    rotation_z: float,
) -> bpy.types.Object:
    """Hide first-bead underside notches with broad six-sided hull panels."""
    panel_segments = 6
    # Keep this as one coarse six-sided canopy.  Letting the fill reach the
    # full thread crest radius created a repeated tooth row after the jaw cuts;
    # the source support is mostly the d=10 six-sided hull rising into the
    # first d=12 round level.
    level_specs = [
        (-0.03, 5.18),
        (3.05, 6.68),
    ]
    verts: list[tuple[float, float, float]] = []
    for z_rel, radius in level_specs:
        for index in range(panel_segments):
            theta = rotation_z + math.tau * index / panel_segments
            verts.append((radius * math.cos(theta), radius * math.sin(theta), z_base + z_rel))

    faces: list[tuple[int, ...]] = []

    def idx(level: int, index: int) -> int:
        return level * panel_segments + (index % panel_segments)

    for level in range(len(level_specs) - 1):
        for index in range(panel_segments):
            faces.append(
                (
                    idx(level, index),
                    idx(level, index + 1),
                    idx(level + 1, index + 1),
                    idx(level + 1, index),
                )
            )
    faces.append(tuple(reversed([idx(0, index) for index in range(panel_segments)])))
    top_level = len(level_specs) - 1
    faces.append(tuple(idx(top_level, index) for index in range(panel_segments)))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _thread_frame(center: Vector, prev_center: Vector, next_center: Vector) -> tuple[Vector, Vector, Vector]:
    tangent = next_center - prev_center
    if tangent.length < 0.0001:
        tangent = Vector((0.0, 0.0, 1.0))
    tangent.normalize()

    radial = Vector((center.x, center.y, 0.0))
    if radial.length < 0.0001:
        radial = Vector((1.0, 0.0, 0.0))
    radial.normalize()

    normal = radial - tangent * radial.dot(tangent)
    if normal.length < 0.0001:
        normal = tangent.cross(Vector((0.0, 0.0, 1.0)))
    normal.normalize()

    binormal = tangent.cross(normal)
    if binormal.length < 0.0001:
        binormal = Vector((0.0, 0.0, 1.0))
    binormal.normalize()
    return tangent, normal, binormal


def apply_boolean(
    target: bpy.types.Object,
    cutter: bpy.types.Object,
    operation: str = "DIFFERENCE",
    *,
    keep_cutter: bool = False,
    solver: str = "EXACT",
) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    modifier = target.modifiers.new(f"{operation.lower()}_{cutter.name}", "BOOLEAN")
    modifier.operation = operation
    modifier.object = cutter
    modifier.solver = solver
    if operation == "DIFFERENCE" and cutter.data.materials and hasattr(modifier, "material_mode"):
        modifier.material_mode = "TRANSFER"
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    target.select_set(False)
    if not keep_cutter:
        bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def apply_boolean_difference(
    target: bpy.types.Object,
    cutters: Iterable[bpy.types.Object],
    *,
    keep_cutters: bool = False,
) -> bpy.types.Object:
    for cutter in cutters:
        apply_boolean(target, cutter, "DIFFERENCE", keep_cutter=keep_cutters)
    return target


def trim_boolean_transfer_artifacts(
    target: bpy.types.Object,
    *,
    max_radius: float,
    max_z: float,
) -> bpy.types.Object:
    """Delete transferred cutter sheet fragments outside the source envelope."""
    mesh = target.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    world_matrix = target.matrix_world.copy()

    def world_point(vert: bmesh.types.BMVert) -> Vector:
        return world_matrix @ vert.co

    def is_outside_source_extent(vert: bmesh.types.BMVert) -> bool:
        point = world_point(vert)
        return math.hypot(point.x, point.y) > max_radius or point.z > max_z

    suspect_faces = [face for face in bm.faces if any(is_outside_source_extent(vert) for vert in face.verts)]
    if suspect_faces:
        bmesh.ops.triangulate(bm, faces=suspect_faces)

    faces_to_delete = [
        face
        for face in bm.faces
        if face.verts and any(is_outside_source_extent(vert) for vert in face.verts)
    ]
    if faces_to_delete:
        bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")
    loose_edges = [edge for edge in bm.edges if not edge.link_faces]
    if loose_edges:
        bmesh.ops.delete(bm, geom=loose_edges, context="EDGES")
    loose_verts = [vert for vert in bm.verts if not vert.link_faces]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")
    bm.to_mesh(mesh)
    mesh.update()
    bm.free()
    return target


def boolean_union(
    name: str,
    objects: Iterable[bpy.types.Object],
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    objects = list(objects)
    if not objects:
        raise ValueError("boolean_union requires at least one object")

    target = objects[0]
    for obj in objects[1:]:
        apply_boolean(target, obj, "UNION")
    target.name = name
    target.data.name = f"{name}_mesh"
    if material:
        assign_material(target, material)
    return target


def combine_cut_mesh_objects(
    name: str,
    objects: Iterable[bpy.types.Object],
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Append already-cut source union members into one reviewable mesh object."""
    objects = list(objects)
    if not objects:
        raise ValueError("combine_cut_mesh_objects requires at least one object")

    material_slots: list[bpy.types.Material] = []
    material_lookup: dict[str, int] = {}
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    face_material_indices: list[int] = []

    def material_index_for(material_obj: bpy.types.Material | None) -> int:
        if material_obj is None:
            material_obj = get_material(material or SCAD_YELLOW)
        if material_obj.name not in material_lookup:
            material_lookup[material_obj.name] = len(material_slots)
            material_slots.append(material_obj)
        return material_lookup[material_obj.name]

    default_material_index = material_index_for(get_material(material or SCAD_YELLOW))
    for obj in objects:
        local_material_indices = [
            material_index_for(slot_material)
            for slot_material in obj.data.materials
        ]
        base = len(verts)
        verts.extend(tuple(obj.matrix_world @ vert.co) for vert in obj.data.vertices)
        for polygon in obj.data.polygons:
            faces.append(tuple(base + vertex_index for vertex_index in polygon.vertices))
            if local_material_indices and polygon.material_index < len(local_material_indices):
                face_material_indices.append(local_material_indices[polygon.material_index])
            else:
                face_material_indices.append(default_material_index)

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon, material_index in zip(mesh.polygons, face_material_indices):
        polygon.material_index = material_index
        polygon.use_smooth = False

    combined = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(combined)
    for material_obj in material_slots:
        combined.data.materials.append(material_obj)
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    return combined


def _make_low_poly_thread_segment(
    name: str,
    center_a: Vector,
    center_b: Vector,
    radius: float,
    material: MaterialSpec,
    ring_segments: int = 10,
) -> bpy.types.Object:
    """Build one low-poly capsule span matching the SCAD hull(sphere, sphere)."""
    axis = center_b - center_a
    if axis.length < 0.0001:
        axis = Vector((0.0, 0.0, 1.0))
    tangent = axis.normalized()
    radial = Vector((center_a.x + center_b.x, center_a.y + center_b.y, 0.0))
    if radial.length < 0.0001:
        radial = Vector((1.0, 0.0, 0.0))
    radial.normalize()
    normal = radial - tangent * radial.dot(tangent)
    if normal.length < 0.0001:
        normal = tangent.cross(Vector((0.0, 0.0, 1.0)))
    normal.normalize()
    binormal = tangent.cross(normal).normalized()

    verts: list[tuple[float, float, float]] = []
    for center in (center_a, center_b):
        for index in range(ring_segments):
            phi = math.tau * index / ring_segments
            outward = math.cos(phi)
            axial = math.sin(phi)
            outer_bias = 1.14 if outward >= 0.0 else 0.64
            axial_scale = 0.78 if outward >= -0.25 else 0.58
            point = center.copy()
            point += normal * (outward * radius * outer_bias)
            point += binormal * (axial * radius * axial_scale)
            point += tangent * (0.10 * radius * math.sin(2.0 * phi))
            verts.append(tuple(point))

    faces: list[tuple[int, ...]] = []
    for index in range(ring_segments):
        faces.append((index, (index + 1) % ring_segments, ring_segments + (index + 1) % ring_segments, ring_segments + index))
    faces.append(tuple(reversed(range(ring_segments))))
    faces.append(tuple(ring_segments + index for index in range(ring_segments)))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _thread_lens_profile_points(
    center: Vector,
    prev_center: Vector,
    next_center: Vector,
    radius: float,
    *,
    envelope_clearance: float,
    z_offset: float,
    max_local_z: float | None,
    ring_segments: int,
    ring_scale: float,
    vertical_fullness: float,
    outward_fullness: float,
    inner_fullness: float,
    phase_twist: float,
) -> list[Vector]:
    """One low-poly sphere-like ring for the segmented source-thread chain."""
    tangent, normal, binormal = _thread_frame(center, prev_center, next_center)
    vertical = Vector((0.0, 0.0, 1.0))
    side_axis = vertical.cross(normal)
    if side_axis.length < 0.0001:
        side_axis = tangent.copy()
    side_axis.normalize()

    points: list[Vector] = []
    for ring_index in range(ring_segments):
        phi = math.tau * ring_index / ring_segments + phase_twist
        outward = math.cos(phi)
        vertical_amount = math.sin(phi)

        radial_scale = outward_fullness if outward >= 0.0 else inner_fullness
        vertical_scale = vertical_fullness if abs(vertical_amount) > 0.12 else 0.70
        side_skew = 0.15 * math.sin(phi + 0.55) + 0.07 * math.sin(2.0 * phi)

        point = center.copy()
        point += normal * (outward * radius * ring_scale * radial_scale)
        point += vertical * (vertical_amount * radius * ring_scale * vertical_scale)
        point += side_axis * (side_skew * radius * ring_scale)
        point += tangent * (0.07 * radius * ring_scale * math.sin(2.0 * phi - 0.35))
        if max_local_z is not None:
            point.z = min(max(point.z, z_offset), z_offset + max_local_z)
        local_z = point.z - z_offset
        envelope_radius = collet_thread_envelope_radius(local_z) - envelope_clearance
        radius_xy = math.hypot(point.x, point.y)
        if radius_xy > envelope_radius:
            scale = envelope_radius / radius_xy
            point.x *= scale
            point.y *= scale
        points.append(point)
    return points


def _make_segmented_thread_lens_chain(
    name: str,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
    turns: float,
    thread_radius: float,
    material: MaterialSpec,
    max_local_z: float | None = None,
    envelope_clearance: float = 0.055,
    ring_segments: int = 10,
) -> bpy.types.Object:
    """Build a connected pearled low-poly chain along the SCAD helix.

    Previous connected sweeps made the ridges read like thin stacked collars,
    while a literal stack of overlapping capsules destabilized the jaw-slot
    booleans. This keeps one connected surface but alternates fuller rings at
    the source sphere centers with smaller neck rings between them, preserving
    the accepted cutter behavior while making the exposed side silhouette more
    like hulled low-poly beads.
    """
    max_step = int(math.floor(36 * turns))
    if max_local_z is not None:
        max_step = min(max_step, int(math.floor(max_local_z * 36.0 / spread)))

    path_substeps = 1
    sample_count = max_step * path_substeps + 1
    centers = [
        scad_thread_center(
            sample / path_substeps,
            z_offset=z_offset,
            spread=spread,
            radius=radius,
            taper=taper,
        )
        for sample in range(sample_count)
    ]

    verts: list[tuple[float, float, float]] = []
    for sample_index, center in enumerate(centers):
        prev_center = centers[max(0, sample_index - 1)]
        next_center = centers[min(sample_count - 1, sample_index + 1)]
        source_step = sample_index / path_substeps
        turn_phase = (source_step % 36.0) / 36.0
        bead_peak = 0.5 + 0.5 * math.cos(math.tau * turn_phase)
        phase_twist = math.radians(18.0) * math.sin(math.tau * turn_phase)
        profile = _thread_lens_profile_points(
            center,
            prev_center,
            next_center,
            thread_radius,
            envelope_clearance=envelope_clearance,
            z_offset=z_offset,
            max_local_z=max_local_z,
            ring_segments=ring_segments,
            ring_scale=0.94 + 0.05 * bead_peak,
            vertical_fullness=0.88 + 0.08 * bead_peak,
            outward_fullness=1.34 + 0.08 * bead_peak,
            inner_fullness=0.42 + 0.08 * bead_peak,
            phase_twist=phase_twist,
        )
        verts.extend(tuple(point) for point in profile)

    faces: list[tuple[int, ...]] = []

    def idx(sample_index: int, ring_index: int) -> int:
        return sample_index * ring_segments + (ring_index % ring_segments)

    for sample_index in range(sample_count - 1):
        for ring_index in range(ring_segments):
            faces.append(
                (
                    idx(sample_index, ring_index),
                    idx(sample_index, ring_index + 1),
                    idx(sample_index + 1, ring_index + 1),
                    idx(sample_index + 1, ring_index),
                )
            )

    faces.append(tuple(reversed([idx(0, ring_index) for ring_index in range(ring_segments)])))
    end_index = sample_count - 1
    faces.append(tuple(idx(end_index, ring_index) for ring_index in range(ring_segments)))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _make_continuous_thread_bead(
    name: str,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
    turns: float,
    thread_radius: float,
    material: MaterialSpec,
    path_substeps: int = 1,
    ring_segments: int = 10,
    max_local_z: float | None = None,
    envelope_clearance: float = 0.025,
) -> bpy.types.Object:
    """Sweep one faceted lobe along the SCAD thread centers.

    The source is a union of 10-degree hull(sphere, sphere) spans clipped by the
    collet envelope. Building one connected low-poly lobe keeps the same
    diagonal center path while avoiding a boolean stack of separate span solids.
    The points are clipped to the same envelope up front so Blender does not add
    a second set of horizontal intersection faces that read as collar bands.
    """
    max_step = int(math.floor(36 * turns))
    if max_local_z is not None:
        max_step = min(max_step, int(math.floor(max_local_z * 36.0 / spread)))
    sample_count = max_step * path_substeps + 1
    centers = [
        scad_thread_center(
            sample / path_substeps,
            z_offset=z_offset,
            spread=spread,
            radius=radius,
            taper=taper,
        )
        for sample in range(sample_count)
    ]

    verts: list[tuple[float, float, float]] = []
    for sample_index, center in enumerate(centers):
        prev_center = centers[max(0, sample_index - 1)]
        next_center = centers[min(sample_count - 1, sample_index + 1)]
        tangent, normal, _binormal = _thread_frame(center, prev_center, next_center)
        vertical = Vector((0.0, 0.0, 1.0))
        tangential = vertical.cross(normal)
        if tangential.length < 0.0001:
            tangential = tangent.copy()
        tangential.normalize()

        source_step = sample_index / path_substeps
        center_lobe = 0.5 + 0.5 * math.cos(math.tau * (source_step % 1.0))
        lobe_scale = 0.96 + 0.07 * center_lobe

        for ring_index in range(ring_segments):
            phi = math.tau * ring_index / ring_segments
            outward = math.cos(phi)
            vertical_amount = math.sin(phi)
            if outward >= 0.0:
                radial_scale = 1.16
            elif outward >= -0.45:
                radial_scale = 0.82
            else:
                radial_scale = 0.58
            vertical_scale = 1.14 - 0.05 * max(outward, 0.0)
            tangent_skew = 0.07 * math.sin(2.0 * phi)
            point = center.copy()
            point += normal * (outward * radial_scale * thread_radius * lobe_scale)
            point += vertical * (vertical_amount * vertical_scale * thread_radius * lobe_scale)
            point += tangential * (tangent_skew * thread_radius)
            if max_local_z is not None:
                point.z = min(max(point.z, z_offset), z_offset + max_local_z)
            local_z = point.z - z_offset
            envelope_radius = collet_thread_envelope_radius(local_z) - envelope_clearance
            radius_xy = math.hypot(point.x, point.y)
            if radius_xy > envelope_radius:
                scale = envelope_radius / radius_xy
                point.x *= scale
                point.y *= scale
            verts.append(tuple(point))

    faces: list[tuple[int, ...]] = []

    def idx(sample_index: int, ring_index: int) -> int:
        return sample_index * ring_segments + (ring_index % ring_segments)

    for sample_index in range(sample_count - 1):
        for ring_index in range(ring_segments):
            faces.append(
                (
                    idx(sample_index, ring_index),
                    idx(sample_index, ring_index + 1),
                    idx(sample_index + 1, ring_index + 1),
                    idx(sample_index + 1, ring_index),
                )
            )

    faces.append(tuple(reversed([idx(0, ring_index) for ring_index in range(ring_segments)])))
    end_index = sample_count - 1
    faces.append(tuple(idx(end_index, ring_index) for ring_index in range(ring_segments)))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _thread_profile_points(
    center: Vector,
    prev_center: Vector,
    next_center: Vector,
    radius: float,
    *,
    envelope_clearance: float,
    z_offset: float,
    max_local_z: float | None,
    ring_segments: int = 10,
) -> list[Vector]:
    tangent, normal, _binormal = _thread_frame(center, prev_center, next_center)
    vertical = Vector((0.0, 0.0, 1.0))
    side_axis = vertical.cross(normal)
    if side_axis.length < 0.0001:
        side_axis = tangent.copy()
    side_axis.normalize()

    points: list[Vector] = []
    for index in range(ring_segments):
        phi = math.tau * index / ring_segments
        outward = math.cos(phi)
        vertical_amount = math.sin(phi)
        if outward >= 0.0:
            radial_scale = 1.18
        elif outward >= -0.35:
            radial_scale = 0.58
        else:
            radial_scale = 0.38
        vertical_scale = 0.76 if abs(vertical_amount) > 0.15 else 0.58
        side_skew = 0.10 * math.sin(phi + 0.45) + 0.04 * math.sin(2.0 * phi)

        point = center.copy()
        point += normal * (outward * radius * radial_scale)
        point += vertical * (vertical_amount * radius * vertical_scale)
        point += side_axis * (side_skew * radius)
        point += tangent * (0.05 * radius * math.sin(2.0 * phi - 0.35))
        if max_local_z is not None:
            point.z = min(max(point.z, z_offset), z_offset + max_local_z)
        local_z = point.z - z_offset
        envelope_radius = collet_thread_envelope_radius(local_z) - envelope_clearance
        radius_xy = math.hypot(point.x, point.y)
        if radius_xy > envelope_radius:
            scale = envelope_radius / radius_xy
            point.x *= scale
            point.y *= scale
        points.append(point)
    return points


def _make_source_thread_bead_span_mesh(
    name: str,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
    turns: float,
    thread_radius: float,
    material: MaterialSpec,
    max_local_z: float | None = None,
    envelope_clearance: float = 0.02,
    ring_segments: int = 10,
    step_stride: int = 2,
) -> bpy.types.Object:
    """Build the thread as explicit low-poly hull spans between SCAD centers."""
    max_step = int(math.floor(36 * turns))
    if max_local_z is not None:
        max_step = min(max_step, int(math.floor(max_local_z * 36.0 / spread)))

    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    def clamp_to_thread_envelope(point: Vector) -> Vector:
        point = point.copy()
        if max_local_z is not None:
            point.z = min(max(point.z, z_offset), z_offset + max_local_z)
        local_z = point.z - z_offset
        envelope_radius = collet_thread_envelope_radius(local_z) - envelope_clearance
        radius_xy = math.hypot(point.x, point.y)
        if radius_xy > envelope_radius:
            scale = envelope_radius / radius_xy
            point.x *= scale
            point.y *= scale
        return point

    def low_poly_sphere_points(center: Vector) -> list[Vector]:
        points: list[Vector] = []
        longitudes = ring_segments
        latitudes = (-66.0, -42.0, -18.0, 18.0, 42.0, 66.0)
        points.append(clamp_to_thread_envelope(center + Vector((0.0, 0.0, thread_radius))))
        points.append(clamp_to_thread_envelope(center + Vector((0.0, 0.0, -thread_radius))))
        for lat_deg in latitudes:
            lat = math.radians(lat_deg)
            z_amount = math.sin(lat) * thread_radius
            ring_radius = math.cos(lat) * thread_radius
            for ring_index in range(longitudes):
                theta = math.tau * ring_index / longitudes
                point = Vector(
                    (
                        center.x + math.cos(theta) * ring_radius,
                        center.y + math.sin(theta) * ring_radius,
                        center.z + z_amount,
                    )
                )
                points.append(clamp_to_thread_envelope(point))
        return points

    def append_convex_hull(points: list[Vector]) -> None:
        bm = bmesh.new()
        for point in points:
            bm.verts.new(point)
        bm.verts.ensure_lookup_table()
        bmesh.ops.convex_hull(bm, input=list(bm.verts), use_existing_faces=False)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        used_verts = sorted(
            {vert for face in bm.faces for vert in face.verts},
            key=lambda vert: vert.index,
        )
        vert_lookup = {vert: len(verts) + index for index, vert in enumerate(used_verts)}
        verts.extend(tuple(vert.co) for vert in used_verts)
        for face in bm.faces:
            if len(face.verts) >= 3:
                faces.append(tuple(vert_lookup[vert] for vert in face.verts))
        bm.free()

    # Neighboring source hull spans share low-poly sphere centers. This render
    # mesh does not boolean-union every span, so use a generous overlap to keep
    # the helical ridges continuous while retaining the low-poly span facets.
    span_overlap = 0.58
    for step in range(0, max_step, step_stride):
        next_step = min(step + step_stride, max_step)
        center_a = scad_thread_center(
            step - span_overlap,
            z_offset=z_offset,
            spread=spread,
            radius=radius,
            taper=taper,
        )
        center_b = scad_thread_center(
            next_step + span_overlap,
            z_offset=z_offset,
            spread=spread,
            radius=radius,
            taper=taper,
        )
        append_convex_hull(low_poly_sphere_points(center_a) + low_poly_sphere_points(center_b))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _make_connected_hulled_thread_mesh(
    name: str,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
    turns: float,
    thread_radius: float,
    material: MaterialSpec,
    max_local_z: float | None = None,
    envelope_clearance: float = 0.08,
    ring_segments: int = 10,
    path_substeps: int = 1,
) -> bpy.types.Object:
    """Build one connected faceted ridge from SCAD hulled-sphere centers.

    The mesh stays connected for stable slot booleans, but the profile is
    biased hard toward the exposed outer lobe. The final thread cleanup removes
    most of the inner collar skin so the rendered valleys are carried by the
    tapered core/support panels rather than by a full circumferential sleeve.
    """
    max_step = int(math.floor(36 * turns))
    if max_local_z is not None:
        max_step = min(max_step, int(math.floor(max_local_z * 36.0 / spread)))

    del path_substeps
    centers = [
        scad_thread_center(
            step,
            z_offset=z_offset,
            spread=spread,
            radius=radius,
            taper=taper,
        )
        for step in range(max_step + 1)
    ]

    def clamp_to_envelope(point: Vector) -> Vector:
        point = point.copy()
        if max_local_z is not None:
            point.z = min(max(point.z, z_offset), z_offset + max_local_z)
        local_z = point.z - z_offset
        envelope_radius = collet_thread_envelope_radius(local_z) - envelope_clearance
        radius_xy = math.hypot(point.x, point.y)
        if radius_xy > envelope_radius:
            scale = envelope_radius / radius_xy
            point.x *= scale
            point.y *= scale
        return point

    verts: list[tuple[float, float, float]] = []
    for step, center in enumerate(centers):
        prev_center = centers[max(0, step - 1)]
        next_center = centers[min(max_step, step + 1)]
        tangent, normal, _binormal = _thread_frame(center, prev_center, next_center)
        vertical = Vector((0.0, 0.0, 1.0))
        side_axis = vertical.cross(normal)
        if side_axis.length < 0.0001:
            side_axis = tangent.copy()
        side_axis.normalize()

        turn_phase = (step % 36) / 36.0
        lobe_phase = math.sin(math.tau * turn_phase)
        ring_twist = math.radians(24.0) * lobe_phase
        center_lift = 0.30 * thread_radius * math.sin(math.tau * turn_phase + 0.45)

        for ring_index in range(ring_segments):
            phi = math.tau * ring_index / ring_segments + ring_twist
            outward = math.cos(phi)
            vertical_amount = math.sin(phi)

            if outward >= 0.0:
                radial_scale = 1.52 + 0.17 * math.cos(math.tau * turn_phase - 0.35)
            elif outward > -0.55:
                radial_scale = 0.25
            else:
                radial_scale = 0.10
            vertical_scale = 1.18 if abs(vertical_amount) > 0.15 else 0.82
            side_skew = 0.46 * math.sin(phi + 0.65) + 0.15 * math.sin(2.0 * phi - 0.2)

            point = center.copy()
            point.z += center_lift
            point += normal * (outward * thread_radius * radial_scale)
            point += vertical * (vertical_amount * thread_radius * vertical_scale)
            point += side_axis * (side_skew * thread_radius)
            point += tangent * (0.23 * thread_radius * math.sin(2.0 * phi - 0.4))
            verts.append(tuple(clamp_to_envelope(point)))

    faces: list[tuple[int, ...]] = []

    def idx(step: int, ring_index: int) -> int:
        return step * ring_segments + (ring_index % ring_segments)

    for step in range(max_step):
        for ring_index in range(ring_segments):
            faces.append(
                (
                    idx(step, ring_index),
                    idx(step, ring_index + 1),
                    idx(step + 1, ring_index + 1),
                    idx(step + 1, ring_index),
                )
            )

    faces.append(tuple(reversed([idx(0, ring_index) for ring_index in range(ring_segments)])))
    faces.append(tuple(idx(max_step, ring_index) for ring_index in range(ring_segments)))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _make_visible_helical_thread_strip(
    name: str,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
    turns: float,
    thread_radius: float,
    material: MaterialSpec,
    max_local_z: float | None = None,
    envelope_clearance: float = 0.035,
) -> bpy.types.Object:
    """Build a source-pitch faceted helical ridge without a collar sleeve.

    The OpenSCAD thread is a union of adjacent low-poly sphere hulls clipped by
    the collet envelope.  A complete tube around the sphere centers rendered as
    horizontal collars after the jaw cuts, so this mesh keeps only the visible
    outer hulled-sphere lobe and closes it back against the tapered core.  The
    diagonal center path remains the source 10-degree cadence.
    """
    max_step = int(math.floor(36 * turns))
    if max_local_z is not None:
        max_step = min(max_step, int(math.floor(max_local_z * 36.0 / spread)))

    centers = [
        scad_thread_center(
            step,
            z_offset=z_offset,
            spread=spread,
            radius=radius,
            taper=taper,
        )
        for step in range(max_step + 1)
    ]

    verts: list[tuple[float, float, float]] = []
    # A partial low-poly sphere profile, expressed in the helix tangent frame.
    # Endpoints close back near the tapered core while the middle points retain
    # the exposed hulled-sphere belly.
    profile_phis = (
        -2.34,
        -1.78,
        -1.20,
        -0.62,
        0.0,
        0.62,
        1.20,
        1.78,
        2.34,
    )
    profile_count = len(profile_phis)

    def core_radius_at(local_z: float) -> float:
        return radius - 0.10 * local_z

    def envelope_radius_at(local_z: float) -> float:
        return collet_thread_envelope_radius(local_z) - envelope_clearance

    for step, center in enumerate(centers):
        prev_center = centers[max(0, step - 1)]
        next_center = centers[min(max_step, step + 1)]
        tangent, normal, binormal = _thread_frame(center, prev_center, next_center)
        side_axis = Vector((0.0, 0.0, 1.0)).cross(normal)
        if side_axis.length < 0.0001:
            side_axis = tangent.copy()
        side_axis.normalize()

        turn_phase = (step % 36) / 36.0
        bead = 0.5 + 0.5 * math.cos(math.tau * turn_phase)
        crest_boost = 0.08 * bead
        side_bias = 0.12 * math.sin(math.tau * turn_phase + 0.35)

        for phi in profile_phis:
            outward = math.cos(phi)
            side_amount = math.sin(phi)
            radial_scale = 1.18 + crest_boost if outward > 0.0 else 0.72
            point = center.copy()
            point += normal * (outward * thread_radius * radial_scale)
            point += binormal * (side_amount * thread_radius * 0.98)
            point += side_axis * (side_bias * thread_radius * max(outward, 0.0))
            point += tangent * (0.07 * thread_radius * math.sin(math.tau * turn_phase + phi))
            local_z = point.z - z_offset
            if max_local_z is not None:
                local_z = min(max(local_z, 0.0), max_local_z)
            else:
                local_z = max(local_z, 0.0)
            point.z = z_offset + local_z

            radius_xy = math.hypot(point.x, point.y)
            min_radius = core_radius_at(local_z) + 0.16
            max_radius = envelope_radius_at(local_z)
            if radius_xy < min_radius:
                scale = min_radius / max(radius_xy, 0.001)
                point.x *= scale
                point.y *= scale
            elif radius_xy > max_radius:
                scale = max_radius / radius_xy
                point.x *= scale
                point.y *= scale
            verts.append(tuple(point))

    faces: list[tuple[int, ...]] = []

    def idx(step: int, profile_index: int) -> int:
        return step * profile_count + profile_index

    for step in range(max_step):
        for profile_index in range(profile_count - 1):
            faces.append(
                (
                    idx(step, profile_index),
                    idx(step, profile_index + 1),
                    idx(step + 1, profile_index + 1),
                    idx(step + 1, profile_index),
                )
            )
        # Close the inner seam against the tapered core.  These faces are later
        # trimmed where exposed so they do not create the rejected collar bands.
        faces.append(
            (
                idx(step, profile_count - 1),
                idx(step, 0),
                idx(step + 1, 0),
                idx(step + 1, profile_count - 1),
            )
        )

    faces.append(tuple(reversed([idx(0, profile_index) for profile_index in range(profile_count)])))
    faces.append(tuple(idx(max_step, profile_index) for profile_index in range(profile_count)))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _make_sampled_hulled_sphere_thread_field(
    name: str,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
    turns: float,
    thread_radius: float,
    material: MaterialSpec,
    max_local_z: float | None = None,
    envelope_clearance: float = 0.015,
    angular_segments: int = 48,
) -> bpy.types.Object:
    """Sample the source hulled-sphere thread as a radial outer field.

    The OpenSCAD thread is a dense union of overlapping low-poly sphere hulls.
    For the visible model, a polar field keeps the true helical
    high-points and deep valleys without building a stack of booleaned capsule
    solids in every render.  Low-poly shading and source-step z sampling retain
    the faceted bead-span character.
    """
    max_step = int(math.floor(36 * turns))
    if max_local_z is not None:
        max_step = min(max_step, int(math.floor(max_local_z * 36.0 / spread)))
    local_z_max = max_local_z if max_local_z is not None else max_step * spread / 36.0
    z_steps = max(36, int(math.ceil(local_z_max * 7.2)))

    centers: list[tuple[float, float, float]] = []
    for step in range(max_step + 1):
        center = scad_thread_center(
            step,
            z_offset=0.0,
            spread=spread,
            radius=radius,
            taper=taper,
        )
        centers.append((math.atan2(center.y, center.x), math.hypot(center.x, center.y), center.z))

    def core_radius_at(local_z: float) -> float:
        return radius - 0.10 * local_z

    def sphere_union_radius(theta: float, local_z: float) -> float:
        best = core_radius_at(local_z) + 0.04
        # Only nearby source centers can affect this z slice.
        for center_theta, center_radius, center_z in centers:
            dz = local_z - center_z
            if abs(dz) > thread_radius * 1.08:
                continue
            delta = math.atan2(math.sin(theta - center_theta), math.cos(theta - center_theta))
            side_distance = center_radius * math.sin(delta)
            available = thread_radius * thread_radius - dz * dz - side_distance * side_distance
            if available <= 0.0:
                continue
            candidate = center_radius * math.cos(delta) + math.sqrt(available)
            if candidate > best:
                best = candidate
        return min(best, collet_thread_envelope_radius(local_z) - envelope_clearance)

    verts: list[tuple[float, float, float]] = []
    # Outer sampled source thread surface.
    for z_index in range(z_steps + 1):
        local_z = local_z_max * z_index / z_steps
        for angle_index in range(angular_segments):
            theta = math.tau * angle_index / angular_segments
            outer_radius = sphere_union_radius(theta, local_z)
            verts.append(
                (
                    outer_radius * math.cos(theta),
                    outer_radius * math.sin(theta),
                    z_offset + local_z,
                )
            )

    # Inner support surface keeps the mesh watertight for the accepted cutters.
    inner_base = len(verts)
    for z_index in range(z_steps + 1):
        local_z = local_z_max * z_index / z_steps
        inner_radius = max(core_radius_at(local_z) - 0.32, 0.1)
        for angle_index in range(angular_segments):
            theta = math.tau * angle_index / angular_segments
            verts.append(
                (
                    inner_radius * math.cos(theta),
                    inner_radius * math.sin(theta),
                    z_offset + local_z,
                )
            )

    faces: list[tuple[int, ...]] = []

    def outer_idx(z_index: int, angle_index: int) -> int:
        return z_index * angular_segments + (angle_index % angular_segments)

    def inner_idx(z_index: int, angle_index: int) -> int:
        return inner_base + z_index * angular_segments + (angle_index % angular_segments)

    for z_index in range(z_steps):
        for angle_index in range(angular_segments):
            faces.append(
                (
                    outer_idx(z_index, angle_index),
                    outer_idx(z_index, angle_index + 1),
                    outer_idx(z_index + 1, angle_index + 1),
                    outer_idx(z_index + 1, angle_index),
                )
            )
            faces.append(
                (
                    inner_idx(z_index + 1, angle_index),
                    inner_idx(z_index + 1, angle_index + 1),
                    inner_idx(z_index, angle_index + 1),
                    inner_idx(z_index, angle_index),
                )
            )

    for angle_index in range(angular_segments):
        faces.append(
            (
                outer_idx(0, angle_index + 1),
                outer_idx(0, angle_index),
                inner_idx(0, angle_index),
                inner_idx(0, angle_index + 1),
            )
        )
        faces.append(
            (
                outer_idx(z_steps, angle_index),
                outer_idx(z_steps, angle_index + 1),
                inner_idx(z_steps, angle_index + 1),
                inner_idx(z_steps, angle_index),
            )
        )

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _make_chunky_helical_thread_lobe_mesh(
    name: str,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
    turns: float,
    thread_radius: float,
    material: MaterialSpec,
    max_local_z: float | None = None,
    envelope_clearance: float = 0.03,
    source_step_stride: float = 2.0,
) -> bpy.types.Object:
    """Build the visible source thread as one faceted helical cord.

    The rejected radial field sampled every angle/z slice and kept resolving as
    horizontal collar shelves.  This mesh follows the SCAD sphere-center helix
    directly and keeps only the broad outer lobe of that hulled-sphere cord,
    closing the hidden side back near the tapered collet core for stable cuts.
    """
    max_step = int(math.floor(36 * turns))
    if max_local_z is not None:
        max_step = min(max_step, int(math.floor(max_local_z * 36.0 / spread)))

    step_values: list[float] = []
    step = 0.0
    while step < max_step:
        step_values.append(step)
        step += source_step_stride
    if not step_values or step_values[-1] != float(max_step):
        step_values.append(float(max_step))

    centers = [
        scad_thread_center(
            step_value,
            z_offset=z_offset,
            spread=spread,
            radius=radius,
            taper=taper,
        )
        for step_value in step_values
    ]

    # Partial low-poly sphere profile: the middle is the visible chunky crest,
    # while the endpoints close inside the tapered core and are removed from the
    # rendered silhouette by the existing inner-collar trim.
    profile_phis = (-2.18, -1.58, -1.05, -0.52, 0.0, 0.52, 1.05, 1.58, 2.18)
    profile_count = len(profile_phis)

    def core_radius_at(local_z: float) -> float:
        return radius - 0.10 * local_z

    def clamp_to_visible_envelope(point: Vector, outward: float) -> Vector:
        point = point.copy()
        local_z = point.z - z_offset
        if max_local_z is not None:
            local_z = min(max(local_z, 0.0), max_local_z)
        else:
            local_z = max(local_z, 0.0)
        point.z = z_offset + local_z

        radius_xy = math.hypot(point.x, point.y)
        min_radius = core_radius_at(local_z) + (0.08 if outward < 0.10 else 0.0)
        if radius_xy < min_radius:
            scale = min_radius / max(radius_xy, 0.001)
            point.x *= scale
            point.y *= scale

        max_radius = collet_thread_envelope_radius(local_z) - envelope_clearance
        radius_xy = math.hypot(point.x, point.y)
        if radius_xy > max_radius:
            scale = max_radius / radius_xy
            point.x *= scale
            point.y *= scale
        return point

    verts: list[tuple[float, float, float]] = []
    for sample_index, (step_value, center) in enumerate(zip(step_values, centers)):
        prev_center = centers[max(0, sample_index - 1)]
        next_center = centers[min(len(centers) - 1, sample_index + 1)]
        tangent, normal, binormal = _thread_frame(center, prev_center, next_center)
        turn_phase = (step_value % 36.0) / 36.0
        phase_lump = math.cos(math.tau * turn_phase - 0.35)

        for phi in profile_phis:
            outward = math.cos(phi)
            side_amount = math.sin(phi)
            if outward >= 0.0:
                radial_scale = 1.43 + 0.08 * phase_lump
                side_scale = 1.10
            elif outward > -0.45:
                radial_scale = 0.34
                side_scale = 0.78
            else:
                radial_scale = 0.18
                side_scale = 0.52

            point = center.copy()
            point += normal * (outward * thread_radius * radial_scale)
            point += binormal * (side_amount * thread_radius * side_scale)
            point += tangent * (0.09 * thread_radius * math.sin(2.0 * phi + math.tau * turn_phase))
            verts.append(tuple(clamp_to_visible_envelope(point, outward)))

    faces: list[tuple[int, ...]] = []

    def idx(sample_index: int, profile_index: int) -> int:
        return sample_index * profile_count + (profile_index % profile_count)

    for sample_index in range(len(step_values) - 1):
        for profile_index in range(profile_count - 1):
            faces.append(
                (
                    idx(sample_index, profile_index),
                    idx(sample_index, profile_index + 1),
                    idx(sample_index + 1, profile_index + 1),
                    idx(sample_index + 1, profile_index),
                )
            )
        faces.append(
            (
                idx(sample_index, profile_count - 1),
                idx(sample_index, 0),
                idx(sample_index + 1, 0),
                idx(sample_index + 1, profile_count - 1),
            )
        )

    faces.append(tuple(reversed([idx(0, profile_index) for profile_index in range(profile_count)])))
    end_index = len(step_values) - 1
    faces.append(tuple(idx(end_index, profile_index) for profile_index in range(profile_count)))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _make_connected_source_thread_ridge_mesh(
    name: str,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
    turns: float,
    thread_radius: float,
    material: MaterialSpec,
    max_local_z: float | None = None,
    envelope_clearance: float = 0.018,
    source_step_stride: float = 3.0,
) -> bpy.types.Object:
    """Build one continuous visible ridge on the SCAD hulled-sphere helix.

    The OpenSCAD source makes the thread from 10-degree hulls between low-poly
    spheres.  A literal append of independent hulls looked like separate blocks
    in renders, while full sleeve meshes became collar bands.  This keeps the
    source center cadence but connects only the outer sphere-belly profile,
    leaving the tapered core visible as the valley surface.
    """
    max_step = int(math.floor(36 * turns))
    if max_local_z is not None:
        max_step = min(max_step, int(math.floor(max_local_z * 36.0 / spread)))

    step_values: list[float] = []
    step = 0.0
    while step < max_step:
        step_values.append(step)
        step += source_step_stride
    if not step_values or step_values[-1] != float(max_step):
        step_values.append(float(max_step))
    centers = [
        scad_thread_center(
            step_value,
            z_offset=z_offset,
            spread=spread,
            radius=radius,
            taper=taper,
        )
        for step_value in step_values
    ]

    # Broad partial low-poly sphere cross-section.  The source is made from
    # adjacent low-poly sphere hulls; keeping more of that rounded belly while
    # leaving a narrow hidden gap avoids the prior flat open strip.
    profile_phis = (
        -2.42,
        -1.92,
        -1.66,
        -1.18,
        -0.72,
        -0.34,
        0.0,
        0.34,
        0.72,
        1.18,
        1.66,
        1.92,
        2.42,
    )
    profile_count = len(profile_phis)

    def core_radius_at(local_z: float) -> float:
        return radius - 0.10 * local_z

    def clamp_point(point: Vector, outward: float) -> Vector:
        point = point.copy()
        local_z = point.z - z_offset
        if max_local_z is not None:
            local_z = min(max(local_z, 0.0), max_local_z)
        else:
            local_z = max(local_z, 0.0)
        point.z = z_offset + local_z

        radius_xy = math.hypot(point.x, point.y)
        if outward < 0.12:
            min_radius = core_radius_at(local_z) + 0.02
            if radius_xy < min_radius:
                scale = min_radius / max(radius_xy, 0.001)
                point.x *= scale
                point.y *= scale

        max_radius = collet_thread_envelope_radius(local_z) - envelope_clearance
        radius_xy = math.hypot(point.x, point.y)
        if radius_xy > max_radius:
            scale = max_radius / radius_xy
            point.x *= scale
            point.y *= scale
        return point

    verts: list[tuple[float, float, float]] = []
    for sample_index, (step_value, center) in enumerate(zip(step_values, centers)):
        prev_center = centers[max(0, sample_index - 1)]
        next_center = centers[min(len(centers) - 1, sample_index + 1)]
        tangent, normal, binormal = _thread_frame(center, prev_center, next_center)
        turn_phase = (step_value % 36.0) / 36.0
        low_poly_lump = math.cos(math.tau * turn_phase - 0.28)
        diagonal_swell = math.sin(math.tau * turn_phase + 0.35)

        for profile_index, phi in enumerate(profile_phis):
            outward = math.cos(phi)
            side_amount = math.sin(phi)

            if outward >= 0.0:
                radial_scale = 1.64 + 0.10 * low_poly_lump
                side_scale = 1.10
            else:
                radial_scale = 0.24 + 0.08 * max(outward + 0.50, 0.0)
                side_scale = 0.58
            if profile_index in (0, profile_count - 1):
                radial_scale = 0.08

            point = center.copy()
            point += normal * (outward * thread_radius * radial_scale)
            point += binormal * (side_amount * thread_radius * side_scale)
            point += tangent * (0.24 * thread_radius * math.sin(2.0 * phi + math.tau * turn_phase))
            point.z += 0.16 * thread_radius * diagonal_swell * max(outward, 0.0)
            verts.append(tuple(clamp_point(point, outward)))

    faces: list[tuple[int, ...]] = []

    def idx(sample_index: int, profile_index: int) -> int:
        return sample_index * profile_count + profile_index

    for sample_index in range(len(step_values) - 1):
        for profile_index in range(profile_count - 1):
            faces.append(
                (
                    idx(sample_index, profile_index),
                    idx(sample_index, profile_index + 1),
                    idx(sample_index + 1, profile_index + 1),
                    idx(sample_index + 1, profile_index),
                )
            )

    faces.append(tuple(reversed([idx(0, profile_index) for profile_index in range(profile_count)])))
    end_index = len(step_values) - 1
    faces.append(tuple(idx(end_index, profile_index) for profile_index in range(profile_count)))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _make_broad_source_thread_ribbon_mesh(
    name: str,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
    turns: float,
    thread_radius: float,
    material: MaterialSpec,
    max_local_z: float | None = None,
    envelope_clearance: float = 0.018,
    angular_segments: int = 72,
) -> bpy.types.Object:
    """Build broad helical sphere-hull ribbons from the source thread cadence.

    Earlier connected tubes still left a full collar-like side silhouette after
    the jaw-slot trim.  This active mesh follows the same 10-degree helix, but
    represents each visible turn as a rounded outer ribbon over the tapered core
    instead of as a complete sleeve around the thread center.
    """
    max_step = int(math.floor(36 * turns))
    if max_local_z is not None:
        max_step = min(max_step, int(math.floor(max_local_z * 36.0 / spread)))
    local_z_max = max_local_z if max_local_z is not None else max_step * spread / 36.0

    profile_offsets = (-1.12, -0.82, -0.48, -0.20, 0.0, 0.20, 0.48, 0.82, 1.12)
    profile_count = len(profile_offsets)
    half_width = thread_radius * 1.02
    theta_values = [math.tau * index / angular_segments for index in range(angular_segments + 1)]

    def source_alpha(theta: float) -> float:
        return (theta + math.pi / 2.0) % math.tau

    def core_radius_at(local_z: float) -> float:
        return radius - 0.10 * local_z

    def envelope_radius_at(local_z: float) -> float:
        return collet_thread_envelope_radius(local_z) - envelope_clearance

    def vertex_for(theta: float, turn_index: int, z_offset_from_center: float) -> tuple[float, float, float] | None:
        alpha = source_alpha(theta)
        source_step = alpha / math.radians(10.0) + 36.0 * turn_index
        if source_step < -1.0 or source_step > max_step + 1.0:
            return None

        center_local_z = source_step * spread / 36.0
        local_z = center_local_z + z_offset_from_center
        if local_z < -0.02 or local_z > local_z_max + 0.02:
            return None
        local_z = min(max(local_z, 0.0), local_z_max)

        normalized_offset = min(abs(z_offset_from_center) / half_width, 1.0)
        sphere_belly = math.sqrt(max(0.0, 1.0 - normalized_offset * normalized_offset))

        # Source side views show chunky diagonal cords rather than complete
        # collar lips.  Give the visible lobe a small phase-following lift so
        # adjacent jaw-slot interruptions expose the helix instead of flat rows.
        diagonal_lift = 0.22 * thread_radius * sphere_belly * math.sin(theta - 0.38)
        local_z = min(max(local_z + diagonal_lift, 0.0), local_z_max)

        center_radius = radius - source_step * taper
        low_poly_flat = 0.94 + 0.08 * math.cos(5.0 * alpha)
        radial_radius = center_radius + thread_radius * (-0.04 + 1.34 * sphere_belly) * low_poly_flat
        radial_radius = max(radial_radius, core_radius_at(local_z) + 0.06 + 0.08 * sphere_belly)
        radial_radius = min(radial_radius, envelope_radius_at(local_z))

        return (
            radial_radius * math.cos(theta),
            radial_radius * math.sin(theta),
            z_offset + local_z,
        )

    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    # The 0..10 mm intersection exposes the central turns plus partial end
    # turns.  Building those separate ribbons makes the pitch visibly diagonal
    # while leaving the core surface as the broad valley between ridges.
    for turn_index in range(-1, int(math.ceil(local_z_max / spread)) + 2):
        index_map: dict[tuple[int, int], int] = {}
        for theta_index, theta in enumerate(theta_values):
            for profile_index, profile_offset in enumerate(profile_offsets):
                point = vertex_for(theta, turn_index, profile_offset * thread_radius)
                if point is None:
                    continue
                index_map[(theta_index, profile_index)] = len(verts)
                verts.append(point)

        for theta_index in range(len(theta_values) - 1):
            for profile_index in range(profile_count - 1):
                keys = (
                    (theta_index, profile_index),
                    (theta_index, profile_index + 1),
                    (theta_index + 1, profile_index + 1),
                    (theta_index + 1, profile_index),
                )
                if all(key in index_map for key in keys):
                    faces.append(tuple(index_map[key] for key in keys))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _make_diagonal_source_thread_cord_mesh(
    name: str,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
    turns: float,
    thread_radius: float,
    material: MaterialSpec,
    max_local_z: float | None = None,
    envelope_clearance: float = 0.024,
    source_step_stride: float = 1.0,
) -> bpy.types.Object:
    """Build the visible thread as a source-cadence diagonal cord.

    The source thread is a chain of hulls between low-poly spheres.  Previous
    angular ribbon meshes produced horizontal shelf bands because each turn was
    parameterized around the whole circumference.  This mesh follows the actual
    sphere-center helix and keeps only a broad outer low-poly cord, leaving the
    tapered core as the open valley between turns.
    """
    max_step = int(math.floor(36 * turns))
    if max_local_z is not None:
        max_step = min(max_step, int(math.floor(max_local_z * 36.0 / spread)))

    step_values: list[float] = []
    step = 0.0
    while step < max_step:
        step_values.append(step)
        step += source_step_stride
    if not step_values or step_values[-1] != float(max_step):
        step_values.append(float(max_step))

    centers = [
        scad_thread_center(
            step_value,
            z_offset=z_offset,
            spread=spread,
            radius=radius,
            taper=taper,
        )
        for step_value in step_values
    ]

    # Full low-poly tube profile.  Hidden inner faces are trimmed after build;
    # keeping a closed section here avoids the flat top/bottom shelf lips that
    # open ribbon profiles produced in the side renders.
    profile_phis = tuple(math.tau * index / 10 for index in range(10))
    profile_count = len(profile_phis)

    def core_radius_at(local_z: float) -> float:
        return radius - 0.10 * local_z

    def clamp_point(point: Vector, outward: float) -> Vector:
        point = point.copy()
        local_z = point.z - z_offset
        if max_local_z is not None:
            local_z = min(max(local_z, 0.0), max_local_z)
        else:
            local_z = max(local_z, 0.0)
        point.z = z_offset + local_z

        radius_xy = math.hypot(point.x, point.y)
        min_radius = core_radius_at(local_z) + (0.05 if outward < 0.08 else 0.02)
        if radius_xy < min_radius:
            scale = min_radius / max(radius_xy, 0.001)
            point.x *= scale
            point.y *= scale

        max_radius = collet_thread_envelope_radius(local_z) - envelope_clearance
        radius_xy = math.hypot(point.x, point.y)
        if radius_xy > max_radius:
            scale = max_radius / radius_xy
            point.x *= scale
            point.y *= scale
        return point

    verts: list[tuple[float, float, float]] = []
    for sample_index, (step_value, center) in enumerate(zip(step_values, centers)):
        prev_center = centers[max(0, sample_index - 1)]
        next_center = centers[min(len(centers) - 1, sample_index + 1)]
        tangent, normal, binormal = _thread_frame(center, prev_center, next_center)
        source_phase = math.tau * ((step_value % 36.0) / 36.0)
        sphere_faceting = 1.0 + 0.055 * math.cos(5.0 * source_phase)

        for phi in profile_phis:
            outward = math.cos(phi)
            side_amount = math.sin(phi)
            if outward >= 0.0:
                radial_scale = 1.14 * sphere_faceting
                side_scale = 1.03
            elif outward > -0.48:
                radial_scale = 0.60
                side_scale = 0.92
            else:
                radial_scale = 0.34
                side_scale = 0.68

            point = center.copy()
            point += normal * (outward * thread_radius * radial_scale)
            point += binormal * (side_amount * thread_radius * side_scale)
            # The source hull spans are diagonal cords.  A small tangent skew
            # gives the low-poly belly a visible helical lean at the jaw cuts.
            point += tangent * (0.10 * thread_radius * math.sin(2.0 * phi + source_phase))
            verts.append(tuple(clamp_point(point, outward)))

    faces: list[tuple[int, ...]] = []

    def idx(sample_index: int, profile_index: int) -> int:
        return sample_index * profile_count + profile_index

    for sample_index in range(len(step_values) - 1):
        for profile_index in range(profile_count - 1):
            faces.append(
                (
                    idx(sample_index, profile_index),
                    idx(sample_index, profile_index + 1),
                    idx(sample_index + 1, profile_index + 1),
                    idx(sample_index + 1, profile_index),
                )
            )
        faces.append(
            (
                idx(sample_index, profile_count - 1),
                idx(sample_index, 0),
                idx(sample_index + 1, 0),
                idx(sample_index + 1, profile_count - 1),
            )
        )

    faces.append(tuple(reversed([idx(0, profile_index) for profile_index in range(profile_count)])))
    end_index = len(step_values) - 1
    faces.append(tuple(idx(end_index, profile_index) for profile_index in range(profile_count)))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _make_outer_helical_thread_cord_mesh(
    name: str,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
    turns: float,
    thread_radius: float,
    material: MaterialSpec,
    max_local_z: float | None = None,
    envelope_clearance: float = 0.010,
    source_step_stride: float = 1.0,
) -> bpy.types.Object:
    """Build only the exposed outer belly of the source helical thread.

    The full closed-tube thread variants kept resolving in side renders as
    circumferential collars.  This surface follows the same 10-degree source
    sphere-center helix, but it keeps just the rounded outer half of the
    hulled-sphere cord and leaves the tapered collet core visible as the valley
    surface between turns.
    """
    max_step = int(math.floor(36 * turns))
    if max_local_z is not None:
        max_step = min(max_step, int(math.floor(max_local_z * 36.0 / spread)))

    step_values: list[float] = []
    step = 0.0
    while step < max_step:
        step_values.append(step)
        step += source_step_stride
    if not step_values or step_values[-1] != float(max_step):
        step_values.append(float(max_step))

    centers = [
        scad_thread_center(
            step_value,
            z_offset=z_offset,
            spread=spread,
            radius=radius,
            taper=taper,
        )
        for step_value in step_values
    ]

    # Narrower than a tube: the side profile is the rounded cord belly plus
    # two valley edges.  Keeping this open avoids the rejected inner sleeve and
    # its horizontal collar lips.
    profile_phis = (-1.30, -0.96, -0.62, -0.32, 0.0, 0.32, 0.62, 0.96, 1.30)
    profile_count = len(profile_phis)

    def core_radius_at(local_z: float) -> float:
        return radius - 0.10 * local_z

    def clamp_point(point: Vector, outward: float) -> Vector:
        point = point.copy()
        local_z = point.z - z_offset
        if max_local_z is not None:
            local_z = min(max(local_z, 0.0), max_local_z)
        else:
            local_z = max(local_z, 0.0)
        point.z = z_offset + local_z

        radius_xy = math.hypot(point.x, point.y)
        min_radius = core_radius_at(local_z) + 0.10 + 0.16 * max(outward, 0.0)
        if radius_xy < min_radius:
            scale = min_radius / max(radius_xy, 0.001)
            point.x *= scale
            point.y *= scale

        max_radius = collet_thread_envelope_radius(local_z) - envelope_clearance
        radius_xy = math.hypot(point.x, point.y)
        if radius_xy > max_radius:
            scale = max_radius / radius_xy
            point.x *= scale
            point.y *= scale
        return point

    verts: list[tuple[float, float, float]] = []
    for sample_index, (step_value, center) in enumerate(zip(step_values, centers)):
        prev_center = centers[max(0, sample_index - 1)]
        next_center = centers[min(len(centers) - 1, sample_index + 1)]
        tangent, normal, binormal = _thread_frame(center, prev_center, next_center)
        source_phase = math.tau * ((step_value % 36.0) / 36.0)
        low_poly_swell = 1.0 + 0.075 * math.cos(5.0 * source_phase)

        for phi in profile_phis:
            outward = max(math.cos(phi), 0.0)
            side_amount = math.sin(phi)
            belly = outward**1.35
            radial_offset = thread_radius * (0.26 + 1.30 * belly) * low_poly_swell
            side_offset = thread_radius * side_amount * (0.82 + 0.15 * belly)

            point = center.copy()
            point += normal * radial_offset
            point += binormal * side_offset
            point += tangent * (0.22 * thread_radius * side_amount)
            # Subtle center lift makes the uninterrupted side silhouette follow
            # the helix instead of reading as level rows.
            point.z += 0.12 * thread_radius * belly * math.sin(source_phase + 0.35)
            verts.append(tuple(clamp_point(point, outward)))

    faces: list[tuple[int, ...]] = []

    def idx(sample_index: int, profile_index: int) -> int:
        return sample_index * profile_count + profile_index

    for sample_index in range(len(step_values) - 1):
        for profile_index in range(profile_count - 1):
            faces.append(
                (
                    idx(sample_index, profile_index),
                    idx(sample_index, profile_index + 1),
                    idx(sample_index + 1, profile_index + 1),
                    idx(sample_index + 1, profile_index),
                )
            )

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _make_source_thread_patch_mesh(
    name: str,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
    turns: float,
    thread_radius: float,
    material: MaterialSpec,
    max_local_z: float | None = None,
    envelope_clearance: float = 0.010,
    angular_segments: int = 96,
    z_segments: int = 88,
) -> bpy.types.Object:
    """Sample the source hulled-sphere helix as outer diagonal patches.

    This keeps the tapered cone as the valley floor and only emits faces where
    the source sphere-hull field protrudes above that core.  The side silhouette
    is therefore driven by diagonal helical cords, not by a continuous
    circumferential sleeve.
    """
    max_step = int(math.floor(36 * turns))
    if max_local_z is not None:
        max_step = min(max_step, int(math.floor(max_local_z * 36.0 / spread)))
    local_z_max = max_local_z if max_local_z is not None else max_step * spread / 36.0

    centers: list[tuple[float, float, float]] = []
    for step in range(max_step + 1):
        center = scad_thread_center(
            step,
            z_offset=0.0,
            spread=spread,
            radius=radius,
            taper=taper,
        )
        centers.append((math.atan2(center.y, center.x), math.hypot(center.x, center.y), center.z))

    def core_radius_at(local_z: float) -> float:
        return radius - 0.10 * local_z

    def sphere_field_radius(theta: float, local_z: float) -> float | None:
        best: float | None = None
        for center_theta, center_radius, center_z in centers:
            dz = local_z - center_z
            if abs(dz) > thread_radius * 1.08:
                continue
            delta = math.atan2(math.sin(theta - center_theta), math.cos(theta - center_theta))
            side_distance = center_radius * math.sin(delta)
            available = thread_radius * thread_radius - dz * dz - side_distance * side_distance
            if available <= 0.0:
                continue
            low_poly_flat = 0.96 + 0.06 * math.cos(5.0 * delta)
            candidate = center_radius * math.cos(delta) + math.sqrt(available) * low_poly_flat
            if best is None or candidate > best:
                best = candidate
        return best

    verts: list[tuple[float, float, float]] = []
    vertex_map: dict[tuple[int, int], int] = {}
    protrusion_threshold = 0.18
    for z_index in range(z_segments + 1):
        local_z = local_z_max * z_index / z_segments
        core_radius = core_radius_at(local_z)
        envelope_radius = collet_thread_envelope_radius(local_z) - envelope_clearance
        for angle_index in range(angular_segments):
            theta = math.tau * angle_index / angular_segments
            source_radius = sphere_field_radius(theta, local_z)
            if source_radius is None or source_radius < core_radius + protrusion_threshold:
                continue
            protrusion = min(source_radius, envelope_radius) - core_radius
            radial = core_radius + max(protrusion, 0.18)
            # Bias high protrusions outward so the rendered ridge has the
            # rounded belly of the source sphere hull rather than a flat shelf.
            radial = min(radial + 0.10 * min(protrusion, thread_radius), envelope_radius)
            vertex_map[(z_index, angle_index)] = len(verts)
            verts.append((radial * math.cos(theta), radial * math.sin(theta), z_offset + local_z))

    faces: list[tuple[int, ...]] = []
    for z_index in range(z_segments):
        for angle_index in range(angular_segments):
            keys = (
                (z_index, angle_index),
                (z_index, (angle_index + 1) % angular_segments),
                (z_index + 1, (angle_index + 1) % angular_segments),
                (z_index + 1, angle_index),
            )
            if all(key in vertex_map for key in keys):
                faces.append(tuple(vertex_map[key] for key in keys))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _make_source_thread_lobe_strip_mesh(
    name: str,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
    turns: float,
    thread_radius: float,
    material: MaterialSpec,
    max_local_z: float | None = None,
    envelope_clearance: float = 0.018,
    source_step_stride: float = 1.0,
) -> bpy.types.Object:
    """Build fuller visible lobes from the source hulled-sphere helix.

    The source thread is a 10-degree chain of low-poly spheres joined by hulls.
    This connected mesh follows the 10-degree source center path, keeps a broad
    faceted sphere-belly cross-section, and closes near the tapered core so the
    valleys stay open without degenerating into isolated rectangular tabs.
    """
    max_step = int(math.floor(36 * turns))
    if max_local_z is not None:
        max_step = min(max_step, int(math.floor(max_local_z * 36.0 / spread)))

    step_values: list[float] = []
    step = 0.0
    while step < max_step:
        step_values.append(step)
        step += source_step_stride
    if not step_values or step_values[-1] != float(max_step):
        step_values.append(float(max_step))

    centers = [
        scad_thread_center(
            step_value,
            z_offset=z_offset,
            spread=spread,
            radius=radius,
            taper=taper,
        )
        for step_value in step_values
    ]

    # Low-poly sphere-belly profile. It keeps more of the source $fn=10 sphere
    # circumference than current_version, but still leaves a hidden inner gap so
    # the side view reads as diagonal lobes instead of horizontal collars.
    profile_phis = (
        -2.48,
        -2.08,
        -1.66,
        -1.24,
        -0.82,
        -0.42,
        0.0,
        0.42,
        0.82,
        1.24,
        1.66,
        2.08,
        2.48,
    )
    profile_count = len(profile_phis)

    def core_radius_at(local_z: float) -> float:
        return radius - local_z / 10.0

    def clamp_point(point: Vector, outward: float, edge_weight: float) -> Vector:
        point = point.copy()
        local_z = point.z - z_offset
        if max_local_z is not None:
            local_z = min(max(local_z, 0.0), max_local_z)
        else:
            local_z = max(local_z, 0.0)
        point.z = z_offset + local_z

        core_radius = core_radius_at(local_z)
        min_radius = core_radius + 0.03 + 0.14 * max(outward, 0.0) + 0.08 * edge_weight
        radius_xy = math.hypot(point.x, point.y)
        if radius_xy < min_radius:
            scale = min_radius / max(radius_xy, 0.001)
            point.x *= scale
            point.y *= scale

        max_radius = collet_thread_envelope_radius(local_z) - envelope_clearance
        radius_xy = math.hypot(point.x, point.y)
        if radius_xy > max_radius:
            scale = max_radius / radius_xy
            point.x *= scale
            point.y *= scale
        return point

    verts: list[tuple[float, float, float]] = []
    for sample_index, (step_value, center) in enumerate(zip(step_values, centers)):
        prev_center = centers[max(0, sample_index - 1)]
        next_center = centers[min(len(centers) - 1, sample_index + 1)]
        tangent, normal, binormal = _thread_frame(center, prev_center, next_center)
        source_phase = math.tau * ((step_value % 36.0) / 36.0)
        low_poly_swell = 1.02 + 0.06 * math.cos(5.0 * source_phase - 0.35)
        diagonal_lift = math.sin(source_phase + 0.45)

        for profile_index, phi in enumerate(profile_phis):
            outward_raw = math.cos(phi)
            outward = max(outward_raw, 0.0)
            side_amount = math.sin(phi)
            edge_weight = abs(profile_index - (profile_count - 1) / 2.0) / ((profile_count - 1) / 2.0)
            belly = outward**0.58
            radial_offset = thread_radius * (0.12 + 1.66 * belly) * low_poly_swell
            if outward_raw < 0.0:
                radial_offset += thread_radius * outward_raw * 0.14
            side_offset = thread_radius * side_amount * (1.12 + 0.12 * belly)

            point = center.copy()
            point += normal * radial_offset
            point += binormal * side_offset
            point += tangent * (0.12 * thread_radius * side_amount)
            point.z += 0.10 * thread_radius * belly * diagonal_lift
            verts.append(tuple(clamp_point(point, outward_raw, edge_weight)))

    faces: list[tuple[int, ...]] = []

    def idx(sample_index: int, profile_index: int) -> int:
        return sample_index * profile_count + profile_index

    for sample_index in range(len(step_values) - 1):
        for profile_index in range(profile_count - 1):
            faces.append(
                (
                    idx(sample_index, profile_index),
                    idx(sample_index, profile_index + 1),
                    idx(sample_index + 1, profile_index + 1),
                    idx(sample_index + 1, profile_index),
                )
            )

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _make_full_source_thread_cord_mesh(
    name: str,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
    turns: float,
    thread_radius: float,
    material: MaterialSpec,
    max_local_z: float | None = None,
    envelope_clearance: float = 0.012,
    source_step_stride: float = 1.0,
    ring_segments: int = 10,
) -> bpy.types.Object:
    """Build a continuous low-poly tube through the source thread centers.

    The SCAD thread is made from 10-degree hulls between $fn=10 spheres.  This
    active model keeps that source cadence and uses a closed faceted
    cord, leaving the tapered core to visually supply the broad valleys between
    turns instead of reducing the thread to exterior strip fragments.
    """
    max_step = int(math.floor(36 * turns))
    if max_local_z is not None:
        max_step = min(max_step, int(math.floor(max_local_z * 36.0 / spread)))

    step_values: list[float] = []
    step = 0.0
    while step < max_step:
        step_values.append(step)
        step += source_step_stride
    if not step_values or step_values[-1] != float(max_step):
        step_values.append(float(max_step))

    centers = [
        scad_thread_center(
            step_value,
            z_offset=z_offset,
            spread=spread,
            radius=radius,
            taper=taper,
        )
        for step_value in step_values
    ]

    def clamp_to_source_envelope(point: Vector) -> Vector:
        point = point.copy()
        local_z = point.z - z_offset
        if max_local_z is not None:
            local_z = min(max(local_z, 0.0), max_local_z)
        else:
            local_z = max(local_z, 0.0)
        point.z = z_offset + local_z

        envelope_radius = collet_thread_envelope_radius(local_z) - envelope_clearance
        radius_xy = math.hypot(point.x, point.y)
        if radius_xy > envelope_radius:
            scale = envelope_radius / radius_xy
            point.x *= scale
            point.y *= scale
        return point

    verts: list[tuple[float, float, float]] = []
    for sample_index, (step_value, center) in enumerate(zip(step_values, centers)):
        prev_center = centers[max(0, sample_index - 1)]
        next_center = centers[min(len(centers) - 1, sample_index + 1)]
        tangent, normal, binormal = _thread_frame(center, prev_center, next_center)
        source_phase = math.tau * ((step_value % 36.0) / 36.0)
        low_poly_swell = 1.0 + 0.045 * math.cos(5.0 * source_phase)

        for ring_index in range(ring_segments):
            phi = math.tau * ring_index / ring_segments
            outward = math.cos(phi)
            side_amount = math.sin(phi)

            # Keep the outer belly materially full while allowing the inner
            # side to sink into the tapered core, where it is hidden except
            # where the bore and jaw slots expose the cut-through thread body.
            radial_scale = (1.08 if outward >= 0.0 else 0.88) * low_poly_swell
            side_scale = 0.98 + 0.07 * max(outward, 0.0)

            point = center.copy()
            point += normal * (outward * thread_radius * radial_scale)
            point += binormal * (side_amount * thread_radius * side_scale)
            point += tangent * (0.08 * thread_radius * math.sin(2.0 * phi + source_phase))
            verts.append(tuple(clamp_to_source_envelope(point)))

    faces: list[tuple[int, ...]] = []

    def idx(sample_index: int, ring_index: int) -> int:
        return sample_index * ring_segments + (ring_index % ring_segments)

    for sample_index in range(len(step_values) - 1):
        for ring_index in range(ring_segments):
            faces.append(
                (
                    idx(sample_index, ring_index),
                    idx(sample_index, ring_index + 1),
                    idx(sample_index + 1, ring_index + 1),
                    idx(sample_index + 1, ring_index),
                )
            )

    faces.append(tuple(reversed([idx(0, ring_index) for ring_index in range(ring_segments)])))
    end_index = len(step_values) - 1
    faces.append(tuple(idx(end_index, ring_index) for ring_index in range(ring_segments)))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def _trim_thread_inner_collar_faces(
    obj: bpy.types.Object,
    *,
    z_offset: float,
    collet_diameter: float,
    collet_length: float,
    thread_radius: float,
    protected_local_z: float = 0.0,
) -> bpy.types.Object:
    """Remove inner thread sleeve faces that make the helix read as collars."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    def core_radius(local_z: float) -> float:
        return collet_diameter / 2.0 - local_z / collet_length

    faces_to_delete = []
    for face in bm.faces:
        center = face.calc_center_median()
        local_z = center.z - z_offset
        if local_z < protected_local_z or local_z > collet_length + 0.05:
            continue
        radius_xy = math.hypot(center.x, center.y)
        if radius_xy < core_radius(local_z) + thread_radius * 0.58:
            faces_to_delete.append(face)

    if faces_to_delete:
        bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")
        loose_edges = [edge for edge in bm.edges if not edge.link_faces]
        if loose_edges:
            bmesh.ops.delete(bm, geom=loose_edges, context="EDGES")
        loose_verts = [vert for vert in bm.verts if not vert.link_faces]
        if loose_verts:
            bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")

    bm.to_mesh(mesh)
    mesh.update()
    bm.free()
    return obj


def _trim_thread_faces_for_source_cuts(
    obj: bpy.types.Object,
    *,
    bore_diameter: float,
    jaw_count: int,
    slot_width: float,
    slot_length: float,
    z_min: float,
    z_max: float,
) -> bpy.types.Object:
    """Trim a non-boolean thread mesh against the source bore and jaw slots."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    bore_radius = bore_diameter / 2.0
    slot_half_width = slot_width / 2.0 + 0.025
    slot_half_length = slot_length / 2.0 + 0.025
    slot_angles = [math.radians(360.0 * index / jaw_count) for index in range(jaw_count)]

    def point_in_slot(point: Vector) -> bool:
        if point.z < z_min - 0.02 or point.z > z_max + 0.02:
            return False
        for angle in slot_angles:
            center_x = -math.sin(angle) * slot_length / 2.0
            center_y = math.cos(angle) * slot_length / 2.0
            dx = point.x - center_x
            dy = point.y - center_y
            local_x = math.cos(angle) * dx + math.sin(angle) * dy
            local_y = -math.sin(angle) * dx + math.cos(angle) * dy
            if abs(local_x) <= slot_half_width and abs(local_y) <= slot_half_length:
                return True
        return False

    def point_in_bore(point: Vector) -> bool:
        return point.z >= -0.02 and point.z <= z_max + 0.02 and math.hypot(point.x, point.y) <= bore_radius

    faces_to_delete: list[bmesh.types.BMFace] = []
    for face in bm.faces:
        center = face.calc_center_median()
        points = [center, *(vert.co for vert in face.verts)]
        if any(point_in_bore(point) or point_in_slot(point) for point in points):
            faces_to_delete.append(face)

    if faces_to_delete:
        bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")
        loose_edges = [edge for edge in bm.edges if not edge.link_faces]
        if loose_edges:
            bmesh.ops.delete(bm, geom=loose_edges, context="EDGES")
        loose_verts = [vert for vert in bm.verts if not vert.link_faces]
        if loose_verts:
            bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")

    bm.to_mesh(mesh)
    mesh.update()
    bm.free()
    return obj


def _replace_lower_transition_with_clean_canopy(
    target: bpy.types.Object,
    *,
    material: MaterialSpec,
    z_base: float,
    rotation_z: float,
    socket_diameter: float,
) -> bpy.types.Object:
    """Replace lower cut remnants with the source six-sided support canopy."""
    material_obj = get_material(material)
    if material_obj.name not in {slot.name for slot in target.data.materials}:
        target.data.materials.append(material_obj)
    material_index = next(
        index
        for index, slot_material in enumerate(target.data.materials)
        if slot_material.name == material_obj.name
    )

    mesh = target.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # Split both ends of the replacement band before deleting remnants. Several
    # source-core and first-thread faces span from the hex stem side into the
    # canopy area, so their centers can sit below the old cleanup range while
    # their upper vertices still form the rejected ledge/pocket tips.
    cleanup_z_floor = -0.12
    cleanup_z_ceiling = 4.08
    socket_radius = socket_diameter / 2.0
    cleanup_inner_radius = max(socket_radius + 0.88, 5.05)
    cleanup_outer_radius = 7.54
    for split_z in (cleanup_z_floor, cleanup_z_ceiling):
        bmesh.ops.bisect_plane(
            bm,
            geom=[*bm.verts, *bm.edges, *bm.faces],
            plane_co=(0.0, 0.0, z_base + split_z),
            plane_no=(0.0, 0.0, 1.0),
            dist=0.0001,
            clear_outer=False,
            clear_inner=False,
        )
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # Clear the exterior lower support band after all core/thread cuts have
    # been applied. The test is vertex-aware so partial faces that dip into the
    # transition band are removed. Cutter-owned bore/socket/jaw faces are
    # protected, but any exterior yellow face crossing this band is deleted
    # before the clean six-panel canopy is inserted.
    faces_to_delete: list[bmesh.types.BMFace] = []
    for face in bm.faces:
        center = face.calc_center_median()
        vert_local_z = [vert.co.z - z_base for vert in face.verts]
        if max(vert_local_z) < cleanup_z_floor - 0.01 or min(vert_local_z) > cleanup_z_ceiling + 0.03:
            continue
        vert_radii = [math.hypot(vert.co.x, vert.co.y) for vert in face.verts]
        radius_xy = math.hypot(center.x, center.y)
        reaches_exterior_band = (
            cleanup_inner_radius <= radius_xy <= cleanup_outer_radius
            or min(vert_radii) <= cleanup_outer_radius and max(vert_radii) >= cleanup_inner_radius
        )
        is_cut_face = face.material_index != material_index
        is_central_cut_surface = min(vert_radii) <= socket_radius + 0.35 or radius_xy <= cleanup_inner_radius
        if is_cut_face and is_central_cut_surface:
            continue
        if reaches_exterior_band:
            faces_to_delete.append(face)
    if faces_to_delete:
        bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")
        loose_edges = [edge for edge in bm.edges if not edge.link_faces]
        if loose_edges:
            bmesh.ops.delete(bm, geom=loose_edges, context="EDGES")
        loose_verts = [vert for vert in bm.verts if not vert.link_faces]
        if loose_verts:
            bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")

    segment_count = 6
    # One coarse sloped skin is closer to the OpenSCAD hull than a skirted
    # overlay: broad facets rise from the hex stem crown into the first round
    # collet/thread level, with no separate horizontal collar ring or downward
    # panel tips.
    ring_specs = [
        (0.02, 5.18),
        (3.92, 6.72),
    ]
    new_faces = []
    panel_verts: list[list[bmesh.types.BMVert]] = []
    for z_rel, radius in ring_specs:
        ring: list[bmesh.types.BMVert] = []
        for index in range(segment_count):
            theta = rotation_z + math.tau * index / segment_count
            ring.append(
                bm.verts.new(
                    (
                        radius * math.cos(theta),
                        radius * math.sin(theta),
                        z_base + z_rel,
                    )
                )
            )
        panel_verts.append(ring)

    bm.verts.ensure_lookup_table()
    for level in range(len(panel_verts) - 1):
        for index in range(segment_count):
            try:
                face = bm.faces.new(
                    (
                        panel_verts[level][index],
                        panel_verts[level][(index + 1) % segment_count],
                        panel_verts[level + 1][(index + 1) % segment_count],
                        panel_verts[level + 1][index],
                    )
                )
                new_faces.append(face)
            except ValueError:
                continue
    for face in new_faces:
        face.material_index = material_index
        face.smooth = False

    bm.to_mesh(mesh)
    mesh.update()
    bm.free()
    return target


def _make_source_thread_union(
    name: str,
    *,
    z_offset: float,
    spread: float,
    radius: float,
    taper: float,
    turns: float,
    thread_radius: float,
    material: MaterialSpec,
    max_local_z: float | None = None,
) -> bpy.types.Object:
    """Approximate the OpenSCAD thread() union as hulled low-poly spans."""
    return _make_source_thread_bead_span_mesh(
        name,
        z_offset=z_offset,
        spread=spread,
        radius=radius,
        taper=taper,
        turns=turns,
        thread_radius=thread_radius,
        material=material,
        max_local_z=max_local_z,
        envelope_clearance=0.018,
        ring_segments=10,
        step_stride=1,
    )


def _make_thread_envelope(
    name: str,
    *,
    collet_diameter: float,
    collet_length: float,
    z_min: float,
    material: MaterialSpec,
    vertices: int = 72,
) -> bpy.types.Object:
    """Build the SCAD intersection envelope around the thread."""
    lower = add_cone(
        f"{name}_lower_flare",
        diameter1=collet_diameter,
        diameter2=collet_diameter + 4.0,
        height=4.0,
        z_min=z_min,
        vertices=vertices,
        material=material,
        smooth=False,
    )
    middle = add_cylinder(
        f"{name}_middle_band",
        diameter=collet_diameter + 4.0,
        height=collet_length - 8.0,
        z_min=z_min + 4.0,
        vertices=vertices,
        material=material,
        smooth=False,
    )
    upper = add_cone(
        f"{name}_upper_flare",
        diameter1=collet_diameter + 4.0,
        diameter2=collet_diameter,
        height=4.0,
        z_min=z_min + collet_length - 4.0,
        vertices=vertices,
        material=material,
        smooth=False,
    )
    return boolean_union(name, [lower, middle, upper], material)


def _make_collet_core_union(
    name: str,
    *,
    bit_diameter: float,
    collet_diameter: float,
    stem_length: float,
    collet_length: float,
    tip_length: float,
    material: MaterialSpec,
    angular_segments: int,
    hex_rotation: float,
) -> bpy.types.Object:
    """Build the non-thread source union with a broad supportless lower hull."""
    stem = add_cylinder(
        f"{name}_hex_stem",
        diameter=collet_diameter,
        height=stem_length,
        z_min=0.0,
        vertices=6,
        material=material,
        smooth=False,
        rotation_z=hex_rotation,
    )
    lower_hull = add_supportless_thread_transition_mesh(
        f"{name}_supportless_hull",
        material,
        z_base=stem_length,
        angular_segments=max(48, angular_segments // 2),
        rotation_z=hex_rotation,
    )
    threaded_core = add_cone(
        f"{name}_tapered_thread_core",
        diameter1=collet_diameter,
        diameter2=collet_diameter - 2.0,
        height=collet_length,
        z_min=stem_length,
        vertices=angular_segments,
        material=material,
        smooth=False,
    )
    tip = add_cone(
        f"{name}_nose_taper",
        diameter1=collet_diameter - 2.0,
        diameter2=bit_diameter * 2.0,
        height=tip_length,
        z_min=stem_length + collet_length,
        vertices=angular_segments,
        material=material,
        smooth=False,
    )
    return boolean_union(name, [stem, lower_hull, threaded_core, tip], material)


def add_direct_cut_collet_model(
    name: str,
    material: MaterialSpec,
    cut_material: MaterialSpec,
    *,
    bit_diameter: float,
    bore_diameter: float,
    shaft_socket_diameter: float,
    collet_diameter: float,
    stem_length: float,
    collet_length: float,
    tip_length: float,
    slot_width: float,
    slot_length: float,
    jaw_count: int,
    thread_radius: float,
    angular_segments: int = 72,
    hex_rotation: float = math.radians(30.0),
) -> bpy.types.Object:
    """Build the collet with source-like solids and exact cutter subtractions.

    The current_version direct sampled mesh kept the socket/bore split but made
    the jaw slots and thread ridges visibly stair-stepped. This active builder
    goes back to the source construction order: union the core, broad lower hull,
    and low-poly hulled helical thread, then subtract the round bore, three
    rectangular slot cubes, and underside-only hex socket.
    """
    total_height = stem_length + collet_length + tip_length
    thread_z_max = stem_length + collet_length
    slot_z_min = stem_length + 0.10
    socket_z_top = stem_length - 0.10
    # Match the source cutter as a half-depth rectangular cube whose far end is
    # beyond the thread envelope, so no square cutter end becomes an exterior
    # model feature.
    slot_cut_length = max(slot_length, collet_diameter / 2.0 + thread_radius + 3.0)

    core = _make_collet_core_union(
        f"{name}_core",
        bit_diameter=bit_diameter,
        collet_diameter=collet_diameter,
        stem_length=stem_length,
        collet_length=collet_length,
        tip_length=tip_length,
        material=material,
        angular_segments=angular_segments,
        hex_rotation=hex_rotation,
    )
    thread = _make_source_thread_union(
        f"{name}_thread",
        z_offset=stem_length,
        spread=4.0,
        radius=collet_diameter / 2.0,
        taper=0.5 / 36.0,
        turns=4.0,
        thread_radius=thread_radius,
        material=material,
        max_local_z=collet_length,
    )
    core_cutters: list[bpy.types.Object] = [
        make_hex_socket_cutter(
            f"{name}_hex_socket_cut",
            diameter=shaft_socket_diameter,
            depth=socket_z_top + 0.015,
            z_min=-0.015,
            rotation_z=hex_rotation,
            material=cut_material,
        ),
        make_bore_cutter(
            f"{name}_round_bit_bore_cut",
            diameter=bore_diameter,
            z_min=0.0,
            z_max=total_height + 0.40,
            vertices=angular_segments,
            material=cut_material,
        ),
    ]
    thread_cutters: list[bpy.types.Object] = [
        make_bore_cutter(
            f"{name}_thread_round_bit_bore_cut",
            diameter=bore_diameter,
            z_min=0.0,
            z_max=total_height + 0.40,
            vertices=angular_segments,
            material=cut_material,
        )
    ]
    for jaw_index in range(jaw_count):
        angle = 360.0 * jaw_index / jaw_count
        core_cutters.append(
            make_slot_cutter(
                f"{name}_jaw_slot_cut_{jaw_index + 1}",
                angle_deg=angle,
                width=slot_width,
                length=slot_cut_length,
                z_min=slot_z_min,
                z_max=total_height + 0.18,
                material=cut_material,
            )
        )
        thread_cutters.append(
            make_slot_cutter(
                f"{name}_thread_jaw_slot_cut_{jaw_index + 1}",
                angle_deg=angle,
                width=slot_width,
                length=slot_cut_length,
                z_min=slot_z_min,
                z_max=total_height + 0.18,
                material=cut_material,
            )
        )

    apply_boolean_difference(core, core_cutters)
    _trim_thread_faces_for_source_cuts(
        thread,
        bore_diameter=bore_diameter,
        jaw_count=jaw_count,
        slot_width=slot_width,
        slot_length=slot_cut_length,
        z_min=slot_z_min,
        z_max=total_height + 0.18,
    )
    for cutter in thread_cutters:
        bpy.data.objects.remove(cutter, do_unlink=True)
    target = combine_cut_mesh_objects(name, [core, thread], material)
    trim_boolean_transfer_artifacts(
        target,
        max_radius=collet_diameter / 2.0 + 2.0,
        max_z=total_height,
    )
    _replace_lower_transition_with_clean_canopy(
        target,
        material=material,
        z_base=stem_length,
        rotation_z=hex_rotation,
        socket_diameter=shaft_socket_diameter,
    )
    target.name = name
    target.data.name = f"{name}_mesh"
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    target.select_set(False)

    for polygon in target.data.polygons:
        polygon.use_smooth = False
        center = sum((target.data.vertices[index].co for index in polygon.vertices), Vector()) / len(polygon.vertices)
        radius_xy = math.hypot(center.x, center.y)
        if stem_length <= center.z <= thread_z_max + 0.04:
            local_z = center.z - stem_length
            core_radius = collet_diameter / 2.0 - local_z / collet_length
            if polygon.material_index == 0 and radius_xy >= core_radius + thread_radius * 0.48:
                polygon.use_smooth = False
        elif center.z >= thread_z_max - 0.02:
            polygon.use_smooth = True

    bpy.context.view_layer.update()
    return target


def make_bore_cutter(
    name: str,
    diameter: float,
    z_min: float,
    z_max: float,
    vertices: int = 72,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    return add_cylinder(
        name=name,
        diameter=diameter,
        height=z_max - z_min,
        z_min=z_min,
        vertices=vertices,
        material=material,
        smooth=True,
    )


def make_slot_cutter(
    name: str,
    angle_deg: float,
    width: float,
    length: float,
    z_min: float,
    z_max: float,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    angle = math.radians(angle_deg)
    center = (
        -math.sin(angle) * length / 2.0,
        math.cos(angle) * length / 2.0,
        z_min + (z_max - z_min) / 2.0,
    )
    cutter = add_box(
        name=name,
        size=(width, length, z_max - z_min),
        location=center,
        material=material,
        rotation=(0.0, 0.0, angle),
    )
    return cutter


def make_hex_socket_cutter(
    name: str,
    diameter: float,
    depth: float,
    z_min: float,
    rotation_z: float = math.radians(30.0),
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    return add_cylinder(
        name=name,
        diameter=diameter,
        height=depth,
        z_min=z_min,
        vertices=6,
        material=material,
        smooth=False,
        rotation_z=rotation_z,
    )


# ============================================================
# Procedural assembly
# ============================================================

"""Assembly logic for the small bit chuck collet model."""






BIT_DIAMETER = 3.175
SHAFT_DIAMETER = 7.1
TOLERANCE = 0.4
COLLET_DIAMETER = 12.0
COLLET_LENGTH = 10.0
TIP_LENGTH = 5.0
STEM_LENGTH = 10.0
JAW_COUNT = 3

BASE_Z = 0.0
THREAD_Z0 = STEM_LENGTH
TIP_Z0 = STEM_LENGTH + COLLET_LENGTH
TOTAL_HEIGHT = STEM_LENGTH + COLLET_LENGTH + TIP_LENGTH

HEX_SOCKET_DIAMETER = SHAFT_DIAMETER + TOLERANCE
SLOT_WIDTH = BIT_DIAMETER / 2.0
BORE_CUT_DIAMETER = BIT_DIAMETER * 1.01
SLOT_CUT_WIDTH = SLOT_WIDTH


def build_model() -> list[bpy.types.Object]:
    """Build the default SCAD `part = "collet"` object."""
    collet = build_integrated_collet_solid()
    return [collet]


def build_integrated_collet_solid() -> bpy.types.Object:
    """Create the visible collet with source-registered cuts built into it."""
    collet = add_direct_cut_collet_model(
        name="direct_slot_cut_threaded_collet",
        material=SCAD_YELLOW,
        cut_material=CUT_GREEN,
        bit_diameter=BIT_DIAMETER,
        bore_diameter=BORE_CUT_DIAMETER,
        shaft_socket_diameter=HEX_SOCKET_DIAMETER * 1.10,
        collet_diameter=COLLET_DIAMETER,
        stem_length=STEM_LENGTH,
        collet_length=COLLET_LENGTH,
        tip_length=TIP_LENGTH,
        slot_width=SLOT_CUT_WIDTH,
        slot_length=COLLET_DIAMETER,
        jaw_count=JAW_COUNT,
        thread_radius=1.50,
        angular_segments=72,
        hex_rotation=math.radians(30.0),
    )
    return collet


def jaw_angles() -> tuple[float, ...]:
    return tuple(index * 360.0 / JAW_COUNT for index in range(JAW_COUNT))


def add_weighted_normals(obj: bpy.types.Object) -> None:
    weighted = obj.modifiers.new("weighted_normals_for_machined_faces", "WEIGHTED_NORMAL")
    weighted.keep_sharp = True


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

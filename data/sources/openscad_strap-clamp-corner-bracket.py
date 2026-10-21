from __future__ import annotations
from dataclasses import dataclass
from mathutils import Matrix, Vector
from typing import Iterable
import bmesh
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable OpenSCAD-like primitive builders for the strap clamp bracket."""





@dataclass(frozen=True)
class ClampSpec:
    """Source parameters and derived dimensions from the OpenSCAD model."""

    strap_width: float = 40.0
    strap_thickness: float = 3.0
    clamp_thickness: float = 2.0
    clamp_length: float = 60.0
    clamp_angle_degrees: float = 90.0
    radial_segments: int = 96

    @property
    def total_height(self) -> float:
        return self.strap_width + self.clamp_thickness * 2.0

    @property
    def channel_offset(self) -> float:
        return self.clamp_thickness * 2.0 + self.strap_thickness

    @property
    def short_lip_height(self) -> float:
        return self.strap_width * 0.1 + self.clamp_thickness

    @property
    def upper_step_z(self) -> float:
        return self.total_height - self.strap_thickness

    @property
    def relief_height(self) -> float:
        return self.total_height


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.62


OPENSCAD_GOLD = MaterialSpec("openscad_cgal_gold", (1.0, 0.72, 0.02, 1.0))
BOOLEAN_GREEN = MaterialSpec("openscad_boolean_cut_green", (0.42, 0.70, 0.18, 1.0))


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(spec: MaterialSpec) -> bpy.types.Material:
    material = bpy.data.materials.get(spec.name) or bpy.data.materials.new(spec.name)
    material.diffuse_color = spec.color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = spec.color
        bsdf.inputs["Roughness"].default_value = spec.roughness
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = spec.color
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 0.12
    return material


def assign_material(obj: bpy.types.Object, spec: MaterialSpec) -> bpy.types.Object:
    material = make_material(spec)
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
    return obj


def cylinder_extreme_points(
    center_xy: tuple[float, float],
    z_min: float,
    height: float,
    radius: float,
    segments: int,
) -> list[Vector]:
    """Sample the top and bottom rims of a vertical SCAD cylinder."""

    cx, cy = center_xy
    points: list[Vector] = []
    for z in (z_min, z_min + height):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            points.append(
                Vector(
                    (
                        cx + radius * math.cos(angle),
                        cy + radius * math.sin(angle),
                        z,
                    )
                )
            )
    return points


def _dedupe_points(points: Iterable[Vector], precision: int = 6) -> list[Vector]:
    deduped: list[Vector] = []
    seen: set[tuple[float, float, float]] = set()
    for point in points:
        key = (round(point.x, precision), round(point.y, precision), round(point.z, precision))
        if key not in seen:
            seen.add(key)
            deduped.append(point)
    return deduped


def add_convex_hull_from_points(
    name: str,
    points: Iterable[Vector],
    material: MaterialSpec | None = OPENSCAD_GOLD,
    *,
    smooth_faces: bool = False,
) -> bpy.types.Object:
    """Create a closed 3D convex hull mesh from sampled child primitive points."""

    unique_points = _dedupe_points(points)
    if len(unique_points) < 4:
        raise ValueError(f"{name} needs at least four non-coplanar points")

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    verts = [bm.verts.new(point) for point in unique_points]
    bm.verts.ensure_lookup_table()
    hull = bmesh.ops.convex_hull(bm, input=verts)

    to_delete = []
    seen: set[int] = set()
    for key in ("geom_interior", "geom_unused", "geom_holes"):
        for element in hull.get(key, []):
            if element.is_valid and id(element) not in seen:
                seen.add(id(element))
                to_delete.append(element)
    if to_delete:
        bmesh.ops.delete(bm, geom=to_delete, context="VERTS")

    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    mesh.validate(clean_customdata=False)
    mesh.update(calc_edges=True)

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for polygon in obj.data.polygons:
        polygon.use_smooth = smooth_faces
    if material is not None:
        assign_material(obj, material)
    return obj


def add_cylinder(
    name: str,
    radius: float,
    depth: float,
    location: tuple[float, float, float],
    vertices: int = 96,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        end_fill_type="NGON",
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    if material is not None:
        assign_material(obj, material)
    return obj


def apply_boolean_modifier(
    target: bpy.types.Object,
    cutter: bpy.types.Object,
    operation: str,
    modifier_name: str,
) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    modifier = target.modifiers.new(modifier_name, "BOOLEAN")
    modifier.operation = operation
    modifier.object = cutter
    if hasattr(modifier, "solver"):
        modifier.solver = "EXACT"
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    target.select_set(False)
    recalculate_normals(target)
    return target


def boolean_union(name: str, objects: Iterable[bpy.types.Object]) -> bpy.types.Object:
    objects = list(objects)
    if not objects:
        raise ValueError("boolean_union requires at least one object")
    target = objects[0]
    for obj in objects[1:]:
        apply_boolean_modifier(target, obj, "UNION", f"union_{obj.name}")
        bpy.data.objects.remove(obj, do_unlink=True)
    target.name = name
    target.data.name = f"{name}_mesh"
    recalculate_normals(target)
    return target


def boolean_difference(
    target: bpy.types.Object,
    cutters: Iterable[bpy.types.Object],
    *,
    modifier_prefix: str = "cut",
) -> bpy.types.Object:
    for index, cutter in enumerate(cutters, start=1):
        apply_boolean_modifier(target, cutter, "DIFFERENCE", f"{modifier_prefix}_{index}_{cutter.name}")
        bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def recalculate_normals(obj: bpy.types.Object) -> bpy.types.Object:
    if obj.type != "MESH":
        return obj
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    obj.data.update(calc_edges=True)
    return obj


def remove_doubles(obj: bpy.types.Object, distance: float = 1e-5) -> bpy.types.Object:
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=distance)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update(calc_edges=True)
    return obj


def transform_mesh_object(obj: bpy.types.Object, matrix: Matrix) -> bpy.types.Object:
    obj.data.transform(matrix)
    obj.data.update(calc_edges=True)
    return obj


def mirror_y_then_rotate_minus_z(points: Iterable[Vector], angle_degrees: float) -> list[Vector]:
    matrix = Matrix.Rotation(math.radians(-angle_degrees), 4, "Z") @ Matrix.Diagonal((1.0, -1.0, 1.0, 1.0))
    return [matrix @ point for point in points]


def add_source_hull(
    name: str,
    cylinders: Iterable[tuple[tuple[float, float], float, float, float]],
    spec: ClampSpec,
    material: MaterialSpec | None = OPENSCAD_GOLD,
) -> bpy.types.Object:
    """Hull a set of vertical cylinder children.

    Each cylinder tuple is ``((x, y), z_min, height, radius)`` in the source
    coordinate system, matching the OpenSCAD hull children.
    """

    points: list[Vector] = []
    for center_xy, z_min, height, radius in cylinders:
        points.extend(cylinder_extreme_points(center_xy, z_min, height, radius, spec.radial_segments))
    return add_convex_hull_from_points(name, points, material=material)


def add_relief_cutter(spec: ClampSpec) -> bpy.types.Object:
    """The small hull-subtracted glue squeeze relief at the inside corner."""

    t = spec.clamp_thickness
    points: list[Vector] = []
    points.extend(cylinder_extreme_points((0.0, 0.0), 0.0, spec.relief_height, t * 0.5, spec.radial_segments))
    points.extend(cylinder_extreme_points((t, -t), 0.0, spec.relief_height, t * 0.75, spec.radial_segments))
    return add_convex_hull_from_points("inside_corner_glue_relief_cutter", points, material=None)


def add_green_relief_face(spec: ClampSpec) -> bpy.types.Object:
    """Thin visual insert for the CGAL green boolean face visible in source renders."""

    t = spec.clamp_thickness
    cutter = add_relief_cutter(spec)
    cutter.name = "inside_corner_glue_relief_green_face"
    assign_material(cutter, BOOLEAN_GREEN)

    # Keep the insert slightly inside the real cut so it reads as the exposed
    # boolean surface without changing the bracket's silhouette.
    matrix = Matrix.Diagonal((0.76, 0.76, 0.98, 1.0)) @ Matrix.Translation(Vector((0.0, 0.0, spec.relief_height * 0.01)))
    cutter.data.transform(matrix)
    cutter.data.update(calc_edges=True)
    return cutter


# ============================================================
# Procedural assembly
# ============================================================

"""Readable assembly of the OpenSCAD strap clamp corner bracket."""





def _frame_hulls(prefix: str, spec: ClampSpec, *, mirrored: bool = False) -> list[bpy.types.Object]:
    t = spec.clamp_thickness
    s = spec.strap_thickness
    w = spec.strap_width
    length = spec.clamp_length
    total_height = spec.total_height

    hull_sets = [
        (
            "base_plate",
            [
                ((length, 0.0), 0.0, t, t * 0.5),
                ((-t * 0.5, t * 0.5), 0.0, t, t),
                ((0.0, t * 2.0 + s), 0.0, t, t * 0.5),
                ((length, t + s), 0.0, t, t * 0.5),
            ],
        ),
        (
            "tall_outer_wall",
            [
                ((length, 0.0), 0.0, total_height, t * 0.5),
                ((-t * 0.5, t * 0.5), 0.0, total_height, t),
            ],
        ),
        (
            "low_inner_rail",
            [
                ((0.0, t * 2.0 + s), 0.0, w * 0.1 + t, t * 0.5),
                ((length, t + s), 0.0, w * 0.1 + t, t * 0.5),
            ],
        ),
        (
            "upper_stepped_lip",
            [
                ((0.0, t * 0.5), total_height - s, s, t),
                ((length, 0.0), total_height - s, s, t * 0.5),
                ((-t * 1.5, t * 1.5), total_height - 0.1, 0.1, t * 0.5),
                ((length, t * 0.5), total_height - 0.1, 0.1, t * 0.5),
            ],
        ),
    ]

    objects: list[bpy.types.Object] = []
    for name, cylinders in hull_sets:
        obj = add_source_hull(f"{prefix}_{name}", cylinders, spec, material=OPENSCAD_GOLD)
        if mirrored:
            transform = mirror_y_then_rotate_minus_z
            transformed = transform(
                [vertex.co.copy() for vertex in obj.data.vertices],
                spec.clamp_angle_degrees,
            )
            for vertex, point in zip(obj.data.vertices, transformed):
                vertex.co = point
            obj.data.update(calc_edges=True)
        objects.append(obj)
    return objects


def _outside_corner_hull_points(spec: ClampSpec) -> list:
    t = spec.clamp_thickness
    return [
        ((0.0, 0.0), 0.0, t, t * 0.5),
        ((0.0, t * 2.0 + spec.strap_thickness), 0.0, t, t * 0.5),
    ]


def _add_corner_fill(spec: ClampSpec) -> bpy.types.Object:
    cylinders = _outside_corner_hull_points(spec)
    first = add_source_hull("corner_fill_source_side_points", cylinders, spec, material=OPENSCAD_GOLD)
    second = add_source_hull("corner_fill_rotated_side_points", cylinders, spec, material=OPENSCAD_GOLD)
    transformed = mirror_y_then_rotate_minus_z(
        [vertex.co.copy() for vertex in second.data.vertices],
        spec.clamp_angle_degrees,
    )
    for vertex, point in zip(second.data.vertices, transformed):
        vertex.co = point
    second.data.update(calc_edges=True)

    points = [vertex.co.copy() for vertex in first.data.vertices]
    points.extend(vertex.co.copy() for vertex in second.data.vertices)
    bpy.data.objects.remove(first, do_unlink=True)
    bpy.data.objects.remove(second, do_unlink=True)
    return add_convex_hull_from_points("corner_miter_fill", points, material=OPENSCAD_GOLD)


def build_model() -> list[bpy.types.Object]:
    """Build the bracket as a union of source-equivalent hull solids."""

    spec = ClampSpec()
    objects: list[bpy.types.Object] = []
    objects.extend(_frame_hulls("left_arm", spec, mirrored=False))
    objects.extend(_frame_hulls("right_arm", spec, mirrored=True))
    objects.append(_add_corner_fill(spec))

    bracket = boolean_union("strap_clamp_corner_bracket_body", objects)
    boolean_difference(bracket, [add_relief_cutter(spec)], modifier_prefix="inside_corner_relief")
    remove_doubles(bracket)
    assign_material(bracket, OPENSCAD_GOLD)

    relief_face = add_green_relief_face(spec)
    return [bracket, relief_face]


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

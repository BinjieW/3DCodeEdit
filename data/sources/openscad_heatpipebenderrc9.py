from __future__ import annotations
from dataclasses import dataclass
from mathutils import Vector
from typing import Iterable
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable geometry helpers for the heatpipe bender model."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.48
    metallic: float = 0.0


YELLOW = MaterialSpec("preview_yellow_plastic", (1.0, 0.78, 0.0, 1.0), 0.42)
LIME = MaterialSpec("preview_lime_bracket", (0.58, 1.0, 0.18, 1.0), 0.45)
BLUE = MaterialSpec("preview_blue_press_lever", (0.05, 0.22, 0.95, 1.0), 0.46)
MAGENTA = MaterialSpec("preview_magenta_flat_axle", (1.0, 0.0, 0.78, 1.0), 0.46)
SHADOW = MaterialSpec("recess_shadow", (0.10, 0.10, 0.04, 1.0), 0.70)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def material(spec: MaterialSpec) -> bpy.types.Material:
    mat = bpy.data.materials.get(spec.name) or bpy.data.materials.new(spec.name)
    mat.diffuse_color = spec.color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = spec.color
        bsdf.inputs["Roughness"].default_value = spec.roughness
        bsdf.inputs["Metallic"].default_value = spec.metallic
    return mat


def assign_material(obj: bpy.types.Object, spec: MaterialSpec) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(material(spec))
    return obj


def shade_smooth(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)
    return obj


def shade_flat(obj: bpy.types.Object) -> bpy.types.Object:
    for polygon in obj.data.polygons:
        polygon.use_smooth = False
    return obj


def apply_modifier(obj: bpy.types.Object, modifier: bpy.types.Modifier) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    obj.select_set(False)
    return obj


def configure_boolean(modifier: bpy.types.Modifier) -> None:
    """Use Blender's exact solver when available; these parts rely on real cut-through voids."""
    if hasattr(modifier, "solver"):
        modifier.solver = "EXACT"


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    mat: MaterialSpec | None = None,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat:
        assign_material(obj, mat)
    return obj


def add_cylinder(
    name: str,
    radius: float,
    height: float,
    location: tuple[float, float, float],
    mat: MaterialSpec | None = None,
    vertices: int = 96,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=height,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    if mat:
        assign_material(obj, mat)
    shade_smooth(obj)
    return obj


def add_scaled_cylinder(
    name: str,
    radius: float,
    height: float,
    location: tuple[float, float, float],
    scale_xy: tuple[float, float],
    mat: MaterialSpec | None = None,
    vertices: int = 96,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    obj = add_cylinder(name, radius, height, location, mat, vertices, rotation)
    obj.scale.x *= scale_xy[0]
    obj.scale.y *= scale_xy[1]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.select_set(False)
    return obj


def add_torus(
    name: str,
    major_radius: float,
    minor_radius: float,
    location: tuple[float, float, float],
    mat: MaterialSpec,
    major_segments: int = 128,
    minor_segments: int = 18,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_torus_add(
        major_segments=major_segments,
        minor_segments=minor_segments,
        major_radius=major_radius,
        minor_radius=minor_radius,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    assign_material(obj, mat)
    shade_smooth(obj)
    return obj


def boolean_union(name: str, objects: Iterable[bpy.types.Object], mat: MaterialSpec | None = None) -> bpy.types.Object:
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
        configure_boolean(modifier)
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(obj, do_unlink=True)
    active.name = name
    if mat:
        assign_material(active, mat)
    return active


def boolean_difference(
    name: str,
    base: bpy.types.Object,
    cutters: Iterable[bpy.types.Object],
    mat: MaterialSpec | None = None,
) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = base
    base.select_set(True)
    for cutter in cutters:
        modifier = base.modifiers.new(f"cut_{cutter.name}", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        configure_boolean(modifier)
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
    base.name = name
    if mat:
        assign_material(base, mat)
    base.select_set(False)
    return base


def add_annular_cylinder(
    name: str,
    outer_radius: float,
    inner_radius: float,
    height: float,
    location: tuple[float, float, float],
    mat: MaterialSpec,
    vertices: int = 128,
) -> bpy.types.Object:
    outer = add_cylinder(f"{name}_outer", outer_radius, height, location, mat, vertices)
    cutter = add_cylinder(f"{name}_inner_cut", inner_radius, height + 0.8, location, None, vertices)
    return boolean_difference(name, outer, [cutter], mat)


def add_rounded_box(
    name: str,
    size: tuple[float, float, float],
    radius: float,
    location: tuple[float, float, float],
    mat: MaterialSpec,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    segments: int = 12,
) -> bpy.types.Object:
    obj = add_box(name, size, location, mat, rotation)
    bevel = obj.modifiers.new("small printed edge radius", "BEVEL")
    bevel.width = min(radius, size[0] * 0.22, size[1] * 0.22, size[2] * 0.22)
    bevel.segments = segments
    bevel.profile = 0.5
    apply_modifier(obj, bevel)
    obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))
    return obj


def add_tube_arc(
    name: str,
    major_radius: float,
    tube_radius: float,
    z_center: float,
    start_degrees: float,
    end_degrees: float,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mat: MaterialSpec | None = None,
    major_segments: int = 96,
    tube_segments: int = 16,
    scale_xy: tuple[float, float] = (1.0, 1.0),
) -> bpy.types.Object:
    """Create a capped circular tube along an arc, suitable as a boolean pipe-groove cutter."""
    start = math.radians(start_degrees)
    end = math.radians(end_degrees)
    if end < start:
        end += math.tau

    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for i in range(major_segments + 1):
        theta = start + (end - start) * i / major_segments
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        for j in range(tube_segments):
            phi = math.tau * j / tube_segments
            radius = major_radius + tube_radius * math.cos(phi)
            verts.append(
                (
                    location[0] + radius * cos_theta * scale_xy[0],
                    location[1] + radius * sin_theta * scale_xy[1],
                    location[2] + z_center + tube_radius * math.sin(phi),
                )
            )

    for i in range(major_segments):
        row = i * tube_segments
        next_row = (i + 1) * tube_segments
        for j in range(tube_segments):
            faces.append((row + j, row + (j + 1) % tube_segments, next_row + (j + 1) % tube_segments, next_row + j))

    faces.append(tuple(range(tube_segments - 1, -1, -1)))
    last_row = major_segments * tube_segments
    faces.append(tuple(range(last_row, last_row + tube_segments)))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if mat:
        assign_material(obj, mat)
    shade_smooth(obj)
    return obj


def add_capsule_prism(
    name: str,
    length: float,
    width: float,
    height: float,
    center: tuple[float, float, float],
    mat: MaterialSpec,
    angle_degrees: float = 0.0,
) -> bpy.types.Object:
    radius = width / 2.0
    parts: list[bpy.types.Object] = [
        add_box(f"{name}_web", (length - width, width, height), center, mat)
    ]
    for side in (-1.0, 1.0):
        parts.append(
            add_cylinder(
                f"{name}_cap_{side:g}",
                radius,
                height,
                (center[0] + side * (length / 2.0 - radius), center[1], center[2]),
                mat,
                vertices=72,
            )
        )
    obj = boolean_union(name, parts, mat)
    if angle_degrees:
        obj.rotation_euler[2] = math.radians(angle_degrees)
    return obj


def add_rounded_rect_prism(
    name: str,
    size: tuple[float, float, float],
    corner_radius: float,
    center: tuple[float, float, float],
    mat: MaterialSpec,
    segments: int = 10,
) -> bpy.types.Object:
    """Extrude a flat-topped XY rounded rectangle, matching an OpenSCAD cube/cylinder minkowski bar."""
    sx, sy, sz = size
    radius = min(corner_radius, sx * 0.5 - 0.01, sy * 0.5 - 0.01)
    x0, x1 = -sx / 2.0, sx / 2.0
    y0, y1 = -sy / 2.0, sy / 2.0
    corners = (
        ((x1 - radius, y1 - radius), 0.0, 90.0),
        ((x0 + radius, y1 - radius), 90.0, 180.0),
        ((x0 + radius, y0 + radius), 180.0, 270.0),
        ((x1 - radius, y0 + radius), 270.0, 360.0),
    )

    outline: list[tuple[float, float]] = []
    for (cx, cy), start, end in corners:
        for i in range(segments + 1):
            if outline and i == 0:
                continue
            angle = math.radians(start + (end - start) * i / segments)
            outline.append((center[0] + cx + radius * math.cos(angle), center[1] + cy + radius * math.sin(angle)))

    z0 = center[2] - sz / 2.0
    z1 = center[2] + sz / 2.0
    verts = [(x, y, z0) for x, y in outline] + [(x, y, z1) for x, y in outline]
    count = len(outline)
    faces: list[tuple[int, ...]] = [tuple(range(count - 1, -1, -1)), tuple(range(count, count * 2))]
    for i in range(count):
        j = (i + 1) % count
        faces.append((i, j, count + j, count + i))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, mat)
    return obj


def add_arc_mesh(
    name: str,
    inner_radius: float,
    outer_radius: float,
    height: float,
    start_degrees: float,
    end_degrees: float,
    z_bottom: float,
    mat: MaterialSpec,
    segments: int = 96,
) -> bpy.types.Object:
    start = math.radians(start_degrees)
    end = math.radians(end_degrees)
    if end < start:
        end += math.tau
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for i in range(segments + 1):
        t = start + (end - start) * i / segments
        co = math.cos(t), math.sin(t)
        verts.extend(
            [
                (inner_radius * co[0], inner_radius * co[1], z_bottom),
                (outer_radius * co[0], outer_radius * co[1], z_bottom),
                (inner_radius * co[0], inner_radius * co[1], z_bottom + height),
                (outer_radius * co[0], outer_radius * co[1], z_bottom + height),
            ]
        )
    for i in range(segments):
        a = i * 4
        b = (i + 1) * 4
        faces.append((a, b, b + 2, a + 2))
        faces.append((a + 1, a + 3, b + 3, b + 1))
        faces.append((a + 2, b + 2, b + 3, a + 3))
        faces.append((a, a + 1, b + 1, b))
    faces.extend([(0, 2, 3, 1), (segments * 4, segments * 4 + 1, segments * 4 + 3, segments * 4 + 2)])
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, mat)
    shade_smooth(obj)
    return obj


def add_cylindrical_shell_segment(
    name: str,
    radius: float,
    height: float,
    z_center: float,
    start_degrees: float,
    end_degrees: float,
    mat: MaterialSpec,
    segments: int = 96,
) -> bpy.types.Object:
    """Create a thin visible wall on the inside of a cylindrical recess."""
    start = math.radians(start_degrees)
    end = math.radians(end_degrees)
    if end < start:
        end += math.tau

    z0 = z_center - height / 2.0
    z1 = z_center + height / 2.0
    verts: list[tuple[float, float, float]] = []
    for i in range(segments + 1):
        theta = start + (end - start) * i / segments
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        verts.append((x, y, z0))
        verts.append((x, y, z1))

    faces: list[tuple[int, ...]] = []
    for i in range(segments):
        a = i * 2
        b = (i + 1) * 2
        faces.append((a, b, b + 1, a + 1))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, mat)
    shade_smooth(obj)
    return obj


def add_pipe_groove_sector_body(
    name: str,
    outer_radius: float,
    backing_radius: float,
    cutter_major_radius: float,
    pipe_levels: Iterable[tuple[float, float]],
    z_min: float,
    z_max: float,
    start_degrees: float,
    end_degrees: float,
    mat: MaterialSpec,
    trough_mat: MaterialSpec | None = None,
    trough_depth_threshold: float = 1.0,
    angle_segments: int = 112,
    z_segments: int = 96,
) -> bpy.types.Object:
    """Closed yellow shell whose outer wall follows the source round pipe-groove cutters."""
    levels = tuple(pipe_levels)
    start = math.radians(start_degrees)
    end = math.radians(end_degrees)
    if end < start:
        end += math.tau

    def radius_at_z(z: float) -> float:
        radius = outer_radius
        for pipe_radius, z_center in levels:
            dz = abs(z - z_center)
            if dz >= pipe_radius:
                continue
            cut_radius = cutter_major_radius - math.sqrt(pipe_radius * pipe_radius - dz * dz)
            radius = min(radius, cut_radius)
        return max(backing_radius + 0.35, radius)

    outer_indices: list[list[int]] = []
    inner_indices: list[list[int]] = []
    verts: list[tuple[float, float, float]] = []
    for ai in range(angle_segments + 1):
        theta = start + (end - start) * ai / angle_segments
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        outer_row: list[int] = []
        inner_row: list[int] = []
        for zi in range(z_segments + 1):
            z = z_min + (z_max - z_min) * zi / z_segments
            radius = radius_at_z(z)
            outer_row.append(len(verts))
            verts.append((radius * cos_t, radius * sin_t, z))
            inner_row.append(len(verts))
            verts.append((backing_radius * cos_t, backing_radius * sin_t, z))
        outer_indices.append(outer_row)
        inner_indices.append(inner_row)

    faces: list[tuple[int, ...]] = []
    outer_face_indices: list[int] = []
    for ai in range(angle_segments):
        for zi in range(z_segments):
            outer_face_indices.append(len(faces))
            faces.append(
                (
                    outer_indices[ai][zi],
                    outer_indices[ai + 1][zi],
                    outer_indices[ai + 1][zi + 1],
                    outer_indices[ai][zi + 1],
                )
            )
            faces.append(
                (
                    inner_indices[ai][zi + 1],
                    inner_indices[ai + 1][zi + 1],
                    inner_indices[ai + 1][zi],
                    inner_indices[ai][zi],
                )
            )

    for ai in range(angle_segments):
        faces.append(
            (
                outer_indices[ai][0],
                inner_indices[ai][0],
                inner_indices[ai + 1][0],
                outer_indices[ai + 1][0],
            )
        )
        faces.append(
            (
                outer_indices[ai + 1][z_segments],
                inner_indices[ai + 1][z_segments],
                inner_indices[ai][z_segments],
                outer_indices[ai][z_segments],
            )
        )

    for zi in range(z_segments):
        faces.append(
            (
                outer_indices[0][zi + 1],
                inner_indices[0][zi + 1],
                inner_indices[0][zi],
                outer_indices[0][zi],
            )
        )
        faces.append(
            (
                outer_indices[angle_segments][zi],
                inner_indices[angle_segments][zi],
                inner_indices[angle_segments][zi + 1],
                outer_indices[angle_segments][zi + 1],
            )
        )

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if trough_mat:
        obj.data.materials.append(material(mat))
        obj.data.materials.append(material(trough_mat))
        for face_index in outer_face_indices:
            polygon = obj.data.polygons[face_index]
            avg_radius = sum(math.hypot(mesh.vertices[index].co.x, mesh.vertices[index].co.y) for index in polygon.vertices) / len(polygon.vertices)
            if avg_radius < outer_radius - trough_depth_threshold:
                polygon.material_index = 1
    else:
        assign_material(obj, mat)
    shade_smooth(obj)
    return obj


def add_vertical_hole_marker(
    name: str,
    radius: float,
    height: float,
    location: tuple[float, float, float],
    wall_mat: MaterialSpec,
    recess_mat: MaterialSpec = SHADOW,
) -> list[bpy.types.Object]:
    ring = add_annular_cylinder(f"{name}_rim", radius * 1.24, radius, height, location, wall_mat, vertices=80)
    dark = add_cylinder(f"{name}_open_core", radius * 0.94, height * 1.02, location, recess_mat, vertices=80)
    return [ring, dark]


def add_horizontal_hole_marker(
    name: str,
    radius: float,
    depth: float,
    location: tuple[float, float, float],
    mat: MaterialSpec = SHADOW,
    axis: str = "Y",
) -> bpy.types.Object:
    rotation = (math.pi / 2.0, 0.0, 0.0) if axis.upper() == "Y" else (0.0, math.pi / 2.0, 0.0)
    return add_cylinder(name, radius, depth, location, mat, vertices=64, rotation=rotation)


def add_hull_between_cylinders(
    name: str,
    first_center: tuple[float, float],
    first_radius: float,
    second_center: tuple[float, float],
    second_radius: float,
    height: float,
    z_center: float,
    mat: MaterialSpec,
    samples: int = 96,
) -> bpy.types.Object:
    c1 = Vector((first_center[0], first_center[1], 0.0))
    c2 = Vector((second_center[0], second_center[1], 0.0))
    axis = c2 - c1
    axis.normalize()
    side = Vector((-axis.y, axis.x, 0.0))
    points: list[Vector] = []
    for i in range(samples):
        angle = math.tau * i / samples
        normal = Vector((math.cos(angle), math.sin(angle), 0.0))
        source_center = c1 if normal.dot(axis) < 0.0 else c2
        source_radius = first_radius if normal.dot(axis) < 0.0 else second_radius
        if abs(normal.dot(axis)) < 0.18:
            if normal.dot(side) >= 0:
                source_center = c1 if first_radius > second_radius else c2
                source_radius = max(first_radius, second_radius)
            else:
                source_center = c2 if first_radius > second_radius else c1
                source_radius = max(first_radius, second_radius)
        points.append(source_center + normal * source_radius)

    verts: list[tuple[float, float, float]] = []
    z0 = z_center - height / 2.0
    z1 = z_center + height / 2.0
    for point in points:
        verts.append((point.x, point.y, z0))
    for point in points:
        verts.append((point.x, point.y, z1))
    faces: list[tuple[int, ...]] = []
    count = len(points)
    for i in range(count):
        j = (i + 1) % count
        faces.append((i, j, count + j, count + i))
    faces.append(tuple(range(count - 1, -1, -1)))
    faces.append(tuple(range(count, count * 2)))
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, mat)
    shade_smooth(obj)
    return obj


def set_object_collection(objects: Iterable[bpy.types.Object], collection_name: str) -> None:
    collection = bpy.data.collections.get(collection_name) or bpy.data.collections.new(collection_name)
    if collection.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(collection)
    for obj in objects:
        for existing in obj.users_collection:
            existing.objects.unlink(obj)
        collection.objects.link(obj)


# ============================================================
# Procedural assembly
# ============================================================

"""Procedural assembly for the RC9 heatpipe bender Preview model."""






RAD = 40.0
TOLERANCE = 0.20
MAIN_HEIGHT = 36.0
AXLE_CENTER_X = -RAD - 15.0
OFFSET_STACK_CENTER = (AXLE_CENTER_X, 0.0)
PIPE_LEVELS = ((5.0, 8.0), (4.0, 19.0), (3.0, 28.0))
FLAT_AXLE_ANGLE_DEG = math.degrees(math.atan((20.0 - 6.0) / (RAD + 15.0)))
GROOVE_SIDE_START_DEG = -5.0
GROOVE_SIDE_END_DEG = 184.0
GROOVE_VISIBLE_START_DEG = -3.0
GROOVE_VISIBLE_END_DEG = 187.0
GROOVE_BACKING_RADIUS = 32.4
GROOVE_Z_MIN = 2.1
GROOVE_Z_MAX = 33.8


def _rotate_point(x: float, y: float, angle_degrees: float) -> tuple[float, float]:
    angle = math.radians(angle_degrees)
    return (
        x * math.cos(angle) - y * math.sin(angle),
        x * math.sin(angle) + y * math.cos(angle),
    )


def _press_lever_center(z: float) -> tuple[float, float, float]:
    offset = _rotate_point(35.0 + RAD / 4.0, 0.0, 120.0)
    return (AXLE_CENTER_X - 5.0 + offset[0], offset[1], z)


def _y_axis_cylinder_cutter(
    name: str,
    radius: float,
    depth: float,
    location: tuple[float, float, float],
) -> bpy.types.Object:
    return add_cylinder(name, radius, depth, location, vertices=64, rotation=(math.pi / 2.0, 0.0, 0.0))


def _pipe_arc_cutter(
    name: str,
    major_radius: float,
    pipe_radius: float,
    z: float,
    start: float,
    end: float,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale_y: float = 1.0,
) -> bpy.types.Object:
    return add_tube_arc(
        name,
        major_radius=major_radius,
        tube_radius=pipe_radius + TOLERANCE,
        z_center=z,
        start_degrees=start,
        end_degrees=end,
        location=location,
        major_segments=112,
        tube_segments=18,
        scale_xy=(1.0, scale_y),
    )


def _ausschnitt_cutter(name: str) -> bpy.types.Object:
    """SCAD ausschnitt(): annular relief cutter with three protected radial spokes."""
    outer = add_cylinder(f"{name}_outer_relief", RAD - 10.0, 40.0, (0.0, 0.0, 19.0), vertices=128)
    inner = add_cylinder(f"{name}_inner_keepout", 26.0 + TOLERANCE, 42.0, (0.0, 0.0, 19.0), vertices=128)
    relief = boolean_difference(f"{name}_annular_relief", outer, [inner])

    spoke_cutters: list[bpy.types.Object] = []
    for angle in (0.0, 120.0, 240.0):
        x, y = _rotate_point(RAD / 2.0, 0.0, angle)
        spoke_cutters.append(
            add_box(
                f"{name}_spoke_keepout_{int(angle)}",
                size=(RAD, 8.0, 42.0),
                location=(x, y, 18.0),
                rotation=(0.0, 0.0, math.radians(angle)),
            )
        )
    return boolean_difference(name, relief, spoke_cutters)


def _short_top_crescent_cutter(name: str) -> bpy.types.Object:
    """Shorter top-only annular relief windows used by the preview masking overlay."""
    arcs: list[bpy.types.Object] = []
    for index, (start, end) in enumerate(((32.0, 58.0), (152.0, 178.0), (272.0, 298.0))):
        arcs.append(
            add_arc_mesh(
                f"{name}_short_crescent_{index}",
                inner_radius=26.0,
                outer_radius=29.4,
                height=2.4,
                start_degrees=start,
                end_degrees=end,
                z_bottom=35.85,
                mat=SHADOW,
                segments=24,
            )
        )
    return boolean_union(name, arcs)


def _large_clip_box(
    name: str,
    center_xy: tuple[float, float],
    angle_degrees: float,
    size: tuple[float, float, float] = (220.0, 100.0, 100.0),
) -> bpy.types.Object:
    return add_box(name, size=size, location=(center_xy[0], center_xy[1], 18.0), rotation=(0.0, 0.0, math.radians(angle_degrees)))


def _groove_sector_backing_cutter(name: str) -> bpy.types.Object:
    shell = add_arc_mesh(
        f"{name}_outer_shell",
        inner_radius=GROOVE_BACKING_RADIUS,
        outer_radius=RAD + 4.0,
        height=GROOVE_Z_MAX - GROOVE_Z_MIN,
        start_degrees=GROOVE_VISIBLE_START_DEG,
        end_degrees=GROOVE_VISIBLE_END_DEG,
        z_bottom=GROOVE_Z_MIN,
        mat=SHADOW,
        segments=112,
    )
    center_keep = add_cylinder(f"{name}_center_keepout", 21.8, GROOVE_Z_MAX - GROOVE_Z_MIN + 1.0, (0.0, 0.0, (GROOVE_Z_MIN + GROOVE_Z_MAX) / 2.0), vertices=96)
    return boolean_difference(name, shell, [center_keep])


def _press_outer_jaw_relief_cutter(name: str) -> bpy.types.Object:
    """Annular quadrant removed from presse(), matching the source's clipped r50-r15 cutter."""
    cutter = add_annular_cylinder(
        f"{name}_annular_shell",
        outer_radius=50.0,
        inner_radius=15.0,
        height=44.0,
        location=(AXLE_CENTER_X, 0.0, 20.0),
        mat=SHADOW,
        vertices=96,
    )
    return boolean_difference(
        name,
        cutter,
        [
            add_box(f"{name}_left_limit", (100.0, 120.0, 60.0), (AXLE_CENTER_X - 50.0, -25.0, 20.0)),
            add_box(f"{name}_right_limit", (100.0, 120.0, 60.0), (AXLE_CENTER_X + 100.0, -25.0, 20.0)),
            add_box(f"{name}_rear_limit", (120.0, 100.0, 60.0), (AXLE_CENTER_X + 25.0, 50.0, 20.0)),
            add_box(f"{name}_front_limit", (120.0, 100.0, 60.0), (AXLE_CENTER_X + 25.0, -100.0, 20.0)),
        ],
    )


def _rotate_objects_z(objects: list[bpy.types.Object], angle_degrees: float) -> list[bpy.types.Object]:
    """Apply the visible SCAD preview's shallow Flataxle/clearance rotation to a built subassembly."""
    if not angle_degrees:
        return objects
    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for obj in objects:
        x, y, z = obj.location
        obj.location = (x * cos_a - y * sin_a, x * sin_a + y * cos_a, z)
        obj.rotation_euler.rotate_axis("Z", angle)
    return objects


def build_main_handle() -> list[bpy.types.Object]:
    """Yellow fixed bending body and long handle, following the rad >= 40 SCAD branch."""
    parts: list[bpy.types.Object] = []

    body_base = add_annular_cylinder(
        "main_body_outer_ring_with_center_bore",
        outer_radius=RAD - 0.7,
        inner_radius=20.2,
        height=MAIN_HEIGHT,
        location=(0.0, 0.0, MAIN_HEIGHT / 2.0),
        mat=YELLOW,
        vertices=128,
    )
    holder_low = add_box(
        "main_straight_pipe_holder_low_step",
        size=(RAD + 12.0, RAD / 2.0 + 2.0, 12.0),
        location=(-RAD - 12.0 + (RAD + 12.0) / 2.0, -RAD - 5.0 + (RAD / 2.0 + 2.0) / 2.0, 6.0),
        mat=YELLOW,
    )
    holder_tower = add_box(
        "main_straight_pipe_holder_tall_face",
        size=(25.0 + RAD / 10.0, RAD / 2.0 + 2.0, MAIN_HEIGHT),
        location=(-RAD - 12.0 + (25.0 + RAD / 10.0) / 2.0, -RAD - 5.0 + (RAD / 2.0 + 2.0) / 2.0, MAIN_HEIGHT / 2.0),
        mat=YELLOW,
    )
    body = boolean_union("yellow_fixed_body_and_straight_holder_uncut", [body_base, holder_low, holder_tower], YELLOW)
    body_cutters: list[bpy.types.Object] = [
        add_cylinder("main_center_bore_cut", 20.0 + TOLERANCE, MAIN_HEIGHT + 2.0, (0.0, 0.0, MAIN_HEIGHT / 2.0), vertices=128),
        _ausschnitt_cutter("main_ausschnitt_annular_spoke_relief_cut"),
        _groove_sector_backing_cutter("main_outer_shell_replaced_by_rounded_groove_sector_cut"),
    ]
    for pipe_radius, z in PIPE_LEVELS:
        cutter_radius = pipe_radius + TOLERANCE
        body_cutters.append(
            _y_axis_cylinder_cutter(
                f"main_straight_{int(pipe_radius * 2)}mm_pipe_cut",
                cutter_radius,
                RAD * 2.05,
                (-RAD, -RAD, z),
            )
        )
        body_cutters.append(
            _pipe_arc_cutter(
                f"main_curved_{int(pipe_radius * 2)}mm_pipe_cut",
                RAD,
                pipe_radius,
                z,
                GROOVE_SIDE_START_DEG,
                GROOVE_SIDE_END_DEG,
            )
        )
    body_cutters.extend(
        [
            add_box(
                "main_holder_upper_pocket_clearance_cut",
                size=(21.0, 15.0, 24.0),
                location=(-15.2, -36.8, 24.0),
            ),
            add_box(
                "main_holder_pocket_back_wall_trim_cut",
                size=(12.0, 18.0, 16.0),
                location=(-23.0, -32.2, 26.0),
                rotation=(0.0, 0.0, math.radians(-7.0)),
            ),
        ]
    )
    body = boolean_difference("yellow_fixed_body_with_true_rounded_pipe_channels", body, body_cutters, YELLOW)
    shade_flat(body)
    grooved_sector = add_pipe_groove_sector_body(
        "yellow_integral_rounded_side_pipe_groove_sector",
        outer_radius=RAD - 0.7,
        backing_radius=GROOVE_BACKING_RADIUS,
        cutter_major_radius=RAD,
        pipe_levels=tuple((pipe_radius + TOLERANCE, z) for pipe_radius, z in PIPE_LEVELS),
        z_min=GROOVE_Z_MIN,
        z_max=GROOVE_Z_MAX,
        start_degrees=GROOVE_VISIBLE_START_DEG,
        end_degrees=GROOVE_VISIBLE_END_DEG,
        mat=YELLOW,
        trough_mat=LIME,
        trough_depth_threshold=1.15,
        angle_segments=124,
        z_segments=108,
    )
    parts.append(grooved_sector)

    # Main_part() uses translate([0, -rad+5, 0]) minkowski(cube([80+rad,5,10]), cylinder(r=10,h=5)).
    # This keeps broad flat faces and rounded XY corners instead of a hollow-looking capsule nose.
    handle = add_rounded_rect_prism(
        "yellow_long_minkowski_bar",
        size=(140.0, 25.0, 15.0),
        corner_radius=10.0,
        center=(60.0, -RAD + 7.5, 7.5),
        mat=YELLOW,
        segments=14,
    )
    handle = boolean_difference(
        "yellow_long_minkowski_handle_with_center_and_ausschnitt_cut",
        handle,
        [
            add_cylinder("yellow_handle_inner_radius_clearance", 20.0 + TOLERANCE, 23.0, (0.0, 0.0, 7.5), vertices=96),
            _ausschnitt_cutter("yellow_handle_ausschnitt_relief_cut"),
        ],
        YELLOW,
    )
    fixed = boolean_union("yellow_fixed_body_holder_and_handle_continuous", [body, handle], YELLOW)
    shade_flat(fixed)
    parts.append(fixed)

    for pipe_radius, z in PIPE_LEVELS:
        shadow = add_cylinder(
            f"dark_straight_holder_{int(pipe_radius * 2)}mm_through_bore_shadow",
            radius=pipe_radius * 0.88,
            height=20.5,
            location=(-RAD, -34.2, z),
            mat=SHADOW,
            vertices=48,
            rotation=(math.pi / 2.0, 0.0, 0.0),
        )
        shade_flat(shadow)
        parts.append(shadow)

    return parts


def build_press_lever() -> list[bpy.types.Object]:
    """Blue rotating press/bender lever rebuilt from the source presse() structure."""
    parts: list[bpy.types.Object] = []
    post = add_cylinder(
        "blue_press_source_radius8_post_seed",
        8.0,
        36.0,
        (AXLE_CENTER_X, 0.0, 18.0),
        BLUE,
        vertices=64,
    )
    jaw_lobe = add_cylinder(
        "blue_press_source_intersection_lobe_seed",
        14.6,
        34.0,
        (AXLE_CENTER_X, 0.0, 18.8),
        BLUE,
        vertices=72,
    )
    jaw_web = add_rounded_box(
        "blue_press_source_intersection_cube_seed",
        size=(29.0, 22.0, 33.0),
        radius=2.6,
        location=(AXLE_CENTER_X - 0.3, 4.8, 18.8),
        mat=BLUE,
        segments=5,
    )
    jaw_mass = boolean_difference(
        "blue_press_single_clipped_jaw_mass",
        boolean_union("blue_press_source_intersection_jaw_seed", [jaw_lobe, jaw_web], BLUE),
        [
            add_box("blue_press_source_jaw_negative_y_clip", (46.0, 22.0, 42.0), (AXLE_CENTER_X, -18.5, 19.0)),
            add_box("blue_press_source_jaw_positive_y_face_clip", (46.0, 12.5, 42.0), (AXLE_CENTER_X, 19.8, 19.0)),
            add_box("blue_press_source_jaw_outer_x_clip", (15.0, 48.0, 42.0), (AXLE_CENTER_X - 20.5, 3.0, 19.0)),
            add_box("blue_press_source_jaw_inner_x_clip", (15.0, 48.0, 42.0), (AXLE_CENTER_X + 20.5, 3.0, 19.0)),
            add_box("blue_press_source_low_tab_cluster_remove", (54.0, 42.0, 17.0), (AXLE_CENTER_X, 2.6, 5.2), rotation=(0.0, 0.0, math.radians(6.0))),
            add_box("blue_press_source_upper_front_jaw_bury_clip", (36.0, 14.0, 15.0), (AXLE_CENTER_X + 1.2, -5.5, 29.2), rotation=(0.0, 0.0, math.radians(8.0))),
        ],
        BLUE,
    )
    jaw = boolean_union("blue_press_integrated_source_post_and_jaw", [post, jaw_mass], BLUE)
    lever_center = _press_lever_center(28.7)
    upper_lever = add_rounded_box(
        "blue_press_source_long_rounded_lever_closer_to_minkowski_bar",
        size=(80.0, 17.5, 12.2),
        radius=5.4,
        location=(lever_center[0] + 1.4, lever_center[1] - 0.8, 28.7),
        mat=BLUE,
        rotation=(0.0, 0.0, math.radians(120.0)),
        segments=9,
    )
    root_blend = add_rounded_box(
        "blue_press_short_root_blend_under_bracket",
        size=(27.0, 18.0, 12.0),
        radius=4.0,
        location=(AXLE_CENTER_X - 4.0, 3.2, 28.0),
        mat=BLUE,
        rotation=(0.0, 0.0, math.radians(18.0)),
        segments=8,
    )
    root_palm = add_rounded_box(
        "blue_press_compact_overlapped_root_palm_seed",
        size=(25.0, 20.0, 13.5),
        radius=4.0,
        location=(AXLE_CENTER_X - 1.4, 1.0, 26.5),
        mat=BLUE,
        rotation=(0.0, 0.0, math.radians(5.0)),
        segments=8,
    )
    press_body = boolean_union("blue_press_source_like_single_union_uncut", [jaw, root_blend, root_palm, upper_lever], BLUE)

    press_cutters: list[bpy.types.Object] = [
        _press_outer_jaw_relief_cutter("blue_press_source_outer_jaw_relief_cut"),
        add_cylinder("blue_press_axle_bore_cut", 6.0 + TOLERANCE, 42.0, (AXLE_CENTER_X, 0.0, 20.0), vertices=64),
        add_box("blue_press_source_negative_y_intersection_trim", (54.0, 30.0, 42.0), (AXLE_CENTER_X, -23.0, 18.0)),
        add_box("blue_press_low_root_and_pin_cluster_hidden_trim", (58.0, 42.0, 24.0), (AXLE_CENTER_X, 3.0, 10.0), rotation=(0.0, 0.0, math.radians(6.0))),
        add_box("blue_press_front_view_projecting_root_flush_trim", (42.0, 13.0, 23.0), (AXLE_CENTER_X + 2.4, -7.0, 19.5), rotation=(0.0, 0.0, math.radians(14.0))),
        add_box("blue_press_left_view_outer_tab_cluster_flush_trim", (22.0, 14.0, 21.0), (AXLE_CENTER_X - 11.5, 8.5, 19.8), rotation=(0.0, 0.0, math.radians(-18.0))),
        add_box("blue_press_root_under_top_overlap_shorten_trim", (28.0, 10.0, 14.0), (AXLE_CENTER_X - 6.0, 12.6, 26.5), rotation=(0.0, 0.0, math.radians(6.0))),
        add_box("blue_press_root_hidden_inside_holder_stack_trim", (30.0, 15.0, 19.0), (AXLE_CENTER_X + 4.0, -4.2, 24.0), rotation=(0.0, 0.0, math.radians(11.0))),
        _large_clip_box(
            "blue_press_large_scad_wedge_cut",
            center_xy=(-RAD - 15.0 - 50.0 + _rotate_point(30.0 + RAD / 2.0, 0.0, 120.0)[0], -40.0 + _rotate_point(30.0 + RAD / 2.0, 0.0, 120.0)[1]),
            angle_degrees=120.0,
        ),
    ]
    for pipe_radius, z in PIPE_LEVELS:
        press_cutters.append(_pipe_arc_cutter(f"blue_press_curved_{int(pipe_radius * 2)}mm_pipe_cut", 15.0, pipe_radius * 0.95, z, 180.0, 360.0, (AXLE_CENTER_X, 0.0, 0.0), scale_y=1.2))
        press_cutters.append(_y_axis_cylinder_cutter(f"blue_press_straight_{int(pipe_radius * 2)}mm_pipe_cut", pipe_radius + TOLERANCE, RAD * 2.0, (-RAD, 0.0, z)))
    press_body = boolean_difference("blue_press_compact_boolean_cut_jaw", press_body, press_cutters, BLUE)
    parts.append(press_body)

    return parts


def build_axle() -> list[bpy.types.Object]:
    """Magenta flatted axle linking the main wheel and press lever."""
    parts: list[bpy.types.Object] = []
    axle_main = add_scaled_cylinder(
        "magenta_flat_axle_radius20_filled_center_body_seed",
        radius=20.0,
        height=31.5,
        location=(0.0, 0.0, 13.8),
        mat=MAGENTA,
        scale_xy=(0.96, 0.72),
        vertices=128,
    )
    axle_post = add_scaled_cylinder(
        "magenta_flat_axle_offset_lower_post_seed",
        radius=6.0,
        height=27.0,
        location=(OFFSET_STACK_CENTER[0], OFFSET_STACK_CENTER[1], 12.8),
        mat=MAGENTA,
        scale_xy=(0.72, 0.48),
        vertices=64,
    )
    axle_bridge = add_hull_between_cylinders(
        "magenta_flat_axle_low_bridge_between_posts_seed",
        first_center=(0.0, 0.0),
        first_radius=13.2,
        second_center=OFFSET_STACK_CENTER,
        second_radius=3.8,
        height=5.0,
        z_center=-0.2,
        mat=MAGENTA,
        samples=96,
    )
    axle = boolean_union("magenta_flattened_radius20_axle_stack_uncut", [axle_main, axle_post, axle_bridge], MAGENTA)
    axle = boolean_difference(
        "magenta_flattened_radius20_axle_with_small_vertical_bore",
        axle,
        [
            add_scaled_cylinder(
                "magenta_flat_axle_small_vertical_bore_cut",
                radius=5.2,
                height=70.0,
                location=(0.0, 0.0, 17.0),
                scale_xy=(1.0, 1.7),
                vertices=80,
            ),
            add_box("magenta_flat_axle_lower_flattening_cut", (200.0, 200.0, 100.0), (0.0, 0.0, -51.5)),
            add_box("magenta_flat_axle_top_rear_exposure_clip", (82.0, 56.0, 38.0), (-16.0, 28.0, 23.0), rotation=(0.0, 0.0, math.radians(8.0))),
            add_box("magenta_flat_axle_front_hide_clip", (92.0, 42.0, 50.0), (-10.0, -30.0, 19.0), rotation=(0.0, 0.0, math.radians(-5.0))),
            add_box("magenta_flat_axle_rear_lower_foot_hide_clip", (58.0, 46.0, 18.0), (OFFSET_STACK_CENTER[0] + 2.0, OFFSET_STACK_CENTER[1] + 9.0, 2.0), rotation=(0.0, 0.0, math.radians(7.0))),
            add_box("magenta_flat_axle_offset_face_hide_clip", (38.0, 20.0, 42.0), (OFFSET_STACK_CENTER[0] + 1.3, OFFSET_STACK_CENTER[1] - 13.0, 17.0), rotation=(0.0, 0.0, math.radians(-4.0))),
            add_box("magenta_flat_axle_outer_slab_hide_clip", (30.0, 22.0, 46.0), (OFFSET_STACK_CENTER[0] - 5.0, OFFSET_STACK_CENTER[1] + 11.0, 17.0), rotation=(0.0, 0.0, math.radians(8.0))),
            add_box("magenta_flat_axle_offset_pin_extra_left_rear_hide_clip", (34.0, 27.0, 28.0), (OFFSET_STACK_CENTER[0] - 1.0, OFFSET_STACK_CENTER[1] + 5.8, 11.2), rotation=(0.0, 0.0, math.radians(10.0))),
            add_cylinder("magenta_top_surface_hidden_under_yellow_disk_cut", 24.4, 18.0, (0.0, 0.0, 37.8), vertices=96),
            add_box("magenta_diagonal_stack_hidden_under_top_cover_cut", (84.0, 28.0, 28.0), (-25.5, 2.8, 35.6), rotation=(0.0, 0.0, math.radians(-FLAT_AXLE_ANGLE_DEG))),
        ],
        MAGENTA,
    )
    parts.append(axle)
    rotated = _rotate_objects_z(parts, -FLAT_AXLE_ANGLE_DEG)
    return rotated


def build_bracket() -> list[bpy.types.Object]:
    """Lime bracket plate spanning the main body and the offset press axle."""
    parts: list[bpy.types.Object] = []
    offset_center = OFFSET_STACK_CENTER
    bracket_z = 33.42
    bracket_height = 5.15
    plate = add_hull_between_cylinders(
        "lime_axle_bracket_source_scale_hull_plate",
        first_center=(0.0, 0.0),
        first_radius=25.8,
        second_center=offset_center,
        second_radius=11.8,
        height=bracket_height,
        z_center=bracket_z,
        mat=LIME,
        samples=128,
    )
    big_lobe = add_cylinder("lime_main_bracket_lobe_clipped_seed", 29.6, bracket_height, (0.0, 0.0, bracket_z), LIME, vertices=96)
    shade_flat(big_lobe)
    big_lobe = boolean_difference(
        "lime_main_bracket_lobe_clipped",
        big_lobe,
        [
            add_box(
                "lime_main_lobe_scad_side_clip",
                size=(100.0, 100.0, 18.0),
                location=_rotate_point(-50.0, -100.0, 30.0) + (bracket_z,),
                rotation=(0.0, 0.0, math.radians(30.0)),
            )
        ],
        LIME,
    )
    small_lobe = add_cylinder("lime_small_bracket_lobe_clipped_seed", 15.8, bracket_height, (offset_center[0], offset_center[1], bracket_z), LIME, vertices=72)
    shade_flat(small_lobe)
    small_lobe = boolean_difference(
        "lime_small_bracket_lobe_clipped",
        small_lobe,
        [
            add_box(
                "lime_small_lobe_scad_side_clip",
                size=(100.0, 100.0, 18.0),
                location=(offset_center[0] + _rotate_point(-50.0, -100.0, -30.0)[0], offset_center[1] + _rotate_point(-50.0, -100.0, -30.0)[1], bracket_z),
                rotation=(0.0, 0.0, math.radians(-30.0)),
            )
        ],
        LIME,
    )
    bracket = boolean_union("lime_axle_bracket_solid_before_clearances", [plate, big_lobe, small_lobe], LIME)
    bracket_cutters: list[bpy.types.Object] = [
        add_cylinder("lime_main_axle_clearance_through_cut", 20.4, bracket_height + 1.2, (0.0, 0.0, bracket_z), vertices=96),
        add_cylinder("lime_offset_axle_clearance_through_cut", 6.35, bracket_height + 1.2, (offset_center[0], offset_center[1], bracket_z), vertices=64),
        add_box("lime_main_flataxle_rectangular_slot_cut", (21.2, 41.0, bracket_height + 1.4), (0.0, -9.6, bracket_z), rotation=(0.0, 0.0, math.radians(FLAT_AXLE_ANGLE_DEG))),
        add_box("lime_offset_flataxle_rectangular_slot_cut", (7.2, 40.5, bracket_height + 1.4), (offset_center[0], offset_center[1] - 9.6, bracket_z), rotation=(0.0, 0.0, math.radians(FLAT_AXLE_ANGLE_DEG))),
        add_box("lime_front_face_buried_inside_holder_trim", (76.0, 9.0, bracket_height + 1.4), (-26.0, -19.0, bracket_z), rotation=(0.0, 0.0, math.radians(-3.0))),
        add_box("lime_top_view_midspan_notch_under_yellow_preview_trim", (40.0, 7.5, bracket_height + 1.4), (-25.0, -0.8, bracket_z), rotation=(0.0, 0.0, math.radians(-FLAT_AXLE_ANGLE_DEG))),
    ]
    bracket = boolean_difference("lime_axle_bracket_with_real_cut_through_slots", bracket, bracket_cutters, LIME)
    shade_flat(bracket)
    parts.append(bracket)
    return parts


def build_top_yellow_preview_overlap() -> list[bpy.types.Object]:
    """Yellow faces in the SCAD preview that cover much of the colored internal parts."""
    parts: list[bpy.types.Object] = []
    top_overlay_seed = boolean_union(
        "yellow_broken_top_preview_overlap_seed",
        [
            add_cylinder("yellow_top_overlay_inner_solid_plate_to_relief_seed", 25.7, 0.06, (0.0, 0.0, 36.07), YELLOW, vertices=96),
            add_cylinder("yellow_top_overlay_smaller_center_boss_mask_seed", 15.6, 0.12, (0.0, 0.0, 36.11), YELLOW, vertices=72),
            add_hull_between_cylinders(
                "yellow_top_overlay_short_inner_link_cover_seed",
                first_center=(-6.0, 0.0),
                first_radius=1.25,
                second_center=(-17.5, 0.0),
                second_radius=0.7,
                height=0.055,
                z_center=36.12,
                mat=YELLOW,
                samples=48,
            ),
            add_rounded_box(
                "yellow_top_overlay_small_press_root_tuck_cover_seed",
                size=(8.8, 3.2, 0.09),
                radius=0.9,
                location=(AXLE_CENTER_X - 0.7, 1.1, 36.13),
                mat=YELLOW,
                rotation=(0.0, 0.0, math.radians(120.0)),
                segments=5,
            ),
            add_cylinder("yellow_top_overlay_low_faceted_boss_seed", 6.2, 0.18, (0.0, 0.0, 36.23), YELLOW, vertices=34),
            add_cylinder("yellow_top_overlay_tiny_offset_pin_seed", 0.55, 0.06, (OFFSET_STACK_CENTER[0], OFFSET_STACK_CENTER[1], 36.18), YELLOW, vertices=16),
        ],
        YELLOW,
    )
    top_overlay = boolean_difference(
        "yellow_broken_top_preview_overlap_with_open_body_relief",
        top_overlay_seed,
        [
            add_cylinder("yellow_top_overlay_open_center_bore_cut", 5.05, 2.2, (0.0, 0.0, 36.30), vertices=46),
            add_box("yellow_top_overlay_offset_end_shorten_trim", (17.0, 12.0, 2.0), (OFFSET_STACK_CENTER[0] - 7.2, OFFSET_STACK_CENTER[1] + 4.0, 36.20), rotation=(0.0, 0.0, math.radians(-10.0))),
            add_box("yellow_top_overlay_left_excess_trim", (18.0, 18.0, 2.0), (-22.5, -14.0, 36.20), rotation=(0.0, 0.0, math.radians(-24.0))),
            add_box("yellow_top_overlay_mid_link_exposure_break_trim", (19.0, 8.0, 2.0), (-24.0, 0.2, 36.18), rotation=(0.0, 0.0, math.radians(-FLAT_AXLE_ANGLE_DEG))),
        ],
        YELLOW,
    )
    shade_flat(top_overlay)
    parts.append(top_overlay)
    parts.append(add_cylinder("dark_top_center_bore_recess_shadow", 4.35, 0.06, (0.0, 0.0, 36.56), SHADOW, vertices=44))

    for index, (start, end) in enumerate(((16.0, 100.0), (136.0, 220.0), (256.0, 340.0))):
        crescent_shadow = add_arc_mesh(
            f"dark_top_open_annular_relief_crescent_{index}",
            inner_radius=26.15,
            outer_radius=29.85,
            height=0.045,
            start_degrees=start,
            end_degrees=end,
            z_bottom=36.28,
            mat=SHADOW,
            segments=52,
        )
        shade_flat(crescent_shadow)
        parts.append(crescent_shadow)

    for angle in (0.0, 120.0, 240.0):
        x, y = _rotate_point(28.0, 0.0, angle)
        spoke = add_box(
            f"yellow_top_protected_radial_spoke_{int(angle)}",
            size=(8.2, 7.4, 0.055),
            location=(x, y, 36.34),
            mat=YELLOW,
            rotation=(0.0, 0.0, math.radians(angle)),
        )
        shade_flat(spoke)
        parts.append(spoke)

    lever_cover_center = _press_lever_center(30.85)
    lever_cover = add_rounded_box(
        "yellow_press_lever_preview_occlusion_skin",
        size=(80.0, 18.8, 12.4),
        radius=5.4,
        location=(lever_cover_center[0] + 1.4, lever_cover_center[1] - 0.9, 28.8),
        mat=YELLOW,
        rotation=(0.0, 0.0, math.radians(120.0)),
        segments=9,
    )
    lever_cover = boolean_difference(
        "yellow_press_lever_preview_skin_clipped_at_hinge",
        lever_cover,
        [
            add_cylinder("yellow_press_lever_skin_offset_axle_bite_cut", 5.4, 14.0, (AXLE_CENTER_X, 0.0, 28.8), vertices=48),
            add_box("yellow_press_lever_skin_inner_root_shortening_cut", (6.0, 6.0, 13.0), (AXLE_CENTER_X + 5.2, -2.0, 28.8), rotation=(0.0, 0.0, math.radians(10.0))),
        ],
        YELLOW,
    )
    shade_flat(lever_cover)
    parts.append(lever_cover)

    lever_top_skin = add_rounded_box(
        "yellow_press_lever_top_preview_skin_covering_blue_strip",
        size=(80.0, 14.6, 0.52),
        radius=4.2,
        location=(lever_cover_center[0] + 1.4, lever_cover_center[1] - 0.9, 35.12),
        mat=YELLOW,
        rotation=(0.0, 0.0, math.radians(120.0)),
        segments=8,
    )
    shade_flat(lever_top_skin)
    parts.append(lever_top_skin)

    hinge_neck = add_rounded_box(
        "yellow_press_lever_integrated_hinge_neck_fill",
        size=(31.0, 18.0, 12.0),
        radius=4.0,
        location=(AXLE_CENTER_X - 4.8, 3.6, 27.8),
        mat=YELLOW,
        rotation=(0.0, 0.0, math.radians(18.0)),
        segments=8,
    )
    hinge_neck = boolean_difference(
        "yellow_press_lever_integrated_hinge_neck_with_axle_bite",
        hinge_neck,
        [
            add_cylinder("yellow_hinge_neck_offset_axle_bite_cut", 4.8, 13.0, (AXLE_CENTER_X, 0.0, 27.9), vertices=42),
            add_box("yellow_hinge_neck_front_face_flatten_cut", (18.0, 4.5, 13.0), (AXLE_CENTER_X + 4.2, -5.2, 27.9), rotation=(0.0, 0.0, math.radians(5.0))),
        ],
        YELLOW,
    )
    shade_flat(hinge_neck)
    parts.append(hinge_neck)

    root_shroud = add_rounded_box(
        "yellow_holder_and_bracket_press_root_occlusion_shroud",
        size=(29.0, 18.0, 15.0),
        radius=3.2,
        location=(AXLE_CENTER_X + 1.0, -2.0, 26.4),
        mat=YELLOW,
        rotation=(0.0, 0.0, math.radians(7.0)),
        segments=7,
    )
    root_shroud = boolean_difference(
        "yellow_holder_and_bracket_press_root_occlusion_shroud_with_axle_notch",
        root_shroud,
        [
            add_cylinder("yellow_root_shroud_offset_axle_notch_cut", 4.1, 17.0, (AXLE_CENTER_X, 0.0, 26.7), vertices=36),
            add_box("yellow_root_shroud_rear_exposure_trim", (17.0, 4.5, 10.0), (AXLE_CENTER_X - 8.0, 11.0, 27.5), rotation=(0.0, 0.0, math.radians(4.0))),
        ],
        YELLOW,
    )
    shade_flat(root_shroud)
    parts.append(root_shroud)

    front_face = add_rounded_box(
        "yellow_front_holder_overlap_hiding_press_root",
        size=(28.0, 2.8, 17.5),
        radius=0.6,
        location=(AXLE_CENTER_X + 12.0, -22.25, 25.2),
        mat=YELLOW,
        segments=3,
    )
    shade_flat(front_face)
    parts.append(front_face)

    side_shadow = add_rounded_box(
        "dark_occluded_press_lever_underface_hint",
        size=(54.0, 7.6, 1.4),
        radius=1.6,
        location=(lever_cover_center[0] + 5.0, lever_cover_center[1] - 3.4, 23.4),
        mat=SHADOW,
        rotation=(0.0, 0.0, math.radians(120.0)),
        segments=4,
    )
    shade_flat(side_shadow)
    parts.append(side_shadow)
    return parts


def build_model() -> list[bpy.types.Object]:
    """Build the Preview assembly from the SCAD default parameters."""
    objects: list[bpy.types.Object] = []
    objects.extend(build_press_lever())
    objects.extend(build_axle())
    objects.extend(build_bracket())
    objects.extend(build_main_handle())
    objects.extend(build_top_yellow_preview_overlap())
    return objects


def install_camera_center_alignment_handler() -> None:
    """Compensate for center-offset canonical cameras after the shared renderer normalizes the scene."""

    def align_meshes_to_review_target(scene: bpy.types.Scene) -> None:
        if scene.get("heatpipe_center_alignment_applied"):
            return
        target = bpy.data.objects.get("ReviewTarget")
        if target is None or abs(target.location.x) < 0.01:
            return
        for obj in scene.objects:
            if obj.type == "MESH":
                obj.location.x += target.location.x * 0.92
        scene["heatpipe_center_alignment_applied"] = True
        bpy.context.view_layer.update()

    bpy.app.handlers.render_pre[:] = [
        handler
        for handler in bpy.app.handlers.render_pre
        if getattr(handler, "__name__", "") != "align_meshes_to_review_target"
    ]
    bpy.app.handlers.render_pre.append(align_meshes_to_review_target)


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()
    install_camera_center_alignment_handler()


if __name__ == "__main__":
    main()

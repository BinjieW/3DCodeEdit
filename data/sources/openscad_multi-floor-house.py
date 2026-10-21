from __future__ import annotations
from dataclasses import dataclass
from mathutils import Matrix, Vector
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the multi-floor house model."""





RGBA = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]


WALL_HEIGHT = 2.95
FLOOR_HEIGHT = 0.19
OUTER_WALL_THICKNESS = 0.365
WINDOWFRAME_WIDTH = 0.06
WINDOWFRAME_OFFSET = 0.05
EDGING_WIDTH = 0.10


COLORS: dict[str, RGBA] = {
    "outer_wall": (0.26, 0.26, 0.25, 1.0),
    "inner_wall": (0.80, 0.80, 0.75, 1.0),
    "glass": (0.62, 0.64, 0.59, 0.70),
    "window_frame": (0.18, 0.18, 0.18, 1.0),
    "window_edging": (0.88, 0.88, 0.84, 1.0),
    "window_reveal": (0.12, 0.12, 0.11, 1.0),
    "blind": (0.93, 0.93, 0.90, 1.0),
    "roof_top": (0.006, 0.006, 0.006, 1.0),
    "roof_bottom": (0.70, 0.70, 0.66, 1.0),
    "roof_trim": (0.09, 0.09, 0.09, 1.0),
    "solar": (0.0, 0.0, 0.0, 1.0),
    "solar_grid": (0.10, 0.10, 0.10, 1.0),
    "chimney": (0.10, 0.10, 0.10, 1.0),
    "walk": (0.34, 0.34, 0.32, 1.0),
    "garden": (0.0, 0.18, 0.0, 1.0),
    "entry_mat": (0.16, 0.16, 0.16, 1.0),
    "door": (0.34, 0.16, 0.07, 1.0),
    "handle": (0.42, 0.42, 0.40, 1.0),
    "stair": (0.50, 0.26, 0.12, 1.0),
    "stair_side": (0.92, 0.92, 0.88, 1.0),
}


@dataclass(frozen=True)
class WindowSpec:
    """Window placement in the local coordinates of one facade."""

    width: float
    height: float
    shift: float
    sill_height: float
    blind: float | None = 0.0
    subdivisions: int = 1


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def get_material(name: str, color: RGBA, *, roughness: float = 0.66) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        material.diffuse_color = color
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            if "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = color
            if "Alpha" in bsdf.inputs:
                bsdf.inputs["Alpha"].default_value = color[3]
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = roughness
        material.blend_method = "BLEND" if color[3] < 1.0 else "OPAQUE"
    return material


def assign_material(obj: bpy.types.Object, material_name: str, color: RGBA | None = None) -> bpy.types.Object:
    obj.data.materials.append(get_material(material_name, color or COLORS[material_name]))
    return obj


def add_box(
    name: str,
    size: Vec3,
    location: Vec3,
    material_name: str,
    color: RGBA | None = None,
    rotation: Vec3 = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return assign_material(obj, material_name, color)


def add_scad_box(
    name: str,
    corner: Vec3,
    size: Vec3,
    material_name: str,
    color: RGBA | None = None,
) -> bpy.types.Object:
    location = (
        corner[0] + size[0] / 2.0,
        corner[1] + size[1] / 2.0,
        corner[2] + size[2] / 2.0,
    )
    return add_box(name, size, location, material_name, color)


def add_cylinder(
    name: str,
    diameter: float,
    height: float,
    location: Vec3,
    material_name: str,
    color: RGBA | None = None,
    rotation: Vec3 = (0.0, 0.0, 0.0),
    vertices: int = 32,
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
    return assign_material(obj, material_name, color)


def add_cylinder_between(
    name: str,
    start: Vec3,
    end: Vec3,
    diameter: float,
    material_name: str,
    color: RGBA | None = None,
    vertices: int = 24,
) -> bpy.types.Object:
    start_vec = Vector(start)
    end_vec = Vector(end)
    direction = end_vec - start_vec
    length = direction.length
    if length <= 0.0001:
        raise ValueError(f"Cannot create zero-length cylinder {name}")

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=diameter / 2.0,
        depth=length,
        location=(start_vec + end_vec) * 0.5,
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    return assign_material(obj, material_name, color)


def add_mesh_object(
    name: str,
    vertices: list[Vec3],
    faces: list[tuple[int, ...]],
    material_name: str,
    color: RGBA | None = None,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return assign_material(obj, material_name, color)


def transform_vertices(vertices: list[Vec3], matrix: Matrix) -> list[Vec3]:
    return [tuple(matrix @ Vector(vertex)) for vertex in vertices]


def translation_matrix(offset: Vec3) -> Matrix:
    return Matrix.Translation(Vector(offset))


def rotation_matrix_degrees(rotation: Vec3) -> Matrix:
    rx, ry, rz = (math.radians(value) for value in rotation)
    return Matrix.Rotation(rz, 4, "Z") @ Matrix.Rotation(ry, 4, "Y") @ Matrix.Rotation(rx, 4, "X")


def add_transformed_box(
    name: str,
    corner: Vec3,
    size: Vec3,
    material_name: str,
    matrix: Matrix,
    color: RGBA | None = None,
) -> bpy.types.Object:
    sx, sy, sz = size
    x0, y0, z0 = corner
    vertices = [
        (x0, y0, z0),
        (x0 + sx, y0, z0),
        (x0 + sx, y0 + sy, z0),
        (x0, y0 + sy, z0),
        (x0, y0, z0 + sz),
        (x0 + sx, y0, z0 + sz),
        (x0 + sx, y0 + sy, z0 + sz),
        (x0, y0 + sy, z0 + sz),
    ]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return add_mesh_object(name, transform_vertices(vertices, matrix), faces, material_name, color)


def subtract_rect_cutouts(
    rect: tuple[float, float, float, float],
    cutouts: list[tuple[float, float, float, float]] | tuple[tuple[float, float, float, float], ...],
    *,
    min_size: float = 0.01,
) -> list[tuple[float, float, float, float]]:
    """Return x/y rectangles left after subtracting local roof cutouts."""

    rects = [rect]
    for cut_x0, cut_x1, cut_y0, cut_y1 in cutouts:
        next_rects: list[tuple[float, float, float, float]] = []
        for x0, x1, y0, y1 in rects:
            overlap_x0 = max(x0, cut_x0)
            overlap_x1 = min(x1, cut_x1)
            overlap_y0 = max(y0, cut_y0)
            overlap_y1 = min(y1, cut_y1)
            if overlap_x1 <= overlap_x0 or overlap_y1 <= overlap_y0:
                next_rects.append((x0, x1, y0, y1))
                continue

            candidates = [
                (x0, overlap_x0, y0, y1),
                (overlap_x1, x1, y0, y1),
                (overlap_x0, overlap_x1, y0, overlap_y0),
                (overlap_x0, overlap_x1, overlap_y1, y1),
            ]
            for candidate in candidates:
                cx0, cx1, cy0, cy1 = candidate
                if cx1 - cx0 > min_size and cy1 - cy0 > min_size:
                    next_rects.append(candidate)
        rects = next_rects
    return rects


def add_clipped_transformed_box(
    name: str,
    corner: Vec3,
    size: Vec3,
    material_name: str,
    matrix: Matrix,
    cutouts: list[tuple[float, float, float, float]] | tuple[tuple[float, float, float, float], ...],
    color: RGBA | None = None,
) -> list[bpy.types.Object]:
    x, y, z = corner
    sx, sy, sz = size
    kept_rects = subtract_rect_cutouts((x, x + sx, y, y + sy), cutouts)
    parts: list[bpy.types.Object] = []
    for index, (x0, x1, y0, y1) in enumerate(kept_rects):
        suffix = "" if len(kept_rects) == 1 else f"_clip_{index:02d}"
        parts.append(add_transformed_box(f"{name}{suffix}", (x0, y0, z), (x1 - x0, y1 - y0, sz), material_name, matrix, color))
    return parts


def add_prism_from_xy(
    name: str,
    points: list[tuple[float, float]],
    z_bottom: float,
    height: float,
    material_name: str,
    matrix: Matrix | None = None,
    color: RGBA | None = None,
) -> bpy.types.Object:
    vertices: list[Vec3] = [(x, y, z_bottom) for x, y in points]
    vertices += [(x, y, z_bottom + height) for x, y in points]
    if matrix is not None:
        vertices = transform_vertices(vertices, matrix)

    count = len(points)
    faces: list[tuple[int, ...]] = [tuple(reversed(range(count))), tuple(range(count, count * 2))]
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, next_index + count, index + count))
    return add_mesh_object(name, vertices, faces, material_name, color)


def add_facade_wall(
    name: str,
    width: float,
    height: float,
    orientation: str,
    shift_x: float,
    shift_y: float,
    *,
    thickness: float = OUTER_WALL_THICKNESS,
    windows: list[WindowSpec] | None = None,
    material_name: str = "outer_wall",
    z_offset: float = 0.0,
) -> list[bpy.types.Object]:
    """Build a facade as wall strips around its window openings.

    The SCAD source cuts windows from solid walls. For readability in BPY this
    helper composes the same facades from rectangular strips and places reusable
    window assemblies in the cutouts.
    """

    openings: list[tuple[float, float, float, float, WindowSpec]] = []
    for window in windows or []:
        edge = EDGING_WIDTH if window.height > 0.95 else 0.0
        z0 = FLOOR_HEIGHT + window.sill_height - edge
        z1 = z0 + window.height + edge * 2.0
        openings.append((window.shift - edge, window.shift + window.width + edge, z0, z1, window))
    openings.sort(key=lambda item: item[0])

    segments: list[bpy.types.Object] = []
    edge_points = [0.0, width]
    for x0, x1, _z0, _z1, _window in openings:
        edge_points.extend([max(0.0, x0), min(width, x1)])
    edge_points = sorted(set(round(value, 4) for value in edge_points if 0.0 <= value <= width))

    def blocked(mid_x: float, mid_z: float) -> bool:
        return any(x0 <= mid_x <= x1 and z0 <= mid_z <= z1 for x0, x1, z0, z1, _w in openings)

    z_breaks = sorted(set([0.0, height] + [value for item in openings for value in (item[2], item[3]) if 0.0 <= value <= height]))

    for left, right in zip(edge_points, edge_points[1:]):
        if right - left <= 0.001:
            continue
        for bottom, top in zip(z_breaks, z_breaks[1:]):
            if top - bottom <= 0.001:
                continue
            if blocked((left + right) * 0.5, (bottom + top) * 0.5):
                continue
            local_center = (left + (right - left) / 2.0, thickness / 2.0, z_offset + bottom + (top - bottom) / 2.0)
            segments.append(
                add_oriented_box(
                    f"{name}_wall_strip",
                    (right - left, thickness, top - bottom),
                    local_center,
                    orientation,
                    shift_x,
                    shift_y,
                    material_name,
                )
            )

    for index, (_x0, _x1, _z0, _z1, window) in enumerate(openings):
        segments.extend(
            add_window_assembly(
                f"{name}_window_{index + 1}",
                window,
                orientation,
                shift_x,
                shift_y,
                wall_thickness=thickness,
                z_offset=z_offset,
            )
        )
    return segments


def add_sloped_facade_wall(
    name: str,
    width: float,
    height_left: float,
    height_right: float,
    orientation: str,
    shift_x: float,
    shift_y: float,
    *,
    z_bottom: float,
    thickness: float = OUTER_WALL_THICKNESS,
    material_name: str = "outer_wall",
) -> bpy.types.Object:
    local_vertices = [
        (0.0, 0.0, z_bottom),
        (width, 0.0, z_bottom),
        (width, 0.0, z_bottom + height_right),
        (0.0, 0.0, z_bottom + height_left),
        (0.0, thickness, z_bottom),
        (width, thickness, z_bottom),
        (width, thickness, z_bottom + height_right),
        (0.0, thickness, z_bottom + height_left),
    ]
    vertices = [local_to_world(vertex, orientation, shift_x, shift_y) for vertex in local_vertices]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return add_mesh_object(name, vertices, faces, material_name)


def add_sloped_wall_segment(
    name: str,
    left: float,
    right: float,
    bottom_left: float,
    bottom_right: float,
    top_left: float,
    top_right: float,
    orientation: str,
    shift_x: float,
    shift_y: float,
    *,
    thickness: float,
    material_name: str,
) -> bpy.types.Object:
    local_vertices = [
        (left, 0.0, bottom_left),
        (right, 0.0, bottom_right),
        (right, 0.0, top_right),
        (left, 0.0, top_left),
        (left, thickness, bottom_left),
        (right, thickness, bottom_right),
        (right, thickness, top_right),
        (left, thickness, top_left),
    ]
    vertices = [local_to_world(vertex, orientation, shift_x, shift_y) for vertex in local_vertices]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return add_mesh_object(name, vertices, faces, material_name)


def add_sloped_facade_wall_with_windows(
    name: str,
    width: float,
    height_left: float,
    height_right: float,
    orientation: str,
    shift_x: float,
    shift_y: float,
    *,
    z_bottom: float,
    thickness: float = OUTER_WALL_THICKNESS,
    windows: list[WindowSpec] | None = None,
    material_name: str = "outer_wall",
) -> list[bpy.types.Object]:
    openings: list[tuple[float, float, float, float, WindowSpec]] = []
    for window in windows or []:
        edge = EDGING_WIDTH if window.height > 0.95 else 0.0
        z0 = z_bottom + FLOOR_HEIGHT + window.sill_height - edge
        z1 = z0 + window.height + edge * 2.0
        openings.append((window.shift - edge, window.shift + window.width + edge, z0, z1, window))
    openings.sort(key=lambda item: item[0])

    def wall_top(local_x: float) -> float:
        return z_bottom + height_left + (height_right - height_left) * (local_x / width)

    def blocked(mid_x: float, mid_z: float) -> bool:
        return any(x0 <= mid_x <= x1 and z0 <= mid_z <= z1 for x0, x1, z0, z1, _w in openings)

    edge_points = [0.0, width]
    for x0, x1, _z0, _z1, _window in openings:
        edge_points.extend([max(0.0, x0), min(width, x1)])
    edge_points = sorted(set(round(value, 4) for value in edge_points if 0.0 <= value <= width))

    z_breaks = [z_bottom, z_bottom + max(height_left, height_right)]
    for _x0, _x1, z0, z1, _window in openings:
        z_breaks.extend([z0, z1])
    z_breaks = sorted(set(round(value, 4) for value in z_breaks))

    parts: list[bpy.types.Object] = []
    for left, right in zip(edge_points, edge_points[1:]):
        if right - left <= 0.001:
            continue
        for bottom, top in zip(z_breaks, z_breaks[1:]):
            if top - bottom <= 0.001:
                continue
            if blocked((left + right) * 0.5, (bottom + top) * 0.5):
                continue
            top_left = min(top, wall_top(left))
            top_right = min(top, wall_top(right))
            bottom_left = min(bottom, wall_top(left))
            bottom_right = min(bottom, wall_top(right))
            if max(top_left - bottom_left, top_right - bottom_right) <= 0.001:
                continue
            parts.append(
                add_sloped_wall_segment(
                    f"{name}_wall_strip",
                    left,
                    right,
                    bottom_left,
                    bottom_right,
                    top_left,
                    top_right,
                    orientation,
                    shift_x,
                    shift_y,
                    thickness=thickness,
                    material_name=material_name,
                )
            )

    for index, (_x0, _x1, _z0, _z1, window) in enumerate(openings):
        parts.extend(
            add_window_assembly(
                f"{name}_window_{index + 1}",
                window,
                orientation,
                shift_x,
                shift_y,
                wall_thickness=thickness,
                z_offset=z_bottom,
            )
        )
    return parts


def local_to_world(point: Vec3, orientation: str, shift_x: float, shift_y: float) -> Vec3:
    x, y, z = point
    if orientation == "S":
        return (shift_x + x, shift_y + y, z)
    if orientation == "N":
        return (shift_x + x, shift_y - y, z)
    if orientation == "W":
        return (shift_x + y, shift_y + x, z)
    if orientation == "E":
        return (shift_x - y, shift_y + x, z)
    raise ValueError(f"Unknown orientation: {orientation}")


def orientation_rotation(orientation: str) -> float:
    if orientation == "S":
        return 0.0
    if orientation == "N":
        return math.radians(180.0)
    if orientation == "W":
        return math.radians(90.0)
    if orientation == "E":
        return math.radians(90.0)
    raise ValueError(f"Unknown orientation: {orientation}")


def add_oriented_box(
    name: str,
    size: Vec3,
    local_center: Vec3,
    orientation: str,
    shift_x: float,
    shift_y: float,
    material_name: str,
    color: RGBA | None = None,
) -> bpy.types.Object:
    world = local_to_world(local_center, orientation, shift_x, shift_y)
    if orientation in {"S", "N"}:
        final_size = size
    else:
        final_size = (size[1], size[0], size[2])
    return add_box(name, final_size, world, material_name, color)


def add_window_assembly(
    name: str,
    spec: WindowSpec,
    orientation: str,
    shift_x: float,
    shift_y: float,
    *,
    wall_thickness: float = OUTER_WALL_THICKNESS,
    z_offset: float = 0.0,
) -> list[bpy.types.Object]:
    parts: list[bpy.types.Object] = []
    z_bottom = z_offset + FLOOR_HEIGHT + spec.sill_height
    center_x = spec.shift + spec.width / 2.0
    center_z = z_bottom + spec.height / 2.0
    reveal_y = wall_thickness * 0.5
    frame_y = wall_thickness * 0.46
    glass_y = wall_thickness * 0.58

    def place(local_center: Vec3, size: Vec3, material: str) -> bpy.types.Object:
        part = add_oriented_box(f"{name}_{material}", size, local_center, orientation, shift_x, shift_y, material)
        parts.append(part)
        return part

    edge = EDGING_WIDTH if spec.height > 0.95 else 0.0
    frame = WINDOWFRAME_WIDTH
    depth = max(0.055, wall_thickness - 0.06)
    glass_depth = 0.012
    reveal_edge = edge if edge > 0.0 else 0.045
    reveal_depth = max(0.10, wall_thickness * 0.86)
    opening_w = spec.width + reveal_edge * 2.0
    opening_h = spec.height + reveal_edge * 2.0

    place((center_x, reveal_y, z_bottom - reveal_edge / 2.0), (opening_w, reveal_depth, reveal_edge), "window_reveal")
    place((center_x, reveal_y, z_bottom + spec.height + reveal_edge / 2.0), (opening_w, reveal_depth, reveal_edge), "window_reveal")
    place((spec.shift - reveal_edge / 2.0, reveal_y, center_z), (reveal_edge, reveal_depth, opening_h), "window_reveal")
    place((spec.shift + spec.width + reveal_edge / 2.0, reveal_y, center_z), (reveal_edge, reveal_depth, opening_h), "window_reveal")

    if edge > 0.0:
        outer_w = spec.width + edge * 2.0
        outer_h = spec.height + edge * 2.0
        place((center_x, reveal_y, z_bottom - edge / 2.0), (outer_w, depth, edge), "window_edging")
        place((center_x, reveal_y, z_bottom + spec.height + edge / 2.0), (outer_w, depth, edge), "window_edging")
        place((spec.shift - edge / 2.0, reveal_y, center_z), (edge, depth, outer_h), "window_edging")
        place((spec.shift + spec.width + edge / 2.0, reveal_y, center_z), (edge, depth, outer_h), "window_edging")

    place((center_x, glass_y, center_z), (max(0.05, spec.width - frame * 1.6), glass_depth, max(0.05, spec.height - frame * 1.6)), "glass")
    place((center_x, frame_y, z_bottom + spec.height - frame / 2.0), (spec.width, 0.055, frame), "window_frame")
    place((center_x, frame_y, z_bottom + frame / 2.0), (spec.width, 0.055, frame), "window_frame")
    place((spec.shift + frame / 2.0, frame_y, center_z), (frame, 0.055, spec.height), "window_frame")
    place((spec.shift + spec.width - frame / 2.0, frame_y, center_z), (frame, 0.055, spec.height), "window_frame")

    inner_width = max(0.05, spec.width - WINDOWFRAME_OFFSET * 2.0)
    inner_height = max(0.05, spec.height - WINDOWFRAME_OFFSET * 2.0)
    place((center_x, frame_y + 0.035, z_bottom + WINDOWFRAME_OFFSET + inner_height - frame / 2.0), (inner_width, 0.055, frame), "window_frame")
    place((center_x, frame_y + 0.035, z_bottom + WINDOWFRAME_OFFSET + frame / 2.0), (inner_width, 0.055, frame), "window_frame")
    place((spec.shift + WINDOWFRAME_OFFSET + frame / 2.0, frame_y + 0.035, center_z), (frame, 0.055, inner_height), "window_frame")
    place((spec.shift + spec.width - WINDOWFRAME_OFFSET - frame / 2.0, frame_y + 0.035, center_z), (frame, 0.055, inner_height), "window_frame")

    if spec.subdivisions >= 2:
        for split in range(1, spec.subdivisions):
            split_x = spec.shift + WINDOWFRAME_OFFSET + split * inner_width / spec.subdivisions
            place((split_x, frame_y + 0.055, center_z), (frame * 0.45, 0.055, inner_height), "window_frame")

    if spec.height > 0.8:
        place((center_x, reveal_y + 0.02, z_bottom - 0.045), (spec.width + 0.12, 0.16, 0.055), "window_frame")

    if spec.blind is not None and spec.blind > 0.0:
        slat_count = max(3, int(spec.height * spec.blind / 0.03))
        for slat in range(slat_count):
            z = z_bottom + spec.height - 0.055 - slat * 0.038
            place((center_x, frame_y + 0.095, z), (spec.width * 0.92, 0.018, 0.016), "blind")

    return parts


def add_house_door(
    name: str,
    orientation: str,
    shift_x: float,
    shift_y: float,
    local_shift: float,
    *,
    width: float = 1.10,
    height: float = 2.10,
    wall_thickness: float = OUTER_WALL_THICKNESS,
) -> list[bpy.types.Object]:
    z_bottom = FLOOR_HEIGHT
    frame_w = 0.12
    y = -0.08
    parts = [
        add_oriented_box(
            f"{name}_leaf",
            (width, 0.075, height),
            (local_shift + width / 2.0 + frame_w, y + 0.035, z_bottom + height / 2.0),
            orientation,
            shift_x,
            shift_y,
            "door",
        ),
    ]
    center_x = local_shift + width / 2.0 + frame_w
    parts.extend(
        [
            add_oriented_box(f"{name}_frame_top", (width + frame_w * 2.0, 0.09, frame_w), (center_x, y - 0.018, z_bottom + height + frame_w / 2.0), orientation, shift_x, shift_y, "window_edging"),
            add_oriented_box(f"{name}_frame_left", (frame_w, 0.09, height + frame_w), (local_shift + frame_w / 2.0, y - 0.018, z_bottom + (height + frame_w) / 2.0), orientation, shift_x, shift_y, "window_edging"),
            add_oriented_box(f"{name}_frame_right", (frame_w, 0.09, height + frame_w), (local_shift + width + frame_w * 1.5, y - 0.018, z_bottom + (height + frame_w) / 2.0), orientation, shift_x, shift_y, "window_edging"),
        ]
    )
    handle_x = local_shift + width + frame_w - 0.18
    handle_z = z_bottom + 1.03
    handle_center = local_to_world((handle_x, y + 0.10, handle_z), orientation, shift_x, shift_y)
    parts.append(add_cylinder(f"{name}_handle", 0.045, 0.16, handle_center, "handle", rotation=(math.radians(90), 0, 0)))
    return parts


def roof_transform_matrix(angle_degrees: float = 20.0, offset: float = 0.0) -> Matrix:
    return (
        Matrix.Translation(Vector((-0.36, -0.30, 4.13 + WALL_HEIGHT)))
        @ Matrix.Rotation(math.radians(angle_degrees), 4, "Y")
        @ Matrix.Translation(Vector((0.0, 0.0, offset)))
    )


def dormer_transform_matrix(angle_degrees: float = 9.0, offset: float = 0.0) -> Matrix:
    return (
        Matrix.Translation(Vector((0.0, -0.25 + 2.575, 4.03 + WALL_HEIGHT)))
        @ Matrix.Rotation(math.radians(angle_degrees), 4, "Y")
        @ Matrix.Translation(Vector((0.0, 0.0, offset)))
    )


def add_roof_slab(
    name: str,
    matrix: Matrix,
    size: Vec3,
    *,
    top_thickness: float = 0.13,
    underside_thickness: float = 0.02,
) -> list[bpy.types.Object]:
    return [
        add_transformed_box(f"{name}_white_underlay", (0.0, 0.0, 0.0), (size[0], size[1], underside_thickness), "roof_bottom", matrix),
        add_transformed_box(
            f"{name}_black_slab",
            (0.0, 0.0, underside_thickness),
            (size[0], size[1], top_thickness),
            "roof_top",
            matrix,
        ),
    ]


def add_roof_slab_with_opening(
    name: str,
    matrix: Matrix,
    size: Vec3,
    opening: tuple[float, float, float, float],
    *,
    top_thickness: float = 0.13,
    underside_thickness: float = 0.02,
) -> list[bpy.types.Object]:
    x0, x1, y0, y1 = opening
    x0 = max(0.0, min(size[0], x0))
    x1 = max(0.0, min(size[0], x1))
    y0 = max(0.0, min(size[1], y0))
    y1 = max(0.0, min(size[1], y1))
    spans = [
        ("west", 0.0, x0, 0.0, size[1]),
        ("east", x1, size[0], 0.0, size[1]),
        ("front", x0, x1, 0.0, y0),
        ("rear", x0, x1, y1, size[1]),
    ]

    parts: list[bpy.types.Object] = []
    for suffix, left, right, front, rear in spans:
        if right - left <= 0.001 or rear - front <= 0.001:
            continue
        parts.append(
            add_transformed_box(
                f"{name}_{suffix}_white_underlay",
                (left, front, 0.0),
                (right - left, rear - front, underside_thickness),
                "roof_bottom",
                matrix,
            )
        )
        parts.append(
            add_transformed_box(
                f"{name}_{suffix}_black_slab",
                (left, front, underside_thickness),
                (right - left, rear - front, top_thickness),
                "roof_top",
                matrix,
            )
        )
    return parts


def add_clipped_wall_along_x(
    name: str,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z_bottom0: float,
    z_bottom1: float,
    z_top0: float,
    z_top1: float,
    material_name: str = "outer_wall",
) -> bpy.types.Object:
    vertices = [
        (x0, y0, z_bottom0),
        (x1, y0, z_bottom1),
        (x1, y0, z_top1),
        (x0, y0, z_top0),
        (x0, y1, z_bottom0),
        (x1, y1, z_bottom1),
        (x1, y1, z_top1),
        (x0, y1, z_top0),
    ]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return add_mesh_object(name, vertices, faces, material_name)


def add_solar_panel(
    name: str,
    x: float,
    y: float,
    matrix: Matrix,
    *,
    width: float = 1.640,
    height: float = 0.992,
) -> list[bpy.types.Object]:
    parts = [
        add_transformed_box(name, (x, y, 0.18), (width, height, 0.035), "solar", matrix),
    ]
    rail = 0.018
    parts.append(add_transformed_box(f"{name}_vertical_grid", (x + width / 2.0 - rail / 2.0, y, 0.218), (rail, height, 0.01), "solar_grid", matrix))
    parts.append(add_transformed_box(f"{name}_horizontal_grid", (x, y + height / 2.0 - rail / 2.0, 0.219), (width, rail, 0.01), "solar_grid", matrix))
    return parts


def add_solar_panel_clipped(
    name: str,
    x: float,
    y: float,
    matrix: Matrix,
    cutouts: list[tuple[float, float, float, float]] | tuple[tuple[float, float, float, float], ...],
    *,
    width: float = 1.640,
    height: float = 0.992,
) -> list[bpy.types.Object]:
    parts = add_clipped_transformed_box(name, (x, y, 0.18), (width, height, 0.035), "solar", matrix, cutouts)
    rail = 0.018
    parts.extend(
        add_clipped_transformed_box(
            f"{name}_vertical_grid",
            (x + width / 2.0 - rail / 2.0, y, 0.218),
            (rail, height, 0.01),
            "solar_grid",
            matrix,
            cutouts,
        )
    )
    parts.extend(
        add_clipped_transformed_box(
            f"{name}_horizontal_grid",
            (x, y + height / 2.0 - rail / 2.0, 0.219),
            (width, rail, 0.01),
            "solar_grid",
            matrix,
            cutouts,
        )
    )
    return parts


def add_panel_array(
    name: str,
    matrix: Matrix,
    origin_x: float,
    origin_y: float,
    columns: int,
    rows: int,
) -> list[bpy.types.Object]:
    parts: list[bpy.types.Object] = []
    for row in range(rows):
        for column in range(columns):
            parts.extend(add_solar_panel(f"{name}_{row}_{column}", origin_x + column * 1.68, origin_y + row * 1.03, matrix))
    return parts


def add_panel_array_clipped(
    name: str,
    matrix: Matrix,
    origin_x: float,
    origin_y: float,
    columns: int,
    rows: int,
    cutouts: list[tuple[float, float, float, float]] | tuple[tuple[float, float, float, float], ...],
) -> list[bpy.types.Object]:
    parts: list[bpy.types.Object] = []
    for row in range(rows):
        for column in range(columns):
            parts.extend(add_solar_panel_clipped(f"{name}_{row}_{column}", origin_x + column * 1.68, origin_y + row * 1.03, matrix, cutouts))
    return parts


def add_roof_ladder(
    name: str,
    matrix: Matrix,
    y: float,
    cutouts: list[tuple[float, float, float, float]] | tuple[tuple[float, float, float, float], ...] | None = None,
) -> list[bpy.types.Object]:
    parts: list[bpy.types.Object] = []
    for rung in range(18):
        if cutouts:
            parts.extend(
                add_clipped_transformed_box(
                    f"{name}_rung_{rung:02d}",
                    (11.8 - rung * 0.30, y + (rung % 2) * 0.08, 0.205),
                    (0.15, 0.15, 0.05),
                    "roof_trim",
                    matrix,
                    cutouts,
                )
            )
        else:
            parts.append(
                add_transformed_box(
                    f"{name}_rung_{rung:02d}",
                    (11.8 - rung * 0.30, y + (rung % 2) * 0.08, 0.205),
                    (0.15, 0.15, 0.05),
                    "roof_trim",
                    matrix,
                )
            )
    return parts


def add_downpipe_vertical(name: str, x: float, y: float, z_bottom: float, height: float) -> list[bpy.types.Object]:
    parts = [
        add_cylinder(
            f"{name}_pipe",
            0.10,
            height,
            (x, y, z_bottom + height / 2.0),
            "chimney",
            vertices=24,
        )
    ]
    for band in range(5):
        parts.append(add_cylinder(f"{name}_band_{band}", 0.13, 0.025, (x, y, z_bottom + 0.35 + band * 0.48), "chimney", vertices=24))
    return parts


def add_elbowed_downpipe(
    name: str,
    x: float,
    y: float,
    z_bottom: float,
    height: float,
    *,
    diameter: float = 0.10,
) -> list[bpy.types.Object]:
    z_top = z_bottom + height
    parts = [
        add_cylinder_between(f"{name}_vertical_pipe", (x, y, z_bottom), (x, y, z_top), diameter, "chimney"),
        add_cylinder_between(f"{name}_upper_elbow_a", (x, y, z_top), (x - 0.16, y, z_top + 0.13), diameter, "chimney"),
        add_cylinder_between(f"{name}_upper_elbow_b", (x - 0.16, y, z_top + 0.13), (x - 0.36, y, z_top + 0.13), diameter, "chimney"),
        add_cylinder_between(f"{name}_lower_elbow", (x, y, z_bottom), (x - 0.20, y, z_bottom - 0.12), diameter, "chimney"),
    ]
    for band in range(5):
        z = z_bottom + 0.35 + band * 0.48
        parts.append(add_cylinder_between(f"{name}_band_{band}", (x, y, z - 0.012), (x, y, z + 0.012), diameter * 1.28, "chimney"))
    return parts


def add_chimney(name: str, x: float = 6.0, y: float = 2.7, z_bottom: float = 0.0, height: float = WALL_HEIGHT * 2.0 + 0.6) -> list[bpy.types.Object]:
    parts = [
        add_scad_box(f"{name}_shaft", (x, y, z_bottom), (0.40, 0.40, height), "chimney"),
        add_scad_box(f"{name}_cap", (x - 0.04, y - 0.04, z_bottom + height), (0.48, 0.48, 0.08), "chimney"),
        add_cylinder(f"{name}_flue", 0.16, height + 0.10, (x + 0.20, y + 0.20, z_bottom + height / 2.0 + 0.05), "roof_top", vertices=32),
    ]
    return parts


def add_lamp(name: str, location: Vec3) -> list[bpy.types.Object]:
    x, y, z = location
    return [
        add_cylinder(f"{name}_shade", 0.16, 0.05, (x, y, z), "chimney", vertices=24),
        add_cylinder(f"{name}_bulb", 0.06, 0.04, (x, y, z - 0.06), "inner_wall", vertices=16),
    ]


def add_funnel_grille(name: str, corner: Vec3, size: Vec3, orientation: str = "S") -> list[bpy.types.Object]:
    x, y, z = corner
    width, depth, height = size
    parts = [add_scad_box(f"{name}_back", corner, size, "chimney")]
    for slat in range(5):
        if orientation in {"S", "N"}:
            parts.append(add_scad_box(f"{name}_slat_{slat}", (x + 0.03, y + depth + 0.006, z + 0.05 + slat * 0.09), (width - 0.06, 0.025, 0.025), "blind"))
        else:
            parts.append(add_scad_box(f"{name}_slat_{slat}", (x + depth + 0.006, y + 0.03, z + 0.05 + slat * 0.09), (0.025, width - 0.06, 0.025), "blind"))
    return parts


# ============================================================
# Procedural assembly
# ============================================================

"""Assembly logic for the current multi-floor house model."""





HOUSE_WIDTH = 10.49
HOUSE_DEPTH = 8.615
SECOND_FLOOR_HEIGHT = 4.135
ROOF_WIDTH = 12.0
ROOF_DEPTH = 8.61 + 0.3 + 0.3
ROOF_OPENING = (0.565, 6.365, 2.575, 6.035)
DORMER_X0 = 0.565
DORMER_EAST_OUTER_X = 6.01 + 0.365 + 0.24
DORMER_Y0 = 2.575
DORMER_Y1 = 6.035
DORMER_SOUTH_WALL_Y0 = 2.76 - 0.24
DORMER_SOUTH_WALL_Y1 = 2.76
DORMER_NORTH_WALL_Y0 = 2.76 + 3.46 - 0.175 - 0.24
DORMER_NORTH_WALL_Y1 = 2.76 + 3.46 - 0.175
DORMER_WALL_THICKNESS = 0.24
ROOF_DETAIL_CUTOUTS = ((DORMER_X0 - 0.10, DORMER_EAST_OUTER_X + 0.08, DORMER_Y0 - 0.26, DORMER_Y1 + 0.30),)


def scad_transform_matrix(translate: tuple[float, float, float], rotate_z_degrees: float = 0.0) -> Matrix:
    return Matrix.Translation(Vector(translate)) @ Matrix.Rotation(math.radians(rotate_z_degrees), 4, "Z")


def main_roof_plane_z(world_x: float) -> float:
    return WALL_HEIGHT + 4.13 - math.tan(math.radians(20.0)) * (world_x + 0.36)


def dormer_roof_plane_z(world_x: float) -> float:
    return WALL_HEIGHT + 4.03 - math.tan(math.radians(9.0)) * world_x


def build_outdoors() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    objects.append(add_scad_box("garden_slab", (-3.5, -3.0, -FLOOR_HEIGHT), (19.0, 13.0, FLOOR_HEIGHT), "garden"))

    path_specs = [
        ("front_walk", (-2.0, -1.4, 0.0), (17.0, 1.2, 0.04)),
        ("entry_walk", (-2.0, 0.8, 0.0), (2.0, 1.2, 0.04)),
        ("front_door_pad", (-0.35, 0.90, 0.0), (1.35, 1.05, 0.055)),
    ]
    for name, corner, size in path_specs:
        objects.append(add_scad_box(name, corner, size, "walk"))

    rotated_paths = [
        ("left_side_walk", (-1.8, -1.4, 0.0), (11.0, 1.2, 0.04), "walk"),
        ("right_side_walk", (14.0, -1.4, 0.0), (5.0, 1.2, 0.04), "walk"),
        ("terrace", (14.3, 3.4, 0.0), (5.0, 3.9, 0.10), "walk"),
    ]
    for name, translate, size, material in rotated_paths:
        objects.append(add_transformed_box(name, (0.0, 0.0, 0.0), size, material, scad_transform_matrix(translate, 90.0)))

    entry_points = [
        (-0.90, 0.55),
        (0.00, 0.55),
        (0.00, 2.35),
        (-0.90, 2.35),
        (-1.30, 2.10),
        (-1.50, 1.45),
        (-1.30, 0.80),
    ]
    objects.append(add_prism_from_xy("semi_round_entry_step", entry_points, 0.0, FLOOR_HEIGHT * 0.5, "walk"))
    objects.append(add_scad_box("entry_mat", (-0.65, 1.02, FLOOR_HEIGHT * 0.5), (0.60, 0.90, 0.02), "entry_mat"))
    return objects


def first_floor_window_specs() -> dict[str, list[WindowSpec]]:
    return {
        "south": [
            WindowSpec(1.13, 1.42, 0.86, 0.84, 0.0),
            WindowSpec(1.51, 1.42, 4.115, 0.84, 0.0),
            WindowSpec(1.01, 1.42, 8.365, 0.84, 0.3),
        ],
        "north": [
            WindowSpec(1.13, 0.635, 1.49, 1.63, 0.0),
            WindowSpec(0.76, 0.635, 3.86, 1.63, 0.0),
            WindowSpec(1.13, 0.635, 5.235, 1.63, 0.0),
            WindowSpec(1.13, 0.50, 8.865, 0.95, 0.0),
        ],
        "east": [
            WindowSpec(1.51, 1.42, 1.11, 0.84, 0.0),
            WindowSpec(3.01, 2.26, 4.485, 0.065, 0.0, 2),
        ],
    }


def second_floor_window_specs() -> dict[str, list[WindowSpec]]:
    return {
        "south": [
            WindowSpec(1.13, 1.42, 0.86, 0.84, 0.0),
        ],
        "west": [
            WindowSpec(0.885, 2.26, 3.74, 0.21, None),
            WindowSpec(1.51, 1.72, 0.74, 0.84, 0.0),
            WindowSpec(1.51, 1.72, 6.36, 0.84, 0.0),
        ],
    }


def build_floor_slabs() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    objects.append(add_scad_box("ground_floor_slab", (OUTER_WALL_THICKNESS, OUTER_WALL_THICKNESS, 0.0), (9.76, 7.885, FLOOR_HEIGHT), "inner_wall"))
    objects.append(add_scad_box("upper_floor_thin_plate", (OUTER_WALL_THICKNESS, OUTER_WALL_THICKNESS, WALL_HEIGHT), (9.76, 7.885, FLOOR_HEIGHT * 0.45), "inner_wall"))
    return objects


def build_first_floor_shell() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    windows = first_floor_window_specs()
    objects.extend(add_facade_wall("first_south", HOUSE_WIDTH, WALL_HEIGHT, "S", 0.0, 0.0, windows=windows["south"]))
    objects.extend(add_facade_wall("first_north", HOUSE_WIDTH, WALL_HEIGHT, "N", 0.0, HOUSE_DEPTH, windows=windows["north"]))
    objects.extend(add_facade_wall("first_west", HOUSE_DEPTH, WALL_HEIGHT, "W", 0.0, 0.0, windows=[]))
    objects.extend(add_facade_wall("first_east", HOUSE_DEPTH, WALL_HEIGHT, "E", HOUSE_WIDTH, 0.0, windows=windows["east"]))

    objects.extend(add_house_door("front_house_door", "W", 0.0, 0.0, 0.80))
    objects.extend(add_lamp("front_wall_lamp", (-0.13, 0.52, 2.30)))
    objects.extend(add_lamp("terrace_wall_lamp", (10.63, 8.00, 2.30)))
    objects.extend(add_funnel_grille("rear_wall_funnel", (0.58, 8.50, 1.32), (0.60, 0.035, 0.60), "N"))
    objects.extend(add_funnel_grille("west_wall_funnel", (-0.02, 7.40, 1.32), (0.60, 0.035, 0.60), "W"))
    return objects


def build_second_floor_shell() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    windows = second_floor_window_specs()
    z0 = WALL_HEIGHT
    objects.extend(
        add_sloped_facade_wall_with_windows(
            "second_south_sloped_wall",
            HOUSE_WIDTH,
            4.13,
            0.68,
            "S",
            0.0,
            0.0,
            z_bottom=z0,
            windows=windows["south"],
        )
    )
    objects.append(add_sloped_facade_wall("second_north_sloped_wall", HOUSE_WIDTH, 4.13, 0.68, "N", 0.0, HOUSE_DEPTH, z_bottom=z0))
    objects.extend(
        add_facade_wall(
            "second_west",
            HOUSE_DEPTH,
            SECOND_FLOOR_HEIGHT,
            "W",
            0.0,
            0.0,
            windows=windows["west"],
            z_offset=z0,
        )
    )
    objects.extend(add_facade_wall("second_east_low_wall", HOUSE_DEPTH, 0.68, "E", HOUSE_WIDTH, 0.0, windows=[], z_offset=z0))
    return objects


def build_interior_hints() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    objects.append(add_scad_box("living_room_floor", (6.28, 0.365, FLOOR_HEIGHT + 0.004), (3.51, 5.07, 0.025), "stair"))
    objects.append(add_scad_box("kitchen_floor", (6.28, 5.435, FLOOR_HEIGHT + 0.006), (3.51, 2.815, 0.025), "walk"))
    objects.append(add_scad_box("entry_partition_hint", (2.50, 2.94, FLOOR_HEIGHT), (3.95, 0.10, 1.25), "inner_wall"))
    objects.append(add_scad_box("bathroom_partition_hint", (6.05, 5.20, WALL_HEIGHT + FLOOR_HEIGHT), (0.12, 2.70, 1.50), "inner_wall"))
    return objects


def build_stairs_hint() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    base_x = OUTER_WALL_THICKNESS
    base_y = 2.53
    step_width = 0.268
    step_height = 0.181
    width = 1.0
    for step in range(9):
        depth = step_width
        objects.append(
            add_scad_box(
                f"stair_lower_step_{step:02d}",
                (base_x, base_y + step * step_width, FLOOR_HEIGHT + step * step_height),
                (width, depth, 0.05),
                "stair",
            )
        )
    for step in range(5):
        objects.append(
            add_scad_box(
                f"stair_turn_step_{step:02d}",
                (base_x + 1.0 + step * step_width, base_y + 1.92, FLOOR_HEIGHT + (9 + step) * step_height),
                (step_width, width, 0.05),
                "stair",
            )
        )
    objects.append(add_scad_box("stair_side_rail_lower", (base_x + width - 0.08, base_y, FLOOR_HEIGHT + 0.35), (0.06, 2.3, 0.08), "stair_side"))
    objects.append(add_scad_box("stair_side_rail_upper", (base_x + 1.0, base_y + 1.92, WALL_HEIGHT + 0.42), (1.35, 0.06, 0.08), "stair_side"))
    return objects


def build_roof() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    main_roof_matrix = roof_transform_matrix()
    dormer_matrix = dormer_transform_matrix()
    side_x0 = DORMER_X0
    side_x1 = DORMER_EAST_OUTER_X
    side_bottom0 = main_roof_plane_z(side_x0) + 0.035
    side_bottom1 = main_roof_plane_z(side_x1) + 0.035
    side_top0 = dormer_roof_plane_z(side_x0) - 0.045
    side_top1 = dormer_roof_plane_z(side_x1) - 0.045
    east_bottom = main_roof_plane_z(DORMER_EAST_OUTER_X) + 0.06
    east_top = dormer_roof_plane_z(DORMER_EAST_OUTER_X) - 0.06
    chimney_base = dormer_roof_plane_z(6.0) + 0.02

    objects.extend(add_chimney("roof_chimney_stub", z_bottom=chimney_base, height=0.52))
    objects.extend(add_roof_slab_with_opening("main_sloped_roof", main_roof_matrix, (ROOF_WIDTH, ROOF_DEPTH, 0.15), ROOF_OPENING))
    objects.extend(add_panel_array_clipped("roof_panels_lower_left", main_roof_matrix, 0.20, 0.20, 7, 2, ROOF_DETAIL_CUTOUTS))
    objects.extend(add_panel_array_clipped("roof_panels_upper_left", main_roof_matrix, 0.20, 7.00, 7, 2, ROOF_DETAIL_CUTOUTS))
    objects.extend(add_panel_array_clipped("roof_panels_right_bank", main_roof_matrix, 8.00, 3.10, 2, 3, ROOF_DETAIL_CUTOUTS))
    objects.extend(add_roof_ladder("roof_center_ladder_low", main_roof_matrix, 2.35, ROOF_DETAIL_CUTOUTS))
    objects.extend(add_roof_ladder("roof_center_ladder_high", main_roof_matrix, 6.70, ROOF_DETAIL_CUTOUTS))

    objects.extend(add_roof_slab("dormer_roof", dormer_matrix, (7.0, 3.96, 0.15)))
    objects.extend(add_panel_array("dormer_panel_grid", dormer_matrix, 0.30, 0.50, 3, 3))

    # Dormer walls are clipped between the main roof cut and the shallower dormer roof.
    objects.append(
        add_clipped_wall_along_x(
            "dormer_south_clipped_face",
            side_x0,
            side_x1,
            DORMER_SOUTH_WALL_Y0,
            DORMER_SOUTH_WALL_Y1,
            side_bottom0,
            side_bottom1,
            side_top0,
            side_top1,
        )
    )
    objects.append(
        add_clipped_wall_along_x(
            "dormer_north_clipped_face",
            side_x0,
            side_x1,
            DORMER_NORTH_WALL_Y0,
            DORMER_NORTH_WALL_Y1,
            side_bottom0,
            side_bottom1,
            side_top0,
            side_top1,
        )
    )
    objects.extend(
        add_facade_wall(
            "dormer_east_clipped_face",
            DORMER_Y1 - DORMER_Y0,
            max(0.2, east_top - east_bottom),
            "E",
            DORMER_EAST_OUTER_X,
            DORMER_Y0,
            windows=[WindowSpec(1.51, 0.72, 0.95, 0.10, None)],
            thickness=DORMER_WALL_THICKNESS,
            z_offset=east_bottom,
        )
    )
    objects.extend(add_elbowed_downpipe("right_front_downpipe", 10.55, 8.52, 0.34, 2.38))
    return objects


def build_roof_edge_details() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    matrix = roof_transform_matrix()
    objects.append(add_transformed_box("front_eave_dark_edge", (0.0, 0.0, 0.178), (ROOF_WIDTH, 0.022, 0.022), "roof_trim", matrix))
    objects.append(add_transformed_box("rear_eave_dark_gutter", (0.0, ROOF_DEPTH - 0.022, 0.178), (ROOF_WIDTH, 0.022, 0.022), "roof_trim", matrix))
    objects.append(add_transformed_box("left_rake_dark_edge", (0.0, 0.0, 0.178), (0.022, ROOF_DEPTH, 0.022), "roof_trim", matrix))
    objects.append(add_transformed_box("right_rake_dark_edge", (ROOF_WIDTH - 0.022, 0.0, 0.178), (0.022, ROOF_DEPTH, 0.022), "roof_trim", matrix))
    objects.append(add_transformed_box("roof_downpipe_half_round_gutter", (11.945, 0.10, 0.158), (0.040, 8.80, 0.040), "roof_trim", matrix))
    for band in range(13):
        objects.append(add_transformed_box(f"roof_downpipe_gutter_band_{band:02d}", (11.930, 0.26 + band * 0.64, 0.192), (0.065, 0.014, 0.026), "roof_trim", matrix))
    objects.append(add_transformed_box("roof_gutter_outlet_to_downpipe", (11.74, 8.75, 0.145), (0.26, 0.055, 0.042), "roof_trim", matrix))
    return objects


def build_model() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    objects.extend(build_outdoors())
    objects.extend(build_floor_slabs())
    objects.extend(build_first_floor_shell())
    objects.extend(build_second_floor_shell())
    objects.extend(build_interior_hints())
    objects.extend(build_stairs_hint())
    objects.extend(build_roof())
    objects.extend(build_roof_edge_details())
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

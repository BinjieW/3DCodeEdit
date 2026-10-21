from __future__ import annotations
from dataclasses import dataclass
from mathutils import Matrix, Vector
from mathutils.geometry import tessellate_polygon
from typing import Iterable, Sequence
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the standing-desk model."""





Point2D = tuple[float, float]


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.58
    metallic: float = 0.0


PLYWOOD_GOLD = MaterialSpec("openscad_preview_yellow_gold", (0.96, 0.78, 0.03, 1.0), 0.62)
PLYWOOD_EDGE = MaterialSpec("slightly_darker_plywood_edges", (0.58, 0.48, 0.07, 1.0), 0.66)
SLOT_SHADOW = MaterialSpec("dark_shadow_inside_joinery", (0.08, 0.065, 0.015, 1.0), 0.75)


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
        bsdf.inputs["Metallic"].default_value = spec.metallic
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


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: MaterialSpec | None = PLYWOOD_GOLD,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.0))
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.location = location
    obj.rotation_euler = rotation
    if material:
        assign_material(obj, material)
    return obj


def add_cylinder(
    name: str,
    radius: float,
    depth: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 48,
    material: MaterialSpec | None = PLYWOOD_GOLD,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    if material:
        assign_material(obj, material)
    if vertices > 16:
        shade_smooth(obj)
    return obj


def _transform_points(points: Sequence[Point2D], matrix: Matrix) -> list[Vector]:
    return [matrix @ Vector((x, y, 0.0)) for x, y in points]


def _signed_area(points: Sequence[Point2D]) -> float:
    return sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, [*points[1:], points[0]])
    ) * 0.5


def _oriented(points: Sequence[Point2D], clockwise: bool) -> list[Point2D]:
    ordered = list(points)
    is_clockwise = _signed_area(ordered) < 0.0
    if is_clockwise != clockwise:
        ordered.reverse()
    return ordered


def _make_prism_mesh(
    name: str,
    base_points: Sequence[Point2D],
    height: float,
    transform: Matrix,
    holes: Sequence[Sequence[Point2D]] | None = None,
) -> bpy.types.Mesh:
    contours = [_oriented(base_points, clockwise=False)]
    contours.extend(_oriented(hole, clockwise=True) for hole in (holes or []))
    flat_points = [point for contour in contours for point in contour]

    bottom = _transform_points(flat_points, transform)
    normal_matrix = Matrix.Translation((0.0, 0.0, height))
    top = [transform @ (normal_matrix @ Vector((x, y, 0.0))) for x, y in flat_points]
    vertices = bottom + top
    count = len(flat_points)

    tessellation_contours = [
        [Vector((x, y, 0.0)) for x, y in contour]
        for contour in contours
    ]
    triangles = tessellate_polygon(tessellation_contours)
    faces: list[tuple[int, ...]] = []
    faces.extend(tuple(reversed(tri)) for tri in triangles)
    faces.extend(tuple(index + count for index in tri) for tri in triangles)

    offset = 0
    for contour_index, contour in enumerate(contours):
        contour_count = len(contour)
        for index in range(contour_count):
            current = offset + index
            nxt = offset + ((index + 1) % contour_count)
            if contour_index == 0:
                faces.append((current, nxt, nxt + count, current + count))
            else:
                faces.append((current, current + count, nxt + count, nxt))
        offset += contour_count

    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata([tuple(v) for v in vertices], [], faces)
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    mesh.update()
    return mesh


def add_prism_from_points(
    name: str,
    points: Sequence[Point2D],
    height: float,
    material: MaterialSpec = PLYWOOD_GOLD,
    transform: Matrix | None = None,
) -> bpy.types.Object:
    mesh = _make_prism_mesh(name, points, height, transform or Matrix.Identity(4))
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def add_prism_from_points_with_holes(
    name: str,
    points: Sequence[Point2D],
    holes: Sequence[Sequence[Point2D]],
    height: float,
    material: MaterialSpec = PLYWOOD_GOLD,
    transform: Matrix | None = None,
) -> bpy.types.Object:
    mesh = _make_prism_mesh(name, points, height, transform or Matrix.Identity(4), holes)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    obj.modifiers.new("weighted sheet normals", "WEIGHTED_NORMAL")
    return obj


def rounded_rect_points(width: float, depth: float, radius: float, segments: int = 10) -> list[Point2D]:
    radius = min(radius, width / 2.0, depth / 2.0)
    centers = [
        (width - radius, depth - radius, 0.0, math.pi / 2.0),
        (radius, depth - radius, math.pi / 2.0, math.pi),
        (radius, radius, math.pi, 3.0 * math.pi / 2.0),
        (width - radius, radius, 3.0 * math.pi / 2.0, 2.0 * math.pi),
    ]
    points: list[Point2D] = []
    for cx, cy, start, end in centers:
        for step in range(segments + 1):
            angle = start + (end - start) * step / segments
            points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
    return points


def rounded_rect_hole_points(
    x0: float,
    y0: float,
    width: float,
    depth: float,
    radius: float,
    segments: int = 8,
) -> list[Point2D]:
    radius = min(radius, width / 2.0, depth / 2.0)
    centers = [
        (x0 + width - radius, y0 + radius, -math.pi / 2.0, 0.0),
        (x0 + width - radius, y0 + depth - radius, 0.0, math.pi / 2.0),
        (x0 + radius, y0 + depth - radius, math.pi / 2.0, math.pi),
        (x0 + radius, y0 + radius, math.pi, 3.0 * math.pi / 2.0),
    ]
    points: list[Point2D] = []
    for cx, cy, start, end in centers:
        for step in range(segments + 1):
            angle = start + (end - start) * step / segments
            points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
    return points


def _append_arc(
    points: list[Point2D],
    center: Point2D,
    radius: float,
    start_angle: float,
    end_angle: float,
    segments: int,
) -> None:
    cx, cy = center
    for step in range(segments + 1):
        angle = start_angle + (end_angle - start_angle) * step / segments
        point = (cx + math.cos(angle) * radius, cy + math.sin(angle) * radius)
        if points and math.dist(points[-1], point) < 1e-7:
            continue
        points.append(point)


def dogbone_slot_points(
    x0: float,
    y0: float,
    width: float,
    depth: float,
    radius: float,
    segments: int = 8,
) -> list[Point2D]:
    """2D outline of the source dogboneCube() footprint."""
    if width <= 0.0 or depth <= 0.0:
        raise ValueError("Dogbone slot dimensions must be positive")

    if radius <= 0.0:
        return [
            (x0, y0),
            (x0 + width, y0),
            (x0 + width, y0 + depth),
            (x0, y0 + depth),
        ]

    offset_factor = math.sin(math.pi / 4.0)
    radius = min(radius, min(width, depth) / (2.0 * offset_factor))
    offset = offset_factor * radius
    x1 = x0 + width
    y1 = y0 + depth

    points: list[Point2D] = [
        (x0 + 2.0 * offset, y0),
        (x1 - 2.0 * offset, y0),
    ]
    _append_arc(
        points,
        (x1 - offset, y0 + offset),
        radius,
        5.0 * math.pi / 4.0,
        9.0 * math.pi / 4.0,
        segments,
    )
    points.append((x1, y1 - 2.0 * offset))
    _append_arc(
        points,
        (x1 - offset, y1 - offset),
        radius,
        -math.pi / 4.0,
        3.0 * math.pi / 4.0,
        segments,
    )
    points.append((x0 + 2.0 * offset, y1))
    _append_arc(
        points,
        (x0 + offset, y1 - offset),
        radius,
        math.pi / 4.0,
        5.0 * math.pi / 4.0,
        segments,
    )
    points.append((x0, y0 + 2.0 * offset))
    _append_arc(
        points,
        (x0 + offset, y0 + offset),
        radius,
        3.0 * math.pi / 4.0,
        7.0 * math.pi / 4.0,
        segments,
    )
    return points


def rounded_end_tab_points(
    length: float,
    width: float,
    radius: float,
    segments: int = 10,
) -> list[Point2D]:
    """Rectangular joint tongue with only the two far outside corners rounded."""
    radius = min(radius, length * 0.45, width * 0.45)
    points: list[Point2D] = [(0.0, 0.0), (length - radius, 0.0)]
    _append_arc(points, (length - radius, radius), radius, -math.pi / 2.0, 0.0, segments)
    points.append((length, width - radius))
    _append_arc(points, (length - radius, width - radius), radius, 0.0, math.pi / 2.0, segments)
    points.append((0.0, width))
    return points


def axis_transform(
    origin: Vector,
    x_axis: Vector,
    y_axis: Vector,
    z_axis: Vector,
) -> Matrix:
    return Matrix(
        (
            (x_axis.x, y_axis.x, z_axis.x, origin.x),
            (x_axis.y, y_axis.y, z_axis.y, origin.y),
            (x_axis.z, y_axis.z, z_axis.z, origin.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def add_joint_tab(
    name: str,
    base_center: tuple[float, float, float],
    length_axis: tuple[float, float, float],
    width_axis: tuple[float, float, float],
    thickness_axis: tuple[float, float, float],
    length: float,
    width: float,
    thickness: float,
    slot_width: float,
    slot_length: float,
    slot_offset: float,
    corner_radius: float,
    fillet_radius: float,
    material: MaterialSpec = PLYWOOD_GOLD,
) -> bpy.types.Object:
    length_v = Vector(length_axis).normalized()
    width_v = Vector(width_axis).normalized()
    thickness_v = Vector(thickness_axis).normalized()
    base_v = Vector(base_center)
    origin = base_v - width_v * (width / 2.0) - thickness_v * (thickness / 2.0)
    transform = axis_transform(origin, length_v, width_v, thickness_v)
    slot_y = (width - slot_width) / 2.0
    outer = rounded_end_tab_points(length, width, corner_radius)
    key_slot = dogbone_slot_points(
        slot_offset,
        slot_y,
        slot_length,
        slot_width,
        fillet_radius,
        segments=8,
    )
    return add_prism_from_points_with_holes(
        name,
        outer,
        [key_slot],
        thickness,
        material,
        transform,
    )


def add_rounded_sheet(
    name: str,
    width: float,
    depth: float,
    thickness: float,
    radius: float,
    transform: Matrix,
    material: MaterialSpec = PLYWOOD_GOLD,
) -> bpy.types.Object:
    return add_prism_from_points(name, rounded_rect_points(width, depth, radius), thickness, material, transform)


def add_sheet_box_between(
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    width: float,
    thickness: float,
    material: MaterialSpec = PLYWOOD_GOLD,
) -> bpy.types.Object:
    """Add a long rectangular strip between two 3D points with a vertical face."""
    start_v = Vector(start)
    end_v = Vector(end)
    length_vec = end_v - start_v
    length = length_vec.length
    if length < 1e-6:
        raise ValueError(f"Cannot build zero-length strip {name}")
    center = (start_v + end_v) * 0.5
    direction = length_vec.normalized()
    angle_z = math.atan2(direction.y, direction.x)
    pitch = math.atan2(direction.z, math.hypot(direction.x, direction.y))
    obj = add_box(name, (length, thickness, width), tuple(center), material)
    obj.rotation_euler = (0.0, -pitch, angle_z)
    return obj


def add_tab(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    axis: str,
    material: MaterialSpec = PLYWOOD_GOLD,
) -> bpy.types.Object:
    rotation = (0.0, 0.0, 0.0)
    if axis == "x":
        rotation = (0.0, math.pi / 2.0, 0.0)
    elif axis == "y":
        rotation = (math.pi / 2.0, 0.0, 0.0)
    return add_box(name, size, center, material, rotation=rotation)


def add_slot_marker(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    axis: str = "x",
    material: MaterialSpec = SLOT_SHADOW,
) -> bpy.types.Object:
    """Dark inset showing a female slot where the SCAD subtracts a dogbone hole."""
    rotation = (0.0, 0.0, 0.0)
    if axis == "x":
        rotation = (0.0, math.pi / 2.0, 0.0)
    elif axis == "y":
        rotation = (math.pi / 2.0, 0.0, 0.0)
    obj = add_box(name, size, center, material, rotation=rotation)
    return obj


def add_edge_band(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    material: MaterialSpec = PLYWOOD_EDGE,
) -> bpy.types.Object:
    return add_box(name, size, center, material)


def group_objects(name: str, objects: Iterable[bpy.types.Object]) -> None:
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    for obj in objects:
        for coll in list(obj.users_collection):
            coll.objects.unlink(obj)
        collection.objects.link(obj)


# ============================================================
# Procedural assembly
# ============================================================

"""Assembly logic for the standing-desk model."""






DESK_HEIGHT = 1100.0
DESK_WIDTH = 1120.0
DESK_DEPTH = 800.0
STOCK_THICKNESS = 12.0
BASE_HEIGHT = 200.0
STAND_DEPTH = 180.0
STAND_SPACE = 200.0
REST_DEPTH = 130.0
REST_LEVEL = 240.0
REST_ANGLE = math.radians(15.0)
TOPSUPPORT_DEPTH = 110.0
CROSS_CUT = max(DESK_HEIGHT / 7.0, 140.0)
CROSS_CLEARANCE = 0.0
JOINT_WIDTH = 60.0
JOINT_LENGTH = 45.0
JOINT_SLOT_WIDTH = STOCK_THICKNESS
JOINT_SLOT_LENGTH = STOCK_THICKNESS * 1.2
JOINT_SLOT_OFFSET = STOCK_THICKNESS * 0.8
JOINT_SLOT_TOLERANCE = 0.2
JOINT_BUFFER = max(3.3 * STOCK_THICKNESS, 6.0)
CORNER_RADIUS = 10.0
FILLET_RADIUS = 4.0 / 1.9
TOTAL_STAND_HEIGHT = DESK_HEIGHT + JOINT_BUFFER


def _top_transform() -> Matrix:
    return Matrix.Translation((-DESK_WIDTH / 2.0, -DESK_DEPTH / 2.0, DESK_HEIGHT))


def _xy_sheet_transform(x: float, y: float, z: float, rot_x: float = 0.0) -> Matrix:
    return Matrix.Translation((x, y, z)) @ Matrix.Rotation(rot_x, 4, "X")


def _xz_sheet_transform(x: float, y: float, z: float = 0.0) -> Matrix:
    return (
        Matrix.Translation((x, y, z))
        @ Matrix.Rotation(math.radians(90.0), 4, "Z")
        @ Matrix.Rotation(math.radians(90.0), 4, "X")
    )


def _side_panel_x(side: str) -> float:
    if side == "left":
        return -DESK_WIDTH / 2.0 - STOCK_THICKNESS
    if side == "right":
        return DESK_WIDTH / 2.0
    raise ValueError(side)


def _part_x_center(side: str) -> float:
    return _side_panel_x(side) + STOCK_THICKNESS / 2.0


def _world_y(source_y: float) -> float:
    return source_y - DESK_DEPTH / 2.0


def _joint_tab(
    name: str,
    base_center: Vector,
    length_axis: Vector,
    width_axis: Vector,
    thickness_axis: Vector,
) -> object:
    return add_joint_tab(
        name=name,
        base_center=tuple(base_center),
        length_axis=tuple(length_axis),
        width_axis=tuple(width_axis),
        thickness_axis=tuple(thickness_axis),
        length=JOINT_LENGTH,
        width=JOINT_WIDTH,
        thickness=STOCK_THICKNESS,
        slot_width=JOINT_SLOT_WIDTH,
        slot_length=JOINT_SLOT_LENGTH,
        slot_offset=JOINT_SLOT_OFFSET,
        corner_radius=CORNER_RADIUS,
        fillet_radius=FILLET_RADIUS,
        material=PLYWOOD_GOLD,
    )


def _axis_dogbone_hole(y_min: float, z_min: float, y_max: float, z_max: float) -> list[tuple[float, float]]:
    return dogbone_slot_points(
        y_min,
        z_min,
        y_max - y_min,
        z_max - z_min,
        FILLET_RADIUS,
        segments=8,
    )


def _rest_depth_axis() -> Vector:
    return Vector((0.0, math.cos(REST_ANGLE), math.sin(REST_ANGLE)))


def _rest_normal_axis() -> Vector:
    return Vector((0.0, -math.sin(REST_ANGLE), math.cos(REST_ANGLE)))


def _rest_source_yz(local_depth: float, local_thickness: float) -> tuple[float, float]:
    y0 = STAND_SPACE + (STAND_DEPTH - REST_DEPTH) / 2.0
    y = y0 + math.cos(REST_ANGLE) * local_depth - math.sin(REST_ANGLE) * local_thickness
    z = REST_LEVEL + math.sin(REST_ANGLE) * local_depth + math.cos(REST_ANGLE) * local_thickness
    return y, z


def _rest_world_point(local_x: float, local_depth: float, local_thickness: float) -> Vector:
    y, z = _rest_source_yz(local_depth, local_thickness)
    return Vector((-DESK_WIDTH / 2.0 + local_x, _world_y(y), z))


def _rest_socket_hole() -> list[tuple[float, float]]:
    left_edge_depth = (REST_DEPTH + JOINT_WIDTH) / 2.0
    right_edge_depth = (REST_DEPTH - JOINT_WIDTH) / 2.0
    d_min = min(left_edge_depth - JOINT_WIDTH, right_edge_depth)
    d_max = max(left_edge_depth, right_edge_depth + JOINT_WIDTH)
    local_points = dogbone_slot_points(
        d_min - JOINT_SLOT_TOLERANCE / 2.0,
        -JOINT_SLOT_TOLERANCE / 2.0,
        d_max - d_min + JOINT_SLOT_TOLERANCE,
        STOCK_THICKNESS + JOINT_SLOT_TOLERANCE,
        FILLET_RADIUS,
        segments=8,
    )
    return [_rest_source_yz(local_depth, local_thickness) for local_depth, local_thickness in local_points]


def _base_side_outline() -> list[tuple[float, float]]:
    """Union-like outline of the base plank and the two tall uprights in Y/Z."""
    y_front = 0.0
    y_back = DESK_DEPTH
    front_a = STAND_SPACE
    front_b = STAND_SPACE + STAND_DEPTH
    rear_a = DESK_DEPTH - STAND_DEPTH
    rear_b = DESK_DEPTH
    z_low = 0.0
    z_base = BASE_HEIGHT
    z_top = TOTAL_STAND_HEIGHT
    return [
        (y_front + CORNER_RADIUS, z_low),
        (y_back - CORNER_RADIUS, z_low),
        (y_back, z_low + CORNER_RADIUS),
        (y_back, z_base - CORNER_RADIUS),
        (y_back - CORNER_RADIUS, z_base),
        (rear_b, z_base),
        (rear_b, z_top - CORNER_RADIUS),
        (rear_b - CORNER_RADIUS, z_top),
        (rear_a + CORNER_RADIUS, z_top),
        (rear_a, z_top - CORNER_RADIUS),
        (rear_a, z_base),
        (front_b, z_base),
        (front_b, z_top - CORNER_RADIUS),
        (front_b - CORNER_RADIUS, z_top),
        (front_a + CORNER_RADIUS, z_top),
        (front_a, z_top - CORNER_RADIUS),
        (front_a, z_base),
        (y_front + CORNER_RADIUS, z_base),
        (y_front, z_base - CORNER_RADIUS),
        (y_front, z_low + CORNER_RADIUS),
    ]


def _panel_socket_holes(side: str) -> list[list[tuple[float, float]]]:
    tol = JOINT_SLOT_TOLERANCE
    holes: list[list[tuple[float, float]]] = []

    top_edges = [
        STAND_SPACE + (STAND_DEPTH - JOINT_WIDTH) / 2.0,
        DESK_DEPTH - (STAND_DEPTH + JOINT_WIDTH) / 2.0,
    ]
    for edge_y in top_edges:
        holes.append(
            _axis_dogbone_hole(
                edge_y - tol / 2.0,
                DESK_HEIGHT - tol / 2.0,
                edge_y + JOINT_WIDTH + tol / 2.0,
                DESK_HEIGHT + STOCK_THICKNESS + tol / 2.0,
            )
        )

    if side == "left":
        lower_cross_y = DESK_DEPTH - STAND_DEPTH / 2.0 + CROSS_CLEARANCE
        upper_cross_y = DESK_DEPTH - STAND_DEPTH / 2.0 - (STOCK_THICKNESS + CROSS_CLEARANCE)
    else:
        lower_cross_y = DESK_DEPTH - STAND_DEPTH / 2.0 - (STOCK_THICKNESS + CROSS_CLEARANCE)
        upper_cross_y = DESK_DEPTH - STAND_DEPTH / 2.0 + CROSS_CLEARANCE
    cross_specs = [
        (lower_cross_y, (CROSS_CUT + JOINT_WIDTH) / 2.0 - JOINT_WIDTH, (CROSS_CUT + JOINT_WIDTH) / 2.0),
        (
            upper_cross_y,
            DESK_HEIGHT - (CROSS_CUT - JOINT_WIDTH) / 2.0 - JOINT_WIDTH,
            DESK_HEIGHT - (CROSS_CUT - JOINT_WIDTH) / 2.0,
        ),
    ]
    for edge_y, z_min, z_max in cross_specs:
        holes.append(
            _axis_dogbone_hole(
                edge_y - tol / 2.0,
                z_min - tol / 2.0,
                edge_y + STOCK_THICKNESS + tol / 2.0,
                z_max + tol / 2.0,
            )
        )

    holes.append(_rest_socket_hole())

    support_y_edge = STAND_SPACE + (STAND_DEPTH + STOCK_THICKNESS) / 2.0
    support_z_min = DESK_HEIGHT - TOPSUPPORT_DEPTH + (TOPSUPPORT_DEPTH - JOINT_WIDTH) / 2.0
    support_z_max = support_z_min + JOINT_WIDTH
    holes.append(
        _axis_dogbone_hole(
            support_y_edge - STOCK_THICKNESS - tol / 2.0,
            support_z_min - tol / 2.0,
            support_y_edge + tol / 2.0,
            support_z_max + tol / 2.0,
        )
    )

    return holes


def build_side_panel(side: str) -> list:
    x0 = _side_panel_x(side)
    objects = [
        add_prism_from_points_with_holes(
            f"{side}_continuous_c_shaped_side_panel",
            _base_side_outline(),
            _panel_socket_holes(side),
            STOCK_THICKNESS,
            PLYWOOD_GOLD,
            _xz_sheet_transform(x0, -DESK_DEPTH / 2.0),
        )
    ]
    x_center = _part_x_center(side)
    objects.append(
        add_edge_band(
            f"{side}_front_base_dark_edge",
            (x_center, -DESK_DEPTH / 2.0 + 1.0, BASE_HEIGHT / 2.0),
            (STOCK_THICKNESS + 0.25, 2.0, BASE_HEIGHT - 18.0),
        )
    )
    objects.append(
        add_edge_band(
            f"{side}_rear_base_dark_edge",
            (x_center, DESK_DEPTH / 2.0 - 1.0, BASE_HEIGHT / 2.0),
            (STOCK_THICKNESS + 0.25, 2.0, BASE_HEIGHT - 18.0),
        )
    )
    return objects


def build_table_top() -> list:
    objects = [
        add_rounded_sheet(
            "rounded_desktop_sheet",
            DESK_WIDTH,
            DESK_DEPTH,
            STOCK_THICKNESS,
            CORNER_RADIUS,
            _top_transform(),
        )
    ]
    tab_specs = [
        ("left_rear", -DESK_WIDTH / 2.0, DESK_DEPTH - (STAND_DEPTH - JOINT_WIDTH) / 2.0, Vector((-1.0, 0.0, 0.0)), -1.0),
        (
            "left_front",
            -DESK_WIDTH / 2.0,
            STAND_SPACE + STAND_DEPTH - (STAND_DEPTH - JOINT_WIDTH) / 2.0,
            Vector((-1.0, 0.0, 0.0)),
            -1.0,
        ),
        ("right_rear", DESK_WIDTH / 2.0, DESK_DEPTH - (STAND_DEPTH + JOINT_WIDTH) / 2.0, Vector((1.0, 0.0, 0.0)), 1.0),
        (
            "right_front",
            DESK_WIDTH / 2.0,
            STAND_SPACE + (STAND_DEPTH - JOINT_WIDTH) / 2.0,
            Vector((1.0, 0.0, 0.0)),
            1.0,
        ),
    ]
    for label, x_edge, source_y_edge, length_axis, width_sign in tab_specs:
        base_center = Vector(
            (
                x_edge,
                _world_y(source_y_edge + width_sign * JOINT_WIDTH / 2.0),
                DESK_HEIGHT + STOCK_THICKNESS / 2.0,
            )
        )
        objects.append(
            _joint_tab(
                f"desktop_visible_male_tab_{label}",
                base_center,
                length_axis,
                Vector((0.0, width_sign, 0.0)),
                Vector((0.0, 0.0, 1.0)),
            )
        )
    objects.append(
        add_edge_band(
            "desktop_front_laminated_edge",
            (0.0, -DESK_DEPTH / 2.0 + 1.2, DESK_HEIGHT + STOCK_THICKNESS / 2.0),
            (DESK_WIDTH - 22.0, 2.4, STOCK_THICKNESS),
        )
    )
    return objects


def build_foot_rest() -> list:
    y0 = _world_y(STAND_SPACE + (STAND_DEPTH - REST_DEPTH) / 2.0)
    transform = _xy_sheet_transform(-DESK_WIDTH / 2.0, y0, REST_LEVEL, REST_ANGLE)
    objects = [
        add_rounded_sheet(
            "angled_front_foot_rest_sheet",
            DESK_WIDTH,
            REST_DEPTH,
            STOCK_THICKNESS,
            CORNER_RADIUS * 0.6,
            transform,
        )
    ]
    depth_axis = _rest_depth_axis()
    normal_axis = _rest_normal_axis()
    tab_specs = [
        ("left", 0.0, (REST_DEPTH + JOINT_WIDTH) / 2.0, Vector((-1.0, 0.0, 0.0)), -1.0),
        ("right", DESK_WIDTH, (REST_DEPTH - JOINT_WIDTH) / 2.0, Vector((1.0, 0.0, 0.0)), 1.0),
    ]
    for label, local_x, source_depth_edge, length_axis, width_sign in tab_specs:
        base_center = _rest_world_point(
            local_x,
            source_depth_edge + width_sign * JOINT_WIDTH / 2.0,
            STOCK_THICKNESS / 2.0,
        )
        objects.append(
            _joint_tab(
                f"foot_rest_visible_male_tab_{label}",
                base_center,
                length_axis,
                depth_axis * width_sign,
                normal_axis,
            )
        )
    return objects


def build_top_support() -> list:
    support_source_y_edge = STAND_SPACE + (STAND_DEPTH + STOCK_THICKNESS) / 2.0
    y = _world_y(support_source_y_edge - STOCK_THICKNESS / 2.0)
    z_top = DESK_HEIGHT - TOPSUPPORT_DEPTH
    objects = [
        add_box(
            "vertical_top_support_strut",
            (DESK_WIDTH, STOCK_THICKNESS, TOPSUPPORT_DEPTH),
            (0.0, y, z_top + TOPSUPPORT_DEPTH / 2.0),
            PLYWOOD_GOLD,
        )
    ]
    tab_specs = [
        ("left", -DESK_WIDTH / 2.0, (TOPSUPPORT_DEPTH + JOINT_WIDTH) / 2.0, Vector((-1.0, 0.0, 0.0)), -1.0),
        ("right", DESK_WIDTH / 2.0, (TOPSUPPORT_DEPTH - JOINT_WIDTH) / 2.0, Vector((1.0, 0.0, 0.0)), 1.0),
    ]
    for label, x_edge, source_depth_edge, length_axis, width_sign in tab_specs:
        base_center = Vector(
            (
                x_edge,
                y,
                z_top + source_depth_edge + width_sign * JOINT_WIDTH / 2.0,
            )
        )
        objects.append(
            _joint_tab(
                f"top_support_visible_male_tab_{label}",
                base_center,
                length_axis,
                Vector((0.0, 0.0, width_sign)),
                Vector((0.0, -1.0, 0.0)),
            )
        )
    return objects


def build_cross_brace() -> list:
    y = _world_y(DESK_DEPTH - STAND_DEPTH / 2.0 + STOCK_THICKNESS / 2.0)
    brace_points = [
        (0.0, 0.0),
        (DESK_WIDTH, DESK_HEIGHT - CROSS_CUT),
        (DESK_WIDTH, DESK_HEIGHT),
        (0.0, CROSS_CUT),
    ]
    brace_transform = Matrix.Translation((-DESK_WIDTH / 2.0, y, 0.0)) @ Matrix.Rotation(math.radians(90.0), 4, "X")
    objects = [
        add_prism_from_points(
            "single_rear_diagonal_cross_brace",
            brace_points,
            STOCK_THICKNESS,
            PLYWOOD_GOLD,
            brace_transform,
        )
    ]
    objects.append(
        _joint_tab(
            "cross_brace_left_male_tab",
            Vector((-DESK_WIDTH / 2.0, y, CROSS_CUT / 2.0)),
            Vector((-1.0, 0.0, 0.0)),
            Vector((0.0, 0.0, 1.0)),
            Vector((0.0, 1.0, 0.0)),
        )
    )
    objects.append(
        _joint_tab(
            "cross_brace_right_male_tab",
            Vector((DESK_WIDTH / 2.0, y, DESK_HEIGHT - CROSS_CUT / 2.0)),
            Vector((1.0, 0.0, 0.0)),
            Vector((0.0, 0.0, -1.0)),
            Vector((0.0, 1.0, 0.0)),
        )
    )
    return objects


def build_key_wedges() -> list:
    objects = []
    wedge_points = [(0.0, 0.0), (50.0, 0.0), (0.0, STOCK_THICKNESS * 1.2)]
    locations = [
        ("front_left_lower", -DESK_WIDTH / 2.0 - 54.0, _world_y(STAND_SPACE + 24.0), REST_LEVEL - 18.0, REST_ANGLE),
        ("front_right_lower", DESK_WIDTH / 2.0 + 6.0, _world_y(STAND_SPACE + 24.0), REST_LEVEL - 18.0, REST_ANGLE),
        ("rear_left_cross", -DESK_WIDTH / 2.0 - 48.0, _world_y(DESK_DEPTH - 104.0), 138.0, 0.0),
        ("rear_right_cross", DESK_WIDTH / 2.0 + 2.0, _world_y(DESK_DEPTH - 104.0), DESK_HEIGHT - 120.0, 0.0),
    ]
    for label, x, y, z, rot_x in locations:
        transform = (
            Matrix.Translation((x, y, z))
            @ Matrix.Rotation(math.radians(90.0), 4, "X")
            @ Matrix.Rotation(rot_x, 4, "Z")
        )
        objects.append(
            add_prism_from_points(
                f"triangular_lock_key_{label}",
                wedge_points,
                STOCK_THICKNESS,
                PLYWOOD_EDGE,
                transform,
            )
        )
    return objects


def build_model() -> list:
    objects = []
    objects.extend(build_side_panel("left"))
    objects.extend(build_side_panel("right"))
    objects.extend(build_table_top())
    objects.extend(build_cross_brace())
    objects.extend(build_foot_rest())
    objects.extend(build_top_support())
    group_objects("standing_desk_model", objects)
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

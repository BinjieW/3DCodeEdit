from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable geometry builders for the Opendesk bedside table model."""





RELIEF_NONE = 0
RELIEF_LEAD_IN = 1
RELIEF_LEAD_OUT = 2

FN = 128
WIDTH = 400.0
DEPTH = 500.0
HEIGHT = 600.0
MATERIAL_THICKNESS = 18.0
TONGUE_WIDTH = WIDTH * 2.0 / 3.0
SHELF_HEIGHT = HEIGHT * 2.0 / 3.0
RELIEF_RADIUS = 3.5
DRILL_RADIUS = 3.5
SLOP = 0.03

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]
MarkedPoint = tuple[float, float, int]
Transform3D = Callable[[float, float, float], Point3]


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_yellow_material() -> bpy.types.Material:
    """Match OpenSCAD's default preview yellow closely enough for comparison."""
    material = bpy.data.materials.get("openscad_preview_yellow")
    if material is None:
        material = bpy.data.materials.new("openscad_preview_yellow")
    material.diffuse_color = (1.0, 0.76, 0.02, 1.0)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1.0, 0.76, 0.02, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.62
    return material


def _distance_2d(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clean_contour(points: Iterable[Point2]) -> list[Point2]:
    cleaned: list[Point2] = []
    for x_value, y_value in points:
        point = (float(x_value), float(y_value))
        if not cleaned or _distance_2d(cleaned[-1], point) > 1e-5:
            cleaned.append(point)
    if len(cleaned) > 1 and _distance_2d(cleaned[0], cleaned[-1]) <= 1e-5:
        cleaned.pop()
    return cleaned


def _signed_area(points: Sequence[Point2]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area * 0.5


def _oriented(points: Sequence[Point2], *, ccw: bool) -> list[Point2]:
    result = list(points)
    if (_signed_area(result) > 0.0) != ccw:
        result.reverse()
    return result


def _matrix_vec(matrix: Sequence[Sequence[float]], point: Point2) -> Point2:
    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1],
        matrix[1][0] * point[0] + matrix[1][1] * point[1],
    )


def arc_points(
    centre: Point2 = (0.0, 0.0),
    *,
    start: float = 0.0,
    angle: float = 180.0,
    radius: float = 5.0,
) -> list[Point2]:
    segments = max(4, int(math.ceil(abs(angle) / 360.0 * FN)))
    return [
        (
            centre[0] + radius * math.cos(math.radians(start + angle * index / segments)),
            centre[1] + radius * math.sin(math.radians(start + angle * index / segments)),
        )
        for index in range(segments + 1)
    ]


def circle_points(centre: Point2 = (0.0, 0.0), *, radius: float = 5.0) -> list[Point2]:
    return [
        (
            centre[0] + radius * math.cos(2.0 * math.pi * index / FN),
            centre[1] + radius * math.sin(2.0 * math.pi * index / FN),
        )
        for index in range(FN)
    ]


def relief_points(corner: Point2 = (0.0, 0.0), *, radius: float = RELIEF_RADIUS) -> list[Point2]:
    line_start = (corner[0] - radius, corner[1])
    notch_centre = (line_start[0], line_start[1] - radius)
    fillet_centre = (line_start[0], line_start[1] - 3.0 * radius)
    return (
        arc_points(fillet_centre, start=0.0, angle=90.0, radius=radius)
        + arc_points(notch_centre, start=270.0, angle=-180.0, radius=radius)
        + [line_start, corner]
    )


def _corner_points(points: Sequence[MarkedPoint], index: int) -> list[Point2]:
    last = len(points) - 1
    previous = points[last] if index == 0 else points[index - 1]
    current = points[index]
    next_point = points[0] if index == last else points[index + 1]
    return [
        (previous[0], previous[1]),
        (current[0], current[1]),
        (next_point[0], next_point[1]),
    ]


def relief_lead_in_points(
    corner: Sequence[Point2],
    *,
    radius: float = RELIEF_RADIUS,
) -> list[Point2]:
    rot_matrix = ((1.0, 0.0), (0.0, 1.0))
    if abs(corner[0][0] - corner[1][0]) > 1e-7:
        rot_matrix = ((0.0, 1.0), (1.0, 0.0))

    before = _matrix_vec(rot_matrix, corner[0])
    at = _matrix_vec(rot_matrix, corner[1])
    after = _matrix_vec(rot_matrix, corner[2])

    mirror_x = 1.0 if at[0] < after[0] else -1.0
    mirror_y = 1.0 if at[1] > before[1] else -1.0
    mirror_matrix = ((mirror_x, 0.0), (0.0, mirror_y))

    generated: list[Point2] = []
    for point in relief_points((0.0, 0.0), radius=radius):
        mirrored = _matrix_vec(mirror_matrix, point)
        rotated = _matrix_vec(rot_matrix, mirrored)
        generated.append((rotated[0] + corner[1][0], rotated[1] + corner[1][1]))
    return generated


def relief_lead_out_points(
    corner: Sequence[Point2],
    *,
    radius: float = RELIEF_RADIUS,
) -> list[Point2]:
    return list(reversed(relief_lead_in_points(list(reversed(corner)), radius=radius)))


def generate_outline(points: Sequence[MarkedPoint]) -> list[Point2]:
    outline: list[Point2] = []
    for index, marked in enumerate(points):
        relief = marked[2]
        if relief == RELIEF_NONE:
            outline.append((marked[0], marked[1]))
        elif relief == RELIEF_LEAD_IN:
            outline.extend(relief_lead_in_points(_corner_points(points, index)))
        else:
            outline.extend(relief_lead_out_points(_corner_points(points, index)))
    return _clean_contour(outline)


def drill_holes(centres: Iterable[Point2]) -> list[list[Point2]]:
    return [circle_points(centre, radius=DRILL_RADIUS) for centre in centres]


def mirror_x_contour(points: Sequence[Point2]) -> list[Point2]:
    return [(-point[0], point[1]) for point in points]


def mirror_x_holes(holes: Sequence[Sequence[Point2]]) -> list[list[Point2]]:
    return [mirror_x_contour(hole) for hole in holes]


def scad_transform(
    translation: Point3 = (0.0, 0.0, 0.0),
    rotation_degrees: Point3 = (0.0, 0.0, 0.0),
) -> Transform3D:
    """Apply OpenSCAD-style rotate([x, y, z]) then translate()."""
    rx, ry, rz = (math.radians(value) for value in rotation_degrees)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    def apply(x_value: float, y_value: float, z_value: float) -> Point3:
        x1 = x_value
        y1 = y_value * cx - z_value * sx
        z1 = y_value * sx + z_value * cx

        x2 = x1 * cy + z1 * sy
        y2 = y1
        z2 = -x1 * sy + z1 * cy

        x3 = x2 * cz - y2 * sz
        y3 = x2 * sz + y2 * cz
        z3 = z2
        return (
            x3 + translation[0],
            y3 + translation[1],
            z3 + translation[2],
        )

    return apply


def add_extruded_panel(
    name: str,
    outline: Sequence[Point2],
    *,
    holes: Sequence[Sequence[Point2]] = (),
    thickness: float = MATERIAL_THICKNESS,
    transform: Transform3D | None = None,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Create a flat CNC panel from a 2D outline plus through-drill contours."""
    transform = transform or scad_transform()
    outer = _oriented(_clean_contour(outline), ccw=True)
    if len(outer) < 3:
        raise ValueError(f"{name} outline has fewer than three usable points")

    contours: list[list[Point2]] = [outer]
    for hole in holes:
        cleaned_hole = _clean_contour(hole)
        if len(cleaned_hole) >= 3:
            contours.append(_oriented(cleaned_hole, ccw=False))

    flat_points = [point for contour in contours for point in contour]
    contour_vectors = [[Vector((point[0], point[1], 0.0)) for point in contour] for contour in contours]
    triangles = tessellate_polygon(contour_vectors)

    vertices = [transform(point[0], point[1], 0.0) for point in flat_points]
    vertices.extend(transform(point[0], point[1], thickness) for point in flat_points)
    vertex_count = len(flat_points)

    faces: list[tuple[int, ...]] = []
    for tri in triangles:
        if len(set(tri)) == 3:
            faces.append(tuple(reversed(tri)))
            faces.append(tuple(index + vertex_count for index in tri))

    contour_offset = 0
    for contour in contours:
        count = len(contour)
        for index in range(count):
            next_index = (index + 1) % count
            faces.append(
                (
                    contour_offset + index,
                    contour_offset + next_index,
                    vertex_count + contour_offset + next_index,
                    vertex_count + contour_offset + index,
                )
            )
        contour_offset += count

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material is not None:
        mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    return obj


def top_outline() -> list[Point2]:
    w, d, t, slop = WIDTH, DEPTH, MATERIAL_THICKNESS, SLOP
    return generate_outline(
        [
            (0.0, 0.0, RELIEF_NONE),
            (w, 0.0, RELIEF_NONE),
            (w, d / 3.0 - slop, RELIEF_NONE),
            (w - t - slop, d / 3.0 - slop, RELIEF_LEAD_OUT),
            (w - t - slop, 2.0 * d / 3.0 - slop, RELIEF_LEAD_IN),
            (w, 2.0 * d / 3.0 + slop, RELIEF_NONE),
            (w, d, RELIEF_NONE),
            (2.0 * w / 3.0 + slop, d, RELIEF_NONE),
            (2.0 * w / 3.0 + slop, d - t - slop, RELIEF_LEAD_OUT),
            (w / 3.0 - slop, d - t - slop, RELIEF_LEAD_IN),
            (w / 3.0 - slop, d, RELIEF_NONE),
            (0.0, d, RELIEF_NONE),
            (0.0, 2.0 * d / 3.0 + slop, RELIEF_NONE),
            (t + slop, 2.0 * d / 3.0 + slop, RELIEF_LEAD_OUT),
            (t + slop, d / 3.0 - slop, RELIEF_LEAD_IN),
            (0.0, d / 3.0 - slop, RELIEF_NONE),
        ]
    )


def top_holes() -> list[list[Point2]]:
    w, d, t = WIDTH, DEPTH, MATERIAL_THICKNESS
    return drill_holes(
        [
            (w - t / 2.0, d / 6.0),
            (w - t / 2.0, 5.0 * d / 6.0),
            (t / 2.0, d / 6.0),
            (t / 2.0, 5.0 * d / 6.0),
            (w - w / 6.0, d - t / 2.0),
            (w / 6.0, d - t / 2.0),
        ]
    )


def shelf_outline() -> list[Point2]:
    w, d, t, slop = WIDTH, DEPTH, MATERIAL_THICKNESS, SLOP
    x_value = (w - TONGUE_WIDTH) / 2.0
    y_value = (d - (d / w) * TONGUE_WIDTH) / 2.0
    return generate_outline(
        [
            (t + slop, t + slop, RELIEF_NONE),
            (x_value, t + slop, RELIEF_LEAD_IN),
            (x_value, 0.0, RELIEF_NONE),
            (w - x_value, 0.0, RELIEF_NONE),
            (w - x_value, t + slop, RELIEF_LEAD_OUT),
            (w - t - slop, t + slop, RELIEF_NONE),
            (w - t - slop, y_value, RELIEF_LEAD_IN),
            (w, y_value, RELIEF_NONE),
            (w, d - y_value, RELIEF_NONE),
            (w - t - slop, d - y_value, RELIEF_LEAD_OUT),
            (w - t - slop, d - t - slop, RELIEF_NONE),
            (w - x_value, d - t - slop, RELIEF_LEAD_IN),
            (w - x_value, d, RELIEF_NONE),
            (x_value, d, RELIEF_NONE),
            (x_value, d - t - slop, RELIEF_LEAD_OUT),
            (t + slop, d - t - slop, RELIEF_NONE),
            (t + slop, d - y_value, RELIEF_LEAD_IN),
            (0.0, d - y_value, RELIEF_NONE),
            (0.0, y_value, RELIEF_NONE),
            (t + slop, y_value, RELIEF_LEAD_OUT),
        ]
    )


def side_outline() -> list[Point2]:
    d, h, t, slop = DEPTH, HEIGHT, MATERIAL_THICKNESS, SLOP
    x_value = (d - (d / WIDTH) * TONGUE_WIDTH) / 2.0
    return generate_outline(
        [
            (0.0, 0.0, RELIEF_NONE),
            (2.0 * t - slop, 0.0, RELIEF_NONE),
            (2.0 * t - slop, t, RELIEF_LEAD_OUT),
            (4.0 * t, t, RELIEF_NONE),
            (4.0 * t + d / 10.0, SHELF_HEIGHT - t - slop, RELIEF_NONE),
            (x_value - slop, SHELF_HEIGHT - t - slop, RELIEF_LEAD_IN),
            (x_value - slop, SHELF_HEIGHT, RELIEF_LEAD_OUT),
            (d - x_value + slop, SHELF_HEIGHT, RELIEF_LEAD_IN),
            (d - x_value + slop, SHELF_HEIGHT - t - slop, RELIEF_LEAD_OUT),
            (d - (4.0 * t + d / 10.0), SHELF_HEIGHT - t - slop, RELIEF_NONE),
            (d - 4.0 * t, t, RELIEF_NONE),
            (d - 2.0 * t + slop, t, RELIEF_LEAD_IN),
            (d - 2.0 * t + slop, 0.0, RELIEF_NONE),
            (d - t - slop, 0.0, RELIEF_NONE),
            (d - t - slop, SHELF_HEIGHT / 2.0 + slop, RELIEF_LEAD_IN),
            (d, SHELF_HEIGHT / 2.0 + slop, RELIEF_NONE),
            (d, h - t, RELIEF_NONE),
            (2.0 * d / 3.0, h - t, RELIEF_LEAD_IN),
            (2.0 * d / 3.0, h, RELIEF_NONE),
            (d / 3.0, h, RELIEF_NONE),
            (d / 3.0, h - t, RELIEF_LEAD_OUT),
            (0.0, h - t, RELIEF_NONE),
            (0.0, h - (h - SHELF_HEIGHT) / 2.0 + slop, RELIEF_NONE),
            (t + slop, h - (h - SHELF_HEIGHT) / 2.0 + slop, RELIEF_LEAD_OUT),
            (t + slop, SHELF_HEIGHT / 2.0 - slop, RELIEF_LEAD_IN),
            (0.0, SHELF_HEIGHT / 2.0 - slop, RELIEF_NONE),
        ]
    )


def side_holes() -> list[list[Point2]]:
    d, h, t = DEPTH, HEIGHT, MATERIAL_THICKNESS
    x_value = (d - (d / WIDTH) * TONGUE_WIDTH) / 2.0
    return drill_holes(
        [
            (t / 2.0, SHELF_HEIGHT / 4.0),
            (x_value / 2.0, SHELF_HEIGHT - t / 2.0),
            (t / 2.0, h - (h - SHELF_HEIGHT) / 4.0),
            (d / 2.0, h - t / 2.0),
            (d - x_value / 2.0, SHELF_HEIGHT - t / 2.0),
            (d - t / 2.0, h / 2.0 - 1.5 * t),
        ]
    )


def back_outline() -> list[Point2]:
    w, h, t, slop = WIDTH, HEIGHT, MATERIAL_THICKNESS, SLOP
    x_value = (w - TONGUE_WIDTH) / 2.0
    arc_radius = TONGUE_WIDTH / 3.0
    arc_y_offset = TONGUE_WIDTH * 0.13
    theta = math.degrees(math.acos(arc_y_offset / arc_radius))
    start_t = 90.0 - theta
    angle = theta * 2.0
    arch = list(
        reversed(
            [
                (point[0], point[1], RELIEF_NONE)
                for point in arc_points(
                    (w / 2.0, SHELF_HEIGHT - arc_y_offset),
                    start=start_t,
                    angle=angle,
                    radius=arc_radius,
                )
            ]
        )
    )
    return generate_outline(
        [
            (t + slop, 0.0, RELIEF_NONE),
            (2.0 * t - slop, 0.0, RELIEF_NONE),
            (2.0 * t - slop, t, RELIEF_LEAD_OUT),
            (4.0 * t, t, RELIEF_NONE),
            (4.0 * t + w / 10.0, SHELF_HEIGHT - t - slop, RELIEF_NONE),
            (x_value - slop, SHELF_HEIGHT - t - slop, RELIEF_LEAD_IN),
            (x_value - slop, SHELF_HEIGHT, RELIEF_LEAD_OUT),
            (w / 2.0 - arc_radius * math.sin(math.radians(theta)), SHELF_HEIGHT, RELIEF_NONE),
            *arch,
            (w / 2.0 + arc_radius * math.sin(math.radians(theta)), SHELF_HEIGHT, RELIEF_NONE),
            (w - x_value + slop, SHELF_HEIGHT, RELIEF_LEAD_IN),
            (w - x_value + slop, SHELF_HEIGHT - t - slop, RELIEF_LEAD_OUT),
            (w - (4.0 * t + w / 10.0), SHELF_HEIGHT - t - slop, RELIEF_NONE),
            (w - 4.0 * t, t, RELIEF_NONE),
            (w - 2.0 * t + slop, t, RELIEF_LEAD_IN),
            (w - 2.0 * t + slop, 0.0, RELIEF_NONE),
            (w - t - slop, 0.0, RELIEF_NONE),
            (w - t - slop, SHELF_HEIGHT / 2.0, RELIEF_LEAD_IN),
            (w, SHELF_HEIGHT / 2.0, RELIEF_NONE),
            (w, h - (h - SHELF_HEIGHT) / 2.0, RELIEF_NONE),
            (w - t - slop, h - (h - SHELF_HEIGHT) / 2.0, RELIEF_LEAD_OUT),
            (w - t - slop, h - t, RELIEF_NONE),
            (2.0 * w / 3.0, h - t, RELIEF_LEAD_IN),
            (2.0 * w / 3.0, h, RELIEF_NONE),
            (w / 3.0, h, RELIEF_NONE),
            (w / 3.0, h - t, RELIEF_LEAD_OUT),
            (t + slop, h - t, RELIEF_NONE),
            (t + slop, h - (h - SHELF_HEIGHT) / 2.0, RELIEF_LEAD_IN),
            (0.0, h - (h - SHELF_HEIGHT) / 2.0, RELIEF_NONE),
            (0.0, SHELF_HEIGHT / 2.0, RELIEF_NONE),
            (t + slop, SHELF_HEIGHT / 2.0, RELIEF_LEAD_OUT),
            (t + slop, 0.0, RELIEF_NONE),
        ]
    )


def back_holes() -> list[list[Point2]]:
    w, h, t = WIDTH, HEIGHT, MATERIAL_THICKNESS
    x_value = (w - TONGUE_WIDTH) / 2.0
    return drill_holes(
        [
            (w - t / 2.0, h / 2.0 - 1.5 * t),
            (w - x_value / 2.0, SHELF_HEIGHT - t / 2.0),
            (w / 2.0, h - t / 2.0),
            (x_value / 2.0, SHELF_HEIGHT - t / 2.0),
            (t / 2.0, h / 2.0 - 1.5 * t),
        ]
    )


def front_outline() -> list[Point2]:
    w, t, slop = WIDTH, MATERIAL_THICKNESS, SLOP
    x_value = (w - TONGUE_WIDTH) / 2.0
    return generate_outline(
        [
            (0.0, 0.0, RELIEF_NONE),
            (2.0 * t - slop, 0.0, RELIEF_NONE),
            (2.0 * t - slop, t, RELIEF_LEAD_OUT),
            (4.0 * t, t, RELIEF_NONE),
            (4.0 * t + w / 10.0, SHELF_HEIGHT - t - slop, RELIEF_NONE),
            (x_value - slop, SHELF_HEIGHT - t, RELIEF_LEAD_IN),
            (x_value - slop, SHELF_HEIGHT, RELIEF_NONE),
            (t + slop, SHELF_HEIGHT, RELIEF_NONE),
            (t + slop, SHELF_HEIGHT / 2.0 - slop, RELIEF_LEAD_IN),
            (0.0, SHELF_HEIGHT / 2.0, RELIEF_NONE),
        ]
    )


def front_holes() -> list[list[Point2]]:
    t = MATERIAL_THICKNESS
    x_value = (WIDTH - TONGUE_WIDTH) / 2.0
    return drill_holes(
        [
            (t + (x_value - t) / 2.0, SHELF_HEIGHT - t / 2.0),
            (t / 2.0, SHELF_HEIGHT / 4.0),
        ]
    )


def foot_outline() -> list[Point2]:
    t, slop = MATERIAL_THICKNESS, SLOP
    return generate_outline(
        [
            (2.0 * t, 0.0, RELIEF_NONE),
            (4.0 * t, 0.0, RELIEF_NONE),
            (4.0 * t, t, RELIEF_NONE),
            (t, 4.0 * t, RELIEF_NONE),
            (0.0, 4.0 * t, RELIEF_NONE),
            (0.0, 2.0 * t, RELIEF_NONE),
            (t + slop, 2.0 * t, RELIEF_LEAD_OUT),
            (t + slop, t + slop, RELIEF_NONE),
            (2.0 * t, t + slop, RELIEF_LEAD_IN),
        ]
    )


def foot_holes() -> list[list[Point2]]:
    t = MATERIAL_THICKNESS
    return drill_holes(
        [
            (3.5 * t, t / 2.0),
            (t / 2.0, 3.5 * t),
        ]
    )


def add_top_panel(material: bpy.types.Material) -> bpy.types.Object:
    return add_extruded_panel(
        "top_panel",
        top_outline(),
        holes=top_holes(),
        transform=scad_transform((0.0, 0.0, HEIGHT - MATERIAL_THICKNESS)),
        material=material,
    )


def add_shelf_panel(material: bpy.types.Material) -> bpy.types.Object:
    return add_extruded_panel(
        "middle_shelf_panel",
        shelf_outline(),
        transform=scad_transform((0.0, 0.0, SHELF_HEIGHT - MATERIAL_THICKNESS)),
        material=material,
    )


def add_front_left_panel(material: bpy.types.Material) -> bpy.types.Object:
    return add_extruded_panel(
        "front_left_panel",
        front_outline(),
        holes=front_holes(),
        transform=scad_transform((0.0, MATERIAL_THICKNESS, 0.0), (90.0, 0.0, 0.0)),
        material=material,
    )


def add_front_right_panel(material: bpy.types.Material) -> bpy.types.Object:
    return add_extruded_panel(
        "front_right_panel",
        mirror_x_contour(front_outline()),
        holes=mirror_x_holes(front_holes()),
        transform=scad_transform((WIDTH, MATERIAL_THICKNESS, 0.0), (90.0, 0.0, 0.0)),
        material=material,
    )


def add_left_side_panel(material: bpy.types.Material) -> bpy.types.Object:
    return add_extruded_panel(
        "left_side_panel",
        side_outline(),
        holes=side_holes(),
        transform=scad_transform((MATERIAL_THICKNESS, DEPTH, 0.0), (90.0, 0.0, -90.0)),
        material=material,
    )


def add_right_side_panel(material: bpy.types.Material) -> bpy.types.Object:
    return add_extruded_panel(
        "right_side_panel",
        mirror_x_contour(side_outline()),
        holes=mirror_x_holes(side_holes()),
        transform=scad_transform((WIDTH - MATERIAL_THICKNESS, DEPTH, 0.0), (90.0, 0.0, 90.0)),
        material=material,
    )


def add_back_panel(material: bpy.types.Material) -> bpy.types.Object:
    return add_extruded_panel(
        "back_panel",
        back_outline(),
        holes=back_holes(),
        transform=scad_transform((0.0, DEPTH, 0.0), (90.0, 0.0, 0.0)),
        material=material,
    )


def add_foot_panel(
    name: str,
    translation: Point3,
    rotation_z_degrees: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    return add_extruded_panel(
        name,
        foot_outline(),
        holes=foot_holes(),
        transform=scad_transform(translation, (0.0, 0.0, rotation_z_degrees)),
        material=material,
    )


# ============================================================
# Procedural assembly
# ============================================================

"""High-level assembly for the current Opendesk bedside table iteration."""




def build_model() -> list:
    """Compose the ten flat CNC panels in the same order as the source SCAD."""
    material = make_yellow_material()
    objects = [
        add_front_left_panel(material),
        add_front_right_panel(material),
        add_shelf_panel(material),
        add_left_side_panel(material),
        add_right_side_panel(material),
        add_back_panel(material),
        add_top_panel(material),
        add_foot_panel("front_left_foot", (0.0, 0.0, 0.0), 0.0, material),
        add_foot_panel("front_right_foot", (WIDTH, 0.0, 0.0), 90.0, material),
        add_foot_panel("rear_right_foot", (WIDTH, DEPTH, 0.0), 180.0, material),
        add_foot_panel("rear_left_foot", (0.0, DEPTH, 0.0), 270.0, material),
    ]
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

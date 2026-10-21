from __future__ import annotations
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the common truss model."""





SOURCE_GOLD = (0.84, 0.68, 0.04, 1.0)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(
    name: str,
    color: tuple[float, float, float, float] = SOURCE_GOLD,
    roughness: float = 0.55,
) -> bpy.types.Material:
    """Create or reuse a simple diffuse material."""
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        material.use_nodes = True

    material.diffuse_color = color
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = roughness
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def signed_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area * 0.5


def line_segment_2d_polygon(
    p0: tuple[float, float],
    p1: tuple[float, float],
    width: float,
    y_offset: float = 0.0,
) -> list[tuple[float, float]]:
    """Match the OpenSCAD Line_Seg_2d polygon for a constant-width 2D bar."""
    half_width = width / 2.0
    x0, y0 = p0
    x1, y1 = p1

    if abs(y0 - y1) < 1e-9:
        points = [
            (x0, y0 - half_width + y_offset),
            (x1, y1 - half_width + y_offset),
            (x1, y1 + half_width + y_offset),
            (x0, y0 + half_width + y_offset),
        ]
    else:
        slope = (x0 - x1) / (y1 - y0)
        normal_length = math.sqrt(1.0 + slope * slope)
        sx = half_width / normal_length
        sy = half_width * slope / normal_length
        points = [
            (x0 + sx, y0 + sy + y_offset),
            (x1 + sx, y1 + sy + y_offset),
            (x1 - sx, y1 - sy + y_offset),
            (x0 - sx, y0 - sy + y_offset),
        ]

    if signed_area(points) < 0.0:
        points.reverse()
    return points


def clip_polygon_to_rect(
    points: list[tuple[float, float]],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> list[tuple[float, float]]:
    """Clip a convex polygon to an axis-aligned rectangle."""

    def clip_edge(
        input_points: list[tuple[float, float]],
        inside,
        intersect,
    ) -> list[tuple[float, float]]:
        if not input_points:
            return []
        output: list[tuple[float, float]] = []
        previous = input_points[-1]
        previous_inside = inside(previous)
        for current in input_points:
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

    def ix_at_x(x_boundary: float):
        def intersect(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
            dx = b[0] - a[0]
            if abs(dx) < 1e-12:
                return (x_boundary, a[1])
            t = (x_boundary - a[0]) / dx
            return (x_boundary, a[1] + t * (b[1] - a[1]))

        return intersect

    def ix_at_y(y_boundary: float):
        def intersect(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
            dy = b[1] - a[1]
            if abs(dy) < 1e-12:
                return (a[0], y_boundary)
            t = (y_boundary - a[1]) / dy
            return (a[0] + t * (b[0] - a[0]), y_boundary)

        return intersect

    clipped = points
    clipped = clip_edge(clipped, lambda p: p[0] >= x_min, ix_at_x(x_min))
    clipped = clip_edge(clipped, lambda p: p[0] <= x_max, ix_at_x(x_max))
    clipped = clip_edge(clipped, lambda p: p[1] >= y_min, ix_at_y(y_min))
    clipped = clip_edge(clipped, lambda p: p[1] <= y_max, ix_at_y(y_max))
    return clipped


def add_extruded_polygon(
    name: str,
    points: list[tuple[float, float]],
    z_min: float,
    z_max: float,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Create a flat-sided prism by extruding a 2D polygon along the source Z axis."""
    if len(points) < 3:
        raise ValueError("An extruded polygon needs at least three points")

    vertices = [(x, y, z_min) for x, y in points] + [(x, y, z_max) for x, y in points]
    count = len(points)
    faces: list[tuple[int, ...]] = [
        tuple(reversed(range(count))),
        tuple(range(count, count * 2)),
    ]
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, next_index + count, index + count))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    if material is not None:
        assign_material(obj, material)
    return obj


def add_line_segment_prism(
    name: str,
    p0: tuple[float, float],
    p1: tuple[float, float],
    width: float,
    z_min: float,
    z_max: float,
    y_offset: float = 0.0,
    clip_rect: tuple[float, float, float, float] | None = None,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Build the source truss bar primitive: Line_Seg_2d followed by linear_extrude."""
    points = line_segment_2d_polygon(p0, p1, width, y_offset=y_offset)
    if clip_rect is not None:
        points = clip_polygon_to_rect(points, *clip_rect)
    return add_extruded_polygon(
        name=name,
        points=points,
        z_min=z_min,
        z_max=z_max,
        material=material,
    )


# ============================================================
# Procedural assembly
# ============================================================

"""Readable assembly logic for the common Pratt truss model."""



TR_HEIGHT = 36.0
TR_WIDTH = 16.0
TR_LENGTH = 170.0
TR_SEGS = 8
TR_CAP_WIDTH = 16.0
TR_CAP_THICKNESS = 3.0
TR_SUPPORT_WIDTH = 3.0

HALF_SUPPORT = TR_SUPPORT_WIDTH / 2.0
WEB_HEIGHT = TR_HEIGHT - TR_CAP_THICKNESS
DIAGONAL_X_SPAN = (TR_LENGTH - HALF_SUPPORT) / TR_SEGS
POST_STEP = (TR_LENGTH - TR_SUPPORT_WIDTH) / TR_SEGS
CENTER_SWITCH_X = (TR_LENGTH - TR_SUPPORT_WIDTH) / 2.0

SOURCE_Y_OFFSET = TR_CAP_THICKNESS / 2.0
CAP_Z_MIN = 0.0
CAP_Z_MAX = TR_CAP_WIDTH
WEB_Z_MIN = (TR_CAP_WIDTH - TR_SUPPORT_WIDTH) / 2.0
WEB_Z_MAX = WEB_Z_MIN + TR_SUPPORT_WIDTH
TRUSS_CLIP_RECT = (0.0, 0.0, TR_LENGTH, TR_HEIGHT)


def source_loop_values(start: float, step: float, stop: float) -> list[float]:
    """Inclusive OpenSCAD-style range values, robust against small float drift."""
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(current)
        current += step
    return values


def add_cap_chords(material) -> list:
    """Top and bottom long cap plates extruded across the full truss depth."""
    return [
        add_line_segment_prism(
            name="bottom_full_depth_cap",
            p0=(TR_LENGTH, 0.0),
            p1=(0.0, 0.0),
            width=TR_CAP_THICKNESS,
            z_min=CAP_Z_MIN,
            z_max=CAP_Z_MAX,
            y_offset=SOURCE_Y_OFFSET,
            clip_rect=TRUSS_CLIP_RECT,
            material=material,
        ),
        add_line_segment_prism(
            name="top_full_depth_cap",
            p0=(0.0, WEB_HEIGHT),
            p1=(TR_LENGTH, WEB_HEIGHT),
            width=TR_CAP_THICKNESS,
            z_min=CAP_Z_MIN,
            z_max=CAP_Z_MAX,
            y_offset=SOURCE_Y_OFFSET,
            clip_rect=TRUSS_CLIP_RECT,
            material=material,
        ),
    ]


def add_vertical_posts(material) -> list:
    """Nine centered web posts from the source range [SW/2 : (L-SW)/S : L]."""
    objects = []
    for index, x_pos in enumerate(source_loop_values(HALF_SUPPORT, POST_STEP, TR_LENGTH), start=1):
        objects.append(
            add_line_segment_prism(
                name=f"web_vertical_post_{index:02d}",
                p0=(x_pos, 0.0),
                p1=(x_pos, WEB_HEIGHT),
                width=TR_SUPPORT_WIDTH,
                z_min=WEB_Z_MIN,
                z_max=WEB_Z_MAX,
                y_offset=SOURCE_Y_OFFSET,
                clip_rect=TRUSS_CLIP_RECT,
                material=material,
            )
        )
    return objects


def add_pratt_diagonals(material) -> list:
    """Eight Pratt web diagonals, rising to midspan then reversing after center."""
    objects = []
    diagonal_starts = source_loop_values(
        HALF_SUPPORT,
        POST_STEP,
        TR_LENGTH * (1.0 - 1.0 / TR_SEGS),
    )
    for index, x_pos in enumerate(diagonal_starts, start=1):
        if x_pos < CENTER_SWITCH_X:
            p0 = (x_pos, 0.0)
            p1 = (x_pos + DIAGONAL_X_SPAN, WEB_HEIGHT)
            direction = "rising"
        else:
            p0 = (x_pos, WEB_HEIGHT)
            p1 = (x_pos + DIAGONAL_X_SPAN, 0.0)
            direction = "falling"
        objects.append(
            add_line_segment_prism(
                name=f"web_pratt_{direction}_diagonal_{index:02d}",
                p0=p0,
                p1=p1,
                width=TR_SUPPORT_WIDTH,
                z_min=WEB_Z_MIN,
                z_max=WEB_Z_MAX,
                y_offset=SOURCE_Y_OFFSET,
                clip_rect=TRUSS_CLIP_RECT,
                material=material,
            )
        )
    return objects


def build_model() -> list:
    """Build the source truss from the same constants and segment ranges as the SCAD."""
    gold = make_material("openscad_default_gold", SOURCE_GOLD)
    objects = []
    objects.extend(add_cap_chords(gold))
    objects.extend(add_vertical_posts(gold))
    objects.extend(add_pratt_diagonals(gold))
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

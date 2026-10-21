from __future__ import annotations
from dataclasses import dataclass
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
from typing import Iterable
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the flat hinge model."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    metallic: float = 0.0
    roughness: float = 0.45


GREEN = MaterialSpec("painted_leaf_green", (0.0, 0.572549, 0.0, 1.0), 0.0, 0.32)
RED = MaterialSpec("painted_leaf_red", (0.862745, 0.0784314, 0.235294, 1.0), 0.0, 0.32)
PIN_GREY = MaterialSpec("hinge_pin_grey", (0.752941, 0.752941, 0.752941, 1.0), 0.5, 0.24)
SCREW_SILVER = MaterialSpec("screw_silver", (0.88, 0.88, 0.86, 1.0), 0.72, 0.18)
NUT_GREY = MaterialSpec("nut_grey", (0.26, 0.26, 0.255, 1.0), 0.65, 0.28)
SLOT_BLUE = MaterialSpec("screw_slot_blue", (0.254902, 0.411765, 0.882353, 1.0), 0.1, 0.35)


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


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
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
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
    material: MaterialSpec | None = None,
    smooth_sides: bool = True,
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
    for polygon in obj.data.polygons:
        polygon.use_smooth = smooth_sides and abs(polygon.normal.z) < 0.5
    return obj


def add_rounded_rect_prism(
    name: str,
    width: float,
    depth: float,
    height: float,
    radius: float,
    location: tuple[float, float, float],
    material: MaterialSpec,
    bevel_segments: int = 24,
) -> bpy.types.Object:
    core_x = add_box(f"{name}_core_x", (width - 2 * radius, depth, height), location, material)
    core_y = add_box(f"{name}_core_y", (width, depth - 2 * radius, height), location, material)
    cutters: list[bpy.types.Object] = [core_x, core_y]
    for sx in (-1, 1):
        for sy in (-1, 1):
            cutters.append(
                add_cylinder(
                    f"{name}_corner_{sx}_{sy}",
                    diameter=2 * radius,
                    height=height,
                    location=(location[0] + sx * (width / 2 - radius), location[1] + sy * (depth / 2 - radius), location[2]),
                    vertices=bevel_segments,
                    material=material,
                )
            )
    obj = boolean_union(name, cutters)
    return obj


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(set(points))
    if len(ordered) <= 1:
        return ordered

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def hull_leaf_points(
    inner_y: float,
    outer_direction: int,
    strip_half_width: float,
    lobe_offset_x: float,
    lobe_y: float,
    lobe_radius: float,
    x_center: float = 0.0,
    circle_steps: int = 48,
) -> list[tuple[float, float]]:
    """Approximate the SCAD hull of two far circles and a thin hinge-side strip."""
    strip_tip = strip_half_width * 1.026
    strip_setback = lobe_radius * 0.028
    samples = [
        (x_center - strip_tip, inner_y),
        (x_center + strip_tip, inner_y),
        (x_center - strip_tip, inner_y + outer_direction * strip_setback),
        (x_center + strip_tip, inner_y + outer_direction * strip_setback),
    ]
    for lobe_x in (-lobe_offset_x, lobe_offset_x):
        center_x = x_center + lobe_x
        center_y = inner_y + outer_direction * lobe_y
        for step in range(circle_steps):
            angle = 2.0 * math.pi * step / circle_steps
            samples.append((center_x + math.cos(angle) * lobe_radius, center_y + math.sin(angle) * lobe_radius))
    hull = _convex_hull(samples)
    if outer_direction < 0:
        hull = list(reversed(hull))
    return hull


def notched_leaf_points(
    inner_y: float,
    outer_direction: int,
    strip_half_width: float,
    lobe_offset_x: float,
    lobe_y: float,
    lobe_radius: float,
    x_center: float = 0.0,
    notch_spans: list[tuple[float, float]] | None = None,
    hinge_step_depth: float = 3.1,
    hinge_relief_depth: float = 1.0,
    side_x: float | None = None,
    circle_steps: int = 18,
) -> list[tuple[float, float]]:
    """Return a compact solid hull leaf with a crisp hinge-side seam.

    The leaf remains a single solid plate, but the hinge-side edge is clipped
    into short straight runs behind the rolled knuckles. This keeps the source's
    compact SCAD hull silhouette without returning to the open-frame cutouts of
    earlier iterations.
    """
    hull = hull_leaf_points(
        inner_y=inner_y,
        outer_direction=outer_direction,
        strip_half_width=strip_half_width,
        lobe_offset_x=lobe_offset_x,
        lobe_y=lobe_y,
        lobe_radius=lobe_radius,
        x_center=x_center,
        circle_steps=circle_steps * 4,
    )
    if not notch_spans:
        return hull

    inset_y = inner_y + outer_direction * hinge_relief_depth
    seam_half = strip_half_width * 1.014
    start_x = x_center - seam_half
    end_x = x_center + seam_half
    spans = sorted((x_center + start, x_center + end) for start, end in notch_spans)
    seam: list[tuple[float, float]] = [(start_x, inner_y)]
    cursor = start_x
    for span_start, span_end in spans:
        span_start = max(start_x, min(end_x, span_start))
        span_end = max(start_x, min(end_x, span_end))
        if span_end <= cursor:
            continue
        if span_start > cursor:
            seam.extend([(span_start, inner_y), (span_start, inset_y)])
        seam.extend([(span_end, inset_y), (span_end, inner_y)])
        cursor = span_end
    seam.append((end_x, inner_y))

    outer_points = [point for point in hull if outer_direction * (point[1] - inner_y) > 0.002]
    if outer_direction > 0:
        outer_points.sort(key=lambda point: point[0], reverse=True)
        return seam + outer_points
    outer_points.sort(key=lambda point: point[0])
    return list(reversed(seam)) + outer_points

def curled_edge_points(
    x_center: float,
    length: float,
    leaf_edge_y: float,
    roll_y: float,
    diameter: float,
    material_side: int,
) -> list[tuple[float, float]]:
    half = length / 2.0
    radius = diameter / 2.0
    tangent_lip = radius * 0.0002
    root_lip = diameter * 0.0
    inset = min(diameter * 0.018, half * 0.030)
    tangent_y = roll_y - material_side * tangent_lip
    return [
        (x_center - half + inset, leaf_edge_y - material_side * root_lip),
        (x_center + half - inset, leaf_edge_y - material_side * root_lip),
        (x_center + half - inset, tangent_y),
        (x_center - half + inset, tangent_y),
    ]


def curled_wrap_points(
    x_center: float,
    length: float,
    leaf_edge_y: float,
    roll_y: float,
    diameter: float,
    material_side: int,
    steps: int = 8,
) -> list[tuple[float, float]]:
    half = length / 2.0
    radius = diameter / 2.0
    inset = min(diameter * 0.020, half * 0.032)
    root_y = leaf_edge_y
    sweep = [root_y]
    for step in range(1, steps + 1):
        t = step / steps
        angle = t * math.pi * 0.34
        curve_y = roll_y - material_side * (radius * (0.0002 + 0.013 * math.sin(angle)))
        sweep.append(curve_y)
    outer = [(x_center + half - inset, y) for y in sweep]
    inner = [(x_center - half + inset, y) for y in reversed(sweep)]
    return outer + inner


def add_polygon_prism(
    name: str,
    points: list[tuple[float, float]],
    height: float,
    z_center: float,
    material: MaterialSpec,
) -> bpy.types.Object:
    """Build a simple vertical prism from a counter-clockwise 2D outline."""
    bottom_z = z_center - height / 2.0
    top_z = z_center + height / 2.0
    vertices = [(x, y, bottom_z) for x, y in points] + [(x, y, top_z) for x, y in points]
    count = len(points)
    triangles = tessellate_polygon([[Vector((x, y, 0.0)) for x, y in points]])
    faces = [tuple(reversed(triangle)) for triangle in triangles]
    faces.extend(tuple(index + count for index in triangle) for triangle in triangles)
    for index in range(count):
        faces.append((index, (index + 1) % count, count + (index + 1) % count, count + index))
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def add_capsule_prism(
    name: str,
    length: float,
    width: float,
    height: float,
    location: tuple[float, float, float],
    material: MaterialSpec,
    axis: str = "X",
) -> bpy.types.Object:
    radius = width / 2.0
    parts: list[bpy.types.Object]
    if axis.upper() == "X":
        parts = [add_box(f"{name}_web", (length - width, width, height), location, material)]
        for side in (-1, 1):
            parts.append(add_cylinder(f"{name}_cap_{side}", width, height, (location[0] + side * (length / 2 - radius), location[1], location[2]), material=material))
    else:
        parts = [add_box(f"{name}_web", (width, length - width, height), location, material)]
        for side in (-1, 1):
            parts.append(add_cylinder(f"{name}_cap_{side}", width, height, (location[0], location[1] + side * (length / 2 - radius), location[2]), material=material))
    return boolean_union(name, parts)


def boolean_union(name: str, objects: Iterable[bpy.types.Object]) -> bpy.types.Object:
    objects = list(objects)
    if not objects:
        raise ValueError("boolean_union requires at least one object")
    active = objects[0]
    bpy.context.view_layer.objects.active = active
    active.select_set(True)
    for obj in objects[1:]:
        modifier = active.modifiers.new(f"join_{obj.name}", "BOOLEAN")
        modifier.operation = "UNION"
        modifier.object = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(obj, do_unlink=True)
    active.name = name
    return active


def add_hinge_knuckle(
    name: str,
    x_center: float,
    length: float,
    y_center: float,
    z_center: float,
    outer_diameter: float,
    material: MaterialSpec,
) -> bpy.types.Object:
    obj = add_cylinder(
        name=name,
        diameter=outer_diameter,
        height=length,
        location=(x_center, y_center, z_center),
        rotation=(0.0, math.pi / 2.0, 0.0),
        vertices=48,
        material=material,
        smooth_sides=False,
    )
    end_cap_radius = outer_diameter * 0.478
    cap_thickness = min(0.170, max(0.095, length * 0.038))
    left_cap = add_cylinder(
        f"{name}_left_fold_face",
        end_cap_radius * 2.0,
        cap_thickness,
        (x_center - length / 2.0 - cap_thickness * 0.08, y_center, z_center),
        rotation=(0.0, math.pi / 2.0, 0.0),
        vertices=48,
        material=material,
        smooth_sides=False,
    )
    right_cap = add_cylinder(
        f"{name}_right_fold_face",
        end_cap_radius * 2.0,
        cap_thickness,
        (x_center + length / 2.0 + cap_thickness * 0.08, y_center, z_center),
        rotation=(0.0, math.pi / 2.0, 0.0),
        vertices=48,
        material=material,
        smooth_sides=False,
    )
    return boolean_union(name, [obj, left_cap, right_cap])


def add_pin(name: str, length: float, location: tuple[float, float, float], diameter: float = 2.65) -> bpy.types.Object:
    return add_cylinder(name, diameter, length, location, rotation=(0.0, math.pi / 2.0, 0.0), vertices=48, material=PIN_GREY)


def add_screw_stack(
    name: str,
    x: float,
    y: float,
    plate_z: float,
    upward: bool = True,
    scale: float = 1.0,
) -> list[bpy.types.Object]:
    sign = 1.0 if upward else -1.0
    head_z = plate_z + sign * 1.28 * scale
    skirt_z = plate_z + sign * 0.48 * scale
    socket_z = plate_z + sign * 2.05 * scale
    objects = [
        add_cylinder(f"{name}_countersunk_flange", 5.26 * scale, 0.25 * scale, (x, y, skirt_z), vertices=72, material=SCREW_SILVER),
        add_cylinder(f"{name}_low_beveled_head", 4.52 * scale, 0.64 * scale, (x, y, plate_z + sign * 0.96 * scale), vertices=72, material=SCREW_SILVER),
        add_cylinder(f"{name}_shallow_crown", 3.40 * scale, 0.34 * scale, (x, y, head_z + sign * 0.38 * scale), vertices=72, material=SCREW_SILVER),
        add_cylinder(f"{name}_hex_socket", 1.78 * scale, 0.20 * scale, (x, y, socket_z), vertices=6, material=NUT_GREY, smooth_sides=False),
    ]
    if not upward:
        for obj in objects:
            obj.rotation_euler[0] += math.pi
    shank = add_cylinder(f"{name}_short_dark_thread", 1.18 * scale, 2.02 * scale, (x, y, plate_z - 1.02 * scale), vertices=32, material=NUT_GREY)
    lower_washer = add_cylinder(f"{name}_underside_washer", 3.12 * scale, 0.10 * scale, (x, y, plate_z - 1.93 * scale), vertices=48, material=SCREW_SILVER)
    nut = add_cylinder(f"{name}_short_hex_nut", 3.60 * scale, 0.25 * scale, (x, y, plate_z - 2.10 * scale), vertices=6, material=NUT_GREY, smooth_sides=False)
    lower_rim = add_cylinder(f"{name}_hex_nut_lower_rim", 3.18 * scale, 0.050 * scale, (x, y, plate_z - 2.26 * scale), vertices=6, material=NUT_GREY, smooth_sides=False)
    objects.extend([shank, lower_washer, nut, lower_rim])
    return objects


def set_origin_center(objects: Iterable[bpy.types.Object]) -> None:
    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.select_set(False)


# ============================================================
# Procedural assembly
# ============================================================

"""Readable assembly logic for the flat hinge model."""




PLATE_THICKNESS = 1.58
PLATE_Z = PLATE_THICKNESS / 2.0
BARREL_Y = 0.0
BARREL_Z = 3.04
BARREL_DIAMETER = 5.74

LARGE_X = 10.40
LARGE_STRIP_HALF_WIDTH = 19.04
LARGE_LOBE_OFFSET_X = 16.50
LARGE_LOBE_Y = 14.12
LARGE_RADIUS = 4.42
LARGE_GAP = 0.22

SMALL_X = -24.75
SMALL_STRIP_HALF_WIDTH = 8.66
SMALL_LOBE_OFFSET_X = 5.30
SMALL_LOBE_Y = 10.68
SMALL_RADIUS = 4.34
SMALL_BARREL_DIAMETER = 4.94
SMALL_BARREL_Z = 3.02


def build_leaf(
    name: str,
    x_center: float,
    inner_y: float,
    outer_direction: int,
    strip_half_width: float,
    lobe_offset_x: float,
    lobe_y: float,
    radius: float,
    material,
) -> object:
    points = hull_leaf_points(
        inner_y=inner_y,
        outer_direction=outer_direction,
        strip_half_width=strip_half_width,
        lobe_offset_x=lobe_offset_x,
        lobe_y=lobe_y,
        lobe_radius=radius,
        x_center=x_center,
    )
    return add_polygon_prism(name, points, PLATE_THICKNESS, PLATE_Z, material)


def build_notched_leaf(
    name: str,
    x_center: float,
    inner_y: float,
    outer_direction: int,
    strip_half_width: float,
    lobe_offset_x: float,
    lobe_y: float,
    radius: float,
    notch_spans: list[tuple[float, float]],
    hinge_step_depth: float,
    hinge_relief_depth: float,
    side_x: float,
    material,
) -> object:
    points = notched_leaf_points(
        inner_y=inner_y,
        outer_direction=outer_direction,
        strip_half_width=strip_half_width,
        lobe_offset_x=lobe_offset_x,
        lobe_y=lobe_y,
        lobe_radius=radius,
        x_center=x_center,
        notch_spans=notch_spans,
        hinge_step_depth=hinge_step_depth,
        hinge_relief_depth=hinge_relief_depth,
        side_x=side_x,
    )
    return add_polygon_prism(name, points, PLATE_THICKNESS, PLATE_Z, material)


def add_curled_knuckles(
    prefix: str,
    x_center: float,
    segments: list[tuple[float, float]],
    leaf_edge_y: float,
    material_side: int,
    diameter: float,
    z_center: float,
    material,
    pin_diameter: float,
    pin_face_offsets: list[float] | None = None,
    roll_y: float = BARREL_Y,
    terminal_pin_diameter: float | None = None,
) -> list:
    objects = []
    for index, (start, end) in enumerate(segments, start=1):
        length = end - start
        segment_x = x_center + (start + end) / 2.0
        web_points = curled_edge_points(
            x_center=segment_x,
            length=length,
            leaf_edge_y=leaf_edge_y,
            roll_y=roll_y,
            diameter=diameter,
            material_side=material_side,
        )
        wrap_points = curled_wrap_points(
            x_center=segment_x,
            length=length,
            leaf_edge_y=leaf_edge_y,
            roll_y=roll_y,
            diameter=diameter,
            material_side=material_side,
        )
        web_height = PLATE_THICKNESS * 0.22
        web_z = PLATE_Z + 0.02
        wrap_height = diameter * 0.036
        wrap_z = z_center - diameter * 0.414
        objects.append(add_polygon_prism(f"{prefix}_curled_sheet_web_{index}", web_points, web_height, web_z, material))
        objects.append(add_polygon_prism(f"{prefix}_rounded_sheet_wrap_{index}", wrap_points, wrap_height, wrap_z, material))
        objects.append(add_hinge_knuckle(f"{prefix}_rolled_knuckle_{index}", segment_x, length, roll_y, z_center, diameter, material))
    for index, offset in enumerate(pin_face_offsets or [], start=1):
        objects.append(
            add_cylinder(
                f"{prefix}_exposed_pin_face_{index}",
                terminal_pin_diameter or pin_diameter,
                0.080,
                (x_center + offset, roll_y, z_center),
                rotation=(0.0, math.pi / 2.0, 0.0),
                vertices=48,
                material=PIN_GREY,
                smooth_sides=False,
            )
        )
    return objects


def add_countersink_shadows(prefix: str, centers: list[tuple[float, float]], scale: float) -> list:
    objects = []
    for index, (x, y) in enumerate(centers, start=1):
        objects.append(
            add_capsule_prism(
                f"{prefix}_countersink_shadow_{index}",
                4.18 * scale,
                1.98 * scale,
                0.16,
                (x, y, PLATE_THICKNESS + 0.090),
                PIN_GREY,
            )
        )
    return objects


def build_large_hinge() -> list:
    objects: list = []
    green_segments = [(-20.62, -15.90), (-4.18, 2.54), (15.98, 20.62)]
    red_segments = [(-13.28, -6.70), (4.92, 13.44)]
    green_notches = [(-20.42, -16.10), (-3.96, 2.32), (16.18, 20.42)]
    red_notches = [(-13.06, -6.92), (5.12, 13.20)]

    objects.append(
        build_notched_leaf(
            "large_upper_green_notched_leaf",
            LARGE_X,
            LARGE_GAP / 2.0,
            1,
            LARGE_STRIP_HALF_WIDTH,
            LARGE_LOBE_OFFSET_X,
            LARGE_LOBE_Y,
            LARGE_RADIUS,
            green_notches,
            1.28,
            0.82,
            LARGE_STRIP_HALF_WIDTH,
            GREEN,
        )
    )
    objects.append(
        build_notched_leaf(
            "large_lower_red_notched_leaf",
            LARGE_X,
            -LARGE_GAP / 2.0,
            -1,
            LARGE_STRIP_HALF_WIDTH,
            LARGE_LOBE_OFFSET_X,
            LARGE_LOBE_Y,
            LARGE_RADIUS,
            red_notches,
            1.28,
            0.82,
            LARGE_STRIP_HALF_WIDTH,
            RED,
        )
    )
    objects.extend(add_curled_knuckles("large_green", LARGE_X, green_segments, LARGE_GAP / 2.0, 1, BARREL_DIAMETER, BARREL_Z, GREEN, 1.42, [-20.88, 20.88], roll_y=0.0, terminal_pin_diameter=2.78))
    objects.extend(add_curled_knuckles("large_red", LARGE_X, red_segments, -LARGE_GAP / 2.0, -1, BARREL_DIAMETER, BARREL_Z, RED, 1.48, [], roll_y=0.0))
    objects.extend(add_large_screw_pattern())
    return objects


def add_large_screw_pattern() -> list:
    objects: list = []
    green_positions = [
        (LARGE_X - 8.66, 15.00),
        (LARGE_X + 8.66, 15.00),
        (LARGE_X - 17.02, 4.55),
        (LARGE_X, 4.55),
        (LARGE_X + 17.02, 4.55),
    ]
    red_positions = [
        (LARGE_X - 17.02, -4.55),
        (LARGE_X, -4.55),
        (LARGE_X + 17.02, -4.55),
        (LARGE_X - 8.66, -15.00),
        (LARGE_X + 8.66, -15.00),
    ]
    objects.extend(add_countersink_shadows("large_green", green_positions, 1.18))
    objects.extend(add_countersink_shadows("large_red", red_positions, 1.18))
    for index, (x, y) in enumerate(green_positions, start=1):
        objects.extend(add_screw_stack(f"large_green_leaf_screw_{index}", x, y, PLATE_Z, scale=1.20))
    for index, (x, y) in enumerate(red_positions, start=1):
        objects.extend(add_screw_stack(f"large_red_leaf_screw_{index}", x, y, PLATE_Z, scale=1.20))
    return objects


def build_small_hinge() -> list:
    objects: list = []
    green_segments = [(-9.08, -6.24), (6.24, 9.08)]
    red_segments = [(-4.18, 4.18)]
    green_notches = [(-8.82, -6.46), (6.46, 8.82)]
    red_notches = [(-3.94, 3.94)]
    small_gap = 0.22
    scale = 0.93

    objects.append(
        build_notched_leaf(
            "small_upper_green_rect_notched_leaf",
            SMALL_X,
            small_gap / 2.0,
            1,
            SMALL_STRIP_HALF_WIDTH,
            SMALL_LOBE_OFFSET_X,
            SMALL_LOBE_Y,
            SMALL_RADIUS,
            green_notches,
            1.00,
            0.64,
            SMALL_STRIP_HALF_WIDTH,
            GREEN,
        )
    )
    objects.append(
        build_notched_leaf(
            "small_lower_red_rect_notched_leaf",
            SMALL_X,
            -small_gap / 2.0,
            -1,
            SMALL_STRIP_HALF_WIDTH,
            SMALL_LOBE_OFFSET_X,
            SMALL_LOBE_Y,
            SMALL_RADIUS,
            red_notches,
            1.00,
            0.64,
            SMALL_STRIP_HALF_WIDTH,
            RED,
        )
    )
    objects.extend(add_curled_knuckles("small_green", SMALL_X, green_segments, small_gap / 2.0, 1, SMALL_BARREL_DIAMETER, SMALL_BARREL_Z, GREEN, 1.26, [-9.38, 9.38], roll_y=0.0, terminal_pin_diameter=2.62))
    objects.extend(add_curled_knuckles("small_red", SMALL_X, red_segments, -small_gap / 2.0, -1, SMALL_BARREL_DIAMETER, SMALL_BARREL_Z, RED, 1.34, [], roll_y=0.0))

    screw_positions = [
        (SMALL_X - 5.45, 4.60),
        (SMALL_X + 5.45, 11.50),
        (SMALL_X - 5.45, -11.50),
        (SMALL_X + 5.45, -4.60),
    ]
    objects.extend(add_countersink_shadows("small", screw_positions, scale))
    for index, (x, y) in enumerate(screw_positions, start=1):
        objects.extend(add_screw_stack(f"small_hinge_screw_{index}", x, y, PLATE_Z, scale=scale))
    return objects


def build_model() -> list:
    """Build the paired flat hinge display with source-style leaves and compact curled barrels."""
    objects: list = []
    objects.extend(build_small_hinge())
    objects.extend(build_large_hinge())
    for obj in objects:
        if obj.name.endswith("pin"):
            assign_material(obj, PIN_GREY)
    set_origin_center(objects)
    return objects


# ============================================================
# Scene setup and rendering
# ============================================================

"""Minimal render entrypoint for the current flat hinge iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

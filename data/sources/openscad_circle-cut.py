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

"""Reusable primitive builders for the router circle jig model."""





Color = tuple[float, float, float, float]
Point2D = tuple[float, float]


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: Color
    roughness: float = 0.56


OPENSCAD_YELLOW = MaterialSpec("openscad_preview_yellow", (1.0, 0.74, 0.02, 1.0), 0.52)
GROOVE_GREEN = MaterialSpec("cut_recess_shadow_green", (0.43, 0.60, 0.18, 1.0), 0.62)
CUT_FACE_LIGHT = MaterialSpec("fresh_cut_light", (1.0, 0.98, 0.68, 1.0), 0.60)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def get_material(spec: MaterialSpec) -> bpy.types.Material:
    material = bpy.data.materials.get(spec.name) or bpy.data.materials.new(spec.name)
    material.diffuse_color = spec.color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = spec.color
        bsdf.inputs["Roughness"].default_value = spec.roughness
        bsdf.inputs["Metallic"].default_value = 0.0
    return material


def assign_material(obj: bpy.types.Object, spec: MaterialSpec) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(get_material(spec))
    return obj


def shade_flat(obj: bpy.types.Object) -> bpy.types.Object:
    for polygon in obj.data.polygons:
        polygon.use_smooth = False
    return obj


def _new_mesh_object(
    name: str,
    vertices: list[tuple[float, float, float]],
    faces: list[list[int]],
    material: MaterialSpec | None,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material is not None:
        assign_material(obj, material)
    shade_flat(obj)
    return obj


def circle_points(
    center: Point2D = (0.0, 0.0),
    radius: float = 1.0,
    segments: int = 96,
    start_degrees: float = 0.0,
    end_degrees: float = 360.0,
) -> list[Point2D]:
    """Return CCW arc points, inclusive at both ends for partial arcs."""
    if segments < 3:
        raise ValueError("segments must be at least 3")
    cx, cy = center
    span = end_degrees - start_degrees
    if abs(span) >= 360.0:
        return [
            (
                cx + radius * math.cos(math.tau * index / segments),
                cy + radius * math.sin(math.tau * index / segments),
            )
            for index in range(segments)
        ]
    return [
        (
            cx + radius * math.cos(math.radians(start_degrees + span * index / segments)),
            cy + radius * math.sin(math.radians(start_degrees + span * index / segments)),
        )
        for index in range(segments + 1)
    ]


def capsule_points_between(
    start: Point2D,
    end: Point2D,
    radius: float,
    segments_per_cap: int = 24,
) -> list[Point2D]:
    """Return a CCW hull/capsule footprint around two equal circles."""
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length = math.hypot(dx, dy)
    if length <= 1e-7:
        return circle_points(start, radius, segments_per_cap * 2)

    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux

    points: list[Point2D] = []
    for index in range(segments_per_cap + 1):
        theta = math.pi * index / segments_per_cap
        points.append((ex + radius * (px * math.cos(theta) + ux * math.sin(theta)), ey + radius * (py * math.cos(theta) + uy * math.sin(theta))))
    for index in range(segments_per_cap + 1):
        theta = math.pi * index / segments_per_cap
        points.append((sx + radius * (-px * math.cos(theta) - ux * math.sin(theta)), sy + radius * (-py * math.cos(theta) - uy * math.sin(theta))))
    return points


def rounded_slot_profile(length: float, diameter: float, segments: int = 28) -> list[Point2D]:
    """SCAD hull of two circles along X, centered at the origin."""
    radius = diameter / 2.0
    half_straight = (length - diameter) / 2.0
    return capsule_points_between((-half_straight, 0.0), (half_straight, 0.0), radius, segments)


def make_circle_loop(center: Point2D, diameter: float, segments: int = 80, clockwise: bool = True) -> list[Point2D]:
    points = circle_points(center=center, radius=diameter / 2.0, segments=segments)
    return list(reversed(points)) if clockwise else points


def signed_area(points: list[Point2D]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        area += point[0] * nxt[1] - nxt[0] * point[1]
    return area / 2.0


def orient_profile(points: list[Point2D], ccw: bool) -> list[Point2D]:
    is_ccw = signed_area(points) > 0.0
    if is_ccw == ccw:
        return points
    return list(reversed(points))


def extrude_profile_with_holes(
    name: str,
    outer_profile: list[Point2D],
    hole_profiles: list[list[Point2D]],
    z_min: float,
    z_max: float,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Create a vertical prism with real top/bottom holes from 2D loops."""
    loops = [orient_profile(outer_profile, ccw=True)] + [orient_profile(hole, ccw=False) for hole in hole_profiles]
    flat_points = [point for loop in loops for point in loop]
    vertices = [(x, y, z_min) for x, y in flat_points] + [(x, y, z_max) for x, y in flat_points]
    count = len(flat_points)

    triangle_input = [[Vector((x, y, 0.0)) for x, y in loop] for loop in loops]
    triangles = tessellate_polygon(triangle_input)
    faces: list[list[int]] = []
    for triangle in triangles:
        faces.append([triangle[2], triangle[1], triangle[0]])
        faces.append([count + triangle[0], count + triangle[1], count + triangle[2]])

    offset = 0
    for loop_index, loop in enumerate(loops):
        loop_count = len(loop)
        for index in range(loop_count):
            a = offset + index
            b = offset + (index + 1) % loop_count
            if loop_index == 0:
                faces.append([a, b, b + count, a + count])
            else:
                faces.append([a, a + count, b + count, b])
        offset += loop_count

    return _new_mesh_object(name, vertices, faces, material)


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material is not None:
        assign_material(obj, material)
    return shade_flat(obj)


def add_cylinder(
    name: str,
    diameter: float,
    height: float,
    location: tuple[float, float, float],
    material: MaterialSpec | None = None,
    vertices: int = 96,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=diameter / 2.0, depth=height, location=location)
    obj = bpy.context.object
    obj.name = name
    if material is not None:
        assign_material(obj, material)
    return shade_flat(obj)


def add_conical_frustum(
    name: str,
    bottom_diameter: float,
    top_diameter: float,
    height: float,
    location: tuple[float, float, float],
    material: MaterialSpec | None = None,
    vertices: int = 96,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cone_add(
        vertices=vertices,
        radius1=bottom_diameter / 2.0,
        radius2=top_diameter / 2.0,
        depth=height,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    if material is not None:
        assign_material(obj, material)
    return shade_flat(obj)


def add_capsule_prism(
    name: str,
    start: Point2D,
    end: Point2D,
    diameter: float,
    z_min: float,
    z_max: float,
    material: MaterialSpec | None = None,
    segments_per_cap: int = 28,
) -> bpy.types.Object:
    profile = capsule_points_between(start, end, diameter / 2.0, segments_per_cap)
    return extrude_profile_with_holes(name, profile, [], z_min, z_max, material)


def add_cylinder_cutter(
    name: str,
    center: Point2D,
    diameter: float,
    z_min: float,
    z_max: float,
    vertices: int = 80,
) -> bpy.types.Object:
    height = z_max - z_min
    return add_cylinder(name, diameter, height, (center[0], center[1], z_min + height / 2.0), vertices=vertices)


def add_cone_cutter(
    name: str,
    center: Point2D,
    bottom_diameter: float,
    top_diameter: float,
    z_min: float,
    z_max: float,
    vertices: int = 96,
) -> bpy.types.Object:
    height = z_max - z_min
    return add_conical_frustum(
        name,
        bottom_diameter=bottom_diameter,
        top_diameter=top_diameter,
        height=height,
        location=(center[0], center[1], z_min + height / 2.0),
        vertices=vertices,
    )


def boolean_difference(target: bpy.types.Object, cutters: Iterable[bpy.types.Object]) -> None:
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    for cutter in cutters:
        modifier = target.modifiers.new(name=f"cut_{cutter.name}", type="BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        if hasattr(modifier, "solver"):
            modifier.solver = "EXACT"
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
    shade_flat(target)


def boolean_union(target: bpy.types.Object, additions: Iterable[bpy.types.Object]) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    for addition in additions:
        modifier = target.modifiers.new(name=f"union_{addition.name}", type="BOOLEAN")
        modifier.operation = "UNION"
        modifier.object = addition
        if hasattr(modifier, "solver"):
            modifier.solver = "EXACT"
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(addition, do_unlink=True)
    shade_flat(target)
    return target


def add_recess_floor(
    name: str,
    start: Point2D,
    end: Point2D,
    diameter: float,
    z: float,
    material: MaterialSpec,
    segments_per_cap: int = 28,
    thickness: float = 0.035,
) -> bpy.types.Object:
    obj = add_capsule_prism(
        name,
        start=start,
        end=end,
        diameter=diameter,
        z_min=z,
        z_max=z + thickness,
        material=material,
        segments_per_cap=segments_per_cap,
    )
    return obj


def hide_objects(objects: Iterable[bpy.types.Object]) -> None:
    for obj in objects:
        obj.hide_viewport = True
        obj.hide_render = True


# ============================================================
# Procedural assembly
# ============================================================

"""Assembly logic for the complex-circle router jig model."""






@dataclass(frozen=True)
class RouterJigParams:
    m3_head_height: float = 2.6
    m3_head_dia: float = 6.0
    m3_hole_dia: float = 3.8
    m3_tap_dia: float = 2.8
    cutting_dia: float = 12.0
    outer_dia: float = 145.0
    inner_dia: float = 30.0
    inner_width: float = 85.0
    base_height: float = 110.0
    base_thickness: float = 6.0
    holes_x_dist: float = 106.0
    holes_y_dist: float = 56.0
    holes_dia: float = 5.6
    inlay_thickness: float = 4.0
    inlay_width: float = 12.0
    inlay_groove: float = 3.8
    extension_width: float = 50.0
    extension_frame: float = 5.0
    max_circle_dia: float = 250.0

    @property
    def extension_length(self) -> float:
        return self.max_circle_dia / 2.0 - self.cutting_dia / 2.0 + self.extension_frame

    @property
    def extension_width_effective(self) -> float:
        return min(self.extension_width, self.extension_length)

    @property
    def mounting_centers(self) -> list[Point2D]:
        xs = (-self.holes_x_dist / 2.0, self.holes_x_dist / 2.0)
        ys = (-self.holes_y_dist / 2.0, self.holes_y_dist / 2.0)
        return [(x, y) for x in xs for y in ys]


def _base_plate_footprint(params: RouterJigParams, segments: int = 96) -> list[Point2D]:
    """Intersection of the round router base with the centered rectangular crop."""
    radius = params.outer_dia / 2.0
    half_height = params.base_height / 2.0
    x_at_crop = (radius * radius - half_height * half_height) ** 0.5
    angle_top_right = math.degrees(math.atan2(half_height, x_at_crop))
    angle_top_left = 180.0 - angle_top_right
    angle_bottom_left = 180.0 + angle_top_right
    angle_bottom_right = 360.0 - angle_top_right

    points: list[Point2D] = [(x_at_crop, half_height), (-x_at_crop, half_height)]
    for step in range(1, segments):
        angle = angle_top_left + (angle_bottom_left - angle_top_left) * step / segments
        points.append((radius * math.cos(math.radians(angle)), radius * math.sin(math.radians(angle))))
    points.extend([(-x_at_crop, -half_height), (x_at_crop, -half_height)])
    for step in range(1, segments):
        angle = angle_bottom_right + (360.0 + angle_top_right - angle_bottom_right) * step / segments
        points.append((radius * math.cos(math.radians(angle % 360.0)), radius * math.sin(math.radians(angle % 360.0))))
    return points


def _central_cutout(params: RouterJigParams) -> list[Point2D]:
    start_x = -params.inner_width / 2.0 + params.inner_dia / 2.0
    end_x = params.inner_width / 2.0 - params.inner_dia / 2.0
    return capsule_points_between((start_x, 0.0), (end_x, 0.0), params.inner_dia / 2.0, 32)


def _extension_plate(params: RouterJigParams) -> bpy.types.Object:
    base = extrude_profile_with_holes(
        "router_jig_plate",
        outer_profile=_combined_plate_footprint(params),
        hole_profiles=[_central_cutout(params)] + [make_circle_loop(center, params.holes_dia, 64) for center in params.mounting_centers],
        z_min=0.0,
        z_max=params.base_thickness,
        material=OPENSCAD_YELLOW,
    )
    return base


def _combined_plate_footprint(params: RouterJigParams) -> list[Point2D]:
    """Manual outline of union(base_plate, extension_hull) in SCAD coordinates."""
    radius = params.outer_dia / 2.0
    half_height = params.base_height / 2.0
    half_ext = params.extension_width_effective / 2.0
    x_at_crop = (radius * radius - half_height * half_height) ** 0.5
    angle_top_right = math.degrees(math.atan2(half_height, x_at_crop))
    angle_top_left = 180.0 - angle_top_right
    nose_center_y = -params.extension_length + half_ext

    points: list[Point2D] = [(x_at_crop, half_height), (-x_at_crop, half_height)]
    for step in range(1, 96):
        angle = angle_top_left + (360.0 - 2.0 * angle_top_left) * step / 96
        points.append((radius * math.cos(math.radians(angle)), radius * math.sin(math.radians(angle))))

    points.extend([(-x_at_crop, -half_height), (-half_ext, -half_height), (-half_ext, nose_center_y)])
    for step in range(1, 65):
        angle = 180.0 + 180.0 * step / 64
        points.append((half_ext * math.cos(math.radians(angle)), nose_center_y + half_ext * math.sin(math.radians(angle))))
    points.extend([(half_ext, nose_center_y), (half_ext, -half_height), (x_at_crop, -half_height)])

    for step in range(1, 96):
        angle = (360.0 - angle_top_right) + (angle_top_right * 2.0) * step / 96
        points.append((radius * math.cos(math.radians(angle)), radius * math.sin(math.radians(angle))))
    return points


def _countersink_cutters(params: RouterJigParams) -> list[bpy.types.Object]:
    cutters: list[bpy.types.Object] = []
    top_dia = params.holes_dia + 2.0 * params.base_thickness + 2.0
    for index, center in enumerate(params.mounting_centers):
        cutters.append(
            add_cone_cutter(
                f"countersink_{index}",
                center=center,
                bottom_diameter=params.holes_dia,
                top_diameter=top_dia,
                z_min=0.0,
                z_max=params.base_thickness + 1.0,
                vertices=96,
            )
        )
    return cutters


def _groove_cutters(params: RouterJigParams) -> list[bpy.types.Object]:
    full_recess_start = (0.0, 0.0)
    full_recess_end = (0.0, -params.extension_length + params.extension_frame + params.inlay_width / 2.0)
    narrow_start = (0.0, -params.inner_dia / 2.0 - params.extension_frame)
    narrow_end = (0.0, -params.extension_length + params.extension_frame + params.inlay_groove / 2.0)
    return [
        add_capsule_prism(
            "wide_inlay_recess_cutter",
            start=full_recess_start,
            end=full_recess_end,
            diameter=params.inlay_width,
            z_min=params.base_thickness - params.inlay_thickness,
            z_max=params.base_thickness + 0.05,
            segments_per_cap=32,
        ),
        add_capsule_prism(
            "narrow_bottom_slot_cutter",
            start=narrow_start,
            end=narrow_end,
            diameter=params.inlay_groove,
            z_min=-0.05,
            z_max=params.inlay_thickness,
            segments_per_cap=28,
        ),
    ]


def build_model() -> list[bpy.types.Object]:
    params = RouterJigParams()
    plate = _extension_plate(params)

    cutters = _countersink_cutters(params)
    cutters.extend(_groove_cutters(params))
    boolean_difference(plate, cutters)

    return [plate]


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

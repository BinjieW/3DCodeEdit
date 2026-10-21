from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the parametric pulley model."""




Point3 = tuple[float, float, float]
Point2 = tuple[float, float]

MXL_TOOTH_PROFILE: tuple[Point2, ...] = (
    (-0.660421, -0.5),
    (-0.660421, 0.0),
    (-0.621898, 0.006033),
    (-0.587714, 0.023037),
    (-0.560056, 0.049424),
    (-0.541182, 0.083609),
    (-0.417357, 0.424392),
    (-0.398413, 0.458752),
    (-0.370649, 0.48514),
    (-0.336324, 0.502074),
    (-0.297744, 0.508035),
    (0.297744, 0.508035),
    (0.336268, 0.502074),
    (0.370452, 0.48514),
    (0.39811, 0.458752),
    (0.416983, 0.424392),
    (0.540808, 0.083609),
    (0.559752, 0.049424),
    (0.587516, 0.023037),
    (0.621841, 0.006033),
    (0.660421, 0.0),
    (0.660421, -0.5),
)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    existing = bpy.data.materials.get(name)
    if existing:
        return existing

    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        base_color = bsdf.inputs.get("Base Color")
        roughness = bsdf.inputs.get("Roughness")
        metallic = bsdf.inputs.get("Metallic")
        if base_color:
            base_color.default_value = color
        if roughness:
            roughness.default_value = 0.62
        if metallic:
            metallic.default_value = 0.0
    return mat


def pulley_yellow() -> bpy.types.Material:
    return material("openscad_preview_yellow", (1.0, 0.77, 0.02, 1.0))


def cut_green() -> bpy.types.Material:
    return material("openscad_cut_green", (0.47, 0.68, 0.22, 1.0))


def cutter_material() -> bpy.types.Material:
    return material("boolean_cut_surface_green", (0.47, 0.68, 0.22, 1.0))


def assign_material(obj: bpy.types.Object, mat: bpy.types.Material | None) -> bpy.types.Object:
    if mat:
        obj.data.materials.append(mat)
    return obj


def add_box(
    name: str,
    size: Point3,
    location: Point3,
    mat: bpy.types.Material | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return assign_material(obj, mat)


def _ring_vertices(
    vertices: list[Point3],
    radius: float,
    z: float,
    radial_segments: int,
) -> list[int]:
    ring: list[int] = []
    for index in range(radial_segments):
        angle = math.tau * index / radial_segments
        ring.append(len(vertices))
        vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
    return ring


def _build_mesh_object(
    name: str,
    vertices: list[Point3],
    faces: list[tuple[int, ...]],
    materials: list[bpy.types.Material],
    material_indices: list[int],
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for mat in materials:
        mesh.materials.append(mat)
    for polygon, material_index in zip(mesh.polygons, material_indices):
        polygon.material_index = material_index

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def add_revolved_ring(
    name: str,
    outer_profile: list[tuple[float, float]],
    inner_radius: float,
    mat: bpy.types.Material,
    *,
    inner_mat: bpy.types.Material | None = None,
    radial_segments: int = 96,
) -> bpy.types.Object:
    """Create a solid of revolution with one straight central bore.

    ``outer_profile`` is a bottom-to-top list of ``(z, radius)`` points. The
    helper keeps the central shaft hole explicit instead of relying on a later
    boolean, which mirrors the source's full-height motor-shaft subtraction.
    """

    if len(outer_profile) < 2:
        raise ValueError("outer_profile requires at least two points")
    if inner_radius <= 0:
        raise ValueError("inner_radius must be positive")

    inner_mat = inner_mat or mat
    vertices: list[Point3] = []
    faces: list[tuple[int, ...]] = []
    material_indices: list[int] = []

    outer_rings = [
        _ring_vertices(vertices, radius, z, radial_segments) for z, radius in outer_profile
    ]
    inner_bottom = _ring_vertices(vertices, inner_radius, outer_profile[0][0], radial_segments)
    inner_top = _ring_vertices(vertices, inner_radius, outer_profile[-1][0], radial_segments)

    for lower, upper in zip(outer_rings, outer_rings[1:]):
        for index in range(radial_segments):
            next_index = (index + 1) % radial_segments
            faces.append((lower[index], lower[next_index], upper[next_index], upper[index]))
            material_indices.append(0)

    outer_bottom = outer_rings[0]
    outer_top = outer_rings[-1]
    for index in range(radial_segments):
        next_index = (index + 1) % radial_segments
        faces.append((outer_bottom[next_index], outer_bottom[index], inner_bottom[index], inner_bottom[next_index]))
        material_indices.append(0)
        faces.append((outer_top[index], outer_top[next_index], inner_top[next_index], inner_top[index]))
        material_indices.append(0)
        faces.append((inner_bottom[next_index], inner_bottom[index], inner_top[index], inner_top[next_index]))
        material_indices.append(1)

    return _build_mesh_object(name, vertices, faces, [mat, inner_mat], material_indices)


def _nearest_tooth_delta(angle: float, tooth_count: int, phase: float) -> float:
    pitch = math.tau / tooth_count
    return ((angle - phase + pitch * 0.5) % pitch) - pitch * 0.5


def _point_on_segment(
    point: Point2,
    start: Point2,
    end: Point2,
    *,
    epsilon: float = 1e-9,
) -> bool:
    px, py = point
    ax, ay = start
    bx, by = end
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > epsilon:
        return False

    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    if dot < -epsilon:
        return False

    squared_length = (bx - ax) ** 2 + (by - ay) ** 2
    return dot <= squared_length + epsilon


def _point_in_polygon(point: Point2, polygon: tuple[Point2, ...]) -> bool:
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(point, start, end):
            return True

    px, py = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if (current_y > py) != (previous_y > py):
            crossing_x = (previous_x - current_x) * (py - current_y) / (previous_y - current_y) + current_x
            if px < crossing_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _inside_scaled_tooth_profile(
    radius: float,
    delta: float,
    *,
    tooth_distance_from_center: float,
    tooth_width_scale: float,
    tooth_depth_scale: float,
    tooth_profile: tuple[Point2, ...],
) -> bool:
    # Matches translate([0, -tooth_distance, z]) scale([width, depth, 1])
    # around the source cutter before the per-tooth Z rotation.
    scaled_x = radius * math.sin(delta)
    scaled_y = tooth_distance_from_center - radius * math.cos(delta)
    return _point_in_polygon(
        (scaled_x / tooth_width_scale, scaled_y / tooth_depth_scale),
        tooth_profile,
    )


def _source_profile_radius_at(
    angle: float,
    *,
    outer_radius: float,
    tooth_count: int,
    tooth_distance_from_center: float,
    tooth_width_scale: float,
    tooth_depth_scale: float,
    tooth_profile: tuple[Point2, ...],
    phase: float,
) -> tuple[float, bool]:
    delta = _nearest_tooth_delta(angle, tooth_count, phase)
    max_profile_x = max(abs(point[0]) for point in tooth_profile) * tooth_width_scale
    max_cut_angle = math.asin(min(0.999, max_profile_x / outer_radius)) + 1e-6
    if abs(delta) > max_cut_angle:
        return outer_radius, False

    def inside(radius: float) -> bool:
        return _inside_scaled_tooth_profile(
            radius,
            delta,
            tooth_distance_from_center=tooth_distance_from_center,
            tooth_width_scale=tooth_width_scale,
            tooth_depth_scale=tooth_depth_scale,
            tooth_profile=tooth_profile,
        )

    if not inside(outer_radius):
        return outer_radius, False

    low = 0.0
    high = outer_radius
    for _ in range(48):
        middle = (low + high) * 0.5
        if inside(middle):
            high = middle
        else:
            low = middle
    return high, True


def add_toothed_ring(
    name: str,
    *,
    inner_radius: float,
    outer_radius: float,
    z_min: float,
    z_max: float,
    tooth_count: int,
    tooth_distance_from_center: float,
    tooth_width_scale: float,
    tooth_depth_scale: float,
    tooth_profile: tuple[Point2, ...],
    mat: bpy.types.Material,
    groove_mat: bpy.types.Material,
    phase: float = 0.0,
    angular_samples: int = 640,
) -> bpy.types.Object:
    """Build the pulley band from the source OpenSCAD tooth cutter profile."""

    vertices: list[Point3] = []
    outer_bottom: list[int] = []
    outer_top: list[int] = []
    inner_bottom: list[int] = []
    inner_top: list[int] = []
    radii: list[float] = []
    cut_flags: list[bool] = []

    for index in range(angular_samples):
        angle = math.tau * index / angular_samples
        radius, cut_by_tooth = _source_profile_radius_at(
            angle,
            outer_radius=outer_radius,
            tooth_count=tooth_count,
            tooth_distance_from_center=tooth_distance_from_center,
            tooth_width_scale=tooth_width_scale,
            tooth_depth_scale=tooth_depth_scale,
            tooth_profile=tooth_profile,
            phase=phase,
        )
        radii.append(radius)
        cut_flags.append(cut_by_tooth)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        outer_bottom.append(len(vertices))
        vertices.append((x, y, z_min))
        outer_top.append(len(vertices))
        vertices.append((x, y, z_max))

        ix = inner_radius * math.cos(angle)
        iy = inner_radius * math.sin(angle)
        inner_bottom.append(len(vertices))
        vertices.append((ix, iy, z_min))
        inner_top.append(len(vertices))
        vertices.append((ix, iy, z_max))

    faces: list[tuple[int, ...]] = []
    material_indices: list[int] = []
    for index in range(angular_samples):
        next_index = (index + 1) % angular_samples
        groove_face = cut_flags[index] or cut_flags[next_index] or min(radii[index], radii[next_index]) < outer_radius - 1e-4

        faces.append((outer_bottom[index], outer_bottom[next_index], outer_top[next_index], outer_top[index]))
        material_indices.append(1 if groove_face else 0)
        faces.append((outer_bottom[next_index], outer_bottom[index], inner_bottom[index], inner_bottom[next_index]))
        material_indices.append(0)
        faces.append((outer_top[index], outer_top[next_index], inner_top[next_index], inner_top[index]))
        material_indices.append(0)
        faces.append((inner_bottom[next_index], inner_bottom[index], inner_top[index], inner_top[next_index]))
        material_indices.append(1)

    return _build_mesh_object(name, vertices, faces, [mat, groove_mat], material_indices)


def add_regular_prism_x(
    name: str,
    *,
    radius: float,
    depth: float,
    center: Point3,
    sides: int,
    rotation: float = 0.0,
    mat: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Create an n-sided prism whose axis runs along the X axis."""

    cx, cy, cz = center
    x_values = (cx - depth / 2.0, cx + depth / 2.0)
    vertices: list[Point3] = []
    rings: list[list[int]] = []
    for x in x_values:
        ring: list[int] = []
        for index in range(sides):
            angle = rotation + math.tau * index / sides
            ring.append(len(vertices))
            vertices.append((x, cy + radius * math.cos(angle), cz + radius * math.sin(angle)))
        rings.append(ring)

    left, right = rings
    faces: list[tuple[int, ...]] = []
    material_indices: list[int] = []
    for index in range(sides):
        next_index = (index + 1) % sides
        faces.append((left[index], left[next_index], right[next_index], right[index]))
        material_indices.append(0)
    faces.append(tuple(reversed(left)))
    material_indices.append(0)
    faces.append(tuple(right))
    material_indices.append(0)

    return _build_mesh_object(name, vertices, faces, [mat or cutter_material()], material_indices)


def add_radial_prism_cavity_surface(
    name: str,
    *,
    radius: float,
    center_z: float,
    x_min: float,
    x_max: float,
    sides: int,
    rotation: float,
    mat: bpy.types.Material,
) -> bpy.types.Object:
    """Create only the interior wall of an X-axis prism cut for clear review renders."""

    vertices: list[Point3] = []
    rings: list[list[int]] = []
    for x in (x_min, x_max):
        ring: list[int] = []
        for index in range(sides):
            angle = rotation + math.tau * index / sides
            ring.append(len(vertices))
            vertices.append((x, radius * math.cos(angle), center_z + radius * math.sin(angle)))
        rings.append(ring)

    faces: list[tuple[int, ...]] = []
    material_indices: list[int] = []
    left, right = rings
    for index in range(sides):
        next_index = (index + 1) % sides
        faces.append((left[index], left[next_index], right[next_index], right[index]))
        material_indices.append(0)

    return _build_mesh_object(name, vertices, faces, [mat], material_indices)


def boolean_difference(
    target: bpy.types.Object,
    cutters: Iterable[bpy.types.Object],
    *,
    keep_cutters: bool = False,
) -> None:
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    for cutter in cutters:
        modifier = target.modifiers.new(f"cut_{cutter.name}", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        modifier.solver = "EXACT"
        modifier.material_mode = "TRANSFER"
        try:
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        except RuntimeError:
            modifier.solver = "FLOAT"
            bpy.ops.object.modifier_apply(modifier=modifier.name)

        if keep_cutters:
            cutter.display_type = "WIRE"
            cutter.hide_viewport = True
            cutter.hide_render = True
        else:
            bpy.data.objects.remove(cutter, do_unlink=True)
    target.select_set(False)


# ============================================================
# Procedural assembly
# ============================================================

"""Readable assembly for the parametric pulley model."""






@dataclass(frozen=True)
class PulleyParameters:
    teeth: int = 20
    tooth_pitch: float = 2.032
    pitch_line_offset: float = 0.254
    tooth_depth: float = 0.508
    tooth_width: float = 1.321
    additional_tooth_width: float = 0.2
    additional_tooth_depth: float = 0.0
    base_height: float = 8.0
    base_diameter: float = 25.0
    toothed_height: float = 12.0
    lower_retainer_height: float = 1.0
    lower_retainer_flat: float = 1.0
    upper_retainer_height: float = 1.0
    upper_retainer_flat: float = 1.0
    motor_shaft_diameter: float = 5.2
    nut_shaft_distance: float = 1.2
    m3_diameter: float = 3.2
    m3_nut_flats: float = 5.7
    m3_nut_depth: float = 2.7

    @property
    def pulley_od(self) -> float:
        return 2.0 * ((self.teeth * self.tooth_pitch) / (math.pi * 2.0) - self.pitch_line_offset)

    @property
    def pulley_radius(self) -> float:
        return self.pulley_od / 2.0

    @property
    def bore_radius(self) -> float:
        return self.motor_shaft_diameter / 2.0

    @property
    def lower_retainer_z_min(self) -> float:
        return self.base_height

    @property
    def lower_retainer_z_max(self) -> float:
        return self.base_height + self.lower_retainer_height + self.lower_retainer_flat

    @property
    def tooth_z_min(self) -> float:
        return self.lower_retainer_z_max

    @property
    def tooth_z_max(self) -> float:
        return self.tooth_z_min + self.toothed_height

    @property
    def upper_retainer_z_max(self) -> float:
        return self.tooth_z_max + self.upper_retainer_height + self.upper_retainer_flat

    @property
    def adjusted_tooth_width(self) -> float:
        return self.tooth_width + self.additional_tooth_width

    @property
    def adjusted_tooth_depth(self) -> float:
        return self.tooth_depth + self.additional_tooth_depth

    @property
    def tooth_width_scale(self) -> float:
        return self.adjusted_tooth_width / self.tooth_width

    @property
    def tooth_depth_scale(self) -> float:
        return self.adjusted_tooth_depth / self.tooth_depth

    @property
    def tooth_distance_from_center(self) -> float:
        half_width = self.adjusted_tooth_width / 2.0
        return math.sqrt(self.pulley_radius**2 - half_width**2)

    @property
    def nut_points_diameter(self) -> float:
        return 2.0 * ((self.m3_nut_flats / 2.0) / math.cos(math.radians(30.0)))


def _rounded_base_profile(params: PulleyParameters, arc_steps: int = 10) -> list[tuple[float, float]]:
    radius = params.base_diameter / 2.0
    roundover_radius = 1.0
    shoulder_radius = radius - roundover_radius
    shoulder_z = params.base_height - roundover_radius

    profile = [(0.0, radius), (shoulder_z, radius)]
    for step in range(1, arc_steps + 1):
        theta = (math.pi * 0.5) * step / arc_steps
        profile.append(
            (
                shoulder_z + roundover_radius * math.sin(theta),
                shoulder_radius + roundover_radius * math.cos(theta),
            )
        )
    return profile


def _captive_nut_cutters(params: PulleyParameters) -> list[bpy.types.Object]:
    cut_mat = cutter_material()
    nut_center_x = params.bore_radius + params.m3_nut_depth / 2.0 + params.nut_shaft_distance
    nut_elevation = params.base_height / 2.0

    screw = add_regular_prism_x(
        "cutter_radial_m3_grub_screw",
        radius=params.m3_diameter / 2.0,
        depth=params.base_diameter / 2.0 + 4.0,
        center=((params.base_diameter / 2.0 + 4.0) / 2.0 - 1.0, 0.0, nut_elevation),
        sides=8,
        rotation=math.radians(22.5),
        mat=cut_mat,
    )

    nut = add_regular_prism_x(
        "cutter_hex_captive_nut",
        radius=params.nut_points_diameter / 2.0,
        depth=params.m3_nut_depth,
        center=(nut_center_x, 0.0, nut_elevation + 0.25),
        sides=6,
        rotation=math.radians(30.0),
        mat=cut_mat,
    )

    entrance = add_box(
        "cutter_bottom_nut_entrance",
        (params.m3_nut_depth, params.m3_nut_flats, params.base_height / 2.0 + 1.0),
        (nut_center_x, 0.0, nut_elevation - params.base_height / 4.0 - 0.5),
        cut_mat,
    )

    return [screw, nut, entrance]


def _add_visible_grub_screw_cavity(params: PulleyParameters, mat: bpy.types.Material) -> bpy.types.Object:
    return add_radial_prism_cavity_surface(
        "visible_octagonal_grub_screw_cut_wall",
        radius=params.m3_diameter / 2.0,
        center_z=params.base_height / 2.0,
        x_min=params.bore_radius - 0.05,
        x_max=params.base_diameter / 2.0 + 0.03,
        sides=8,
        rotation=math.radians(22.5),
        mat=mat,
    )


def build_model() -> list[bpy.types.Object]:
    params = PulleyParameters()
    yellow = pulley_yellow()
    green = cut_green()

    base = add_revolved_ring(
        "rounded_pulley_base_with_shaft_bore",
        _rounded_base_profile(params),
        params.bore_radius,
        yellow,
        inner_mat=green,
        radial_segments=96,
    )
    boolean_difference(base, _captive_nut_cutters(params))
    screw_cavity = _add_visible_grub_screw_cavity(params, green)

    lower_retainer = add_revolved_ring(
        "lower_belt_retainer_idler_flange",
        [
            (params.lower_retainer_z_min, params.pulley_radius + params.lower_retainer_height),
            (params.lower_retainer_z_min + params.lower_retainer_flat, params.pulley_radius + params.lower_retainer_height),
            (params.lower_retainer_z_max, params.pulley_radius),
        ],
        params.bore_radius,
        yellow,
        inner_mat=green,
        radial_segments=params.teeth * 4,
    )

    toothed_band = add_toothed_ring(
        "twenty_tooth_mxl_pulley_band",
        inner_radius=params.bore_radius,
        outer_radius=params.pulley_radius,
        z_min=params.tooth_z_min,
        z_max=params.tooth_z_max,
        tooth_count=params.teeth,
        tooth_distance_from_center=params.tooth_distance_from_center,
        tooth_width_scale=params.tooth_width_scale,
        tooth_depth_scale=params.tooth_depth_scale,
        tooth_profile=MXL_TOOTH_PROFILE,
        mat=yellow,
        groove_mat=green,
        phase=0.0,
        angular_samples=params.teeth * 40,
    )

    upper_retainer = add_revolved_ring(
        "upper_belt_retainer_flange",
        [
            (params.tooth_z_max, params.pulley_radius),
            (params.tooth_z_max + params.upper_retainer_height, params.pulley_radius + params.upper_retainer_height),
            (params.upper_retainer_z_max, params.pulley_radius + params.upper_retainer_height),
        ],
        params.bore_radius,
        yellow,
        inner_mat=green,
        radial_segments=params.teeth * 4,
    )

    return [base, screw_cavity, lower_retainer, toothed_band, upper_retainer]


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

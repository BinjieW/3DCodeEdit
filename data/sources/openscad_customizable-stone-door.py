from __future__ import annotations
from dataclasses import dataclass
from math import atan, cos, degrees, radians, sin, sqrt
from math import pow, tan, radians
import bmesh
import bpy


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the customizable stone door model."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.62


STONE_GRAY = MaterialSpec("stone_gray", (0.53, 0.53, 0.53, 1.0), roughness=0.72)
DOOR_BROWN = MaterialSpec("door_brown", (0.50, 0.28, 0.15, 1.0), roughness=0.78)
HANDLE_BLACK = MaterialSpec("handle_black", (0.05, 0.05, 0.05, 1.0), roughness=0.45)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def get_material(spec: MaterialSpec) -> bpy.types.Material:
    material = bpy.data.materials.get(spec.name) or bpy.data.materials.new(spec.name)
    material.use_nodes = True
    material.diffuse_color = spec.color
    nodes = material.node_tree.nodes
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new(type="ShaderNodeOutputMaterial")
    shader = nodes.new(type="ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = spec.color
    shader.inputs["Roughness"].default_value = spec.roughness
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def assign_material(obj: bpy.types.Object, spec: MaterialSpec) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(get_material(spec))
    return obj


def _apply_bevel(
    obj: bpy.types.Object,
    width: float,
    segments: int = 2,
    smooth_bevels: bool = False,
) -> bpy.types.Object:
    if width <= 0.0:
        return obj
    if hasattr(obj.data, "use_auto_smooth"):
        obj.data.use_auto_smooth = True
    bevel = obj.modifiers.new(name="Bevel", type="BEVEL")
    bevel.width = width
    bevel.segments = segments
    bevel.limit_method = "NONE"
    bevel.harden_normals = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    for polygon in obj.data.polygons:
        polygon.use_smooth = smooth_bevels
    normal = obj.modifiers.new(name="Weighted seam normals", type="WEIGHTED_NORMAL")
    normal.keep_sharp = True
    normal.weight = 50
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=normal.name)
    obj.select_set(False)
    return obj


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: MaterialSpec | None = None,
    bevel_width: float = 0.0,
    bevel_segments: int = 2,
    smooth_bevels: bool = False,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    _apply_bevel(obj, bevel_width, bevel_segments, smooth_bevels=smooth_bevels)
    if material is not None:
        assign_material(obj, material)
    return obj


def add_hull_rounded_block(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    radius: float,
    material: MaterialSpec | None = None,
    sphere_subdivisions: int = 1,
) -> bpy.types.Object:
    """Build a rounded block using the hull of eight corner spheres."""
    half_x = size[0] / 2.0
    half_y = size[1] / 2.0
    half_z = size[2] / 2.0
    core_x = max(0.0, half_x - radius)
    core_y = max(0.0, half_y - radius)
    core_z = max(0.0, half_z - radius)

    bm = bmesh.new()
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                result = bmesh.ops.create_icosphere(bm, subdivisions=sphere_subdivisions, radius=radius)
                bmesh.ops.translate(
                    bm,
                    verts=result["verts"],
                    vec=(sx * core_x, sy * core_y, sz * core_z),
                )
    hull = bmesh.ops.convex_hull(bm, input=bm.verts)
    for key in ("geom_interior", "geom_unused", "geom_holes"):
        geom = hull.get(key)
        if geom:
            bmesh.ops.delete(bm, geom=geom, context="VERTS")

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    normal = obj.modifiers.new(name="Hull stone normals", type="WEIGHTED_NORMAL")
    normal.keep_sharp = True
    normal.weight = 35
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=normal.name)
    obj.select_set(False)
    if material is not None:
        assign_material(obj, material)
    return obj


def _make_convex_hull_from_spheres(
    name: str,
    centers: list[tuple[float, float, float]],
    radius: float,
    material: MaterialSpec | None,
    sphere_subdivisions: int = 2,
) -> bpy.types.Object:
    bm = bmesh.new()
    for center in centers:
        result = bmesh.ops.create_icosphere(bm, subdivisions=sphere_subdivisions, radius=radius)
        bmesh.ops.translate(bm, verts=result["verts"], vec=center)
    hull = bmesh.ops.convex_hull(bm, input=bm.verts)
    for key in ("geom_interior", "geom_unused", "geom_holes"):
        geom = hull.get(key)
        if geom:
            bmesh.ops.delete(bm, geom=geom, context="VERTS")

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    normal = obj.modifiers.new(name="Hull wedge normals", type="WEIGHTED_NORMAL")
    normal.keep_sharp = True
    normal.weight = 40
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=normal.name)
    obj.select_set(False)
    if material is not None:
        assign_material(obj, material)
    return obj


def add_extruded_profile(
    name: str,
    profile_xz: list[tuple[float, float]],
    depth: float,
    material: MaterialSpec | None = None,
    bevel_width: float = 0.0,
    bevel_segments: int = 2,
    smooth_bevels: bool = False,
) -> bpy.types.Object:
    """Create a prism from an x-z outline extruded along Blender's y axis."""
    if len(profile_xz) < 3:
        raise ValueError("profile_xz needs at least three points")

    half_depth = depth / 2.0
    vertices: list[tuple[float, float, float]] = []
    for y in (-half_depth, half_depth):
        for x, z in profile_xz:
            vertices.append((x, y, z))

    count = len(profile_xz)
    faces: list[tuple[int, ...]] = [
        tuple(range(count - 1, -1, -1)),
        tuple(range(count, count * 2)),
    ]
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, next_index + count, index + count))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    _apply_bevel(obj, bevel_width, bevel_segments, smooth_bevels=smooth_bevels)
    if material is not None:
        assign_material(obj, material)
    return obj


def add_prismoid(
    name: str,
    top_size: tuple[float, float],
    bottom_size: tuple[float, float],
    height: float,
    location: tuple[float, float, float],
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Create a rectangular frustum with independent top and bottom footprints."""
    top_half_x = top_size[0] / 2.0
    top_half_y = top_size[1] / 2.0
    bottom_half_x = bottom_size[0] / 2.0
    bottom_half_y = bottom_size[1] / 2.0
    half_height = height / 2.0

    vertices = [
        (-bottom_half_x, -bottom_half_y, -half_height),
        (bottom_half_x, -bottom_half_y, -half_height),
        (bottom_half_x, bottom_half_y, -half_height),
        (-bottom_half_x, bottom_half_y, -half_height),
        (-top_half_x, -top_half_y, half_height),
        (top_half_x, -top_half_y, half_height),
        (top_half_x, top_half_y, half_height),
        (-top_half_x, top_half_y, half_height),
    ]
    faces = [
        (0, 1, 2, 3),
        (7, 6, 5, 4),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    if material is not None:
        assign_material(obj, material)
    return obj


def add_wedge_segment(
    name: str,
    inner_radius: float,
    outer_radius: float,
    start_degrees: float,
    end_degrees: float,
    center_z: float,
    depth: float,
    material: MaterialSpec | None = None,
    bevel_width: float = 0.0,
    bevel_segments: int = 5,
    arc_segments: int = 8,
    smooth_bevels: bool = False,
) -> bpy.types.Object:
    if arc_segments < 2:
        raise ValueError("arc_segments must be at least 2")

    profile: list[tuple[float, float]] = []
    for step in range(arc_segments + 1):
        angle = radians(start_degrees + (end_degrees - start_degrees) * step / arc_segments)
        profile.append((outer_radius * cos(angle), center_z + outer_radius * sin(angle)))
    for step in range(arc_segments, -1, -1):
        angle = radians(start_degrees + (end_degrees - start_degrees) * step / arc_segments)
        profile.append((inner_radius * cos(angle), center_z + inner_radius * sin(angle)))
    return add_extruded_profile(
        name=name,
        profile_xz=profile,
        depth=depth,
        material=material,
        bevel_width=bevel_width,
        bevel_segments=bevel_segments,
        smooth_bevels=smooth_bevels,
    )


def add_scad_wedge_block(
    name: str,
    inner_radius: float,
    width: float,
    theta_degrees: float,
    rotation_degrees: float,
    center_z: float,
    depth: float,
    curvature: float,
    material: MaterialSpec | None = None,
    seam_gap_degrees: float = 0.0,
) -> bpy.types.Object:
    """Approximate SCAD's wedge() hull corner construction.

    The source places the hull spheres at corrected inner/outer angular offsets
    before rotating each wedge. This keeps the side seams radial while the
    rounded corners stay inside the requested angular span.
    """
    outer_radius = inner_radius + width
    start = seam_gap_degrees / 2.0
    end = theta_degrees - seam_gap_degrees / 2.0
    phi_outer = degrees(atan(curvature / max(0.001, outer_radius - curvature)))
    phi_inner = degrees(atan(curvature / (inner_radius + curvature)))

    outer_start = radians(start + phi_outer)
    outer_end = radians(end - phi_outer)
    inner_end = radians(end - phi_inner)
    inner_start = radians(start + phi_inner)

    profile = [
        ((outer_radius - curvature) * cos(outer_start), center_z + (outer_radius - curvature) * sin(outer_start)),
        ((outer_radius - curvature) * cos(outer_end), center_z + (outer_radius - curvature) * sin(outer_end)),
        ((inner_radius + curvature) * cos(inner_end), center_z + (inner_radius + curvature) * sin(inner_end)),
        ((inner_radius + curvature) * cos(inner_start), center_z + (inner_radius + curvature) * sin(inner_start)),
    ]
    obj = add_extruded_profile(
        name=name,
        profile_xz=profile,
        depth=max(0.1, depth - 2.0 * curvature),
        material=material,
        bevel_width=curvature,
        bevel_segments=3,
        smooth_bevels=False,
    )
    obj.rotation_euler[1] = 0.0
    obj.rotation_euler[2] = 0.0
    obj.rotation_euler[0] = 0.0
    obj.name = name
    if rotation_degrees:
        # The profile is already in x/z space, so rotate around the arch center
        # by rebuilding via object origin at world zero is not appropriate here.
        # Coordinates are emitted with rotation baked below instead.
        angle = radians(rotation_degrees)
        for vertex in obj.data.vertices:
            x, y, z = vertex.co.x, vertex.co.y, vertex.co.z - center_z
            vertex.co.x = x * cos(angle) - z * sin(angle)
            vertex.co.z = center_z + x * sin(angle) + z * cos(angle)
        obj.data.update()
    return obj


def add_scad_hull_wedge_block(
    name: str,
    inner_radius: float,
    width: float,
    theta_degrees: float,
    rotation_degrees: float,
    center_z: float,
    depth: float,
    curvature: float,
    material: MaterialSpec | None = None,
    seam_gap_degrees: float = 0.0,
    sphere_subdivisions: int = 2,
) -> bpy.types.Object:
    """Build the source wedge as a convex hull of eight rounded corner spheres."""
    outer_radius = inner_radius + width
    radius = curvature
    start = seam_gap_degrees / 2.0
    end = theta_degrees - seam_gap_degrees / 2.0
    phi_outer = degrees(atan(radius / max(0.001, outer_radius - radius)))
    phi_inner = degrees(atan(radius / (inner_radius + radius)))
    half_depth_core = max(0.0, depth - 2.0 * radius) / 2.0

    local_points = []
    for angle_degrees, radial in (
        (end - phi_outer, outer_radius - radius),
        (end - phi_inner, inner_radius + radius),
        (start + phi_inner, inner_radius + radius),
        (start + phi_outer, outer_radius - radius),
    ):
        angle = radians(angle_degrees + rotation_degrees)
        x = radial * cos(angle)
        arch_y = radial * sin(angle)
        for depth_sign in (-1.0, 1.0):
            local_points.append((x, -depth_sign * half_depth_core, center_z + arch_y))

    return _make_convex_hull_from_spheres(
        name=name,
        centers=local_points,
        radius=radius,
        material=material,
        sphere_subdivisions=sphere_subdivisions,
    )


def profile_rectangle(
    half_width: float,
    bottom_z: float,
    top_z: float,
) -> list[tuple[float, float]]:
    return [
        (-half_width, bottom_z),
        (half_width, bottom_z),
        (half_width, top_z),
        (-half_width, top_z),
    ]


def profile_trapezoid(
    bottom_half_width: float,
    top_half_width: float,
    bottom_z: float,
    top_z: float,
) -> list[tuple[float, float]]:
    return [
        (-bottom_half_width, bottom_z),
        (bottom_half_width, bottom_z),
        (top_half_width, top_z),
        (-top_half_width, top_z),
    ]


def profile_cylinder_rectangle_door(
    half_width: float,
    rect_top_z: float,
    radius: float,
    circle_center_z: float,
    segments: int = 28,
) -> list[tuple[float, float]]:
    """Door outline for the source cylinder(radius=12) union cube(width=22)."""
    shoulder_z = circle_center_z + sqrt(max(0.0, radius * radius - half_width * half_width))
    profile = [(-half_width, 0.0), (half_width, 0.0), (half_width, shoulder_z)]
    start_angle = degrees(atan((shoulder_z - circle_center_z) / half_width))
    for step in range(1, segments):
        angle = radians(start_angle + (180.0 - 2.0 * start_angle) * step / segments)
        x = radius * cos(angle)
        z = circle_center_z + radius * sin(angle)
        profile.append((x, z))
    profile.append((-half_width, shoulder_z))
    profile.append((-half_width, rect_top_z))
    return profile


# ============================================================
# Procedural assembly
# ============================================================

"""Procedural assembly for the customizable stone door model."""





ARCH_NUM = 7
BLOCK_WIDTH = 12.0
BLOCK_THICKNESS = 10.0
DOOR_WIDTH = 20.0
BLOCK_HEIGHT = 6.0
BLOCKS_HIGH = 5
CURVE = 1.0
DOOR_THICKNESS = 3.0
BASE_SCALE = 0.92
BASE_ANGLE_DEGREES = 70.0


def stone_block(
    name: str,
    width: float,
    height: float,
    thickness: float,
    bevel: float,
    location: tuple[float, float, float],
):
    return add_hull_rounded_block(
        name=name,
        size=(width, thickness, height),
        location=location,
        material=STONE_GRAY,
        radius=bevel,
        sphere_subdivisions=3,
    )


def build_base() -> object:
    height = BASE_SCALE * BLOCK_HEIGHT
    top_width = DOOR_WIDTH + 2.0 * BLOCK_WIDTH
    top_depth = BLOCK_THICKNESS
    base_adj = height / max(0.01, tan(radians(BASE_ANGLE_DEGREES)))
    base = add_prismoid(
        name="stone_base",
        top_size=(top_width, top_depth),
        bottom_size=(top_width + 1.65 * base_adj, top_depth + 0.85 * base_adj),
        height=height,
        location=(0.0, 0.0, -height / 2.0 + 0.001),
        material=STONE_GRAY,
    )
    return base


def build_side_stack(x_sign: float) -> list:
    parts = []
    x = x_sign * (DOOR_WIDTH / 2.0 + BLOCK_WIDTH / 2.0)
    for index in range(BLOCKS_HIGH):
        z = (index + 0.5) * BLOCK_HEIGHT
        part = stone_block(
            name=f"side_block_{'r' if x_sign > 0 else 'l'}_{index}",
            width=BLOCK_WIDTH,
            height=BLOCK_HEIGHT + 0.002,
            thickness=BLOCK_THICKNESS,
            bevel=CURVE,
            location=(x, 0.0, z),
        )
        parts.append(part)
    return parts


def build_arch() -> list:
    parts = []
    inner_radius = DOOR_WIDTH / 2.0
    center_z = BLOCKS_HIGH * BLOCK_HEIGHT
    seam_degrees = 0.72
    for index in range(ARCH_NUM):
        theta = 180.0 / ARCH_NUM
        part = add_scad_hull_wedge_block(
            name=f"arch_block_{index}",
            inner_radius=inner_radius,
            width=BLOCK_WIDTH,
            theta_degrees=theta,
            rotation_degrees=index * theta,
            center_z=center_z,
            depth=BLOCK_THICKNESS,
            curvature=CURVE,
            material=STONE_GRAY,
            seam_gap_degrees=seam_degrees,
            sphere_subdivisions=3,
        )
        parts.append(part)
    return parts


def build_door() -> list:
    parts = []
    half_width = DOOR_WIDTH / 2.0 + CURVE
    circle_center_z = BLOCKS_HIGH * BLOCK_HEIGHT
    handle_thickness = pow(
        (3.0 * DOOR_THICKNESS)
        * (DOOR_THICKNESS + 2.0 * CURVE)
        * (0.8 * BLOCK_THICKNESS + 0.2 * DOOR_THICKNESS),
        0.33,
    )
    profile = profile_cylinder_rectangle_door(
        half_width=half_width,
        rect_top_z=circle_center_z,
        radius=DOOR_WIDTH / 2.0 + 2.0,
        circle_center_z=circle_center_z,
        segments=28,
    )
    door = add_extruded_profile(
        name="wood_door",
        profile_xz=profile,
        depth=DOOR_THICKNESS,
        material=DOOR_BROWN,
        bevel_width=0.12,
        bevel_segments=2,
    )
    door.location = (0.0, -0.5, 0.0)
    parts.append(door)

    handle = add_box(
        name="door_handle",
        size=(DOOR_WIDTH / 10.0, handle_thickness, BLOCK_HEIGHT),
        location=(
            -0.75 * (DOOR_WIDTH / 2.0),
            0.0,
            BLOCK_HEIGHT * BLOCKS_HIGH
            + 0.6 * ((DOOR_WIDTH / 2.0) - (BLOCK_HEIGHT * BLOCKS_HIGH)),
        ),
        material=HANDLE_BLACK,
        bevel_width=0.08,
        bevel_segments=1,
    )
    parts.append(handle)
    return parts


def build_model() -> list:
    parts = [build_base()]
    parts.extend(build_side_stack(-1.0))
    parts.extend(build_side_stack(1.0))
    parts.extend(build_arch())
    parts.extend(build_door())
    return parts


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

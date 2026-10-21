from __future__ import annotations
from mathutils import Vector
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the camera-housing model."""





MaterialMap = dict[str, bpy.types.Material]


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def get_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.55
        bsdf.inputs["Metallic"].default_value = 0.0
    return material


def create_materials() -> MaterialMap:
    return {
        "red": get_material("source red housing", (0.88, 0.02, 0.12, 1.0)),
        "red_dark": get_material("dark red side shadow", (0.45, 0.0, 0.05, 1.0)),
        "green": get_material("source green bracket", (0.0, 0.48, 0.0, 1.0)),
        "black": get_material("matte black camera parts", (0.01, 0.01, 0.01, 1.0)),
        "dark_grey": get_material("dark lens glass", (0.04, 0.04, 0.04, 1.0)),
        "metal": get_material("light grey metal", (0.72, 0.72, 0.70, 1.0)),
        "metal_dark": get_material("dark screw recess", (0.32, 0.32, 0.32, 1.0)),
    }


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(material)
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
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material:
        assign_material(obj, material)
    return obj


def add_cylinder(
    name: str,
    radius: float,
    depth: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
    material: bpy.types.Material | None = None,
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




def add_torus(
    name: str,
    major_radius: float,
    minor_radius: float,
    location: tuple[float, float, float],
    material: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_torus_add(
        major_segments=72,
        minor_segments=12,
        major_radius=major_radius,
        minor_radius=minor_radius,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    assign_material(obj, material)
    shade_smooth(obj)
    return obj


def add_rounded_box(
    name: str,
    size: tuple[float, float, float],
    radius: float,
    location: tuple[float, float, float],
    material: bpy.types.Material,
) -> bpy.types.Object:
    obj = add_box(name, size, location, material)
    bevel = obj.modifiers.new("tiny molded edge radius", "BEVEL")
    bevel.width = min(radius, 0.022)
    bevel.segments = 1
    bevel.affect = "EDGES"
    bevel.profile = 0.28
    obj.modifiers.new("weighted source-like normals", "WEIGHTED_NORMAL")
    return obj


def add_capsule_box(
    name: str,
    size: tuple[float, float, float],
    radius: float,
    location: tuple[float, float, float],
    material: bpy.types.Material,
) -> bpy.types.Object:
    obj = add_box(name, size, location, material)
    bevel = obj.modifiers.new("crisp molded edge break", "BEVEL")
    bevel.width = min(radius, 0.014)
    bevel.segments = 1
    bevel.affect = "EDGES"
    bevel.profile = 0.08
    obj.modifiers.new("crisp slab weighted normals", "WEIGHTED_NORMAL")
    return obj


def add_rounded_yoke_side_plate(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: bpy.types.Material,
    segments: int = 18,
) -> bpy.types.Object:
    thickness_x, width_y, height_z = size
    x, y, z = location
    half_x = thickness_x / 2
    half_y = width_y / 2
    bottom_z = z - height_z / 2
    top_z = z + height_z / 2
    cap_radius = half_y
    cap_center_z = top_z - cap_radius
    profile: list[tuple[float, float]] = [(-half_y, bottom_z), (half_y, bottom_z), (half_y, cap_center_z)]
    for index in range(segments + 1):
        angle = math.radians(index * 180 / segments)
        profile.append((math.cos(angle) * half_y, cap_center_z + math.sin(angle) * cap_radius))
    profile.append((-half_y, bottom_z))

    vertices = []
    for x_offset in (-half_x, half_x):
        vertices.extend((x + x_offset, y + y_offset, z_pos) for y_offset, z_pos in profile)
    count = len(profile)
    faces = [tuple(range(count)), tuple(range(count, count * 2))]
    for index in range(count - 1):
        faces.append((index, index + 1, index + 1 + count, index + count))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    bevel = obj.modifiers.new("continuous molded yoke edge", "BEVEL")
    bevel.width = 0.018
    bevel.segments = 2
    bevel.affect = "EDGES"
    obj.modifiers.new("smooth yoke normals", "WEIGHTED_NORMAL")
    return obj


def add_plate_with_front_slot(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    materials: MaterialMap,
) -> list[bpy.types.Object]:
    x, y, z = center
    width, depth, height = size
    objects: list[bpy.types.Object] = []
    large_plate = width > 5.4
    slab_height = height * (0.88 if large_plate else 0.90)
    slab_center = (x, y, z - height * (0.035 if large_plate else 0.025))
    cap_radius = 0.018 if large_plate else 0.016
    objects.append(add_capsule_box(f"{name}_boxy_red_cover_slab", (width, depth, slab_height), cap_radius, slab_center, materials["red"]))

    if not large_plate:
        side_wall_width = 0.034
        side_wall_depth = depth * 0.50
        rear_lip_depth = 0.065
        edge_height = slab_height * 0.48
        edge_y = y + depth * 0.18
        objects.append(add_box(f"{name}_left_subtle_red_side_edge", (side_wall_width, side_wall_depth, edge_height), (x - width / 2 + side_wall_width / 2, edge_y, slab_center[2] - slab_height * 0.08), materials["red_dark"]))
        objects.append(add_box(f"{name}_right_subtle_red_side_edge", (side_wall_width, side_wall_depth, edge_height), (x + width / 2 - side_wall_width / 2, edge_y, slab_center[2] - slab_height * 0.08), materials["red_dark"]))
        objects.append(add_box(f"{name}_thin_rear_wall_shadow", (width * 0.48, rear_lip_depth, slab_height * 0.46), (x, y + depth / 2 - rear_lip_depth / 2, slab_center[2] - slab_height * 0.07), materials["red_dark"]))

    front_face_y = y - depth / 2
    lower_z = z - height / 2
    opening_width = width * (0.70 if large_plate else 0.65)
    opening_height = slab_height * (0.36 if large_plate else 0.35)
    opening_z = lower_z + height * (0.235 if large_plate else 0.224)
    cut_y = front_face_y - 0.125

    frame_depth = 0.115
    frame_thickness = width * (0.044 if large_plate else 0.047)
    frame_material = materials["red"]
    frame_y = front_face_y + 0.006
    jamb_height = opening_height * 1.42
    jamb_width = frame_thickness * 1.12
    objects.append(add_box(f"{name}_left_tall_integrated_red_slot_leg", (jamb_width, frame_depth, jamb_height), (x - opening_width / 2 + jamb_width / 2, frame_y, opening_z), frame_material))
    objects.append(add_box(f"{name}_right_tall_integrated_red_slot_leg", (jamb_width, frame_depth, jamb_height), (x + opening_width / 2 - jamb_width / 2, frame_y, opening_z), frame_material))
    objects.append(add_box(f"{name}_upper_integrated_red_slot_lip", (opening_width, frame_depth, frame_thickness * 0.92), (x, frame_y, opening_z + opening_height / 2 - frame_thickness * 0.18), frame_material))
    objects.append(add_box(f"{name}_lower_integrated_red_slot_sill", (opening_width, frame_depth, frame_thickness * 0.72), (x, frame_y, opening_z - opening_height / 2 + frame_thickness * 0.22), frame_material))

    black_width = opening_width * (0.74 if large_plate else 0.72)
    black_height = opening_height * 0.50
    objects.append(add_rounded_box(f"{name}_deep_black_cutout_cavity", (black_width, 0.035, black_height), 0.008, (x, cut_y, opening_z - opening_height * 0.03), materials["black"]))

    insert_width = opening_width * (0.68 if large_plate else 0.66)
    insert_height = opening_height * 0.24
    objects.append(add_box(f"{name}_setback_green_bottom_strip_inside_slot", (insert_width, 0.024, insert_height), (x, cut_y - 0.035, opening_z - opening_height * 0.27), materials["green"]))
    return objects

def add_screw_stack(
    name: str,
    location: tuple[float, float, float],
    materials: MaterialMap,
    radius: float = 0.62,
    height: float = 0.70,
) -> list[bpy.types.Object]:
    x, y, z = location
    objects: list[bpy.types.Object] = []
    objects.append(add_cylinder(f"{name}_washer", radius * 1.10, 0.10, (x, y, z), material=materials["metal"], vertices=40))
    objects.append(add_cylinder(f"{name}_head", radius, height, (x, y, z + height / 2), material=materials["metal"], vertices=40))
    objects.append(add_cylinder(f"{name}_inner_hex", radius * 0.32, 0.035, (x, y, z + height + 0.02), material=materials["metal_dark"], vertices=6))
    return objects


def add_under_pin(
    name: str,
    location: tuple[float, float, float],
    materials: MaterialMap,
) -> list[bpy.types.Object]:
    x, y, z = location
    return [
        add_cylinder(f"{name}_lower_post", 0.15, 0.62, (x, y, z - 0.31), material=materials["metal"], vertices=20),
        add_cylinder(f"{name}_bottom_nut", 0.34, 0.14, (x, y, z - 0.69), material=materials["metal"], vertices=6),
    ]


def add_small_camera_module(
    name: str,
    center: tuple[float, float, float],
    materials: MaterialMap,
) -> list[bpy.types.Object]:
    x, y, z = center
    objects: list[bpy.types.Object] = []
    objects.append(add_rounded_box(f"{name}_black_square_mount", (1.02, 0.88, 0.09), 0.008, (x, y, z - 0.02), materials["black"]))
    objects.append(add_cylinder(f"{name}_low_lens_body", 0.25, 0.10, (x, y, z + 0.075), material=materials["black"], vertices=32))
    objects.append(add_cylinder(f"{name}_small_flat_glass", 0.13, 0.020, (x, y, z + 0.135), material=materials["dark_grey"], vertices=24))
    return objects


def add_large_lens(name: str, center: tuple[float, float, float], materials: MaterialMap) -> list[bpy.types.Object]:
    x, y, z = center
    objects: list[bpy.types.Object] = []
    objects.append(add_cylinder(f"{name}_thin_base_flange", 1.05, 0.065, (x, y, z + 0.032), material=materials["black"], vertices=64))
    objects.append(add_cylinder(f"{name}_tall_straight_black_barrel", 0.87, 1.56, (x, y, z + 0.81), material=materials["black"], vertices=64))
    objects.append(add_cylinder(f"{name}_flat_front_lip", 0.88, 0.040, (x, y, z + 1.61), material=materials["black"], vertices=64))
    objects.append(add_cylinder(f"{name}_flat_recessed_front_glass", 0.51, 0.014, (x, y, z + 1.637), material=materials["dark_grey"], vertices=64))
    return objects


def add_rear_hinge_bracket(
    name: str,
    plate_center: tuple[float, float, float],
    plate_width: float,
    plate_depth: float,
    materials: MaterialMap,
) -> list[bpy.types.Object]:
    x, y, z = plate_center
    rear_y = y + plate_depth / 2
    base_z = z - 0.57
    objects: list[bpy.types.Object] = []
    large_plate = plate_width > 5.4

    bracket_span = 1.68 if large_plate else 1.42
    base_size = (bracket_span, 0.82 if large_plate else 0.72, 0.20 if large_plate else 0.18)
    upright_size = (0.40, 0.58, 1.12) if large_plate else (0.34, 0.50, 0.96)
    fork_gap = 0.56 if large_plate else 0.48
    base_y = rear_y + (0.36 if large_plate else 0.31)
    yoke_y = rear_y + 0.075
    fork_center_x = x + bracket_span * (-0.05 if large_plate else 0.06)
    upright_z = base_z + upright_size[2] * 0.50 + 0.28
    pivot_z = base_z + upright_size[2] * 0.68 + 0.21

    objects.append(add_rounded_box(f"{name}_wide_horizontal_green_rear_base_rail", base_size, 0.020, (fork_center_x, base_y, base_z + 0.15), materials["green"]))
    objects.append(add_rounded_box(f"{name}_green_base_rail_front_tab_to_body", (fork_gap + upright_size[0], 0.36, base_size[2] * 0.92), 0.014, (fork_center_x, rear_y + 0.12, base_z + 0.18), materials["green"]))

    near_x = fork_center_x - fork_gap / 2
    far_x = fork_center_x + fork_gap / 2
    for suffix, plate_x in (("near", near_x), ("far", far_x)):
        objects.append(add_rounded_yoke_side_plate(f"{name}_{suffix}_thick_green_rear_yoke_cheek", upright_size, (plate_x, yoke_y, upright_z), materials["green"]))
        objects.append(add_rounded_box(f"{name}_{suffix}_green_yoke_foot_on_base_rail", (upright_size[0] * 1.05, 0.54, upright_size[2] * 0.16), 0.012, (plate_x, rear_y + 0.18, base_z + upright_size[2] * 0.28), materials["green"]))

    bridge_size = (fork_gap + upright_size[0] * 0.45, 0.30, 0.18)
    objects.append(add_rounded_box(f"{name}_green_yoke_lower_bridge_tucked_to_rear", bridge_size, 0.012, (fork_center_x, rear_y + 0.035, base_z + 0.69), materials["green"]))

    disk_radius = 0.31 if large_plate else 0.26
    disk_y = rear_y + 0.010
    for suffix, disk_x, offset in (("near", near_x - upright_size[0] * 0.48, -0.004), ("far", far_x + upright_size[0] * 0.48, 0.004)):
        objects.append(add_cylinder(f"{name}_{suffix}_large_grey_pivot_disk_tucked_to_body", disk_radius, 0.078, (disk_x + offset, disk_y, pivot_z), rotation=(0, math.pi / 2, 0), material=materials["metal"], vertices=56))
        objects.append(add_cylinder(f"{name}_{suffix}_dark_pivot_center", disk_radius * 0.24, 0.086, (disk_x + offset, disk_y, pivot_z), rotation=(0, math.pi / 2, 0), material=materials["metal_dark"], vertices=32))

    bolt_radius = 0.16 if large_plate else 0.135
    bolt_height = 0.12 if large_plate else 0.10
    bolt_y = base_y + 0.04
    for index, bolt_x in enumerate((fork_center_x - bracket_span * 0.28, fork_center_x + bracket_span * 0.28), 1):
        bolt_z = base_z + base_size[2] + 0.16
        objects.append(add_cylinder(f"{name}_base_rail_screw_{index}_washer", bolt_radius * 1.22, 0.052, (bolt_x, bolt_y, bolt_z), material=materials["metal"], vertices=36))
        objects.append(add_cylinder(f"{name}_base_rail_screw_{index}_head", bolt_radius, bolt_height, (bolt_x, bolt_y, bolt_z + bolt_height / 2), material=materials["metal"], vertices=36))
        objects.append(add_cylinder(f"{name}_base_rail_screw_{index}_dark_hex", bolt_radius * 0.34, 0.030, (bolt_x, bolt_y, bolt_z + bolt_height + 0.018), material=materials["metal_dark"], vertices=6))
        objects.extend(add_under_pin(f"{name}_base_rail_screw_{index}", (bolt_x, bolt_y, base_z + 0.09), materials))
    return objects

def add_tiny_status_marks(name: str, plate_center: tuple[float, float, float], materials: MaterialMap) -> list[bpy.types.Object]:
    x, y, z = plate_center
    return [
        add_box(f"{name}_tiny_green_window", (0.20, 0.22, 0.05), (x + 2.05, y + 0.33, z + 0.92), materials["green"]),
        add_box(f"{name}_tiny_yellow_dot", (0.13, 0.12, 0.06), (x + 2.22, y + 0.18, z + 0.95), get_material("small yellow status", (1.0, 0.72, 0.02, 1.0))),
        add_box(f"{name}_tiny_black_slot", (0.35, 0.06, 0.04), (x + 1.78, y - 0.35, z + 0.95), materials["black"]),
    ]


def group_objects(name: str, objects: list[bpy.types.Object]) -> bpy.types.Collection:
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    for obj in objects:
        for existing in obj.users_collection:
            existing.objects.unlink(obj)
        collection.objects.link(obj)
    return collection


# ============================================================
# Procedural assembly
# ============================================================

"""High-level assembly for the camera-housing model."""




def _add_cover_fasteners(name: str, center: tuple[float, float, float], size: tuple[float, float, float], screw_layout: str, materials) -> list:
    x, y, z = center
    width, depth, height = size
    top_z = z + height / 2 + 0.04
    objects = []
    if screw_layout == "large_four":
        points = [
            (x - width / 2 + 1.25, y - depth / 2 + 0.94),
            (x + width / 2 - 1.25, y - depth / 2 + 0.94),
            (x - width / 2 + 1.25, y + depth / 2 - 1.05),
            (x + width / 2 - 1.25, y + depth / 2 - 1.05),
        ]
        radius = 0.31
    else:
        points = [
            (x - width / 2 + 1.00, y - depth / 2 + 0.88),
            (x + width / 2 - 1.00, y - depth / 2 + 0.88),
            (x - width / 2 + 1.00, y + depth / 2 - 0.86),
            (x + width / 2 - 1.00, y + depth / 2 - 0.86),
        ]
        radius = 0.23
    for index, (sx, sy) in enumerate(points, 1):
        objects.extend(add_screw_stack(f"{name}_top_screw_{index}", (sx, sy, top_z), materials, radius=radius, height=0.18))
    return objects


def _build_large_camera_housing(materials) -> list:
    center = (2.88, -0.50, 0.30)
    size = (5.92, 4.92, 2.04)
    objects = []
    objects.extend(add_plate_with_front_slot("large_camera", center, size, materials))
    objects.extend(_add_cover_fasteners("large_camera", center, size, "large_four", materials))
    objects.extend(add_large_lens("large_camera", (center[0], center[1] - 0.05, center[2] + size[2] / 2), materials))
    objects.extend(add_rear_hinge_bracket("large_camera", center, size[0], size[1], materials))
    objects.extend(add_tiny_status_marks("large_camera", center, materials))
    return objects


def _build_small_housing(name: str, center: tuple[float, float, float], materials) -> list:
    size = (4.82, 3.98, 1.56)
    objects = []
    objects.extend(add_plate_with_front_slot(name, center, size, materials))
    objects.extend(_add_cover_fasteners(name, center, size, "small_four", materials))
    objects.extend(add_small_camera_module(f"{name}_camera", (center[0], center[1] - 0.06, center[2] + size[2] / 2 + 0.11), materials))
    objects.extend(add_rear_hinge_bracket(name, center, size[0], size[1], materials))
    objects.extend(add_tiny_status_marks(name, center, materials))
    return objects


def build_model() -> list:
    """Build the three repeated camera housings and rear bracket hardware."""
    materials = create_materials()
    objects = []
    objects.extend(_build_small_housing("left_small_camera", (-8.02, 1.78, -0.05), materials))
    objects.extend(_build_small_housing("middle_small_camera", (-2.98, 0.35, 0.03), materials))
    objects.extend(_build_large_camera_housing(materials))
    group_objects("camera_housing_model", objects)
    return objects


# ============================================================
# Scene setup and rendering
# ============================================================

"""Minimal render entrypoint for current_version."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

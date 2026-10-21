from __future__ import annotations
from collections.abc import Iterable, Sequence
from mathutils import Matrix, Vector
import bmesh
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the robot claw model."""





def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.55
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material | None) -> bpy.types.Object:
    if material is not None:
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
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign_material(obj, material)
    return obj


def add_cylinder(
    name: str,
    radius: float,
    height: float,
    location: tuple[float, float, float],
    material: bpy.types.Material | None = None,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
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
    assign_material(obj, material)
    shade_smooth(obj)
    return obj


def apply_boolean_difference(target: bpy.types.Object, cutters: Iterable[bpy.types.Object]) -> bpy.types.Object:
    """Apply Blender boolean difference modifiers and remove the temporary cutters."""
    for cutter in cutters:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = target
        target.select_set(True)
        modifier = target.modifiers.new(f"cut_{cutter.name}", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        modifier.solver = "EXACT"
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        target.select_set(False)
        bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def make_cylinder_cutter(
    name: str,
    radius: float,
    height: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 32,
) -> bpy.types.Object:
    cutter = add_cylinder(f"{name}_cutter", radius, height, location, None, rotation=rotation, vertices=vertices)
    cutter.display_type = "WIRE"
    cutter.hide_render = True
    return cutter


def make_box_cutter(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    cutter = add_box(f"{name}_cutter", size, location, None, rotation=rotation)
    cutter.display_type = "WIRE"
    cutter.hide_render = True
    return cutter


def extrude_polygon(
    name: str,
    points: Sequence[tuple[float, float]],
    z_min: float,
    z_max: float,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Create an extruded XY footprint. Points may be concave but must be ordered."""
    if len(points) < 3:
        raise ValueError("extrude_polygon requires at least three points")

    verts = [(x, y, z_min) for x, y in points] + [(x, y, z_max) for x, y in points]
    n = len(points)
    faces: list[tuple[int, ...]] = []
    faces.append(tuple(reversed(range(n))))
    faces.append(tuple(range(n, n * 2)))
    for i in range(n):
        faces.append((i, (i + 1) % n, ((i + 1) % n) + n, i + n))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    assign_material(obj, material)
    return obj


def extrude_profile_with_center_hole(
    name: str,
    outer_points: Sequence[tuple[float, float]],
    center: tuple[float, float],
    inner_radius: float,
    z_min: float,
    z_max: float,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Create an extruded outer profile with one circular center opening."""
    if len(outer_points) < 3:
        raise ValueError("outer profile requires at least three points")

    inner_points: list[tuple[float, float]] = []
    for x, y in outer_points:
        angle = math.atan2(y - center[1], x - center[0])
        inner_points.append((center[0] + inner_radius * math.cos(angle), center[1] + inner_radius * math.sin(angle)))

    verts = (
        [(x, y, z_min) for x, y in outer_points]
        + [(x, y, z_min) for x, y in inner_points]
        + [(x, y, z_max) for x, y in outer_points]
        + [(x, y, z_max) for x, y in inner_points]
    )
    n = len(outer_points)
    ob, ib, ot, it = 0, n, n * 2, n * 3
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((ob + i, ob + j, ot + j, ot + i))
        faces.append((ib + j, ib + i, it + i, it + j))
        faces.append((ot + i, ot + j, it + j, it + i))
        faces.append((ob + j, ob + i, ib + i, ib + j))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    assign_material(obj, material)
    return obj


def transform_points(
    points: Sequence[tuple[float, float]],
    location: tuple[float, float] = (0.0, 0.0),
    rotation_z: float = 0.0,
) -> list[tuple[float, float]]:
    cos_a = math.cos(rotation_z)
    sin_a = math.sin(rotation_z)
    tx, ty = location
    return [(tx + x * cos_a - y * sin_a, ty + x * sin_a + y * cos_a) for x, y in points]


def capsule_points(length: float, width: float, segments: int = 14) -> list[tuple[float, float]]:
    """Return a rounded OpenSCAD beam footprint from x=0 to x=length."""
    radius = width / 2.0
    points: list[tuple[float, float]] = []
    for i in range(segments + 1):
        angle = math.radians(-90 + 180 * i / segments)
        points.append((length + radius * math.cos(angle), radius * math.sin(angle)))
    for i in range(segments + 1):
        angle = math.radians(90 + 180 * i / segments)
        points.append((radius * math.cos(angle), radius * math.sin(angle)))
    return points


def add_beam(
    name: str,
    length: float,
    width: float,
    thickness: float,
    location: tuple[float, float, float],
    rotation_z: float = 0.0,
    material: bpy.types.Material | None = None,
    with_hole_markers: bool = True,
) -> list[bpy.types.Object]:
    footprint = transform_points(capsule_points(length, width), location[:2], rotation_z)
    body = extrude_polygon(name, footprint, location[2], location[2] + thickness, material)
    if with_hole_markers:
        cutters = []
        for idx, x_local in enumerate((0.0, length)):
            px, py = transform_points([(x_local, 0.0)], location[:2], rotation_z)[0]
            cutters.append(make_cylinder_cutter(
                f"{name}_pivot_hole_{idx}",
                radius=1.55,
                height=thickness + 0.30,
                location=(px, py, location[2] + thickness / 2.0),
                vertices=24,
            ))
        apply_boolean_difference(body, cutters)
    return [body]


def add_boss_ring(
    name: str,
    center: tuple[float, float, float],
    outer_radius: float,
    inner_radius: float,
    height: float,
    material: bpy.types.Material | None = None,
    vertices: int = 48,
) -> bpy.types.Object:
    """Create a short ring with an actual center opening."""
    points_outer = []
    points_inner = []
    for i in range(vertices):
        angle = 2.0 * math.pi * i / vertices
        points_outer.append((outer_radius * math.cos(angle), outer_radius * math.sin(angle)))
        points_inner.append((inner_radius * math.cos(angle), inner_radius * math.sin(angle)))

    verts: list[tuple[float, float, float]] = []
    for z in (-height / 2.0, height / 2.0):
        verts.extend((center[0] + x, center[1] + y, center[2] + z) for x, y in points_outer)
        verts.extend((center[0] + x, center[1] + y, center[2] + z) for x, y in points_inner)

    faces = []
    n = vertices
    outer_bottom = 0
    inner_bottom = n
    outer_top = n * 2
    inner_top = n * 3
    for i in range(n):
        j = (i + 1) % n
        faces.append((outer_bottom + i, outer_bottom + j, outer_top + j, outer_top + i))
        faces.append((inner_bottom + j, inner_bottom + i, inner_top + i, inner_top + j))
        faces.append((outer_top + i, outer_top + j, inner_top + j, inner_top + i))
        faces.append((outer_bottom + j, outer_bottom + i, inner_bottom + i, inner_bottom + j))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    shade_smooth(obj)
    return obj


def gear_tooth_points(
    teeth: int,
    root_radius: float,
    tip_radius: float,
    phase: float = 0.0,
) -> list[tuple[float, float]]:
    """Approximate the OpenSCAD involute gears as a readable toothed outline."""
    points: list[tuple[float, float]] = []
    for tooth in range(teeth):
        base = phase + 2.0 * math.pi * tooth / teeth
        pitch = 2.0 * math.pi / teeth
        samples = [
            (base - 0.45 * pitch, root_radius),
            (base - 0.24 * pitch, tip_radius),
            (base + 0.24 * pitch, tip_radius),
            (base + 0.45 * pitch, root_radius),
        ]
        for angle, radius in samples:
            points.append((radius * math.cos(angle), radius * math.sin(angle)))
    return points


def add_gear(
    name: str,
    teeth: int,
    pitch_radius: float,
    thickness: float,
    center: tuple[float, float, float],
    material: bpy.types.Material | None = None,
    phase: float = 0.0,
    bore_radius: float = 1.55,
    hub_radius: float | None = None,
    hub_height: float = 0.0,
) -> list[bpy.types.Object]:
    root_radius = pitch_radius * 0.82
    tip_radius = pitch_radius * 1.16
    outline = [(center[0] + x, center[1] + y) for x, y in gear_tooth_points(teeth, root_radius, tip_radius, phase)]
    body = extrude_profile_with_center_hole(
        name,
        outline,
        center[:2],
        bore_radius,
        center[2] - thickness / 2.0,
        center[2] + thickness / 2.0,
        material,
    )
    objects = [body]
    if hub_radius and hub_height > thickness:
        objects.append(add_boss_ring(f"{name}_raised_hub", center, hub_radius, bore_radius, hub_height, material, vertices=48))
    return objects


def add_crescent_rotate_plate(
    name: str,
    location: tuple[float, float, float],
    material: bpy.types.Material | None = None,
) -> list[bpy.types.Object]:
    """Rebuild claw_rotate_plate() under the assembly rotate([0, 90, 0])."""

    def local_to_world(x: float, y: float, z: float) -> tuple[float, float, float]:
        return (location[0] + z, location[1] + y, location[2] - x)

    disk_center = local_to_world(0.0, 0.0, 2.0)
    clamp_center = local_to_world(0.0, 0.0, 8.0)
    disk = add_cylinder(
        f"{name}_round_base_with_bores",
        12.5,
        4.0,
        disk_center,
        material,
        rotation=(0.0, math.radians(90.0), 0.0),
        vertices=72,
    )
    clamp = add_box(f"{name}_split_upright_clamp", (12.0, 20.0, 7.0), clamp_center, material)

    slot_size = (20.0, 6.0, 8.0)
    slot_center = local_to_world(0.0, 0.0, 8.0)
    apply_boolean_difference(disk, [make_box_cutter(f"{name}_disk_center_slot", slot_size, slot_center)])

    apply_boolean_difference(
        disk,
        [
            make_cylinder_cutter(
                f"{name}_disk_bore_upper",
                1.5,
                30.0,
                local_to_world(-9.0, 0.0, 14.0),
                rotation=(0.0, math.radians(90.0), 0.0),
                vertices=24,
            ),
            make_cylinder_cutter(
                f"{name}_disk_bore_lower",
                1.5,
                30.0,
                local_to_world(9.0, 0.0, 14.0),
                rotation=(0.0, math.radians(90.0), 0.0),
                vertices=24,
            ),
        ],
    )
    apply_boolean_difference(
        clamp,
        [
            make_box_cutter(f"{name}_central_clamp_slot", slot_size, slot_center),
            make_cylinder_cutter(
                f"{name}_cross_hole_left",
                1.5,
                30.0,
                local_to_world(0.0, 5.0, 11.0),
                vertices=24,
            ),
            make_cylinder_cutter(
                f"{name}_cross_hole_right",
                1.5,
                30.0,
                local_to_world(0.0, -5.0, 11.0),
                vertices=24,
            ),
        ],
    )
    return [disk, clamp]


def add_plate_hole_marker(
    name: str,
    center: tuple[float, float, float],
    radius: float,
    height: float,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    return add_boss_ring(name, center, radius * 1.65, radius, height, material, vertices=32)


def rotate_object_set(objects: Iterable[bpy.types.Object], origin: tuple[float, float, float], rotation_z: float) -> None:
    matrix = Matrix.Translation(Vector(origin)) @ Matrix.Rotation(rotation_z, 4, "Z") @ Matrix.Translation(-Vector(origin))
    for obj in objects:
        obj.matrix_world = matrix @ obj.matrix_world


# ============================================================
# Procedural assembly
# ============================================================

"""High-level assembly for the Thingiverse robot claw model."""






ROT = 0.0
GEAR_STAGES = 1
TARGET_SIDE = 50.0
TARGET_TRANSLATION = 110.0
ARM_LENGTH = 45.0
ARM_WIDTH = 8.0
ARM_THICKNESS = 5.0
GRIP_ANGLE = math.radians(30.0)
FREE_ARM_STAY_OFFSET = 32.0
HAND_LENGTH = 30.0
GRIP_PLATE_HEIGHT = 15.0
GEAR_THICKNESS = ARM_THICKNESS
GEAR_TEETH = 10
GEAR_PLAY = 0.2
SCREW_HOLE_RADIUS = (3.0 + 0.1) / 2.0
PLATE_THICKNESS = 3.0

SMALL_GEAR_RAD = (GEAR_TEETH * 400.0 / 180.0) / 2.0
LARGE_GEAR_RAD = (GEAR_TEETH * 2.0 * 400.0 / 180.0) / 2.0
ARM_GEAR_POS = (SMALL_GEAR_RAD * 2.0) * GEAR_STAGES + LARGE_GEAR_RAD + GEAR_PLAY * (GEAR_STAGES * 2.0)
GRIP_OFFSET = ARM_LENGTH + ARM_GEAR_POS

GEAR_POS_SMALL_L1 = (0.0, SMALL_GEAR_RAD + GEAR_PLAY / 2.0, GEAR_THICKNESS)
GEAR_POS_SMALL_R1 = (0.0, -(SMALL_GEAR_RAD + GEAR_PLAY / 2.0), 0.0)
GEAR_POS_LARGE_R = (0.0, -ARM_GEAR_POS, 0.0)
GEAR_POS_LARGE_L = (0.0, ARM_GEAR_POS, 0.0)
FREE_ARM_STAY_POS_R = (
    FREE_ARM_STAY_OFFSET * math.cos(math.radians(30.0)),
    -GRIP_OFFSET + FREE_ARM_STAY_OFFSET * math.sin(math.radians(30.0)) + ARM_LENGTH,
    0.0,
)
FREE_ARM_STAY_POS_L = (
    FREE_ARM_STAY_OFFSET * math.cos(math.radians(30.0)),
    GRIP_OFFSET - FREE_ARM_STAY_OFFSET * math.sin(math.radians(30.0)) - ARM_LENGTH,
    0.0,
)
GRIP_POS_L = (0.0, GRIP_OFFSET, -ARM_THICKNESS - 0.3)
GRIP_POS_R = (0.0, -GRIP_OFFSET, -ARM_THICKNESS - 0.3)
GRIP_PLATE_POS_L = (
    FREE_ARM_STAY_OFFSET * math.cos(GRIP_ANGLE) + HAND_LENGTH,
    GRIP_OFFSET + FREE_ARM_STAY_OFFSET * -math.sin(GRIP_ANGLE),
    0.0,
)
GRIP_PLATE_POS_R = (
    FREE_ARM_STAY_OFFSET * math.cos(GRIP_ANGLE) + HAND_LENGTH,
    -GRIP_OFFSET + FREE_ARM_STAY_OFFSET * math.sin(GRIP_ANGLE),
    0.0,
)


def build_materials() -> dict[str, bpy.types.Material]:
    return {
        "source_gold": make_material("source_preview_gold", (0.95, 0.72, 0.05, 1.0)),
        "target_gold": make_material("target_preview_gold", (0.95, 0.72, 0.05, 1.0)),
        "servo_green": make_material("servo_internal_green", (0.23, 0.43, 0.18, 1.0)),
        "dark_void": make_material("dark_cut_faces", (0.05, 0.05, 0.03, 1.0)),
    }


def empty_plate_footprint() -> list[tuple[float, float]]:
    p0 = (GEAR_POS_LARGE_R[0], GEAR_POS_LARGE_R[1] - 5.0)
    p1 = (GEAR_POS_LARGE_L[0], GEAR_POS_LARGE_L[1] + 5.0)
    p2 = (GEAR_POS_LARGE_R[0] - SMALL_GEAR_RAD * 1.5, GEAR_POS_LARGE_R[1])
    p3 = (GEAR_POS_LARGE_L[0] - SMALL_GEAR_RAD * 1.5, GEAR_POS_LARGE_L[1])
    p4 = (FREE_ARM_STAY_POS_R[0] + 5.0, FREE_ARM_STAY_POS_R[1] - 5.0)
    p5 = (FREE_ARM_STAY_POS_L[0] + 5.0, FREE_ARM_STAY_POS_L[1] + 5.0)
    return [p4, p5, p1, p3, p2, p0]


def add_servo_plate(materials: dict[str, bpy.types.Material]) -> list[bpy.types.Object]:
    mat = materials["source_gold"]
    plate = extrude_polygon("lower_servo_plate_source_polyhedron", empty_plate_footprint(), -3.5, -0.5, mat)
    tab = add_box("rear_small_gear_mount_tab_with_screw_bores", (20.0, 20.0, PLATE_THICKNESS), (-SMALL_GEAR_RAD * 1.5, 0.0, -2.0), mat)
    objects = [plate, tab]

    plate_cutters = []
    for name, pos in [
        ("small_left_clearance", GEAR_POS_SMALL_L1),
        ("small_right_plate_pivot", GEAR_POS_SMALL_R1),
        ("large_right_plate_pivot", GEAR_POS_LARGE_R),
        ("large_left_plate_pivot", GEAR_POS_LARGE_L),
        ("free_right_plate_pivot", FREE_ARM_STAY_POS_R),
        ("free_left_plate_pivot", FREE_ARM_STAY_POS_L),
    ]:
        plate_cutters.append(make_cylinder_cutter(name, 6.0 if "clearance" in name else SCREW_HOLE_RADIUS, 8.0, (pos[0], pos[1], -2.0), vertices=32))
    plate_cutters.append(
        make_box_cutter(
            "small_right_rotated_square_clearance",
            (6.0, 6.0, 8.0),
            (GEAR_POS_SMALL_R1[0], GEAR_POS_SMALL_R1[1], -2.0),
            rotation=(0.0, 0.0, math.radians(-60.0)),
        )
    )
    apply_boolean_difference(plate, plate_cutters)
    apply_boolean_difference(
        tab,
        [
            make_cylinder_cutter("rear_tab_upper_screw_bore", SCREW_HOLE_RADIUS, 8.0, (-SMALL_GEAR_RAD * 1.5 - 5.0, 5.0, -2.0), vertices=24),
            make_cylinder_cutter("rear_tab_lower_screw_bore", SCREW_HOLE_RADIUS, 8.0, (-SMALL_GEAR_RAD * 1.5 - 5.0, -5.0, -2.0), vertices=24),
        ],
    )
    objects.extend(add_servo_holder(materials))
    return objects


def add_servo_holder(materials: dict[str, bpy.types.Material]) -> list[bpy.types.Object]:
    """Visible underside servo pocket from med_servo_holder(), placed as in Servo_plate()."""
    mat = materials["source_gold"]
    green = materials["servo_green"]
    objects: list[bpy.types.Object] = []
    base_x, base_y, base_z = (4.5, 2.0, -3.9)
    angle = math.radians(-60.0)

    def local_to_world(x: float, y: float, z: float) -> tuple[float, float, float]:
        return (
            base_x - 0.5 * x + math.sqrt(3.0) * 0.5 * y,
            base_y + math.sqrt(3.0) * 0.5 * x + 0.5 * y,
            base_z - z,
        )

    def rot() -> tuple[float, float, float]:
        return (0.0, math.radians(180.0), angle)

    outer = add_box("underside_servo_holder_open_shell", (45.0, 18.0, 11.0), local_to_world(0.0, 0.0, 0.0), mat, rotation=rot())
    apply_boolean_difference(
        outer,
        [
            make_cylinder_cutter("servo_holder_large_round_pocket", 4.5, 13.0, local_to_world(8.0, 0.0, 0.0), rotation=rot(), vertices=32),
            make_cylinder_cutter("servo_holder_small_round_pocket", 3.4, 13.0, local_to_world(4.0, 0.0, 0.0), rotation=rot(), vertices=32),
            make_box_cutter("servo_holder_left_side_slot", (38.0, 1.0, 3.0), local_to_world(0.0, -6.5, 5.0), rotation=rot()),
            make_box_cutter("servo_holder_right_side_slot", (38.0, 1.0, 3.0), local_to_world(0.0, 6.5, 5.0), rotation=rot()),
            make_box_cutter("servo_holder_rectangular_servo_pocket", (31.0, 14.0, 11.0), local_to_world(0.0, 0.0, 3.0), rotation=rot()),
            make_cylinder_cutter("servo_holder_left_screw_bore", 1.7, 13.0, local_to_world(-17.5, 0.0, 0.0), rotation=rot(), vertices=24),
            make_cylinder_cutter("servo_holder_right_screw_bore", 1.7, 13.0, local_to_world(17.5, 0.0, 0.0), rotation=rot(), vertices=24),
            make_box_cutter("servo_holder_bottom_relief", (46.0, 19.0, 4.0), local_to_world(0.0, 0.0, -4.4), rotation=rot()),
        ],
    )
    objects.append(outer)

    lug = add_box("servo_end_lug_with_rectangular_relief", (7.0, 18.0, 5.0), local_to_world(19.0, 0.0, 7.0), mat, rotation=rot())
    apply_boolean_difference(lug, [make_box_cutter("servo_end_lug_inner_relief", (5.0, 14.0, 3.1), local_to_world(17.3, 0.0, 6.5), rotation=rot())])
    objects.append(lug)
    objects.append(add_box("thin_servo_back_flange", (3.0, 18.0, 2.0), local_to_world(-14.0, 0.0, -1.4), mat, rotation=rot()))
    objects.append(add_box("recessed_green_servo_core_visible_through_pocket", (28.0, 10.0, 2.0), local_to_world(0.0, 0.0, -1.8), green, rotation=rot()))
    return objects


def add_large_gear_arm(
    name: str,
    center: tuple[float, float, float],
    side: str,
    material: bpy.types.Material,
) -> list[bpy.types.Object]:
    objects = add_gear(
        f"{name}_twenty_tooth_gear",
        teeth=GEAR_TEETH * 2,
        pitch_radius=LARGE_GEAR_RAD,
        thickness=GEAR_THICKNESS,
        center=(center[0], center[1], center[2] + GEAR_THICKNESS / 2.0),
        material=material,
        phase=math.radians(9.0 if side == "left" else 54.0),
        bore_radius=SCREW_HOLE_RADIUS,
    )
    rotation = math.radians(90.0 if side == "left" else -90.0)
    objects.extend(add_beam(f"{name}_integral_drive_arm", ARM_LENGTH, ARM_WIDTH, ARM_THICKNESS, center, rotation, material))
    return objects


def add_free_arm(name: str, position: tuple[float, float, float], side: str, material: bpy.types.Material) -> list[bpy.types.Object]:
    rotation = math.radians(90.0 if side == "left" else -90.0)
    return add_beam(name, ARM_LENGTH, ARM_WIDTH, ARM_THICKNESS, position, rotation, material)


def add_gripper_linkage(name: str, position: tuple[float, float, float], side: str, material: bpy.types.Material) -> list[bpy.types.Object]:
    assembly_rotation = -GRIP_ANGLE if side == "left" else GRIP_ANGLE
    hand_bend = GRIP_ANGLE if side == "left" else -GRIP_ANGLE
    objects = add_beam(f"{name}_rear_link", FREE_ARM_STAY_OFFSET, ARM_WIDTH, ARM_THICKNESS, position, assembly_rotation, material)
    tip_x, tip_y = transform_points([(FREE_ARM_STAY_OFFSET, 0.0)], position[:2], assembly_rotation)[0]
    objects.extend(
        add_beam(
            f"{name}_horizontal_grip_finger",
            HAND_LENGTH,
            ARM_WIDTH,
            ARM_THICKNESS,
            (tip_x, tip_y, position[2]),
            assembly_rotation + hand_bend,
            material,
        )
    )
    return objects


def add_grip_plate(
    name: str,
    position: tuple[float, float, float],
    rotation_z: float,
    material: bpy.types.Material,
) -> list[bpy.types.Object]:
    objects = add_beam(
        f"{name}_short_mount_beam",
        8.0,
        ARM_WIDTH,
        ARM_THICKNESS,
        position,
        rotation_z,
        material,
        with_hole_markers=True,
    )
    plate_x, plate_y = transform_points([(8.0, 0.0)], position[:2], rotation_z)[0]
    plate = add_box(
        name,
        (4.0, HAND_LENGTH * 0.7 * 2.0, GRIP_PLATE_HEIGHT),
        (plate_x, plate_y, position[2]),
        material,
        rotation=(0.0, 0.0, rotation_z),
    )

    def notch_cutter(label: str) -> bpy.types.Object:
        notch_x, notch_y = transform_points([(13.8, 0.0)], position[:2], rotation_z)[0]
        return make_box_cutter(
            label,
            (8.4, 20.0, 20.0),
            (notch_x, notch_y, position[2]),
            rotation=(0.0, 0.0, rotation_z),
        )

    apply_boolean_difference(plate, [notch_cutter(f"{name}_outside_bite")])
    for index, obj in enumerate(objects):
        apply_boolean_difference(obj, [notch_cutter(f"{name}_beam_bite_{index}")])
    objects.append(plate)
    return objects


def add_gear_train(material: bpy.types.Material) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    objects.extend(
        add_gear(
            "raised_servo_drive_gear",
            teeth=GEAR_TEETH,
            pitch_radius=SMALL_GEAR_RAD,
            thickness=GEAR_THICKNESS,
            center=(GEAR_POS_SMALL_L1[0], GEAR_POS_SMALL_L1[1], GEAR_POS_SMALL_L1[2]),
            material=material,
            phase=math.radians(108.0),
            bore_radius=3.1,
            hub_radius=4.5,
            hub_height=GEAR_THICKNESS + 4.0,
        )
    )
    objects.extend(
        add_gear(
            "lower_small_idler_gear",
            teeth=GEAR_TEETH,
            pitch_radius=SMALL_GEAR_RAD,
            thickness=GEAR_THICKNESS,
            center=(GEAR_POS_SMALL_R1[0], GEAR_POS_SMALL_R1[1], GEAR_POS_SMALL_R1[2] + GEAR_THICKNESS / 2.0),
            material=material,
            phase=math.radians(90.0),
            bore_radius=SCREW_HOLE_RADIUS,
        )
    )
    objects.extend(add_large_gear_arm("left_closing", GEAR_POS_LARGE_L, "left", material))
    objects.extend(add_large_gear_arm("right_closing", GEAR_POS_LARGE_R, "right", material))
    return objects


def add_target_cube(materials: dict[str, bpy.types.Material]) -> bpy.types.Object:
    return add_box(
        "source_sizing_target_cube_50mm",
        (TARGET_SIDE, TARGET_SIDE, TARGET_SIDE),
        (TARGET_TRANSLATION, 0.0, 0.0),
        materials["target_gold"],
    )


def build_model() -> list[bpy.types.Object]:
    materials = build_materials()
    gold = materials["source_gold"]
    objects: list[bpy.types.Object] = []

    objects.append(add_target_cube(materials))
    objects.extend(add_servo_plate(materials))
    objects.extend(add_gear_train(gold))
    objects.extend(add_free_arm("right_outer_free_arm", FREE_ARM_STAY_POS_R, "right", gold))
    objects.extend(add_free_arm("left_outer_free_arm", FREE_ARM_STAY_POS_L, "left", gold))
    objects.extend(add_gripper_linkage("left_gripper_linkage", GRIP_POS_L, "left", gold))
    objects.extend(add_gripper_linkage("right_gripper_linkage", GRIP_POS_R, "right", gold))
    objects.extend(add_grip_plate("left_vertical_grip_plate", GRIP_PLATE_POS_L, math.radians(-90.0), gold))
    objects.extend(add_grip_plate("right_vertical_grip_plate", GRIP_PLATE_POS_R, math.radians(90.0), gold))
    objects.extend(add_crescent_rotate_plate("rear_servo_rotation_mount", (-SMALL_GEAR_RAD * 1.5 - 16.0, 0.0, 2.0), gold))

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

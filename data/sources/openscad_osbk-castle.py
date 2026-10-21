from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from typing import NamedTuple
import bpy
import math
import random


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable geometry helpers for the OSBK castle model."""





INCH = 2.54
BASE_Z = 0.40
MODEL_Z = 0.40
STONE = (0.95, 0.72, 0.06, 1.0)
STONE_DARK = (0.52, 0.36, 0.035, 1.0)
CRACK_GREEN = (0.06, 0.45, 0.18, 1.0)


@dataclass(frozen=True)
class MaterialSet:
    stone: bpy.types.Material
    dark_stone: bpy.types.Material
    crack: bpy.types.Material
    shadow: bpy.types.Material


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_materials() -> MaterialSet:
    def material(name: str, color: tuple[float, float, float, float], roughness: float) -> bpy.types.Material:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        return mat

    return MaterialSet(
        stone=material("source_yellow_ochre_stone", STONE, 0.72),
        dark_stone=material("recessed_shadow_yellow_brown", STONE_DARK, 0.85),
        crack=material("green_crack_infill", CRACK_GREEN, 0.8),
        shadow=material("dark_gate_and_voids", (0.18, 0.13, 0.035, 1.0), 0.9),
    )


def add_box(
    name: str,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
    material: bpy.types.Material | None = None,
    rotation_z: float = 0.0,
) -> bpy.types.Object:
    hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    c = math.cos(rotation_z)
    s = math.sin(rotation_z)

    def world(local: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = local
        return (
            center[0] + x * c - y * s,
            center[1] + x * s + y * c,
            center[2] + z,
        )

    vertices = [
        world((-hx, -hy, -hz)),
        world((hx, -hy, -hz)),
        world((hx, hy, -hz)),
        world((-hx, hy, -hz)),
        world((-hx, -hy, hz)),
        world((hx, -hy, hz)),
        world((hx, hy, hz)),
        world((-hx, hy, hz)),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
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
    if material:
        obj.data.materials.append(material)
    return obj


def add_box_min(
    name: str,
    min_xyz: tuple[float, float, float],
    size: tuple[float, float, float],
    material: bpy.types.Material | None = None,
    rotation_z: float = 0.0,
) -> bpy.types.Object:
    center = (
        min_xyz[0] + size[0] * 0.5,
        min_xyz[1] + size[1] * 0.5,
        min_xyz[2] + size[2] * 0.5,
    )
    return add_box(name, size, center, material, rotation_z)


def add_beveled_box(
    name: str,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
    material: bpy.types.Material,
    bevel: float = 0.012,
    rotation_z: float = 0.0,
) -> bpy.types.Object:
    obj = add_box(name, size, center, material, rotation_z)
    if bevel > 0:
        mod = obj.modifiers.new("tiny_chipped_edges", "BEVEL")
        mod.width = bevel
        mod.segments = 1
        mod.affect = "EDGES"
    return obj


def add_brick(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    material: bpy.types.Material,
    seed: int,
    jitter: float = 0.015,
    rot_max_deg: float = 1.2,
) -> bpy.types.Object:
    rng = random.Random(seed)
    sx = size[0] * rng.uniform(0.96, 1.04)
    sy = size[1] * rng.uniform(0.94, 1.06)
    sz = size[2] * rng.uniform(0.96, 1.04)
    loc = (
        center[0] + rng.uniform(-jitter, jitter),
        center[1] + rng.uniform(-jitter, jitter),
        center[2] + rng.uniform(-jitter * 0.4, jitter * 0.4),
    )
    rot = math.radians(rng.uniform(-rot_max_deg, rot_max_deg))
    return add_beveled_box(name, (sx, sy, sz), loc, material, bevel=0.008, rotation_z=rot)


def add_crack_marks(
    prefix: str,
    face: str,
    base_center: tuple[float, float, float],
    span: tuple[float, float],
    material: bpy.types.Material,
    seed: int,
    count: int,
) -> list[bpy.types.Object]:
    rng = random.Random(seed)
    objects: list[bpy.types.Object] = []
    for idx in range(count):
        u = rng.uniform(-0.46, 0.46) * span[0]
        v = rng.uniform(-0.44, 0.44) * span[1]
        length = rng.uniform(0.09, 0.20)
        width = rng.uniform(0.010, 0.018)
        if face in {"front", "back"}:
            y_off = -0.004 if face == "front" else 0.004
            obj = add_box(
                f"{prefix}_crack_{idx:02d}",
                (length, 0.012, width),
                (base_center[0] + u, base_center[1] + y_off, base_center[2] + v),
                material,
                rotation_z=math.radians(rng.uniform(-35, 35)),
            )
        else:
            x_off = 0.004 if face == "right" else -0.004
            obj = add_box(
                f"{prefix}_crack_{idx:02d}",
                (0.012, length, width),
                (base_center[0] + x_off, base_center[1] + u, base_center[2] + v),
                material,
                rotation_z=math.radians(rng.uniform(-35, 35)),
            )
        objects.append(obj)
    return objects


def build_brick_wall_face(
    prefix: str,
    orientation: str,
    origin: tuple[float, float, float],
    length: float,
    height: float,
    thickness: float,
    material: bpy.types.Material,
    seed: int,
    brick_len: float = 0.46,
    course_h: float = 0.25,
) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    gap = 0.025
    courses = max(1, int(round(height / course_h)))
    for row in range(courses):
        z0 = origin[2] + row * course_h
        row_h = min(course_h - gap, origin[2] + height - z0)
        offset = (brick_len * 0.5) if row % 2 else 0.0
        pos = -offset
        col = 0
        while pos < length - 0.01:
            seg_start = max(0.0, pos)
            seg_end = min(length, pos + brick_len)
            seg_len = seg_end - seg_start - gap
            if seg_len > 0.06:
                if orientation == "y":
                    size = (thickness, seg_len, row_h)
                    center = (origin[0], origin[1] + seg_start + seg_len * 0.5, z0 + row_h * 0.5)
                else:
                    size = (seg_len, thickness, row_h)
                    center = (origin[0] + seg_start + seg_len * 0.5, origin[1], z0 + row_h * 0.5)
                objects.append(add_brick(f"{prefix}_r{row:02d}_c{col:02d}", center, size, material, seed + row * 101 + col))
            pos += brick_len
            col += 1
    return objects


def build_cap_tiles(
    prefix: str,
    min_x: float,
    min_y: float,
    z: float,
    size_x: float,
    size_y: float,
    tile: float,
    material: bpy.types.Material,
    seed: int,
) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    rng = random.Random(seed)
    nx = max(1, int(round(size_x / tile)))
    ny = max(1, int(round(size_y / tile)))
    step_x = size_x / nx
    step_y = size_y / ny
    for ix in range(nx):
        for iy in range(ny):
            inset = 0.018
            cx = min_x + ix * step_x + step_x * 0.5 + rng.uniform(-0.012, 0.012)
            cy = min_y + iy * step_y + step_y * 0.5 + rng.uniform(-0.012, 0.012)
            objects.append(
                add_beveled_box(
                    f"{prefix}_cap_{ix:02d}_{iy:02d}",
                    (step_x - inset, step_y - inset, 0.08),
                    (cx, cy, z + 0.04),
                    material,
                    bevel=0.006,
                    rotation_z=math.radians(rng.uniform(-1.5, 1.5)),
                )
            )
    return objects


def build_transformed_cap_tiles(
    prefix: str,
    point_fn: Callable[[float, float, float], tuple[float, float, float]],
    rotation_z: float,
    min_x: float,
    min_y: float,
    z: float,
    size_x: float,
    size_y: float,
    tile: float,
    material: bpy.types.Material,
    seed: int,
    thickness: float = 0.08,
) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    rng = random.Random(seed)
    nx = max(1, int(round(size_x / tile)))
    ny = max(1, int(round(size_y / tile)))
    step_x = size_x / nx
    step_y = size_y / ny
    inset = 0.018
    for ix in range(nx):
        for iy in range(ny):
            local_x = min_x + ix * step_x + step_x * 0.5 + rng.uniform(-0.010, 0.010)
            local_y = min_y + iy * step_y + step_y * 0.5 + rng.uniform(-0.010, 0.010)
            local_z = z + thickness * 0.5 + rng.uniform(-0.004, 0.004)
            objects.append(
                add_beveled_box(
                    f"{prefix}_cap_{ix:02d}_{iy:02d}",
                    (step_x - inset, step_y - inset, thickness),
                    point_fn(local_x, local_y, local_z),
                    material,
                    bevel=0.006,
                    rotation_z=rotation_z + math.radians(rng.uniform(-1.1, 1.1)),
                )
            )
    return objects


def build_transformed_brick_deck(
    prefix: str,
    point_fn: Callable[[float, float, float], tuple[float, float, float]],
    rotation_z: float,
    min_x: float,
    min_y: float,
    z: float,
    size_x: float,
    size_y: float,
    brick_x: float,
    brick_y: float,
    material: bpy.types.Material,
    seed: int,
    thickness: float = 0.13,
    gap: float = 0.018,
) -> list[bpy.types.Object]:
    """Build a sparse staggered horizontal brick field with visible air between runs."""
    objects: list[bpy.types.Object] = []
    rng = random.Random(seed)
    rows = max(1, int(round(size_y / brick_y)))
    row_step = size_y / rows
    for row in range(rows):
        y0 = min_y + row * row_step
        row_h = max(0.022, row_step * 0.28 - gap)
        offset = brick_x * 0.5 if row % 2 else 0.0
        x = min_x - offset
        col = 0
        while x < min_x + size_x - 0.01:
            start = max(min_x, x)
            end = min(min_x + size_x, x + brick_x)
            raw_span = end - start
            span = raw_span * 0.34 - gap
            if span > 0.05 and (row + col) % 3 == 0:
                local_x = start + raw_span * 0.5 + rng.uniform(-0.006, 0.006)
                local_y = y0 + row_h * 0.5 + rng.uniform(-0.006, 0.006)
                local_z = z + thickness * 0.5 + rng.uniform(-0.003, 0.003)
                objects.append(
                    add_beveled_box(
                        f"{prefix}_deck_r{row:02d}_c{col:02d}",
                        (span, row_h, thickness),
                        point_fn(local_x, local_y, local_z),
                        material,
                        bevel=0.006,
                        rotation_z=rotation_z + math.radians(rng.uniform(-0.8, 0.8)),
                    )
                )
            x += brick_x
            col += 1
    return objects


def build_transformed_running_bond_deck(
    prefix: str,
    point_fn: Callable[[float, float, float], tuple[float, float, float]],
    rotation_z: float,
    min_x: float,
    min_y: float,
    z: float,
    size_x: float,
    size_y: float,
    brick_x: float,
    brick_y: float,
    material: bpy.types.Material,
    seed: int,
    thickness: float = 0.06,
    gap: float = 0.014,
) -> list[bpy.types.Object]:
    """Low, sparse running-bond field, separated enough to read as discrete bricks."""
    objects: list[bpy.types.Object] = []
    rng = random.Random(seed)
    rows = max(1, int(round(size_y / brick_y)))
    row_step = size_y / rows
    for row in range(rows):
        y0 = min_y + row * row_step
        row_h = max(0.024, row_step * 0.38 - gap)
        nominal = brick_x * (0.82 if row % 3 == 1 else 1.0)
        offset = nominal * 0.5 if row % 2 else 0.0
        x = min_x - offset
        col = 0
        while x < min_x + size_x - 0.01:
            start = max(min_x, x)
            end = min(min_x + size_x, x + nominal)
            raw_span = end - start
            span = raw_span * 0.46 - gap
            if span > 0.045 and (row + col) % 3 != 1:
                local_x = start + raw_span * 0.5 + rng.uniform(-0.004, 0.004)
                local_y = y0 + row_h * 0.5 + rng.uniform(-0.004, 0.004)
                local_z = z + thickness * 0.5 + rng.uniform(-0.002, 0.002)
                objects.append(
                    add_beveled_box(
                        f"{prefix}_bond_r{row:02d}_c{col:02d}",
                        (span, row_h, thickness),
                        point_fn(local_x, local_y, local_z),
                        material,
                        bevel=0.004,
                        rotation_z=rotation_z + math.radians(rng.uniform(-0.55, 0.55)),
                    )
                )
            x += nominal
            col += 1
    return objects


def add_rail_lip(
    prefix: str,
    min_x: float,
    min_y: float,
    z: float,
    size_x: float,
    size_y: float,
    material: bpy.types.Material,
) -> list[bpy.types.Object]:
    lip = 0.08
    h = 0.12
    return [
        add_box_min(f"{prefix}_lip_front", (min_x, min_y, z), (size_x, lip, h), material),
        add_box_min(f"{prefix}_lip_back", (min_x, min_y + size_y - lip, z), (size_x, lip, h), material),
        add_box_min(f"{prefix}_lip_left", (min_x, min_y, z), (lip, size_y, h), material),
        add_box_min(f"{prefix}_lip_right", (min_x + size_x - lip, min_y, z), (lip, size_y, h), material),
    ]


# ============================================================
# Procedural assembly
# ============================================================

"""High-level assembly for the OSBK castle model.

This iteration follows the source `theCastle()` module transforms directly.
"""






B_PILLAR = (0.20 * INCH, 0.20 * INCH, 0.20 * INCH)
B_WALL = (0.10 * INCH, 0.20 * INCH, 0.10 * INCH)
WALL_GAP = 0.05
WALL_LEN = 2.75 * INCH
WALL_HEIGHT = 1.50 * INCH
TOWER_HEIGHT = 2.00 * INCH
GATE_LOW_HEIGHT = 0.75 * INCH
GATE_TOP_Z = 0.95 * INCH
GATE_WALK_Z = 1.15 * INCH
PILLAR_HEIGHT = 1.00 * INCH
TOWER_SIDE = 0.75 * INCH
FACE_T = 0.10
WALL_DEPTH = 0.15 * INCH
FACE_BACKING_T = FACE_T * 0.46
FACE_BLOCK_DEPTH = FACE_T * 0.86
FACE_BLOCK_OFFSET = FACE_BACKING_T + FACE_BLOCK_DEPTH * 0.5
FACE_SURFACE_OFFSET = FACE_BACKING_T + FACE_BLOCK_DEPTH + 0.004


class Xf(NamedTuple):
    x: float = 0.0
    y: float = 0.0
    rot: float = 0.0

    def then(self, dx: float = 0.0, dy: float = 0.0, drot: float = 0.0) -> "Xf":
        c = math.cos(self.rot)
        s = math.sin(self.rot)
        return Xf(
            self.x + dx * c - dy * s,
            self.y + dx * s + dy * c,
            self.rot + drot,
        )

    def point(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        c = math.cos(self.rot)
        s = math.sin(self.rot)
        return (self.x + x * c - y * s, self.y + x * s + y * c, MODEL_Z + z)


def _box_local(
    name: str,
    xf: Xf,
    min_xyz: tuple[float, float, float],
    size: tuple[float, float, float],
    material: bpy.types.Material,
) -> bpy.types.Object:
    center = (
        min_xyz[0] + size[0] * 0.5,
        min_xyz[1] + size[1] * 0.5,
        min_xyz[2] + size[2] * 0.5,
    )
    return add_box(name, size, xf.point(*center), material, rotation_z=xf.rot)


def _brick_wall_local(
    prefix: str,
    xf: Xf,
    face: str,
    origin: tuple[float, float, float],
    length: float,
    height: float,
    material: bpy.types.Material,
    seed: int,
    brick_len: float = 0.42,
    course_h: float = 0.22,
    gap: float = 0.028,
    joint_material: bpy.types.Material | None = None,
) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    rng_seed = seed
    courses = max(1, math.ceil(height / course_h))
    normal_sign = -1.0 if face in {"front", "left"} else 1.0
    relief_depth = FACE_BLOCK_DEPTH
    normal_offset = normal_sign * FACE_BLOCK_OFFSET
    joint_offset = normal_sign * FACE_SURFACE_OFFSET
    for row in range(courses):
        z0 = origin[2] + row * course_h
        row_h = min(course_h - gap, origin[2] + height - z0)
        row_len = brick_len * (0.94 if row % 4 == 2 else 1.03 if row % 3 == 1 else 1.08)
        offset = row_len * (0.50 if row % 2 else 0.15 if row % 3 == 1 else 0.0)
        pos = -offset
        col = 0
        while pos < length - 0.01:
            seg_start = max(0.0, pos)
            seg_end = min(length, pos + row_len)
            seg_len = seg_end - seg_start - gap
            if seg_len > 0.050 and row_h > 0.040:
                local_seed = rng_seed + row * 101 + col * 17
                jitter = random.Random(local_seed)
                u = seg_start + seg_len * 0.5 + jitter.uniform(-0.006, 0.006)
                zc = z0 + row_h * 0.5 + jitter.uniform(-0.003, 0.003)
                span_scale = jitter.uniform(0.92, 0.99)
                height_scale = jitter.uniform(0.84, 0.94)
                if face in {"front", "back"}:
                    lx = origin[0] + normal_offset + jitter.uniform(-0.0015, 0.0015)
                    ly = origin[1] + u
                    size = (relief_depth, seg_len * span_scale, row_h * height_scale)
                else:
                    lx = origin[0] + u
                    ly = origin[1] + normal_offset + jitter.uniform(-0.0015, 0.0015)
                    size = (seg_len * span_scale, relief_depth, row_h * height_scale)
                objects.append(
                    add_beveled_box(
                        f"{prefix}_r{row:02d}_b{col:02d}",
                        size,
                        xf.point(lx, ly, zc),
                        material,
                        bevel=0.006,
                        rotation_z=xf.rot + math.radians(jitter.uniform(-1.2, 1.2)),
                    )
                )
                if joint_material and col > 0 and (row + col) % 4 in {1, 3}:
                    if face in {"front", "back"}:
                        objects.append(
                            add_box(
                                f"{prefix}_short_recess_{row:02d}_{col:02d}",
                                (0.018, 0.036, row_h * 0.82),
                                xf.point(origin[0] + joint_offset, origin[1] + seg_start, zc),
                                joint_material,
                                rotation_z=xf.rot,
                            )
                        )
                    else:
                        objects.append(
                            add_box(
                                f"{prefix}_short_recess_{row:02d}_{col:02d}",
                                (0.036, 0.018, row_h * 0.82),
                                xf.point(origin[0] + seg_start, origin[1] + joint_offset, zc),
                                joint_material,
                                rotation_z=xf.rot,
                            )
                        )
            pos += row_len
            col += 1
    return objects


def _deck_rim_blocks(
    prefix: str,
    xf: Xf,
    min_x: float,
    min_y: float,
    z: float,
    size_x: float,
    size_y: float,
    seed: int,
    mats,
) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    rng = random.Random(seed)
    rim_w = B_WALL[0] * 0.08
    brick_len = B_PILLAR[1] * 1.10
    gap = 0.055
    for edge_idx, x in enumerate((min_x + rim_w * 0.55, min_x + size_x - rim_w * 0.55)):
        y = min_y
        col = 0
        while y < min_y + size_y - 0.02:
            span = min(brick_len, min_y + size_y - y) - gap
            if span > 0.10:
                objects.append(
                    add_box(
                        f"{prefix}_rim_{edge_idx}_{col:02d}",
                        (rim_w, span, B_WALL[2] * 0.25),
                        xf.point(x + rng.uniform(-0.004, 0.004), y + span * 0.5, z + B_WALL[2] * 0.18),
                        mats.stone,
                        rotation_z=xf.rot + math.radians(rng.uniform(-0.5, 0.5)),
                    )
                )
            y += brick_len
            col += 1
    return objects


def _tower_cap_rim_blocks(prefix: str, xf: Xf, seed: int, mats) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    rng = random.Random(seed)
    z = TOWER_HEIGHT - 1.25 * B_PILLAR[0]
    cap_h = B_WALL[2] * 0.22
    objects.append(
        add_box(
            f"{prefix}_solid_flat_cap",
            (TOWER_SIDE * 0.94, TOWER_SIDE * 0.94, cap_h),
            xf.point(-TOWER_SIDE * 0.5, TOWER_SIDE * 0.5, z + cap_h * 0.5),
            mats.stone,
            rotation_z=xf.rot,
        )
    )
    objects.extend(
        [
            add_box(f"{prefix}_south_rim_groove", (TOWER_SIDE * 0.76, 0.018, 0.012), xf.point(-TOWER_SIDE * 0.5, 0.13, z + cap_h + 0.006), mats.dark_stone, rotation_z=xf.rot),
            add_box(f"{prefix}_north_rim_groove", (TOWER_SIDE * 0.76, 0.018, 0.012), xf.point(-TOWER_SIDE * 0.5, TOWER_SIDE - 0.13, z + cap_h + 0.006), mats.dark_stone, rotation_z=xf.rot),
            add_box(f"{prefix}_west_rim_groove", (0.018, TOWER_SIDE * 0.76, 0.012), xf.point(-TOWER_SIDE + 0.13, TOWER_SIDE * 0.5, z + cap_h + 0.006), mats.dark_stone, rotation_z=xf.rot),
            add_box(f"{prefix}_east_rim_groove", (0.018, TOWER_SIDE * 0.76, 0.012), xf.point(-0.13, TOWER_SIDE * 0.5, z + cap_h + 0.006), mats.dark_stone, rotation_z=xf.rot),
        ]
    )
    x_min = -TOWER_SIDE + 0.20
    x_max = -0.20
    y_min = 0.20
    y_max = TOWER_SIDE - 0.20
    for idx, frac in enumerate((0.30, 0.46, 0.62, 0.78)):
        x = x_min + (x_max - x_min) * frac + rng.uniform(-0.025, 0.025)
        objects.append(
            add_box(
                f"{prefix}_top_grid_x_{idx:02d}",
                (0.012, y_max - y_min, 0.014),
                xf.point(x, (y_min + y_max) * 0.5, z + cap_h + 0.010),
                mats.dark_stone,
                rotation_z=xf.rot + math.radians(rng.uniform(-2.0, 2.0)),
            )
        )
    for idx, frac in enumerate((0.27, 0.43, 0.59, 0.75)):
        y = y_min + (y_max - y_min) * frac + rng.uniform(-0.025, 0.025)
        objects.append(
            add_box(
                f"{prefix}_top_grid_y_{idx:02d}",
                (x_max - x_min, 0.012, 0.014),
                xf.point((x_min + x_max) * 0.5, y, z + cap_h + 0.011),
                mats.dark_stone,
                rotation_z=xf.rot + math.radians(rng.uniform(-2.0, 2.0)),
            )
        )
    for idx in range(8):
        if idx % 2:
            size = (TOWER_SIDE * rng.uniform(0.16, 0.30), 0.010, 0.012)
            center = (-TOWER_SIDE * rng.uniform(0.28, 0.74), TOWER_SIDE * rng.uniform(0.24, 0.76), z + cap_h + 0.013)
        else:
            size = (0.010, TOWER_SIDE * rng.uniform(0.14, 0.28), 0.012)
            center = (-TOWER_SIDE * rng.uniform(0.28, 0.74), TOWER_SIDE * rng.uniform(0.24, 0.76), z + cap_h + 0.013)
        objects.append(
            add_box(
                f"{prefix}_top_shallow_joint_{idx:02d}",
                size,
                xf.point(center[0], center[1], center[2]),
                mats.dark_stone,
                rotation_z=xf.rot + math.radians(rng.uniform(-7.0, 7.0)),
            )
        )
    return objects


def _face_backing_local(
    prefix: str,
    xf: Xf,
    face: str,
    origin: tuple[float, float, float],
    length: float,
    height: float,
    material: bpy.types.Material,
    course_h: float = B_WALL[2],
) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    normal_sign = -1.0 if face in {"front", "left"} else 1.0
    normal_offset = normal_sign * (FACE_BACKING_T * 0.5)
    courses = max(1, math.ceil(height / course_h))
    for row in range(courses):
        z0 = origin[2] + row * course_h
        row_h = min(course_h, origin[2] + height - z0)
        if row_h <= 0.01:
            continue
        if face in {"front", "back"}:
            objects.append(
                add_box(
                    f"{prefix}_recessed_course_back_{row:02d}",
                    (FACE_BACKING_T, length, row_h),
                    xf.point(origin[0] + normal_offset, origin[1] + length * 0.5, z0 + row_h * 0.5),
                    material,
                    rotation_z=xf.rot,
                )
            )
        else:
            objects.append(
                add_box(
                    f"{prefix}_recessed_course_back_{row:02d}",
                    (length, FACE_BACKING_T, row_h),
                    xf.point(origin[0] + length * 0.5, origin[1] + normal_offset, z0 + row_h * 0.5),
                    material,
                    rotation_z=xf.rot,
                )
            )
    return objects


def _shallow_joint_marks(
    prefix: str,
    xf: Xf,
    face: str,
    origin: tuple[float, float, float],
    length: float,
    height: float,
    material: bpy.types.Material,
    seed: int,
) -> list[bpy.types.Object]:
    """Readable dark mortar/joint cues on top of raised source-scale blocks."""
    objects: list[bpy.types.Object] = []
    rng = random.Random(seed)
    normal_sign = -1.0 if face in {"front", "left"} else 1.0
    normal_offset = normal_sign * FACE_SURFACE_OFFSET
    course_count = max(3, int(round(height / B_WALL[2])))
    row_h = height / course_count
    for row in range(1, course_count):
        z = origin[2] + row_h * row
        segment_count = max(2, int(length / 0.75))
        for seg in range(segment_count):
            start = length * seg / segment_count + rng.uniform(0.01, 0.05)
            run = length / segment_count * rng.uniform(0.76, 1.02)
            if face in {"front", "back"}:
                objects.append(
                    add_box(
                        f"{prefix}_shallow_horizontal_joint_{row:02d}_{seg:02d}",
                        (0.024, run, 0.050),
                        xf.point(origin[0] + normal_offset, origin[1] + start + run * 0.5, z),
                        material,
                        rotation_z=xf.rot,
                    )
                )
            else:
                objects.append(
                    add_box(
                        f"{prefix}_shallow_horizontal_joint_{row:02d}_{seg:02d}",
                        (run, 0.024, 0.050),
                        xf.point(origin[0] + start + run * 0.5, origin[1] + normal_offset, z),
                        material,
                        rotation_z=xf.rot,
                    )
                )
    idx = 0
    for row in range(course_count):
        z = origin[2] + row_h * (row + 0.50)
        joint_h = row_h * rng.uniform(0.70, 0.94)
        step = B_WALL[1] * (1.00 if length > 2.2 else 0.82)
        u = step * (0.55 if row % 2 else 0.15)
        while u < length - 0.10:
            if rng.random() > 0.08:
                uj = u + rng.uniform(-0.045, 0.045)
                if face in {"front", "back"}:
                    objects.append(
                        add_box(
                            f"{prefix}_shallow_vertical_joint_{idx:02d}",
                            (0.024, 0.042, joint_h),
                            xf.point(origin[0] + normal_offset, origin[1] + uj, z),
                            material,
                            rotation_z=xf.rot,
                        )
                    )
                else:
                    objects.append(
                        add_box(
                            f"{prefix}_shallow_vertical_joint_{idx:02d}",
                            (0.042, 0.024, joint_h),
                            xf.point(origin[0] + uj, origin[1] + normal_offset, z),
                            material,
                            rotation_z=xf.rot,
                        )
                    )
                idx += 1
            u += step * rng.uniform(0.82, 1.18)
    return objects


def _source_scale_mortar_overlay(
    prefix: str,
    xf: Xf,
    face: str,
    origin: tuple[float, float, float],
    length: float,
    height: float,
    material: bpy.types.Material,
    seed: int,
    brick_len: float,
    course_h: float,
) -> list[bpy.types.Object]:
    """High-contrast surface mortar that keeps the wall solid but reads as OSBK courses."""
    objects: list[bpy.types.Object] = []
    rng = random.Random(seed)
    normal_sign = -1.0 if face in {"front", "left"} else 1.0
    normal_offset = normal_sign * (FACE_SURFACE_OFFSET + 0.010)
    horizontal_w = 0.034
    vertical_w = 0.030
    courses = max(1, math.ceil(height / course_h))

    for row in range(1, courses):
        z = origin[2] + row * course_h + rng.uniform(-0.010, 0.010)
        if z >= origin[2] + height - 0.025:
            continue
        u = 0.03 + rng.uniform(0.0, 0.04)
        idx = 0
        while u < length - 0.06:
            run = min(length - u - 0.025, brick_len * rng.uniform(0.88, 1.55))
            if run > 0.12:
                if face in {"front", "back"}:
                    objects.append(
                        add_box(
                            f"{prefix}_mortar_h_{row:02d}_{idx:02d}",
                            (0.030, run, horizontal_w),
                            xf.point(origin[0] + normal_offset, origin[1] + u + run * 0.5, z),
                            material,
                            rotation_z=xf.rot + math.radians(rng.uniform(-0.25, 0.25)),
                        )
                    )
                else:
                    objects.append(
                        add_box(
                            f"{prefix}_mortar_h_{row:02d}_{idx:02d}",
                            (run, 0.030, horizontal_w),
                            xf.point(origin[0] + u + run * 0.5, origin[1] + normal_offset, z),
                            material,
                            rotation_z=xf.rot + math.radians(rng.uniform(-0.25, 0.25)),
                        )
                    )
            u += run + rng.uniform(0.06, 0.16)
            idx += 1

    idx = 0
    for row in range(courses):
        z = origin[2] + row * course_h + course_h * 0.50 + rng.uniform(-0.006, 0.006)
        row_h = min(course_h, origin[2] + height - (origin[2] + row * course_h))
        if row_h <= 0.05:
            continue
        u = brick_len * (0.50 if row % 2 else 0.10) + rng.uniform(-0.025, 0.025)
        while u < length - 0.08:
            if rng.random() > 0.12:
                joint_h = row_h * rng.uniform(0.56, 0.86)
                if face in {"front", "back"}:
                    objects.append(
                        add_box(
                            f"{prefix}_mortar_v_{idx:03d}",
                            (0.030, vertical_w, joint_h),
                            xf.point(origin[0] + normal_offset, origin[1] + u, z),
                            material,
                            rotation_z=xf.rot,
                        )
                    )
                else:
                    objects.append(
                        add_box(
                            f"{prefix}_mortar_v_{idx:03d}",
                            (vertical_w, 0.030, joint_h),
                            xf.point(origin[0] + u, origin[1] + normal_offset, z),
                            material,
                            rotation_z=xf.rot,
                        )
                    )
                idx += 1
            u += brick_len * rng.uniform(0.82, 1.20)
    return objects


def _opposite_face(face: str) -> str:
    return {
        "front": "back",
        "back": "front",
        "left": "right",
        "right": "left",
    }[face]


def _opposite_block_origin(origin: tuple[float, float, float], face: str) -> tuple[float, float, float]:
    opposite = _opposite_face(face)
    shift = FACE_BACKING_T
    if opposite in {"front", "left"}:
        shift = -shift
    if face in {"front", "back"}:
        return (origin[0] - shift, origin[1], origin[2])
    return (origin[0], origin[1] - shift, origin[2])


def _pillar_stack(
    prefix: str,
    xf: Xf,
    min_x: float,
    min_y: float,
    height: float,
    seed: int,
    mats,
    sx: float = B_PILLAR[0],
    sy: float = B_PILLAR[1],
    course_h: float = B_PILLAR[2],
) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    courses = max(1, int(round(height / course_h)))
    actual_h = height / courses
    for row in range(courses):
        center = xf.point(min_x + sx * 0.5, min_y + sy * 0.5, row * actual_h + actual_h * 0.5)
        objects.append(
            add_brick(
                f"{prefix}_course_{row:02d}",
                center,
                (sx * 0.96, sy * 0.96, actual_h * 0.94),
                mats.stone,
                seed + row,
                jitter=0.006,
                rot_max_deg=0.6,
            )
        )
    return objects


def _wall_panel(
    prefix: str,
    xf: Xf,
    y0: float,
    length: float,
    z0: float,
    height: float,
    seed: int,
    mats,
    back: bool = False,
) -> list[bpy.types.Object]:
    x_face = -(B_PILLAR[0] + B_WALL[0]) if back else 0.0
    face_name = "back" if back else "front"
    origin = (x_face, y0, z0)
    brick_len = B_WALL[1] * (1.08 if length > 2.2 else 1.02)
    course_h = B_WALL[2] * 1.04
    gap = WALL_GAP
    objects = _face_backing_local(
        f"{prefix}_{face_name}",
        xf,
        face_name,
        origin,
        length,
        height,
        mats.dark_stone,
        course_h=course_h,
    )
    objects.extend(_brick_wall_local(
        f"{prefix}_{face_name}",
        xf,
        face_name,
        origin,
        length,
        height,
        mats.stone,
        seed,
        brick_len=brick_len,
        course_h=course_h,
        gap=gap,
        joint_material=mats.dark_stone,
    ))
    opposite_face = _opposite_face(face_name)
    opposite_origin = _opposite_block_origin(origin, face_name)
    objects.extend(
        _brick_wall_local(
            f"{prefix}_{opposite_face}_exterior",
            xf,
            opposite_face,
            opposite_origin,
            length,
            height,
            mats.stone,
            seed + 1500,
            brick_len=brick_len,
            course_h=course_h,
            gap=gap,
            joint_material=mats.dark_stone,
        )
    )
    objects.extend(
        _shallow_joint_marks(
            f"{prefix}_{face_name}",
            xf,
            face_name,
            origin,
            length,
            height,
            mats.dark_stone,
            seed + 1000,
        )
    )
    objects.extend(
        _source_scale_mortar_overlay(
            f"{prefix}_{face_name}",
            xf,
            face_name,
            origin,
            length,
            height,
            mats.dark_stone,
            seed + 2000,
            brick_len=brick_len,
            course_h=course_h,
        )
    )
    objects.extend(
        _source_scale_mortar_overlay(
            f"{prefix}_{opposite_face}_exterior_cuts",
            xf,
            opposite_face,
            opposite_origin,
            length,
            height,
            mats.dark_stone,
            seed + 3000,
            brick_len=brick_len,
            course_h=course_h,
        )
    )
    return objects


def _gate_side_lining(
    prefix: str,
    xf: Xf,
    y: float,
    face: str,
    height: float,
    seed: int,
    mats,
) -> list[bpy.types.Object]:
    origin = (-B_PILLAR[0] - 0.02, y, 0.0)
    length = B_PILLAR[0] + B_WALL[0] + 0.04
    objects = _face_backing_local(f"{prefix}_{face}", xf, face, origin, length, height, mats.dark_stone)
    objects.extend(
        _brick_wall_local(
            f"{prefix}_{face}",
            xf,
            face,
            origin,
            length,
            height,
            mats.stone,
            seed,
            brick_len=0.24,
            course_h=0.18,
            gap=0.042,
            joint_material=mats.dark_stone,
        )
    )
    return objects


def _portal_face_patch(
    prefix: str,
    xf: Xf,
    y0: float,
    length: float,
    z0: float,
    height: float,
    seed: int,
    mats,
) -> list[bpy.types.Object]:
    """Brick skin set just behind the outer front plane inside the gate opening."""
    origin = (-B_PILLAR[0] + 0.015, y0, z0)
    objects: list[bpy.types.Object] = []
    objects.extend(
        _brick_wall_local(
            prefix,
            xf,
            "back",
            origin,
            length,
            height,
            mats.stone,
            seed,
            brick_len=0.30,
            course_h=0.18,
            gap=0.042,
            joint_material=mats.dark_stone,
        )
    )
    return objects


def _gate_recess_shadow(
    prefix: str,
    xf: Xf,
    y0: float,
    length: float,
    z0: float,
    height: float,
    mats,
) -> list[bpy.types.Object]:
    """Tiny rear passage-depth cues kept well behind the open lower gate."""
    x = -B_PILLAR[0] - B_WALL[0] * 3.20
    cue_len = max(0.05, length * 0.16)
    cue_y = y0 + length * 0.5 - cue_len * 0.5
    return [
        _box_local(
            f"{prefix}_inner_floor_shadow",
            xf,
            (x, cue_y, z0 + 0.015),
            (0.006, cue_len, min(height, B_WALL[2] * 0.045)),
            mats.shadow,
        ),
        _box_local(
            f"{prefix}_inner_roof_shadow",
            xf,
            (x, cue_y, z0 + height - B_WALL[2] * 0.09),
            (0.006, cue_len, B_WALL[2] * 0.055),
            mats.shadow,
        ),
    ]


def build_front_pillars(xf: Xf, seed: int, mats, prefix: str) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    for idx, y in enumerate((0.0, 0.75 * INCH + B_PILLAR[1])):
        objects.extend(_pillar_stack(f"{prefix}_pair_{idx}", xf, 0.0, y, PILLAR_HEIGHT, seed + idx * 12, mats))
        objects.extend(
            add_crack_marks(
                f"{prefix}_cracks_{idx}",
                "front",
                xf.point(B_PILLAR[0] * 0.5, y - 0.004, PILLAR_HEIGHT * 0.5),
                (B_PILLAR[0], PILLAR_HEIGHT),
                mats.crack,
                seed + 100 + idx,
                2,
            )
        )
    return objects


def build_front_wall(xf: Xf, seed: int, mats) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []

    objects.extend(_wall_panel("front_wall_lower_a", xf, 0.0, 1.0 * INCH, 0.0, GATE_LOW_HEIGHT, seed, mats))
    objects.extend(
        _wall_panel(
            "front_wall_lower_b",
            xf,
            1.75 * INCH,
            1.0 * INCH,
            0.0,
            GATE_LOW_HEIGHT,
            seed + 1,
            mats,
        )
    )
    objects.extend(_wall_panel("front_wall_bridge", xf, 0.0, WALL_LEN, GATE_TOP_Z, WALL_HEIGHT - GATE_TOP_Z, seed + 2, mats))

    objects.extend(_pillar_stack("front_gate_left_side_pillar", xf, -B_PILLAR[0], 0.75 * INCH, PILLAR_HEIGHT, seed + 3, mats))
    objects.extend(
        _pillar_stack(
            "front_gate_right_side_pillar",
            xf,
            -B_PILLAR[0],
            2.0 * INCH - B_PILLAR[1],
            PILLAR_HEIGHT,
            seed + 4,
            mats,
        )
    )
    objects.extend(
        _brick_wall_local(
            "front_gate_inner_lintel",
            xf,
            "back",
            (-B_PILLAR[0], 0.75 * INCH, 1.0 * INCH - B_PILLAR[2]),
            0.75 * INCH + 2.5 * B_PILLAR[1],
            B_PILLAR[2],
            mats.stone,
            seed + 5,
            brick_len=B_PILLAR[1],
            course_h=B_PILLAR[2],
            gap=0.010,
            joint_material=mats.dark_stone,
        )
    )
    objects.extend(_wall_panel("front_wall_back_lower_a", xf, 0.0, 1.0 * INCH, 0.0, GATE_LOW_HEIGHT, seed + 7, mats, back=True))
    objects.extend(
        _wall_panel(
            "front_wall_back_lower_b",
            xf,
            1.75 * INCH,
            1.0 * INCH,
            0.0,
            GATE_LOW_HEIGHT,
            seed + 8,
            mats,
            back=True,
        )
    )
    return objects


def build_wall(xf: Xf, seed: int, mats, prefix: str) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    objects.extend(_wall_panel(f"{prefix}_front_face", xf, 0.0, WALL_LEN, 0.0, WALL_HEIGHT, seed, mats))
    objects.extend(_wall_panel(f"{prefix}_back_face", xf, 0.0, WALL_LEN, 0.0, WALL_HEIGHT, seed + 1, mats, back=True))
    objects.extend(
        _deck_rim_blocks(
            f"{prefix}_walkway_edge_courses",
            xf,
            -B_PILLAR[0] + B_WALL[0] * 0.40,
            0.0,
            GATE_WALK_Z + B_PILLAR[2] * 0.08,
            B_WALL[0] * 0.24,
            WALL_LEN,
            seed + 91,
            mats,
        )
    )
    objects.extend(
        add_crack_marks(
            f"{prefix}_front_cracks",
            "front",
            xf.point(0.0, WALL_LEN * 0.5, WALL_HEIGHT * 0.5),
            (WALL_LEN, WALL_HEIGHT),
            mats.crack,
            seed + 500,
            12,
        )
    )
    return objects


def build_tower(xf: Xf, seed: int, mats, prefix: str) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    txf = xf.then(0.75 * INCH, 0.0)

    objects.extend(_wall_panel(f"{prefix}_south_wall", txf, 0.0, TOWER_SIDE, 0.0, TOWER_HEIGHT, seed, mats))
    objects.extend(_wall_panel(f"{prefix}_east_wall", txf.then(0.0, TOWER_SIDE, math.radians(90)), 0.0, TOWER_SIDE, 0.0, TOWER_HEIGHT, seed + 1, mats))
    objects.extend(_wall_panel(f"{prefix}_north_wall", txf.then(-TOWER_SIDE, TOWER_SIDE, math.radians(180)), 0.0, TOWER_SIDE, 0.0, TOWER_HEIGHT, seed + 2, mats))
    objects.extend(_wall_panel(f"{prefix}_west_wall", txf.then(-TOWER_SIDE, 0.0, math.radians(270)), 0.0, TOWER_SIDE, 0.0, TOWER_HEIGHT, seed + 3, mats))

    strip_w = B_WALL[0] * 1.075
    strip_d = B_WALL[1] * 0.575
    for idx, (px, py, rot) in enumerate(
        (
            (0.0, 0.0, math.radians(270)),
            (0.0, TOWER_SIDE, 0.0),
            (-TOWER_SIDE - B_WALL[0] * 1.05, TOWER_SIDE, 0.0),
            (-TOWER_SIDE - B_WALL[0] * 1.05, 0.0, math.radians(270)),
        )
    ):
        pxf = txf.then(px, py, rot)
        objects.extend(
            _pillar_stack(
                f"{prefix}_vertical_corner_strip_{idx}",
                pxf,
                0.0,
                0.0,
                TOWER_HEIGHT,
                seed + 4 + idx,
                mats,
                sx=strip_w,
                sy=strip_d,
                course_h=B_WALL[2],
            )
        )

    objects.extend(_tower_cap_rim_blocks(f"{prefix}_top_integrated_edge_courses", txf, seed + 81, mats))
    objects.extend(
        add_crack_marks(
            f"{prefix}_visible_cracks",
            "front",
            txf.point(-TOWER_SIDE * 0.5, -0.01, TOWER_HEIGHT * 0.5),
            (TOWER_SIDE, TOWER_HEIGHT),
            mats.crack,
            seed + 120,
            8,
        )
    )
    return objects


def build_base(mats) -> list[bpy.types.Object]:
    main_min_x = -3.75 * INCH
    main_min_y = -1.00 * INCH
    main_size_x = 4.50 * INCH
    return [
        add_box_min(
            "base_main_source_plate",
            (main_min_x, main_min_y, 0.0),
            (main_size_x, 4.75 * INCH, BASE_Z),
            mats.stone,
        ),
        add_box_min(
            "base_front_approach_plate",
            (0.0, 1.0 * INCH - B_PILLAR[1] - 0.15 * INCH, 0.0),
            (1.80 * INCH, 1.50 * INCH, BASE_Z),
            mats.stone,
        ),
    ]


def build_model() -> list[bpy.types.Object]:
    mats = make_materials()
    objects: list[bpy.types.Object] = []
    objects.extend(build_base(mats))

    root = Xf()
    objects.extend(build_front_wall(root, 123, mats))

    for idx, (x, seed) in enumerate(((0.5 * INCH, 7945), (1.0 * INCH, 48987), (1.5 * INCH, 8971))):
        objects.extend(build_front_pillars(root.then(x, 1.0 * INCH - B_PILLAR[1]), seed, mats, f"front_free_pillars_{idx}"))

    objects.extend(build_tower(root.then(-0.25 * INCH, 0.0, math.radians(-90)), 87894, mats, "front_right_tower"))
    objects.extend(build_tower(root.then(-0.25 * INCH, WALL_LEN, 0.0), 3146, mats, "back_right_tower"))
    objects.extend(build_wall(root.then(0.0, -0.25 * B_WALL[1], math.radians(90)), 87263, mats, "right_side_wall"))
    objects.extend(build_wall(root.then(-B_PILLAR[0], WALL_LEN + 1.25 * B_PILLAR[1], math.radians(90)), 98765136, mats, "left_side_wall"))
    objects.extend(build_tower(root.then(-WALL_LEN, 0.0, math.radians(-180)), 7516587, mats, "front_left_tower"))
    objects.extend(build_tower(root.then(-WALL_LEN, WALL_LEN, math.radians(-270)), 634798, mats, "back_left_tower"))
    objects.extend(build_wall(root.then(-3.0 * INCH, WALL_LEN - 0.25 * B_WALL[1], math.radians(180)), 735641, mats, "rear_wall"))

    for obj in objects:
        obj.select_set(True)
    bpy.ops.object.shade_flat()
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

from __future__ import annotations
from dataclasses import dataclass
from mathutils import Vector
from typing import Iterable
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable Blender primitive builders for the miniature house model."""





Color = tuple[float, float, float, float]


@dataclass(frozen=True)
class MaterialPalette:
    wall_a: bpy.types.Material
    wall_b: bpy.types.Material
    wall_dark: bpy.types.Material
    roof_a: bpy.types.Material
    roof_b: bpy.types.Material
    roof_dark: bpy.types.Material
    trim: bpy.types.Material
    shadow: bpy.types.Material


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(name: str, color: Color, roughness: float = 0.65) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = color
    nodes = mat.node_tree.nodes
    nodes.clear()
    shader = nodes.new(type="ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = roughness
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return mat


def make_palette() -> MaterialPalette:
    return MaterialPalette(
        wall_a=make_material("varied burnt lumber", (0.46, 0.13, 0.015, 1.0)),
        wall_b=make_material("orange brown lumber", (0.62, 0.20, 0.02, 1.0)),
        wall_dark=make_material("dark board recess", (0.20, 0.055, 0.006, 1.0)),
        roof_a=make_material("deep terracotta roof", (0.58, 0.13, 0.015, 1.0)),
        roof_b=make_material("burnt orange roof", (0.78, 0.20, 0.02, 1.0)),
        roof_dark=make_material("dark roof cap", (0.30, 0.065, 0.012, 1.0)),
        trim=make_material("pale gray cut trim", (0.78, 0.78, 0.73, 1.0)),
        shadow=make_material("deep gray recess", (0.34, 0.34, 0.31, 1.0)),
    )


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
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material is not None:
        obj.data.materials.append(material)
    return obj


def add_triangular_prism(
    name: str,
    length: float,
    thickness: float,
    height: float,
    location: tuple[float, float, float],
    material: bpy.types.Material,
    axis: str,
) -> bpy.types.Object:
    """Create an upright gable triangle extruded along X or Y."""
    if axis == "x":
        verts = [
            (-length / 2, -thickness / 2, 0),
            (length / 2, -thickness / 2, 0),
            (0, -thickness / 2, height),
            (-length / 2, thickness / 2, 0),
            (length / 2, thickness / 2, 0),
            (0, thickness / 2, height),
        ]
    elif axis == "y":
        verts = [
            (-thickness / 2, -length / 2, 0),
            (-thickness / 2, length / 2, 0),
            (-thickness / 2, 0, height),
            (thickness / 2, -length / 2, 0),
            (thickness / 2, length / 2, 0),
            (thickness / 2, 0, height),
        ]
    else:
        raise ValueError(f"unknown prism axis {axis!r}")

    faces = [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.data.materials.append(material)
    return obj


def add_cylinder_between(
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    material: bpy.types.Material,
    vertices: int = 12,
) -> bpy.types.Object:
    start_v = Vector(start)
    end_v = Vector(end)
    direction = end_v - start_v
    length = direction.length
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=length, location=(start_v + end_v) / 2)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(material)
    return obj


def board_material(index: int, palette: MaterialPalette) -> bpy.types.Material:
    sequence = [palette.wall_a, palette.wall_b, palette.wall_a, palette.wall_dark, palette.wall_b]
    return sequence[index % len(sequence)]


def stone_material(row: int, col: int, palette: MaterialPalette) -> bpy.types.Material:
    sequence = [palette.wall_a, palette.wall_b, palette.wall_a, palette.wall_dark, palette.wall_b, palette.wall_a]
    return sequence[(row + col * 2) % len(sequence)]


def roof_material(index: int, palette: MaterialPalette) -> bpy.types.Material:
    sequence = [palette.roof_a, palette.roof_a, palette.roof_b, palette.roof_a, palette.roof_dark]
    return sequence[index % len(sequence)]


def add_vertical_board_wall(
    name: str,
    length: float,
    height: float,
    thickness: float,
    board_width: float,
    origin: tuple[float, float, float],
    axis: str,
    palette: MaterialPalette,
    openings: Iterable[tuple[float, float, float, float, str]] = (),
) -> list[bpy.types.Object]:
    """Build a wall face from vertical board strips, skipping door/window spans."""
    objs: list[bpy.types.Object] = []
    x0, y0, z0 = origin
    count = math.ceil(length / board_width)
    opening_list = list(openings)

    for i in range(count):
        start = i * board_width
        width = min(board_width, length - start)
        center = start + width / 2
        spans: list[tuple[float, float]] = [(0.0, height)]
        for open_center, open_width, open_z, open_h, _kind in opening_list:
            if abs(center - open_center) <= open_width / 2 + width / 2:
                next_spans: list[tuple[float, float]] = []
                for low, high in spans:
                    if low < open_z:
                        next_spans.append((low, min(high, open_z)))
                    if open_z + open_h < high:
                        next_spans.append((max(low, open_z + open_h), high))
                spans = [(low, high) for low, high in next_spans if high - low > 1.0]

        for low, high in spans:
            if axis == "x":
                loc = (x0 + center, y0, z0 + (low + high) / 2)
                size = (width * 0.92, thickness, high - low)
            elif axis == "y":
                loc = (x0, y0 + center, z0 + (low + high) / 2)
                size = (thickness, width * 0.92, high - low)
            else:
                raise ValueError(f"unknown wall axis {axis!r}")
            objs.append(add_box(f"{name}_board_{i:02d}", size, loc, board_material(i, palette)))
    return objs


def _span_is_blocked(center: float, width: float, z_center: float, height: float, openings: Iterable[tuple[float, float, float, float, str]]) -> bool:
    for open_center, open_width, open_z, open_h, _kind in openings:
        overlaps_x = abs(center - open_center) < (width + open_width) / 2
        overlaps_z = abs(z_center - (open_z + open_h / 2)) < (height + open_h) / 2
        if overlaps_x and overlaps_z:
            return True
    return False


def add_stone_wall_grid(
    name: str,
    length: float,
    height: float,
    thickness: float,
    stone_width: float,
    stone_height: float,
    origin: tuple[float, float, float],
    axis: str,
    palette: MaterialPalette,
    openings: Iterable[tuple[float, float, float, float, str]] = (),
) -> list[bpy.types.Object]:
    """Approximate wallblock()/stone() as narrow vertical modules with subtle relief."""
    objs: list[bpy.types.Object] = []
    x0, y0, z0 = origin
    opening_list = list(openings)
    cols = math.ceil(length / stone_width)

    for col in range(cols):
        start = col * stone_width
        block_w = min(stone_width, length - start)
        if block_w <= 2.0:
            continue
        center = start + block_w / 2
        spans: list[tuple[float, float]] = [(0.0, height)]
        for open_center, open_width, open_z, open_h, _kind in opening_list:
            if abs(center - open_center) <= (open_width + block_w) / 2:
                next_spans: list[tuple[float, float]] = []
                for low, high in spans:
                    if low < open_z:
                        next_spans.append((low, min(high, open_z)))
                    if open_z + open_h < high:
                        next_spans.append((max(low, open_z + open_h), high))
                spans = [(low, high) for low, high in next_spans if high - low > 2.0]

        relief = thickness * (0.04 if col % 3 else 0.13)
        mat = stone_material(0, col, palette)
        for segment, (low, high) in enumerate(spans):
            segment_h = high - low
            z_center = z0 + low + segment_h / 2
            inset_gap = 1.2 if segment > 0 or low > 0.0 else 0.0
            if axis == "x":
                size = (block_w * 0.86, thickness * 0.72 + relief, segment_h - inset_gap)
                loc = (x0 + center, y0, z_center)
                edge_size = (max(2.8, block_w * 0.12), thickness * 0.80 + relief, segment_h - inset_gap)
                edge_loc = (x0 + center + block_w * 0.31, y0, z_center)
            elif axis == "y":
                size = (thickness * 0.72 + relief, block_w * 0.86, segment_h - inset_gap)
                loc = (x0, y0 + center, z_center)
                edge_size = (thickness * 0.80 + relief, max(2.8, block_w * 0.12), segment_h - inset_gap)
                edge_loc = (x0, y0 + center + block_w * 0.31, z_center)
            else:
                raise ValueError(f"unknown wall axis {axis!r}")
            objs.append(add_box(f"{name}_stone_c{col:02d}_s{segment:02d}", size, loc, mat))
            if col % 2 == 0:
                objs.append(add_box(f"{name}_stone_edge_c{col:02d}_s{segment:02d}", edge_size, edge_loc, palette.wall_dark))
    return objs


def add_opening_trim(
    name: str,
    center: float,
    width: float,
    z: float,
    height: float,
    wall_coord: float,
    axis: str,
    palette: MaterialPalette,
    kind: str,
) -> list[bpy.types.Object]:
    objs: list[bpy.types.Object] = []
    frame = 8.5
    sill = 10.0
    depth = 14.0
    if kind == "door":
        panel_mat = palette.shadow
        split_count = 1
    else:
        panel_mat = palette.shadow
        split_count = 2

    if axis == "x":
        y = wall_coord
        objs.append(add_box(f"{name}_recess", (width * 0.88, depth * 1.00, height * 0.90), (center, y, z + height * 0.48), panel_mat))
        objs.append(add_box(f"{name}_lintel", (width * 1.08, depth * 1.15, frame), (center, y, z + height + frame / 2), palette.trim))
        objs.append(add_box(f"{name}_sill", (width * 1.10, depth * 1.20, sill), (center, y, max(4.0, z - sill / 2)), palette.trim))
        objs.append(add_box(f"{name}_left_jamb", (frame, depth * 1.15, height), (center - width / 2, y, z + height / 2), palette.trim))
        objs.append(add_box(f"{name}_right_jamb", (frame, depth * 1.15, height), (center + width / 2, y, z + height / 2), palette.trim))
        if kind == "window":
            objs.append(add_box(f"{name}_mid_vert", (frame * 0.55, depth * 1.18, height * 0.68), (center, y, z + height * 0.50), palette.trim))
            objs.append(add_box(f"{name}_mid_horiz", (width * 0.68, depth * 1.18, frame * 0.50), (center, y, z + height * 0.50), palette.trim))
    elif axis == "y":
        x = wall_coord
        objs.append(add_box(f"{name}_recess", (depth * 1.00, width * 0.88, height * 0.90), (x, center, z + height * 0.48), panel_mat))
        objs.append(add_box(f"{name}_lintel", (depth * 1.15, width * 1.08, frame), (x, center, z + height + frame / 2), palette.trim))
        objs.append(add_box(f"{name}_sill", (depth * 1.20, width * 1.10, sill), (x, center, max(4.0, z - sill / 2)), palette.trim))
        objs.append(add_box(f"{name}_left_jamb", (depth * 1.15, frame, height), (x, center - width / 2, z + height / 2), palette.trim))
        objs.append(add_box(f"{name}_right_jamb", (depth * 1.15, frame, height), (x, center + width / 2, z + height / 2), palette.trim))
        if kind == "window":
            objs.append(add_box(f"{name}_mid_vert", (depth * 1.18, frame * 0.55, height * 0.68), (x, center, z + height * 0.50), palette.trim))
            objs.append(add_box(f"{name}_mid_horiz", (depth * 1.18, width * 0.68, frame * 0.50), (x, center, z + height * 0.50), palette.trim))
    else:
        raise ValueError(f"unknown trim axis {axis!r}")

    if split_count:
        return objs
    return objs


def add_roof_tile(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    rotation: tuple[float, float, float],
    material: bpy.types.Material,
) -> list[bpy.types.Object]:
    """SCAD-like roofstone: a flat base with one raised downslope rib."""
    tile = add_box(f"{name}_flat_plate", size, center, material, rotation)
    rib_center = (center[0] - size[0] * 0.22, center[1], center[2] + size[2] * 0.01)
    rib = add_box(f"{name}_raised_downslope_rib", (size[0] * 0.20, size[1] * 1.85, size[2] * 0.96), rib_center, material, rotation)
    return [tile, rib]


# ============================================================
# Procedural assembly
# ============================================================

"""High-level assembly for the miniature brick and lumber house model."""






HOUSE_X = 600.0
HOUSE_Y = 500.0
WALL_H = 280.0
ROOF_RISE = 200.0
ROOF_OVERHANG_Y = 70.0
ROOF_OVERHANG_X = 50.0
BOARD_W = 30.0
WALL_THICKNESS = 31.0

WIN_W = 80.0
WIN_H = 100.0
WIN_Z = 82.0
DOOR_W = 100.0
DOOR_H = 220.0
DOOR_Z = 0.0
FIRST_OPENING = 120.0
OPENING_GAP = 100.0


def _opening_pattern(length: float, order: tuple[int, ...]) -> list[tuple[float, float, float, float, str]]:
    openings: list[tuple[float, float, float, float, str]] = []
    cursor = FIRST_OPENING
    for item in order:
        width = WIN_W if item == 1 else DOOR_W
        height = WIN_H if item == 1 else DOOR_H
        z = WIN_Z if item == 1 else DOOR_Z
        if cursor + width / 2 > length - 35:
            break
        openings.append((cursor, width, z, height, "window" if item == 1 else "door"))
        cursor += width + OPENING_GAP
    return openings


def _build_wall(
    name: str,
    length: float,
    axis: str,
    constant: float,
    start: float,
    palette: MaterialPalette,
    order: tuple[int, ...],
    face_sign: float,
) -> list[bpy.types.Object]:
    openings = _opening_pattern(length, order)
    objs: list[bpy.types.Object] = []

    if axis == "x":
        y = constant
        objs.extend(
            add_stone_wall_grid(
                name,
                length,
                WALL_H,
                WALL_THICKNESS,
                BOARD_W,
                WALL_H,
                (start, y, 0.0),
                "x",
                palette,
                openings,
            )
        )
        trim_y = y - face_sign * (WALL_THICKNESS * 0.54)
        for index, (center, width, z, height, kind) in enumerate(openings):
            objs.extend(add_opening_trim(f"{name}_{kind}_{index}", start + center, width, z, height, trim_y, "x", palette, kind))
    else:
        x = constant
        objs.extend(
            add_stone_wall_grid(
                name,
                length,
                WALL_H,
                WALL_THICKNESS,
                BOARD_W,
                WALL_H,
                (x, start, 0.0),
                "y",
                palette,
                openings,
            )
        )
        trim_x = x + face_sign * (WALL_THICKNESS * 0.54)
        for index, (center, width, z, height, kind) in enumerate(openings):
            objs.extend(add_opening_trim(f"{name}_{kind}_{index}", start + center, width, z, height, trim_x, "y", palette, kind))

    return objs


def _add_corner_posts(palette: MaterialPalette) -> list[bpy.types.Object]:
    objs: list[bpy.types.Object] = []
    post_size = 21.0
    for x in (0.0, HOUSE_X):
        for y in (0.0, -HOUSE_Y):
            objs.append(add_box("gray_corner_post", (post_size, post_size, WALL_H), (x, y, WALL_H / 2), palette.trim))
    return objs


def _build_gables(palette: MaterialPalette) -> list[bpy.types.Object]:
    objs: list[bpy.types.Object] = []
    objs.append(add_triangular_prism("right_gable_triangle", HOUSE_Y, 18.0, ROOF_RISE, (HOUSE_X + 7.0, -HOUSE_Y / 2, WALL_H), palette.wall_dark, "y"))
    objs.append(add_triangular_prism("left_gable_triangle", HOUSE_Y, 18.0, ROOF_RISE, (-7.0, -HOUSE_Y / 2, WALL_H), palette.wall_dark, "y"))

    for x, prefix in ((HOUSE_X + 10.0, "right"), (-10.0, "left")):
        for i in range(9):
            y = -HOUSE_Y + i * 58.0 + 19.0
            distance = abs(y + HOUSE_Y / 2)
            local_h = max(0.0, ROOF_RISE * (1.0 - distance / (HOUSE_Y / 2)))
            if local_h < 24.0:
                continue
            mat = palette.wall_a if i % 3 else palette.wall_dark
            objs.append(add_box(f"{prefix}_gable_block_{i:02d}", (10.0, 30.0, local_h + 2.0), (x, y, WALL_H + local_h / 2), mat))
    return objs


def _roof_vectors(side: int, angle: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the rotated roof local-Z axis and the outward face normal."""
    local_z_axis = (0.0, -side * math.sin(angle), math.cos(angle))
    outward_normal = (0.0, side * math.cos(angle), math.sin(angle))
    return local_z_axis, outward_normal


def _build_roof_face(name: str, side: int, palette: MaterialPalette) -> list[bpy.types.Object]:
    objs: list[bpy.types.Object] = []
    half = HOUSE_Y / 2
    base_slope_len = math.sqrt(half * half + ROOF_RISE * ROOF_RISE)
    slope_len = base_slope_len + ROOF_OVERHANG_Y
    roof_width = HOUSE_X + ROOF_OVERHANG_X * 2
    angle = math.atan2(half, ROOF_RISE)
    rot_x = side * angle

    # Thin backing plane gives the clipped sloped silhouette under the repeated roofstone skin.
    local_z_axis, outward_normal = _roof_vectors(side, angle)
    ridge_y = -HOUSE_Y / 2
    ridge_z = WALL_H + ROOF_RISE
    center_y = ridge_y - local_z_axis[1] * (slope_len / 2)
    center_z = ridge_z - local_z_axis[2] * (slope_len / 2)

    def roof_point(x: float, local_z: float, lift: float) -> tuple[float, float, float]:
        return (
            x,
            center_y + local_z_axis[1] * local_z + outward_normal[1] * lift,
            center_z + local_z_axis[2] * local_z + outward_normal[2] * lift,
        )

    surface_lift = 9.5
    objs.append(
        add_box(
            f"{name}_under_roof_plane",
            (roof_width, 7.0, slope_len),
            (HOUSE_X / 2, center_y, center_z),
            palette.roof_dark,
            (rot_x, 0.0, 0.0),
        )
    )

    column_w = 27.5
    col_pitch = 31.0
    segment_h = 39.5
    segment_pitch = 39.5
    ridge_limit = slope_len / 2 - 10.0
    eave_limit = -slope_len / 2 + 8.0
    run_len = ridge_limit - eave_limit
    column_count = int(math.floor(roof_width / col_pitch))
    start_x = HOUSE_X / 2 - ((column_count - 1) * col_pitch) / 2

    for col in range(column_count):
        x = start_x + col * col_pitch
        if x < -ROOF_OVERHANG_X + column_w / 2 or x > HOUSE_X + ROOF_OVERHANG_X - column_w / 2:
            continue
        mat = roof_material(col, palette)
        base_lift = surface_lift + 2.2 + (0.25 if col % 4 == 0 else 0.0)
        objs.extend(
            add_roof_tile(
                f"{name}_roofstone_column_{col:02d}",
                roof_point(x, (ridge_limit + eave_limit) / 2, base_lift),
                (column_w, 6.5, run_len),
                (rot_x, 0.0, 0.0),
                mat,
            )
        )

        segment_count = int(math.ceil(run_len / segment_pitch))
        for seg in range(segment_count):
            local_z = ridge_limit - segment_h / 2 - seg * segment_pitch
            if local_z < eave_limit + segment_h / 2:
                continue
            rib_lift = base_lift + 3.0
            objs.append(
                add_box(
                    f"{name}_roofstone_rib_segment_c{col:02d}_s{seg:02d}",
                    (column_w * 0.17, 12.0, segment_h * 0.86),
                    roof_point(x - column_w * 0.22, local_z, rib_lift),
                    mat,
                    (rot_x, 0.0, 0.0),
                )
            )
            if seg > 0:
                objs.append(
                    add_box(
                        f"{name}_roofstone_segment_seam_c{col:02d}_s{seg:02d}",
                        (column_w * 0.82, 7.4, 1.7),
                        roof_point(x + column_w * 0.02, local_z + segment_h / 2, base_lift + 1.2),
                        palette.roof_dark,
                        (rot_x, 0.0, 0.0),
                    )
                )

    for col in range(column_count + 1):
        x = start_x - col_pitch / 2 + col * col_pitch
        if -ROOF_OVERHANG_X - 1.0 <= x <= HOUSE_X + ROOF_OVERHANG_X + 1.0:
            objs.append(
                add_box(
                    f"{name}_roofstone_column_shadow_{col:02d}",
                    (1.8, 7.8, run_len),
                    roof_point(x, (ridge_limit + eave_limit) / 2, surface_lift + 2.0),
                    palette.roof_dark,
                    (rot_x, 0.0, 0.0),
                )
            )

    for x in (-ROOF_OVERHANG_X - 2.0, HOUSE_X + ROOF_OVERHANG_X + 2.0):
        objs.append(add_box(f"{name}_straight_rake_edge", (4.2, 8.0, slope_len + 5.0), roof_point(x, 0.0, surface_lift + 1.0), palette.trim, (rot_x, 0.0, 0.0)))

    objs.append(add_box(f"{name}_straight_eave_edge", (roof_width + 2.0, 6.0, 4.2), roof_point(HOUSE_X / 2, eave_limit - 3.0, surface_lift + 1.0), palette.roof_dark, (rot_x, 0.0, 0.0)))
    return objs


def _build_roof(palette: MaterialPalette) -> list[bpy.types.Object]:
    objs = []
    objs.extend(_build_roof_face("front_roof", 1, palette))
    objs.extend(_build_roof_face("rear_roof", -1, palette))
    ridge_z = WALL_H + ROOF_RISE + 8.0
    ridge_w = HOUSE_X + ROOF_OVERHANG_X * 2 + 8.0
    objs.append(add_box("tight_dark_roof_ridge_cap", (ridge_w, 18.0, 11.0), (HOUSE_X / 2, -HOUSE_Y / 2, ridge_z), palette.roof_dark))
    return objs


def _build_steps(palette: MaterialPalette) -> list[bpy.types.Object]:
    return [
        add_box("front_door_low_step", (118.0, 28.0, 12.0), (120.0, 22.0, 6.0), palette.trim),
        add_box("side_door_low_step", (28.0, 118.0, 12.0), (HOUSE_X + 22.0, -120.0, 6.0), palette.trim),
    ]


def _join_detail_meshes(prefixes: tuple[str, ...]) -> None:
    for prefix in prefixes:
        matches = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.name.startswith(prefix)]
        if len(matches) < 2:
            continue
        bpy.ops.object.select_all(action="DESELECT")
        for obj in matches:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = matches[0]
        bpy.ops.object.join()
        bpy.context.object.name = f"{prefix}_merged"


def build_model() -> list[bpy.types.Object]:
    palette = make_palette()
    objs: list[bpy.types.Object] = []

    # Front/rear run along X. Side walls run along Y from the front corner to the rear corner.
    objs.extend(_build_wall("front_wall", HOUSE_X, "x", 0.0, 0.0, palette, (2, 1, 1, 1, 1, 1), face_sign=-1.0))
    objs.extend(_build_wall("rear_wall", HOUSE_X, "x", -HOUSE_Y, 0.0, palette, (2, 1, 1, 1, 1, 1), face_sign=1.0))
    objs.extend(_build_wall("right_wall", HOUSE_Y, "y", HOUSE_X, -HOUSE_Y, palette, (2, 1, 1, 1, 1), face_sign=1.0))
    objs.extend(_build_wall("left_wall", HOUSE_Y, "y", 0.0, -HOUSE_Y, palette, (1, 2, 1, 1, 1), face_sign=-1.0))
    objs.extend(_add_corner_posts(palette))
    objs.extend(_build_steps(palette))
    objs.extend(_build_gables(palette))
    objs.extend(_build_roof(palette))

    _join_detail_meshes(
        (
            "front_wall_stone",
            "rear_wall_stone",
            "right_wall_stone",
            "left_wall_stone",
            "front_roof_long_roofstone",
            "rear_roof_long_roofstone",
            "front_roof_roofstone_course",
            "rear_roof_roofstone_course",
            "front_roof_roofstone_plate",
            "rear_roof_roofstone_plate",
            "front_roof_roofstone_rib",
            "rear_roof_roofstone_rib",
            "front_roof_roofstone_transverse",
            "rear_roof_roofstone_transverse",
            "front_roof_roofstone_downslope",
            "rear_roof_roofstone_downslope",
            "front_roof_roofstone_low_valley",
            "rear_roof_roofstone_low_valley",
            "front_roof_roofstone_skin_break",
            "rear_roof_roofstone_skin_break",
            "front_roof_roofstone_short_rib",
            "rear_roof_roofstone_short_rib",
            "front_roof_roofstone_vertical_gap",
            "rear_roof_roofstone_vertical_gap",
            "front_roof_roofstone_course_shadow",
            "rear_roof_roofstone_course_shadow",
            "front_roof_roofstone_column",
            "rear_roof_roofstone_column",
            "front_roof_roofstone_rib_segment",
            "rear_roof_roofstone_rib_segment",
            "front_roof_roofstone_segment_seam",
            "rear_roof_roofstone_segment_seam",
            "front_roof_roofstone_column_shadow",
            "rear_roof_roofstone_column_shadow",
            "right_gable_block",
            "left_gable_block",
        )
    )

    return objs


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

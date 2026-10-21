from __future__ import annotations
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the small-parts cabinet model."""





MaterialMap = dict[str, bpy.types.Material]


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material:
        obj.data.materials.append(material)
    return obj


def add_cylinder(
    name: str,
    diameter: float,
    height: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
    material: bpy.types.Material | None = None,
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
        obj.data.materials.append(material)
    return obj


def add_sphere(
    name: str,
    radius: float,
    location: tuple[float, float, float],
    material: bpy.types.Material | None = None,
    segments: int = 48,
    rings: int = 24,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=rings,
        radius=radius,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    if material:
        obj.data.materials.append(material)
    return obj


def make_mat(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.55,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
    material.diffuse_color = color
    return material


def cabinet_materials() -> MaterialMap:
    """Colors approximate the OpenSCAD preview: yellow shell, olive interiors."""
    return {
        "shell": make_mat("openscad_yellow_shell", (0.93, 0.76, 0.06, 1.0)),
        "inside": make_mat("olive_green_inner_faces", (0.43, 0.62, 0.18, 1.0)),
        "shadow": make_mat("dark_green_voids", (0.08, 0.17, 0.05, 1.0)),
        "highlight": make_mat("nub_highlight_yellow", (1.0, 0.82, 0.08, 1.0)),
    }


def add_perforated_y_wall(
    name: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    holes: list[tuple[float, float]],
    diagonal: float,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Build a thick y-facing wall mesh with real diamond through-holes."""
    half = diagonal / 2.0
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    vertex_indices: dict[tuple[int, int, int], int] = {}

    def vertex_index(point: tuple[float, float, float]) -> int:
        key = tuple(round(value * 1000000.0) for value in point)
        existing = vertex_indices.get(key)
        if existing is not None:
            return existing
        vertex_indices[key] = len(verts)
        verts.append(point)
        return len(verts) - 1

    def add_face(points: list[tuple[float, float, float]]) -> None:
        cleaned: list[tuple[float, float, float]] = []
        for point in points:
            if not cleaned or _distance_squared(cleaned[-1], point) > 1.0e-12:
                cleaned.append(point)
        if len(cleaned) > 1 and _distance_squared(cleaned[0], cleaned[-1]) <= 1.0e-12:
            cleaned.pop()
        if len(cleaned) < 3:
            return
        faces.append(tuple(vertex_index(point) for point in cleaned))

    def hole_edges_at_z(center_x: float, center_z: float, z: float) -> tuple[float, float]:
        span = max(0.0, half - abs(z - center_z))
        return center_x - span, center_x + span

    def add_surface(y: float, outward_negative_y: bool) -> None:
        z_breaks = [z_min, z_max]
        for _, center_z in holes:
            for z in (center_z - half, center_z, center_z + half):
                if z_min < z < z_max:
                    z_breaks.append(z)
        z_breaks = _unique_sorted(z_breaks)

        for lower_z, upper_z in zip(z_breaks, z_breaks[1:]):
            if upper_z - lower_z <= 1.0e-7:
                continue
            mid_z = (lower_z + upper_z) / 2.0
            active = [
                (center_x, center_z)
                for center_x, center_z in holes
                if abs(mid_z - center_z) < half
            ]
            active.sort(key=lambda item: hole_edges_at_z(item[0], item[1], mid_z)[0])

            boundaries: list[tuple[object, object]] = []
            left_boundary: object = x_min
            for hole in active:
                boundaries.append((left_boundary, ("left", hole)))
                left_boundary = ("right", hole)
            boundaries.append((left_boundary, x_max))

            for left, right in boundaries:
                left_lower = _boundary_x(left, lower_z, half)
                right_lower = _boundary_x(right, lower_z, half)
                left_upper = _boundary_x(left, upper_z, half)
                right_upper = _boundary_x(right, upper_z, half)
                if max(right_lower - left_lower, right_upper - left_upper) <= 1.0e-7:
                    continue
                points = [
                    (left_lower, y, lower_z),
                    (right_lower, y, lower_z),
                    (right_upper, y, upper_z),
                    (left_upper, y, upper_z),
                ]
                if not outward_negative_y:
                    points.reverse()
                add_face(points)

    def add_edge_wall(points: list[tuple[float, float]], reverse: bool = False) -> None:
        for (x0, z0), (x1, z1) in zip(points, points[1:] + points[:1]):
            face_points = [
                (x0, y_min, z0),
                (x1, y_min, z1),
                (x1, y_max, z1),
                (x0, y_max, z0),
            ]
            if reverse:
                face_points.reverse()
            add_face(face_points)

    add_surface(y_min, outward_negative_y=True)
    add_surface(y_max, outward_negative_y=False)

    add_edge_wall(
        [(x_min, z_min), (x_max, z_min), (x_max, z_max), (x_min, z_max)],
        reverse=False,
    )
    for center_x, center_z in holes:
        add_edge_wall(
            [
                (center_x, center_z + half),
                (center_x + half, center_z),
                (center_x, center_z - half),
                (center_x - half, center_z),
            ],
            reverse=True,
        )

    mesh = bpy.data.meshes.new(name=f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material:
        obj.data.materials.append(material)
    return obj


def _boundary_x(boundary: object, z: float, half: float) -> float:
    if isinstance(boundary, (float, int)):
        return float(boundary)
    side, hole = boundary
    center_x, center_z = hole
    span = max(0.0, half - abs(z - center_z))
    if side == "left":
        return center_x - span
    return center_x + span


def _distance_squared(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum((a[index] - b[index]) ** 2 for index in range(3))


def _unique_sorted(values: list[float]) -> list[float]:
    sorted_values = sorted(values)
    result: list[float] = []
    for value in sorted_values:
        if not result or abs(value - result[-1]) > 1.0e-7:
            result.append(value)
    return result


def apply_boolean_difference(target: bpy.types.Object, cutter: bpy.types.Object) -> None:
    modifier = target.modifiers.new(name=f"cut_{cutter.name}", type="BOOLEAN")
    modifier.operation = "DIFFERENCE"
    modifier.object = cutter
    try:
        modifier.solver = "EXACT"
    except Exception:
        pass
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    target.select_set(False)


def add_open_tray(
    name: str,
    origin: tuple[float, float, float],
    size: tuple[float, float, float],
    wall: float,
    base_material: bpy.types.Material,
    inner_material: bpy.types.Material,
) -> list[bpy.types.Object]:
    """Build a low open-top tray from rectangular walls and floor."""
    ox, oy, oz = origin
    sx, sy, sz = size
    objects = [
        add_box(
            f"{name}_floor",
            (sx, sy, wall),
            (ox + sx / 2.0, oy + sy / 2.0, oz + wall / 2.0),
            inner_material,
        ),
        add_box(
            f"{name}_front_wall",
            (sx, wall, sz),
            (ox + sx / 2.0, oy + wall / 2.0, oz + sz / 2.0),
            base_material,
        ),
        add_box(
            f"{name}_back_wall",
            (sx, wall, sz),
            (ox + sx / 2.0, oy + sy - wall / 2.0, oz + sz / 2.0),
            base_material,
        ),
        add_box(
            f"{name}_left_wall",
            (wall, sy, sz),
            (ox + wall / 2.0, oy + sy / 2.0, oz + sz / 2.0),
            base_material,
        ),
        add_box(
            f"{name}_right_wall",
            (wall, sy, sz),
            (ox + sx - wall / 2.0, oy + sy / 2.0, oz + sz / 2.0),
            base_material,
        ),
    ]
    return objects


def add_tray_long_dividers(
    name: str,
    origin: tuple[float, float, float],
    size: tuple[float, float, float],
    wall: float,
    fractions: list[float],
    material: bpy.types.Material,
) -> list[bpy.types.Object]:
    """Add dividers running front-to-back, splitting the tray width."""
    ox, oy, oz = origin
    sx, sy, sz = size
    objects: list[bpy.types.Object] = []
    for index, fraction in enumerate(fractions, start=1):
        x = ox + sx * fraction
        objects.append(
            add_box(
                f"{name}_long_divider_{index}",
                (wall, sy - 2.0 * wall, sz - wall),
                (x, oy + sy / 2.0, oz + (sz - wall) / 2.0),
                material,
            )
        )
    return objects


def add_tray_cross_dividers(
    name: str,
    origin: tuple[float, float, float],
    size: tuple[float, float, float],
    wall: float,
    fractions: list[float],
    material: bpy.types.Material,
) -> list[bpy.types.Object]:
    """Add dividers running side-to-side, splitting the tray depth."""
    ox, oy, oz = origin
    sx, sy, sz = size
    objects: list[bpy.types.Object] = []
    for index, fraction in enumerate(fractions, start=1):
        y = oy + sy * fraction
        objects.append(
            add_box(
                f"{name}_cross_divider_{index}",
                (sx - 2.0 * wall, wall, sz - 2.0 * wall),
                (ox + sx / 2.0, y, oz + (sz - 2.0 * wall) / 2.0),
                material,
            )
        )
    return objects


def add_front_pull(
    name: str,
    center_x: float,
    front_y: float,
    z: float,
    width: float,
    material: bpy.types.Material,
) -> list[bpy.types.Object]:
    """Two-block pull handle matching the source's projecting rectangular lip."""
    return [
        add_box(
            f"{name}_flat_pull",
            (width, 10.0, 2.0),
            (center_x, front_y - 4.0, z),
            material,
        ),
        add_box(
            f"{name}_front_lip",
            (width, 2.0, 4.0),
            (center_x, front_y - 8.0, z),
            material,
        ),
    ]


def add_indent_disc(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    material: bpy.types.Material,
    face_axis: str,
) -> bpy.types.Object:
    """Dark circular cue for the four subtractive stacking indentions."""
    if face_axis == "x":
        rotation = (0.0, math.radians(90.0), 0.0)
    elif face_axis == "y":
        rotation = (math.radians(90.0), 0.0, 0.0)
    else:
        rotation = (0.0, 0.0, 0.0)
    return add_cylinder(
        name=name,
        diameter=radius * 2.0,
        height=0.16,
        location=location,
        rotation=rotation,
        vertices=48,
        material=material,
    )


# ============================================================
# Procedural assembly
# ============================================================

"""Assembly logic for the small-parts cabinet model."""





DRAWERS = 5
DRAWER_PITCH = 29.0
CASE_WIDTH_X = DRAWERS * DRAWER_PITCH + 2.0
CASE_DEPTH_Y = 100.0
CASE_HEIGHT_Z = 80.0

WALL = 2.0
OPENING_WIDTH_X = DRAWER_PITCH - 2.0
SOURCE_TRAY_WIDTH_X = 94.0
SOURCE_TRAY_DEPTH_Y = 77.0
TRAY_HEIGHT = 25.0
TRAY_WALL = 2.0


def _bay_x(index: int) -> float:
    return WALL + index * DRAWER_PITCH


def _add_case_frame(materials: MaterialMap) -> list[bpy.types.Object]:
    shell = materials["shell"]
    inside = materials["inside"]
    objects: list[bpy.types.Object] = []

    front_wall = add_perforated_y_wall(
        "cabinet_front_perforated_wall",
        x_min=0.0,
        x_max=CASE_WIDTH_X,
        y_min=0.0,
        y_max=WALL,
        z_min=0.0,
        z_max=CASE_HEIGHT_Z,
        holes=_diamond_centers(),
        diagonal=6.4,
        material=shell,
    )
    back_wall = add_perforated_y_wall(
        "cabinet_back_perforated_wall",
        x_min=0.0,
        x_max=CASE_WIDTH_X,
        y_min=CASE_DEPTH_Y - WALL,
        y_max=CASE_DEPTH_Y,
        z_min=0.0,
        z_max=CASE_HEIGHT_Z,
        holes=_diamond_centers(),
        diagonal=6.4,
        material=shell,
    )
    objects.extend([front_wall, back_wall])

    left_end_wall = add_box(
        "cabinet_left_end_wall",
        (WALL, CASE_DEPTH_Y, CASE_HEIGHT_Z),
        (WALL / 2.0, CASE_DEPTH_Y / 2.0, CASE_HEIGHT_Z / 2.0),
        shell,
    )
    right_end_wall = add_box(
        "cabinet_right_end_wall",
        (WALL, CASE_DEPTH_Y, CASE_HEIGHT_Z),
        (CASE_WIDTH_X - WALL / 2.0, CASE_DEPTH_Y / 2.0, CASE_HEIGHT_Z / 2.0),
        shell,
    )
    bottom_slab = add_box(
        "cabinet_bottom_slab",
        (CASE_WIDTH_X, CASE_DEPTH_Y, WALL),
        (CASE_WIDTH_X / 2.0, CASE_DEPTH_Y / 2.0, WALL / 2.0),
        shell,
    )
    objects.extend([left_end_wall, right_end_wall, bottom_slab])

    for index in range(1, DRAWERS):
        x = WALL + index * DRAWER_PITCH - WALL / 2.0
        objects.append(
            add_box(
                f"cabinet_vertical_bay_wall_{index}",
                (WALL, CASE_DEPTH_Y - 2.0 * WALL, CASE_HEIGHT_Z),
                (x, CASE_DEPTH_Y / 2.0, CASE_HEIGHT_Z / 2.0),
                shell,
            )
        )

    # Inner green liners reproduce the OpenSCAD preview's contrasting inner faces.
    for index in range(DRAWERS):
        x0 = _bay_x(index)
        objects.append(
            add_box(
                f"bay_{index + 1}_green_floor_liner",
                (OPENING_WIDTH_X, CASE_DEPTH_Y - 2.0 * WALL, 0.18),
                (x0 + OPENING_WIDTH_X / 2.0, CASE_DEPTH_Y / 2.0, WALL + 0.09),
                inside,
            )
        )
    objects.append(
        add_box(
            "left_end_green_liner",
            (0.22, CASE_DEPTH_Y - 2.0 * WALL, CASE_HEIGHT_Z - WALL),
            (WALL + 0.11, CASE_DEPTH_Y / 2.0, CASE_HEIGHT_Z / 2.0),
            inside,
        )
    )
    objects.append(
        add_box(
            "right_end_green_liner",
            (0.22, CASE_DEPTH_Y - 2.0 * WALL, CASE_HEIGHT_Z - WALL),
            (CASE_WIDTH_X - WALL - 0.11, CASE_DEPTH_Y / 2.0, CASE_HEIGHT_Z / 2.0),
            inside,
        )
    )
    for index in range(1, DRAWERS):
        x = WALL + index * DRAWER_PITCH - WALL / 2.0
        for side, offset in [("left_face", -WALL / 2.0 - 0.08), ("right_face", WALL / 2.0 + 0.08)]:
            objects.append(
                add_box(
                    f"bay_wall_{index}_{side}_green_liner",
                    (0.16, CASE_DEPTH_Y - 2.0 * WALL, CASE_HEIGHT_Z - WALL),
                    (x + offset, CASE_DEPTH_Y / 2.0, CASE_HEIGHT_Z / 2.0),
                    inside,
                )
            )

    _cut_stacking_dimples(right_end_wall)

    # The SCAD adds four protruding spheres near x-min and subtracts four
    # spherical stacking dimples from the opposite end face.
    for y in (5.0, CASE_DEPTH_Y - 5.0):
        for z in (5.0, CASE_HEIGHT_Z - 5.0):
            objects.append(
                add_sphere(
                    f"x_min_stacking_nub_y{int(y)}_z{int(z)}",
                    1.4,
                    (0.2, y, z),
                    materials["highlight"],
                    segments=32,
                    rings=16,
                )
            )

    return objects


def _diamond_centers() -> list[tuple[float, float]]:
    centers: list[tuple[float, float]] = []
    for hole_z in range(7, 71, 10):
        for drawer in range(DRAWERS):
            bay_start = _bay_x(drawer)
            centers.extend(
                [
                    (bay_start + 5.0, hole_z),
                    (bay_start + 12.0, hole_z + 5.0),
                    (bay_start + 19.0, hole_z),
                ]
            )
    return centers


def _cut_stacking_dimples(right_end_wall: bpy.types.Object) -> None:
    for y in (5.0, CASE_DEPTH_Y - 5.0):
        for z, radius in [(5.0, 1.5), (CASE_HEIGHT_Z - 5.0, 1.55)]:
            cutter = add_sphere(
                f"x_max_stacking_dimple_cut_y{int(y)}_z{int(z)}",
                radius,
                (CASE_WIDTH_X, y, z),
                None,
                segments=32,
                rings=16,
            )
            cutter.hide_viewport = True
            cutter.hide_render = True
            apply_boolean_difference(right_end_wall, cutter)
            bpy.data.objects.remove(cutter, do_unlink=True)


def _add_drawer_trays(materials: MaterialMap) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    shell = materials["shell"]
    inside = materials["inside"]

    origin = (0.0, 0.0, 0.0)
    size = (SOURCE_TRAY_WIDTH_X, SOURCE_TRAY_DEPTH_Y, TRAY_HEIGHT)

    # The SCAD emits five drawer modules at the same origin after the cabinet.
    # Recreate that literal pile: one shared tray footprint with all source
    # divider variants superimposed instead of one drawer per cabinet bay.
    objects.extend(add_open_tray("overlapped_source_drawer_pile", origin, size, TRAY_WALL, shell, inside))

    divider_specs = [
        ("two_long_1", (100.0 / 3.0) - 3.0, SOURCE_TRAY_DEPTH_Y - 4.0, TRAY_HEIGHT),
        ("two_long_2", (100.0 / 3.0) * 2.0 - 3.0, SOURCE_TRAY_DEPTH_Y - 4.0, TRAY_HEIGHT),
        ("single_long_center", 50.0 - 3.0, SOURCE_TRAY_DEPTH_Y - 4.0, TRAY_HEIGHT),
        ("three_long_1", 25.0 - 4.0, SOURCE_TRAY_DEPTH_Y - 4.0, TRAY_HEIGHT),
        ("three_long_2", 50.0 - 4.0, SOURCE_TRAY_DEPTH_Y - 4.0, TRAY_HEIGHT),
        ("three_long_3", 75.0 - 4.0, SOURCE_TRAY_DEPTH_Y - 4.0, TRAY_HEIGHT),
        ("grid_long_center", 50.0 - 4.0, SOURCE_TRAY_DEPTH_Y - 4.0, TRAY_HEIGHT),
    ]
    for name, x0, depth, height in divider_specs:
        objects.append(
            add_box(
                f"overlapped_drawer_{name}",
                (2.0, depth, height),
                (x0 + 1.0, 2.0 + depth / 2.0, height / 2.0),
                shell,
            )
        )

    objects.append(
        add_box(
            "overlapped_drawer_cross_mid",
            (100.0 - 7.0, 2.0, TRAY_HEIGHT - 2.0),
            ((100.0 - 7.0) / 2.0, 80.0 / 2.0 - 4.0 + 1.0, (TRAY_HEIGHT - 2.0) / 2.0),
            shell,
        )
    )

    objects.append(
        add_indent_disc(
            "overlapped_drawer_front_thumb_scoop",
            (50.0 - 2.0, 3.0, -0.02),
            2.0,
            materials["shadow"],
            "z",
        )
    )
    objects.extend(
        add_front_pull(
            "overlapped_drawer_visible_pull",
            center_x=50.0 - 2.0,
            front_y=0.0,
            z=TRAY_HEIGHT / 3.0 + 1.0,
            width=20.0,
            material=shell,
        )
    )

    _add_retaining_nubs(materials, objects)
    return objects


def _add_retaining_nubs(materials: MaterialMap, objects: list[bpy.types.Object]) -> None:
    highlight = materials["highlight"]
    for nub in range(DRAWERS):
        x = DRAWER_PITCH * nub + 2.0
        objects.append(
            add_sphere(
                f"drawer_start_center_retaining_nub_{nub + 1}",
                1.2,
                (x, CASE_DEPTH_Y / 2.0, CASE_HEIGHT_Z - 2.0),
                highlight,
                segments=32,
                rings=16,
            )
        )

    for nub in range(1, DRAWERS + 1):
        x = DRAWER_PITCH * nub - 0.2
        for y, label in [(CASE_DEPTH_Y / 5.0, "front"), ((CASE_DEPTH_Y / 5.0) * 4.0, "back")]:
            objects.append(
                add_sphere(
                    f"drawer_boundary_upper_retaining_nub_{nub}_{label}",
                    2.2,
                    (x, y, CASE_HEIGHT_Z - 4.0),
                    highlight,
                    segments=32,
                    rings=16,
                )
            )


def build_model() -> list[bpy.types.Object]:
    materials = cabinet_materials()
    objects: list[bpy.types.Object] = []
    objects.extend(_add_case_frame(materials))
    objects.extend(_add_drawer_trays(materials))
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

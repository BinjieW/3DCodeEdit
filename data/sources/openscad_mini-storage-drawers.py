from __future__ import annotations
from dataclasses import dataclass
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
import bpy


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the mini storage drawer model."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.55


OPENSCAD_GOLD = MaterialSpec("openscad_preview_gold", (1.0, 0.72, 0.02, 1.0))


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
    return material


def assign_material(obj: bpy.types.Object, spec: MaterialSpec | None) -> bpy.types.Object:
    if spec is not None:
        obj.data.materials.clear()
        obj.data.materials.append(get_material(spec))
    return obj


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Add a cube with source-style dimensions centered at ``location``."""
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return assign_material(obj, material)


def _new_mesh_object(
    name: str,
    vertices: list[tuple[float, float, float]],
    faces: list[list[int]],
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return assign_material(obj, material)


def add_extruded_xy_profile_with_holes(
    name: str,
    outer_profile: list[tuple[float, float]],
    hole_profiles: list[list[tuple[float, float]]],
    z_min: float,
    z_max: float,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Extrude a 2D XY footprint with rectangular compartment holes through Z."""
    loops = [outer_profile] + hole_profiles
    flat_points = [point for loop in loops for point in loop]
    vertices = [(x, y, z_min) for x, y in flat_points] + [(x, y, z_max) for x, y in flat_points]
    point_count = len(flat_points)

    triangles = tessellate_polygon([[Vector((x, y, 0.0)) for x, y in loop] for loop in loops])
    faces: list[list[int]] = []
    for triangle in triangles:
        faces.append([triangle[2], triangle[1], triangle[0]])
        faces.append([point_count + triangle[0], point_count + triangle[1], point_count + triangle[2]])

    offset = 0
    for loop_index, loop in enumerate(loops):
        loop_count = len(loop)
        for index in range(loop_count):
            a = offset + index
            b = offset + (index + 1) % loop_count
            if loop_index == 0:
                faces.append([a, b, b + point_count, a + point_count])
            else:
                faces.append([a, a + point_count, b + point_count, b])
        offset += loop_count

    return _new_mesh_object(name, vertices, faces, material)


def add_rectangular_tube(
    name: str,
    outer_size: tuple[float, float, float],
    inner_size: tuple[float, float, float],
    center: tuple[float, float, float],
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Build an OpenSCAD cube-minus-cube tube, open through the Y/depth axis."""
    outer_x, outer_y, outer_z = outer_size
    inner_x, inner_y, inner_z = inner_size
    wall_x = (outer_x - inner_x) / 2.0
    wall_z = (outer_z - inner_z) / 2.0
    if wall_x <= 0.0 or wall_z <= 0.0:
        raise ValueError("inner_size must leave positive side and top/bottom walls")

    x0, x1 = -outer_x / 2.0, outer_x / 2.0
    y0, y1 = -outer_y / 2.0, outer_y / 2.0
    z0, z1 = -outer_z / 2.0, outer_z / 2.0
    ix0, ix1 = -inner_x / 2.0, inner_x / 2.0
    iz0, iz1 = -inner_z / 2.0, inner_z / 2.0

    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []

    def add_prism(min_corner: tuple[float, float, float], max_corner: tuple[float, float, float]) -> None:
        base_index = len(vertices)
        lx0, ly0, lz0 = min_corner
        lx1, ly1, lz1 = max_corner
        vertices.extend(
            [
                (center[0] + lx0, center[1] + ly0, center[2] + lz0),
                (center[0] + lx1, center[1] + ly0, center[2] + lz0),
                (center[0] + lx1, center[1] + ly1, center[2] + lz0),
                (center[0] + lx0, center[1] + ly1, center[2] + lz0),
                (center[0] + lx0, center[1] + ly0, center[2] + lz1),
                (center[0] + lx1, center[1] + ly0, center[2] + lz1),
                (center[0] + lx1, center[1] + ly1, center[2] + lz1),
                (center[0] + lx0, center[1] + ly1, center[2] + lz1),
            ]
        )
        prism_faces = [
            [0, 1, 2, 3],
            [4, 7, 6, 5],
            [0, 4, 5, 1],
            [1, 5, 6, 2],
            [2, 6, 7, 3],
            [3, 7, 4, 0],
        ]
        faces.extend([[base_index + index for index in face] for face in prism_faces])

    add_prism((x0, y0, z0), (ix0, y1, z1))
    add_prism((ix1, y0, z0), (x1, y1, z1))
    add_prism((ix0, y0, z0), (ix1, y1, iz0))
    add_prism((ix0, y0, iz1), (ix1, y1, z1))

    return _new_mesh_object(name, vertices, faces, material)


# ============================================================
# Procedural assembly
# ============================================================

"""Procedural assembly for the mini storage drawer model."""




DEPTH = 68.0
WALL = 1.0

WALL_HEIGHT = DEPTH
Z_MIN = -WALL_HEIGHT / 2.0
Z_MAX = WALL_HEIGHT / 2.0
BOTTOM_RAIL_Z = -33.5
BOTTOM_RAIL_THICKNESS = 1.0

HALF_ROW_DEPTH = 14.5
NORMAL_ROW_DEPTH = 28.0

HALF_ROW_CENTER_Y = 13.5
NORMAL_ROW_CENTER_Y = 33.75
DOUBLE_ROW_CENTER_Y = 60.75
QUAD_ROW_CENTER_Y = 87.75

NARROW_PITCH = 30.0
NARROW_WIDTH = 31.0
DOUBLE_WIDTH = 61.0
QUAD_WIDTH = 121.0

HALF_RAIL_DEPTH = 3.0
FULL_RAIL_DEPTH = 6.0


def _rect_loop(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> list[tuple[float, float]]:
    return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]


def _hole_loop(
    x_center: float,
    y_center: float,
    width: float,
    depth: float,
) -> list[tuple[float, float]]:
    """Clockwise loop for a rectangular compartment opening in the XY footprint."""
    x_min = x_center - width / 2.0
    x_max = x_center + width / 2.0
    y_min = y_center - depth / 2.0
    y_max = y_center + depth / 2.0
    return [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]


def _narrow_centers() -> tuple[float, ...]:
    return tuple(NARROW_PITCH * col for col in range(1, 5))


def _compartment_holes() -> list[list[tuple[float, float]]]:
    holes: list[list[tuple[float, float]]] = []
    for x_center in _narrow_centers():
        holes.append(_hole_loop(x_center, HALF_ROW_CENTER_Y, NARROW_WIDTH - 2.0 * WALL, HALF_ROW_DEPTH - 2.0 * WALL))
        holes.append(_hole_loop(x_center, NORMAL_ROW_CENTER_Y, NARROW_WIDTH - 2.0 * WALL, NORMAL_ROW_DEPTH - 2.0 * WALL))
    for x_center in (45.0, 105.0):
        holes.append(_hole_loop(x_center, DOUBLE_ROW_CENTER_Y, DOUBLE_WIDTH - 2.0 * WALL, NORMAL_ROW_DEPTH - 2.0 * WALL))
    holes.append(_hole_loop(75.0, QUAD_ROW_CENTER_Y, QUAD_WIDTH - 2.0 * WALL, NORMAL_ROW_DEPTH - 2.0 * WALL))
    return holes


def build_wall_grid() -> bpy.types.Object:
    """Build the continuous outer shell and vertical divider network."""
    outer = _rect_loop(
        30.0 - NARROW_WIDTH / 2.0,
        HALF_ROW_CENTER_Y - HALF_ROW_DEPTH / 2.0,
        75.0 + QUAD_WIDTH / 2.0,
        QUAD_ROW_CENTER_Y + NORMAL_ROW_DEPTH / 2.0,
    )
    return add_extruded_xy_profile_with_holes(
        name="continuous_open_storage_wall_grid",
        outer_profile=outer,
        hole_profiles=_compartment_holes(),
        z_min=Z_MIN,
        z_max=Z_MAX,
        material=OPENSCAD_GOLD,
    )


def _bottom_rail(
    name: str,
    width: float,
    depth: float,
    x_center: float,
    y_center: float,
) -> bpy.types.Object:
    return add_box(
        name=name,
        size=(width, depth, BOTTOM_RAIL_THICKNESS),
        location=(x_center, y_center, BOTTOM_RAIL_Z),
        material=OPENSCAD_GOLD,
    )


def build_bottom_rails() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    for index, x_center in enumerate(_narrow_centers(), start=1):
        objects.append(
            _bottom_rail(
                name=f"half_height_bottom_rail_{index}",
                width=NARROW_WIDTH - 2.0 * WALL,
                depth=HALF_RAIL_DEPTH,
                x_center=x_center,
                y_center=HALF_ROW_CENTER_Y,
            )
        )
        objects.append(
            _bottom_rail(
                name=f"normal_height_bottom_rail_{index}",
                width=NARROW_WIDTH - 2.0 * WALL,
                depth=FULL_RAIL_DEPTH,
                x_center=x_center,
                y_center=NORMAL_ROW_CENTER_Y,
            )
        )
    for index, x_center in enumerate((45.0, 105.0), start=1):
        objects.append(
            _bottom_rail(
                name=f"double_width_bottom_rail_{index}",
                width=DOUBLE_WIDTH - 2.0 * WALL,
                depth=FULL_RAIL_DEPTH,
                x_center=x_center,
                y_center=DOUBLE_ROW_CENTER_Y,
            )
        )
    objects.append(
        _bottom_rail(
            name="quad_width_bottom_rail",
            width=QUAD_WIDTH - 2.0 * WALL,
            depth=FULL_RAIL_DEPTH,
            x_center=75.0,
            y_center=QUAD_ROW_CENTER_Y,
        )
    )
    return objects


def build_model() -> list[bpy.types.Object]:
    """Build the source four-row open storage organizer from SCAD parameters."""
    return [build_wall_grid(), *build_bottom_rails()]


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

from __future__ import annotations
from dataclasses import dataclass
from mathutils import Euler, Vector
import bmesh
import bpy
import math
import random


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the customizable stone wall."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.58


STONE = MaterialSpec("slate_gray_stone", (0.26, 0.31, 0.36, 1.0), 0.72)
MORTAR = MaterialSpec("shadowed_silver_mortar", (0.34, 0.36, 0.36, 1.0), 0.82)
BASE = MaterialSpec("green_base", (0.0, 0.28, 0.0, 1.0), 0.62)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def get_material(spec: MaterialSpec) -> bpy.types.Material:
    material = bpy.data.materials.get(spec.name) or bpy.data.materials.new(spec.name)
    material.diffuse_color = spec.color
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    shader = nodes.new(type="ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = spec.color
    shader.inputs["Roughness"].default_value = spec.roughness
    shader.inputs["Metallic"].default_value = 0.0
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def assign_material(obj: bpy.types.Object, spec: MaterialSpec) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(get_material(spec))
    return obj


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Add an OpenSCAD-style cube using center=false coordinates."""

    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(
            location[0] + size[0] / 2.0,
            location[1] + size[1] / 2.0,
            location[2] + size[2] / 2.0,
        ),
    )
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material is not None:
        assign_material(obj, material)
    return obj


def add_faceted_stone(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    seed: int,
    max_rotation_degrees: float,
    clip_count: int = 8,
    power: float = 0.0,
    material: MaterialSpec = STONE,
    min_world_z: float | None = 0.0,
) -> bpy.types.Object:
    """Create the source-style ``rock()`` as an intersection of rotated cubes."""

    vertices, faces = rotated_cube_intersection_mesh(
        size=size,
        seed=seed,
        max_rotation_degrees=max_rotation_degrees,
        clip_count=clip_count,
        power=power,
    )
    if min_world_z is not None:
        vertices, faces = clip_polyhedron(
            vertices,
            faces,
            normal=Vector((0.0, 0.0, -1.0)),
            distance=location[2] - min_world_z,
        )

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata([tuple(vertex) for vertex in vertices], [], faces)
    mesh.update(calc_edges=True)
    mesh.validate(clean_customdata=False)

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update(calc_edges=True)

    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def rotated_cube_intersection_mesh(
    size: tuple[float, float, float],
    seed: int,
    max_rotation_degrees: float,
    clip_count: int,
    power: float,
) -> tuple[list[Vector], list[list[int]]]:
    """Return polygons for the OpenSCAD rock(): intersection of rotated centered cubes."""

    hx, hy, hz = (axis / 2.0 for axis in size)
    vertices = [
        Vector((-hx, -hy, -hz)),
        Vector((hx, -hy, -hz)),
        Vector((hx, hy, -hz)),
        Vector((-hx, hy, -hz)),
        Vector((-hx, -hy, hz)),
        Vector((hx, -hy, hz)),
        Vector((hx, hy, hz)),
        Vector((-hx, hy, hz)),
    ]
    faces = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]

    rng = random.Random(seed)
    random_values = [rng.uniform(-1.0, 1.0) for _ in range(max(0, clip_count) * 3)]
    half_extents = (hx, hy, hz)
    for index in range(max(0, clip_count)):
        factor = math.pow(index, power) if index != 0 else 1.0
        angles = (
            math.radians(max_rotation_degrees * random_values[3 * index + 0] * factor),
            math.radians(max_rotation_degrees * random_values[3 * index + 1] * factor),
            math.radians(max_rotation_degrees * random_values[3 * index + 2] * factor),
        )
        rotation = Euler(angles, "XYZ").to_matrix()
        for axis_index, half_extent in enumerate(half_extents):
            normal = rotation.col[axis_index].normalized()
            vertices, faces = clip_polyhedron(vertices, faces, normal, half_extent)
            vertices, faces = clip_polyhedron(vertices, faces, -normal, half_extent)

    return vertices, faces


def clip_polyhedron(
    vertices: list[Vector],
    faces: list[list[int]],
    normal: Vector,
    distance: float,
    epsilon: float = 1e-7,
) -> tuple[list[Vector], list[list[int]]]:
    """Clip a convex polygon mesh against ``normal.dot(point) <= distance``."""

    if not faces:
        return vertices, faces

    new_vertices = [vertex.copy() for vertex in vertices]
    new_faces: list[list[int]] = []
    cut_points: list[int] = []

    def add_vertex(point: Vector) -> int:
        for index in cut_points:
            if (new_vertices[index] - point).length <= 1e-6:
                return index
        new_vertices.append(point)
        cut_points.append(len(new_vertices) - 1)
        return len(new_vertices) - 1

    for face in faces:
        clipped: list[int] = []
        for position, current_index in enumerate(face):
            next_index = face[(position + 1) % len(face)]
            current = new_vertices[current_index]
            next_vertex = new_vertices[next_index]
            current_distance = normal.dot(current) - distance
            next_distance = normal.dot(next_vertex) - distance
            current_inside = current_distance <= epsilon
            next_inside = next_distance <= epsilon

            if current_inside:
                clipped.append(current_index)
            if current_inside != next_inside:
                t = current_distance / (current_distance - next_distance)
                clipped.append(add_vertex(current.lerp(next_vertex, t)))

        if len(clipped) >= 3:
            cleaned: list[int] = []
            for index in clipped:
                if not cleaned or cleaned[-1] != index:
                    cleaned.append(index)
            if len(cleaned) >= 3 and cleaned[0] == cleaned[-1]:
                cleaned.pop()
            if len(cleaned) >= 3:
                new_faces.append(cleaned)

    if len(cut_points) >= 3:
        center = sum((new_vertices[index] for index in cut_points), Vector()) / len(cut_points)
        tangent = normal.cross(Vector((0.0, 0.0, 1.0)))
        if tangent.length <= epsilon:
            tangent = normal.cross(Vector((0.0, 1.0, 0.0)))
        tangent.normalize()
        bitangent = normal.cross(tangent).normalized()
        ordered = sorted(
            set(cut_points),
            key=lambda index: math.atan2(
                (new_vertices[index] - center).dot(bitangent),
                (new_vertices[index] - center).dot(tangent),
            ),
        )
        new_faces.append(list(reversed(ordered)))

    used = sorted({index for face in new_faces for index in face})
    remap = {old: new for new, old in enumerate(used)}
    compact_vertices = [new_vertices[index] for index in used]
    compact_faces = [[remap[index] for index in face] for face in new_faces]
    return compact_vertices, compact_faces


# ============================================================
# Procedural assembly
# ============================================================

"""Procedural assembly for the customizable stone wall."""






@dataclass(frozen=True)
class WallParameters:
    stone_length: float = 3.0
    wall_thickness: float = 2.0
    stone_height: float = 1.0
    stones_across: int = 4
    rows: int = 6
    master_seed: int = 14
    max_rotation_angle: float = 12.0
    z_compression: float = 0.9
    clip_count: int = 8
    power: float = 0.0
    mortar_enabled: bool = True
    u_adjust: float = -0.15
    v_ratio: float = 0.5
    base_enabled: bool = True
    bx_adjust: float = 0.1
    by_adjust: float = 0.1
    base_height_ratio: float = 0.5


def uses_half_stone(x_index: int, row_index: int, stones_across: int) -> bool:
    return (x_index == 0 and row_index % 2 == 0) or (
        x_index == stones_across - 1 and row_index % 2 == 1
    )


def stone_size(params: WallParameters, half_stone: bool) -> tuple[float, float, float]:
    length = params.stone_length / 2.0 if half_stone else params.stone_length
    return (length, params.wall_thickness, params.stone_height)


def stone_center(
    x_index: int,
    row_index: int,
    half_stone: bool,
    params: WallParameters,
) -> tuple[float, float, float]:
    if half_stone:
        x = params.stone_length * (x_index + 0.5 - 0.25)
    else:
        x = params.stone_length * (
            x_index + 0.5 - 0.25 * (1.0 + (-1.0 if row_index % 2 else 1.0))
        )
    z = row_index * params.stone_height * params.z_compression
    return (x, 0.0, z)


def add_mortar(params: WallParameters) -> bpy.types.Object:
    return add_box(
        name="continuous_silver_mortar_backing",
        size=(
            (params.stones_across - 0.5 + 2.0 * params.u_adjust) * params.stone_length,
            params.wall_thickness * params.v_ratio,
            params.stone_height * (params.rows - 1) * params.z_compression,
        ),
        location=(
            (-params.u_adjust) * params.stone_length,
            -params.wall_thickness * params.v_ratio / 2.0,
            0.0,
        ),
        material=MORTAR,
    )


def add_base(params: WallParameters) -> bpy.types.Object:
    return add_box(
        name="green_rectangular_base_slab",
        size=(
            params.stone_length * (params.stones_across - 0.5 + 2.0 * params.bx_adjust),
            params.wall_thickness * (1.0 + 2.0 * params.by_adjust),
            params.stone_height * params.base_height_ratio,
        ),
        location=(
            -params.bx_adjust * params.stone_length,
            -(0.5 + params.by_adjust) * params.wall_thickness,
            -params.stone_height * (params.base_height_ratio - 0.01),
        ),
        material=BASE,
    )


def build_wall(params: WallParameters = WallParameters()) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []

    if params.mortar_enabled:
        objects.append(add_mortar(params))
    if params.base_enabled:
        objects.append(add_base(params))

    for row_index in range(params.rows):
        for x_index in range(params.stones_across):
            half_stone = uses_half_stone(x_index, row_index, params.stones_across)
            objects.append(
                add_faceted_stone(
                    name=f"stone_row_{row_index:02d}_col_{x_index:02d}",
                    size=stone_size(params, half_stone),
                    location=stone_center(x_index, row_index, half_stone, params),
                    seed=params.stones_across * row_index + x_index + params.master_seed,
                    max_rotation_degrees=params.max_rotation_angle,
                    clip_count=params.clip_count,
                    power=params.power,
                )
            )

    return objects


def build_model() -> list[bpy.types.Object]:
    return build_wall()


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the customizable stone wall iteration."""




def configure_scene_visibility() -> None:
    """Keep shared Cycles renders readable without washing out facets."""

    scene = bpy.context.scene
    try:
        scene.view_settings.view_transform = "Standard"
    except TypeError:
        pass
    try:
        scene.view_settings.look = "Medium High Contrast"
    except TypeError:
        scene.view_settings.look = "None"
    scene.view_settings.exposure = -0.8
    scene.view_settings.gamma = 1.0


def main() -> None:
    clear_scene()
    configure_scene_visibility()
    build_model()


if __name__ == "__main__":
    main()

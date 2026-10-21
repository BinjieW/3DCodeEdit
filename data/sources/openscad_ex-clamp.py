from __future__ import annotations
from dataclasses import dataclass
from mathutils import Matrix, Vector
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
from typing import Iterable, Sequence
import bmesh
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the quick-release t-slot clamp."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    metallic: float = 0.0
    roughness: float = 0.48
    backface_color: tuple[float, float, float, float] | None = None


BODY_YELLOW = MaterialSpec("source_preview_yellow_plastic", (1.0, 0.76, 0.02, 1.0), 0.0, 0.44)
SHADOW_GREEN = MaterialSpec("preview_internal_shadow_green", (0.18, 0.34, 0.08, 1.0), 0.0, 0.68)
DEEP_SHADOW_GREEN = MaterialSpec("preview_deep_cube_cut_green", (0.025, 0.075, 0.012, 1.0), 0.0, 0.82)
UNDERSIDE_SHADOW_GREEN = MaterialSpec(
    "underside_forward_shadow_yellow_back",
    (0.18, 0.34, 0.08, 1.0),
    0.0,
    0.72,
    BODY_YELLOW.color,
)
UNDERSIDE_DEEP_SHADOW_GREEN = MaterialSpec(
    "underside_forward_deep_shadow_yellow_back",
    (0.025, 0.075, 0.012, 1.0),
    0.0,
    0.84,
    BODY_YELLOW.color,
)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def get_material(spec: MaterialSpec) -> bpy.types.Material:
    material = bpy.data.materials.get(spec.name) or bpy.data.materials.new(spec.name)
    material.use_nodes = True
    if spec.backface_color is None:
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = spec.color
            bsdf.inputs["Metallic"].default_value = spec.metallic
            bsdf.inputs["Roughness"].default_value = spec.roughness
    else:
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        geometry = nodes.new("ShaderNodeNewGeometry")
        mix = nodes.new("ShaderNodeMixShader")
        front_bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        back_bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        front_bsdf.inputs["Base Color"].default_value = spec.color
        front_bsdf.inputs["Metallic"].default_value = spec.metallic
        front_bsdf.inputs["Roughness"].default_value = spec.roughness
        back_bsdf.inputs["Base Color"].default_value = spec.backface_color
        back_bsdf.inputs["Metallic"].default_value = spec.metallic
        back_bsdf.inputs["Roughness"].default_value = spec.roughness
        links.new(geometry.outputs["Backfacing"], mix.inputs[0])
        links.new(front_bsdf.outputs["BSDF"], mix.inputs[1])
        links.new(back_bsdf.outputs["BSDF"], mix.inputs[2])
        links.new(mix.outputs["Shader"], output.inputs["Surface"])
    material.diffuse_color = spec.color
    return material


def assign_material(obj: bpy.types.Object, spec: MaterialSpec) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(get_material(spec))
    return obj


def assign_material_slots(
    obj: bpy.types.Object,
    materials: Sequence[MaterialSpec],
    face_material_indices: Sequence[int] | None = None,
) -> bpy.types.Object:
    obj.data.materials.clear()
    for spec in materials:
        obj.data.materials.append(get_material(spec))
    if face_material_indices is not None:
        for polygon, material_index in zip(obj.data.polygons, face_material_indices, strict=False):
            polygon.material_index = material_index
    return obj


def shade_smooth(obj: bpy.types.Object, auto_smooth: bool = False) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)
    if auto_smooth and hasattr(obj.data, "use_auto_smooth"):
        obj.data.use_auto_smooth = True
    return obj


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material:
        assign_material(obj, material)
    return obj


def add_cylinder(
    name: str,
    diameter: float,
    height: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 72,
    material: MaterialSpec | None = None,
    radius_top: float | None = None,
    radius_bottom: float | None = None,
) -> bpy.types.Object:
    if radius_top is None and radius_bottom is None:
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=vertices,
            radius=diameter / 2.0,
            depth=height,
            location=location,
            rotation=rotation,
        )
    else:
        bpy.ops.mesh.primitive_cone_add(
            vertices=vertices,
            radius1=diameter / 2.0 if radius_bottom is None else radius_bottom,
            radius2=diameter / 2.0 if radius_top is None else radius_top,
            depth=height,
            location=location,
            rotation=rotation,
        )
    obj = bpy.context.object
    obj.name = name
    if material:
        assign_material(obj, material)
    shade_smooth(obj)
    return obj


def add_polygon_prism(
    name: str,
    points: Sequence[tuple[float, float]],
    height: float,
    z_center: float,
    material: MaterialSpec,
) -> bpy.types.Object:
    """Build a vertical prism from a simple counter-clockwise XY outline."""
    bottom_z = z_center - height / 2.0
    top_z = z_center + height / 2.0
    vertices = [(x, y, bottom_z) for x, y in points] + [(x, y, top_z) for x, y in points]
    count = len(points)
    triangles = tessellate_polygon([[Vector((x, y, 0.0)) for x, y in points]])
    faces = [tuple(reversed(triangle)) for triangle in triangles]
    faces.extend(tuple(index + count for index in triangle) for triangle in triangles)
    for index in range(count):
        faces.append((index, (index + 1) % count, count + (index + 1) % count, count + index))
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def add_oriented_polygon_prism(
    name: str,
    points: Sequence[tuple[float, float]],
    depth: float,
    axis: str,
    center: tuple[float, float, float],
    material: MaterialSpec,
) -> bpy.types.Object:
    """Extrude a 2D outline along X, Y, or Z while keeping code-level intent readable."""
    cx, cy, cz = center
    half = depth / 2.0
    vertices: list[tuple[float, float, float]] = []
    axis_key = axis.upper()
    for side in (-half, half):
        for a, b in points:
            if axis_key == "X":
                vertices.append((cx + side, cy + a, cz + b))
            elif axis_key == "Y":
                vertices.append((cx + a, cy + side, cz + b))
            elif axis_key == "Z":
                vertices.append((cx + a, cy + b, cz + side))
            else:
                raise ValueError(f"Unsupported axis {axis!r}")
    count = len(points)
    triangles = tessellate_polygon([[Vector((x, y, 0.0)) for x, y in points]])
    faces = [tuple(reversed(triangle)) for triangle in triangles]
    faces.extend(tuple(index + count for index in triangle) for triangle in triangles)
    for index in range(count):
        faces.append((index, (index + 1) % count, count + (index + 1) % count, count + index))
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def add_multimaterial_mesh(
    name: str,
    vertices: Sequence[tuple[float, float, float]],
    faces: Sequence[Sequence[int]],
    materials: Sequence[MaterialSpec],
    face_material_indices: Sequence[int] | None = None,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(list(vertices), [], [tuple(face) for face in faces])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material_slots(obj, materials, face_material_indices)
    return obj


def add_capsule_prism(
    name: str,
    length: float,
    width: float,
    height: float,
    location: tuple[float, float, float],
    axis: str,
    material: MaterialSpec,
    vertices: int = 48,
) -> bpy.types.Object:
    radius = width / 2.0
    if axis.upper() == "X":
        parts = [add_box(f"{name}_web", (max(length - width, 0.01), width, height), location, material)]
        for side in (-1, 1):
            parts.append(
                add_cylinder(
                    f"{name}_cap_{side}",
                    width,
                    height,
                    (location[0] + side * (length / 2.0 - radius), location[1], location[2]),
                    vertices=vertices,
                    material=material,
                )
            )
    elif axis.upper() == "Y":
        parts = [add_box(f"{name}_web", (width, max(length - width, 0.01), height), location, material)]
        for side in (-1, 1):
            parts.append(
                add_cylinder(
                    f"{name}_cap_{side}",
                    width,
                    height,
                    (location[0], location[1] + side * (length / 2.0 - radius), location[2]),
                    vertices=vertices,
                    material=material,
                )
            )
    else:
        raise ValueError("Capsule prism supports X or Y axis")
    return boolean_union(name, parts)


def boolean_union(name: str, objects: Iterable[bpy.types.Object]) -> bpy.types.Object:
    objects = [obj for obj in objects if obj is not None]
    if not objects:
        raise ValueError("boolean_union requires at least one object")
    active = objects[0]
    bpy.context.view_layer.objects.active = active
    active.select_set(True)
    for obj in objects[1:]:
        modifier = active.modifiers.new(f"union_{obj.name}", "BOOLEAN")
        modifier.operation = "UNION"
        modifier.object = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(obj, do_unlink=True)
    active.name = name
    active.select_set(False)
    return active


def boolean_difference(
    obj: bpy.types.Object,
    cutters: Iterable[bpy.types.Object],
    *,
    solver: str = "FLOAT",
    remove: bool = True,
) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    for cutter in cutters:
        modifier = obj.modifiers.new(f"cut_{cutter.name}", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        modifier.solver = solver
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        if remove:
            bpy.data.objects.remove(cutter, do_unlink=True)
    obj.select_set(False)
    return obj


def add_bevel_modifier(obj: bpy.types.Object, amount: float, segments: int = 1) -> bpy.types.Object:
    modifier = obj.modifiers.new("small_edge_softening", "BEVEL")
    modifier.width = amount
    modifier.segments = segments
    modifier.affect = "EDGES"
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    obj.select_set(False)
    weighted = obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=weighted.name)
    obj.select_set(False)
    return obj


def rotate_object(obj: bpy.types.Object, rotation: tuple[float, float, float]) -> bpy.types.Object:
    obj.rotation_euler = rotation
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    obj.select_set(False)
    return obj


def make_tapered_jaw_mesh(
    name: str,
    *,
    length: float,
    width: float,
    root_height: float,
    tip_height: float,
    top_z: float,
    start_x: float,
    material: MaterialSpec,
) -> bpy.types.Object:
    """Make the main arm wedge with a flat top and tapered lower profile."""
    x0 = start_x
    x1 = start_x + length
    half_y = width / 2.0
    z0_root = top_z - root_height
    z0_tip = top_z - tip_height
    vertices = [
        (x0, -half_y, z0_root),
        (x0, half_y, z0_root),
        (x0, -half_y, top_z),
        (x0, half_y, top_z),
        (x1, -half_y, z0_tip),
        (x1, half_y, z0_tip),
        (x1, -half_y, top_z),
        (x1, half_y, top_z),
    ]
    faces = [
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (2, 6, 7, 3),
        (0, 1, 5, 4),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    ]
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    add_bevel_modifier(obj, 0.10, 1)
    return obj


def make_wedge_tooth(
    name: str,
    *,
    x_center: float,
    y_center: float,
    width_x: float,
    depth_y: float,
    height_z: float,
    base_z: float,
    material: MaterialSpec,
) -> bpy.types.Object:
    half_x = width_x / 2.0
    half_y = depth_y / 2.0
    vertices = [
        (x_center - half_x, y_center - half_y, base_z),
        (x_center + half_x, y_center - half_y, base_z),
        (x_center + half_x, y_center + half_y, base_z),
        (x_center - half_x, y_center + half_y, base_z),
        (x_center, y_center - half_y, base_z - height_z),
        (x_center, y_center + half_y, base_z - height_z),
    ]
    faces = [
        (0, 1, 2, 3),
        (0, 4, 1),
        (3, 2, 5),
        (0, 3, 5, 4),
        (1, 4, 5, 2),
    ]
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def make_trapezoid_t_slot(
    name: str,
    *,
    length: float,
    top_width: float,
    neck_width: float,
    depth: float,
    lip_height: float,
    z_center: float,
    material: MaterialSpec,
) -> bpy.types.Object:
    """Approximate the OpenSCAD groove_clamp hull as a long triangular t-slot tongue."""
    half_top = top_width / 2.0
    half_neck = neck_width / 2.0
    points = [
        (-half_top, -depth * 0.54),
        (half_top, -depth * 0.54),
        (half_neck, lip_height * 0.05),
        (half_neck, depth * 0.46),
        (-half_neck, depth * 0.46),
        (-half_neck, lip_height * 0.05),
    ]
    return add_oriented_polygon_prism(name, points, length, "Z", (0.0, 0.0, z_center), material)


def make_rounded_rect_frame(
    name: str,
    *,
    width: float,
    height: float,
    stroke: float,
    thickness: float,
    location: tuple[float, float, float],
    material: MaterialSpec,
) -> list[bpy.types.Object]:
    x, y, z = location
    parts = [
        add_box(f"{name}_top", (width, stroke, thickness), (x, y + height / 2.0 - stroke / 2.0, z), material),
        add_box(f"{name}_bottom", (width, stroke, thickness), (x, y - height / 2.0 + stroke / 2.0, z), material),
        add_box(f"{name}_left", (stroke, height, thickness), (x - width / 2.0 + stroke / 2.0, y, z), material),
        add_box(f"{name}_right", (stroke, height, thickness), (x + width / 2.0 - stroke / 2.0, y, z), material),
    ]
    for obj in parts:
        add_bevel_modifier(obj, stroke * 0.35, 3)
    return parts


def bend_handle_mesh(
    name: str,
    *,
    length: float,
    root_width: float,
    end_width: float,
    thickness: float,
    root_center: tuple[float, float, float],
    angle_degrees: float,
    material: MaterialSpec,
) -> bpy.types.Object:
    """Flat lever handle: wide round root, narrower rounded end, slight lower slant."""
    cx, cy, cz = root_center
    angle = math.radians(angle_degrees)
    direction = Vector((math.cos(angle), math.sin(angle), 0.0))
    normal = Vector((-direction.y, direction.x, 0.0))
    root = Vector((cx, cy, cz))
    end = root + direction * length
    z_top_root = cz + thickness / 2.0
    z_bottom_root = cz - thickness / 2.0
    z_top_end = z_top_root + 0.8
    z_bottom_end = z_bottom_root - 0.45
    vertices: list[tuple[float, float, float]] = []
    for point, width, z_top, z_bottom in ((root, root_width, z_top_root, z_bottom_root), (end, end_width, z_top_end, z_bottom_end)):
        side = normal * (width / 2.0)
        vertices.extend(
            [
                tuple(point - side + Vector((0.0, 0.0, z_bottom - point.z))),
                tuple(point + side + Vector((0.0, 0.0, z_bottom - point.z))),
                tuple(point - side + Vector((0.0, 0.0, z_top - point.z))),
                tuple(point + side + Vector((0.0, 0.0, z_top - point.z))),
            ]
        )
    faces = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (2, 3, 7, 6),
        (0, 4, 5, 1),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    add_bevel_modifier(obj, min(end_width, thickness) * 0.45, 5)
    return obj


def add_boolean_cylinder_cutter(
    name: str,
    *,
    radius: float,
    depth: float,
    location: tuple[float, float, float],
    axis: str,
    vertices: int = 72,
) -> bpy.types.Object:
    rotation = {
        "X": (0.0, math.pi / 2.0, 0.0),
        "Y": (math.pi / 2.0, 0.0, 0.0),
        "Z": (0.0, 0.0, 0.0),
    }[axis.upper()]
    return add_cylinder(name, radius * 2.0, depth, location, rotation=rotation, vertices=vertices)


def clean_mesh(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj.select_set(False)
    return obj


def prune_mesh_outside_bounds(
    obj: bpy.types.Object,
    *,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    z_bounds: tuple[float, float] | None = None,
) -> bpy.types.Object:
    """Delete boolean artifact faces that escape the known source hull envelope."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    def inside(vertex: bmesh.types.BMVert) -> bool:
        co = vertex.co
        return (
            (x_bounds is None or x_bounds[0] <= co.x <= x_bounds[1])
            and (y_bounds is None or y_bounds[0] <= co.y <= y_bounds[1])
            and (z_bounds is None or z_bounds[0] <= co.z <= z_bounds[1])
        )

    faces_to_delete = [face for face in bm.faces if not all(inside(vertex) for vertex in face.verts)]
    if faces_to_delete:
        bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")
    loose_verts = [vertex for vertex in bm.verts if not vertex.link_faces]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj.select_set(False)
    return obj


def scad_rotation_matrix(rotation_degrees: tuple[float, float, float]) -> Matrix:
    """OpenSCAD-style rotate([x,y,z]) matrix for source-coordinate model."""
    rx, ry, rz = (math.radians(value) for value in rotation_degrees)
    return (
        Matrix.Rotation(rz, 4, "Z")
        @ Matrix.Rotation(ry, 4, "Y")
        @ Matrix.Rotation(rx, 4, "X")
    )


def box_corner_points(size: tuple[float, float, float], center: tuple[float, float, float], matrix: Matrix) -> list[Vector]:
    sx, sy, sz = (axis / 2.0 for axis in size)
    cx, cy, cz = center
    points: list[Vector] = []
    for dx in (-sx, sx):
        for dy in (-sy, sy):
            for dz in (-sz, sz):
                points.append(matrix @ Vector((cx + dx, cy + dy, cz + dz)))
    return points


def add_transformed_box(
    name: str,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
    matrix: Matrix,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    points = box_corner_points(size, center, matrix)
    return add_convex_hull_mesh(name, points, material)


def add_source_arm_box_cutter(
    name: str,
    *,
    size: tuple[float, float, float],
    source_center: tuple[float, float, float],
    clearance: float,
    clamp_len: float,
    rotations: Sequence[tuple[float, float, float]] = (),
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Create a cutter from a cube placed in source arm() coordinates."""
    matrix = source_arm_to_target_matrix(clearance=clearance, clamp_len=clamp_len) @ Matrix.Translation(Vector(source_center))
    for rotation in rotations:
        matrix = matrix @ scad_rotation_matrix(rotation)
    points = box_corner_points(size, (0.0, 0.0, 0.0), matrix)
    return add_convex_hull_mesh(name, points, material)


def add_transformed_cylinder(
    name: str,
    radius: float,
    height: float,
    matrix: Matrix,
    *,
    vertices: int = 72,
    material: MaterialSpec | None = None,
    radius_top: float | None = None,
    radius_bottom: float | None = None,
) -> bpy.types.Object:
    if radius_top is None and radius_bottom is None:
        bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=height)
    else:
        bpy.ops.mesh.primitive_cone_add(
            vertices=vertices,
            radius1=radius if radius_bottom is None else radius_bottom,
            radius2=radius if radius_top is None else radius_top,
            depth=height,
        )
    obj = bpy.context.object
    obj.name = name
    obj.matrix_world = matrix
    if material:
        assign_material(obj, material)
    shade_smooth(obj)
    return obj


def add_convex_hull_mesh(
    name: str,
    points: Sequence[Vector],
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    bm = bmesh.new()
    verts = [bm.verts.new(point) for point in points]
    bm.verts.ensure_lookup_table()
    result = bmesh.ops.convex_hull(bm, input=verts)
    unused = list({geom for key in ("geom_unused", "geom_interior") for geom in result.get(key, [])})
    if unused:
        bmesh.ops.delete(bm, geom=unused, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material:
        assign_material(obj, material)
    return obj


def add_hull_from_transformed_boxes(
    name: str,
    boxes: Sequence[tuple[tuple[float, float, float], tuple[float, float, float], Matrix]],
    material: MaterialSpec,
) -> bpy.types.Object:
    points: list[Vector] = []
    for size, center, matrix in boxes:
        points.extend(box_corner_points(size, center, matrix))
    obj = add_convex_hull_mesh(name, points, material)
    add_bevel_modifier(obj, 0.06, 1)
    return obj


def source_arm_to_target_matrix(*, clearance: float, clamp_len: float) -> Matrix:
    """Transform points from source arm() coordinates into the assembled jaw coordinates."""
    return (
        Matrix.Translation(Vector((0.0, clearance, -clamp_len / 2.0)))
        @ scad_rotation_matrix((0.0, 180.0, -90.0))
        @ Matrix.Translation(Vector((0.0, 0.0, -clamp_len / 2.0)))
    )


def add_source_groove_clamp(
    name: str,
    *,
    length: float,
    groove_max: float,
    groove_min: float,
    groove_lip: float,
    groove_depth: float,
    extrusion_size: float,
    extension_depth: float,
    material: MaterialSpec,
    shadow_material: MaterialSpec,
) -> bpy.types.Object:
    """Source-like groove_clamp() cross-section as one compact t-slot prism."""
    flange_half = groove_max / 2.0
    neck_half = groove_min / 2.0
    lip_y = -groove_lip - 0.30
    bottom_clear_y = -groove_depth
    lower_hull_y = -min(extrusion_size / 2.0, groove_max * math.cos(math.radians(45.0))) + 0.30 - groove_lip
    bottom_t = max(0.0, min(1.0, (bottom_clear_y - lip_y) / (lower_hull_y - lip_y)))
    bottom_half = flange_half * max(0.0, 1.0 - bottom_t)
    extension_y = extension_depth + groove_lip / 2.0
    points = [
        (-neck_half, extension_y),
        (neck_half, extension_y),
        (neck_half, 0.0),
        (flange_half, lip_y),
        (bottom_half, bottom_clear_y),
        (-bottom_half, bottom_clear_y),
        (-flange_half, lip_y),
        (-neck_half, 0.0),
    ]
    half_z = length / 2.0
    vertices = [(x, y, -half_z) for x, y in points] + [(x, y, half_z) for x, y in points]
    count = len(points)
    triangles = tessellate_polygon([[Vector((x, y, 0.0)) for x, y in points]])
    faces: list[tuple[int, ...]] = [tuple(reversed(triangle)) for triangle in triangles]
    face_materials: list[int] = [0 for _ in faces]
    faces.extend(tuple(index + count for index in triangle) for triangle in triangles)
    face_materials.extend(0 for _ in triangles)
    for index in range(count):
        faces.append((index, (index + 1) % count, count + (index + 1) % count, count + index))
        face_materials.append(0)
    obj = add_multimaterial_mesh(name, vertices, faces, [material, shadow_material], face_materials)
    clean_mesh(obj)
    add_bevel_modifier(obj, 0.02, 1)
    return obj


def make_literal_source_arm_body(
    name: str,
    *,
    extrusion_size: float,
    clamp_len: float,
    jaw_len: float,
    pad_len: float,
    wall_thickness: float,
    groove_depth: float,
    clearance: float,
    material: MaterialSpec,
    shadow_material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Rebuild the source arm() hull from the two SCAD hull boxes."""
    matrix = source_arm_to_target_matrix(clearance=clearance, clamp_len=clamp_len)
    body = add_hull_from_transformed_boxes(
        name,
        (
            (
                (extrusion_size / 2.0, extrusion_size, clamp_len),
                (extrusion_size / 4.0, 0.0, 0.0),
                matrix,
            ),
            (
                (pad_len, extrusion_size, wall_thickness),
                (jaw_len - pad_len / 2.0, 0.0, clamp_len / 2.0 - wall_thickness / 2.0),
                matrix,
            ),
        ),
        material,
    )
    if shadow_material is not None:
        body.data.materials.append(get_material(shadow_material))
        body.data.materials.append(get_material(DEEP_SHADOW_GREEN))

    pad_center = jaw_len - pad_len / 2.0
    cutters: list[bpy.types.Object] = [
        add_source_arm_box_cutter(
            "source_strength_slot_left_literal_cut",
            size=(jaw_len, extrusion_size / 5.0, clamp_len),
            source_center=(jaw_len / 2.0 - 0.01 + extrusion_size, extrusion_size / 5.0, -wall_thickness),
            clearance=clearance,
            clamp_len=clamp_len,
        ),
        add_source_arm_box_cutter(
            "source_strength_slot_right_literal_cut",
            size=(jaw_len, extrusion_size / 5.0, clamp_len),
            source_center=(jaw_len / 2.0 - 0.01 + extrusion_size, -extrusion_size / 5.0, -wall_thickness),
            clearance=clearance,
            clamp_len=clamp_len,
        ),
        add_source_arm_box_cutter(
            "source_longitudinal_rotated_cube_v_cut",
            size=(groove_depth, groove_depth, pad_len + 0.02),
            source_center=(pad_center, 0.0, clamp_len / 2.0),
            clearance=clearance,
            clamp_len=clamp_len,
            rotations=((0.0, 90.0, 0.0), (0.0, 0.0, 45.0)),
        ),
    ]
    boolean_difference(body, cutters, solver="FLOAT")
    prune_mesh_outside_bounds(
        body,
        x_bounds=(-extrusion_size / 2.0 - 0.06, extrusion_size / 2.0 + 0.06),
        y_bounds=(clearance - 0.08, clearance + jaw_len + 0.08),
        z_bounds=(-clamp_len / 2.0 - 0.08, clamp_len / 2.0 + 0.08),
    )
    clean_mesh(body)
    weighted = body.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    bpy.ops.object.modifier_apply(modifier=weighted.name)
    body.select_set(False)
    return body


def trim_arm_body_front_projection(
    obj: bpy.types.Object,
    *,
    y_limit: float,
    top_z_limit: float,
    bottom_z_limit: float,
    side_x_limit: float,
) -> bpy.types.Object:
    """Hide the sampled arm from the straight front except for its source-like top/lower bridges."""
    mesh = obj.data
    for vertex in mesh.vertices:
        co = vertex.co
        if co.y <= y_limit and bottom_z_limit < co.z < top_z_limit and abs(co.x) > side_x_limit:
            co.x = side_x_limit if co.x > 0.0 else -side_x_limit
    mesh.update()
    clean_mesh(obj)
    return obj


def make_source_boolean_arm_body(
    name: str,
    *,
    extrusion_size: float,
    clamp_len: float,
    jaw_len: float,
    pad_len: float,
    wall_thickness: float,
    groove_depth: float,
    clearance: float,
    material: MaterialSpec,
    shadow_material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Source-coordinate arm() profile with stable sampled top slots and underside teeth."""
    root_y = clearance
    root_shell_end_y = clearance + extrusion_size * 0.30
    root_width_start_y = root_y
    top_width_full_y = clearance + extrusion_size * 0.86
    lower_width_full_y = clearance + extrusion_size * 2.35
    pad_start_y = clearance + jaw_len - pad_len
    tip_y = clearance + jaw_len
    body_half_width = extrusion_size * 0.430
    root_half_width = extrusion_size * 0.245
    lower_root_half_width = extrusion_size * 0.170
    top_root_z = clamp_len / 2.0
    top_pad_z = -clamp_len / 2.0 + wall_thickness
    bottom_z = -clamp_len / 2.0
    channel_start_y = pad_start_y + extrusion_size * 0.10
    channel_end_y = tip_y - extrusion_size * 0.12
    pitch = extrusion_size / 4.0
    cross_half_width = min(pitch * 0.125, groove_depth * 0.32)
    cross_flat_half_width = cross_half_width * 0.02
    trough_half_width = groove_depth * 0.42
    channel_left_outer = -0.600
    channel_left_center = -0.420
    channel_left_inner = -0.280
    channel_right_inner = 0.280
    channel_right_center = 0.420
    channel_right_outer = 0.600

    x_samples = (
        -1.0,
        -0.78,
        -0.62,
        channel_left_outer,
        -0.515,
        channel_left_center,
        -0.285,
        channel_left_inner,
        -0.12,
        0.0,
        0.12,
        channel_right_inner,
        0.285,
        channel_right_center,
        0.515,
        channel_right_outer,
        0.62,
        0.78,
        1.0,
    )
    y_values: list[float] = [
        root_y,
        root_shell_end_y,
        clearance + extrusion_size * 0.18,
        top_width_full_y - 0.01,
        top_width_full_y,
        top_width_full_y + 0.01,
        lower_width_full_y - 0.01,
        lower_width_full_y,
        lower_width_full_y + 0.01,
        clearance + extrusion_size * 0.66,
        clearance + extrusion_size * 0.72,
        channel_start_y - 2.0,
        channel_start_y - 0.01,
        channel_start_y,
        channel_start_y + 0.01,
        channel_end_y - 0.01,
        channel_end_y,
        channel_end_y + 0.01,
        pad_start_y,
        tip_y,
    ]
    groove_y = pad_start_y
    while groove_y <= tip_y + 0.001:
        y_values.extend([groove_y - cross_half_width, groove_y, groove_y + cross_half_width])
        groove_y += pitch
    y_values = sorted({round(max(root_y, min(tip_y, y)), 4) for y in y_values})

    def smoothstep_t(y_pos: float, start: float, end: float) -> float:
        if y_pos <= start:
            return 0.0
        if y_pos >= end:
            return 1.0
        t = (y_pos - start) / max(end - start, 0.001)
        return t * t * (3.0 - 2.0 * t)

    def top_half_width_at(y_pos: float) -> float:
        if y_pos >= top_width_full_y:
            return body_half_width
        if y_pos <= root_width_start_y:
            return root_half_width
        t = smoothstep_t(y_pos, root_width_start_y, top_width_full_y)
        return root_half_width + (body_half_width - root_half_width) * t

    def bottom_half_width_at(y_pos: float) -> float:
        if y_pos >= lower_width_full_y:
            return body_half_width
        if y_pos <= root_width_start_y:
            return lower_root_half_width
        t = smoothstep_t(y_pos, root_width_start_y, lower_width_full_y)
        return lower_root_half_width + (body_half_width - lower_root_half_width) * t

    def bottom_z_at(y_pos: float) -> float:
        root_floor_end = root_shell_end_y + extrusion_size * 0.24
        if y_pos <= root_floor_end:
            t = smoothstep_t(y_pos, root_y, root_floor_end)
            return bottom_z + groove_depth * 1.18 * (1.0 - t)
        if y_pos < pad_start_y:
            relief = max(0.0, 1.0 - smoothstep_t(y_pos, root_floor_end, pad_start_y))
            return bottom_z + groove_depth * 0.18 * relief
        return bottom_z

    def top_z_at(y_pos: float) -> float:
        if y_pos <= root_shell_end_y:
            return top_root_z
        if y_pos >= pad_start_y:
            return top_pad_z
        t = (y_pos - root_shell_end_y) / max(pad_start_y - root_shell_end_y, 0.001)
        return top_root_z + (top_pad_z - top_root_z) * t

    def top_z_for_x_norm(x_norm: float, y_pos: float) -> float:
        """Current iteration keeps root cleanup in plan width, not false side-top drops."""
        _ = x_norm
        return top_z_at(y_pos)

    def in_top_channel_norm(x_norm: float, y_pos: float) -> bool:
        if y_pos < channel_start_y or y_pos > channel_end_y:
            return False
        return (channel_left_outer <= x_norm <= channel_left_inner) or (
            channel_right_inner <= x_norm <= channel_right_outer
        )

    def in_top_channel(x_pos: float, y_pos: float) -> bool:
        return in_top_channel_norm(x_pos / max(top_half_width_at(y_pos), 0.001), y_pos)

    def top_channel_drop_norm(x_norm: float, y_pos: float) -> float:
        if not in_top_channel_norm(x_norm, y_pos):
            return 0.0
        if x_norm < 0.0:
            center = channel_left_center
            outer = channel_left_outer
            inner = channel_left_inner
        else:
            center = channel_right_center
            outer = channel_right_inner
            inner = channel_right_outer
        half_width = max(abs(center - outer), abs(inner - center), 0.001)
        distance = abs(x_norm - center) / half_width
        if distance >= 1.0:
            return 0.0
        edge_distance = min(abs(x_norm - outer), abs(inner - x_norm))
        edge_band = max(0.018, half_width * 0.12)
        edge_strength = max(0.0, min(1.0, edge_distance / edge_band))
        if edge_strength >= 1.0:
            strength = 1.0
        else:
            strength = edge_strength * edge_strength * (3.0 - 2.0 * edge_strength)
        end_band = groove_depth * 0.70
        end_strength = min(
            1.0,
            max(0.0, (y_pos - channel_start_y) / max(end_band, 0.001)),
            max(0.0, (channel_end_y - y_pos) / max(end_band, 0.001)),
        )
        return groove_depth * 1.30 * strength * end_strength

    def top_channel_strength_norm(x_norm: float, y_pos: float) -> float:
        if not in_top_channel_norm(x_norm, y_pos):
            return 0.0
        if x_norm < 0.0:
            center = channel_left_center
            half_width = max(center - channel_left_outer, channel_left_inner - center, 0.001)
        else:
            center = channel_right_center
            half_width = max(center - channel_right_inner, channel_right_outer - center, 0.001)
        return max(0.0, 1.0 - abs(x_norm - center) / half_width)

    def cross_raise_at(y_pos: float) -> float:
        if y_pos < pad_start_y:
            return 0.0
        local = y_pos - pad_start_y
        nearest_center = round(local / pitch) * pitch
        distance = abs(local - nearest_center)
        if distance >= cross_half_width:
            return 0.0
        if distance <= cross_flat_half_width:
            return groove_depth * 0.62
        t = (distance - cross_flat_half_width) / max(cross_half_width - cross_flat_half_width, 0.001)
        return (1.0 - t) * groove_depth * 0.62

    def trough_raise_at(x_pos: float, y_pos: float) -> float:
        if y_pos < pad_start_y:
            return 0.0
        distance = abs(x_pos)
        if distance >= trough_half_width:
            return 0.0
        t = 1.0 - distance / trough_half_width
        return (t * t) * groove_depth * 0.64

    vertices: list[tuple[float, float, float]] = []
    top_rows: list[list[int]] = []
    bottom_rows: list[list[int]] = []
    for y_pos in y_values:
        top_row: list[int] = []
        bottom_row: list[int] = []
        top_row_half_width = top_half_width_at(y_pos)
        bottom_row_half_width = bottom_half_width_at(y_pos)
        row_bottom_z = bottom_z_at(y_pos)
        for x_norm in x_samples:
            x_pos = x_norm * top_row_half_width
            local_top = top_z_for_x_norm(x_norm, y_pos)
            channel_drop = top_channel_drop_norm(x_norm, y_pos)
            if channel_drop > 0.0:
                local_top = max(row_bottom_z + groove_depth * 0.58, local_top - channel_drop)
            top_row.append(len(vertices))
            vertices.append((x_pos, y_pos, local_top))
        for x_norm in x_samples:
            x_pos = x_norm * bottom_row_half_width
            cross_raise = cross_raise_at(y_pos)
            trough_raise = trough_raise_at(x_pos, y_pos)
            local_top = top_z_for_x_norm(x_norm, y_pos)
            channel_drop = top_channel_drop_norm(x_norm, y_pos)
            if channel_drop > 0.0:
                local_top = max(row_bottom_z + groove_depth * 0.58, local_top - channel_drop)
            max_raise = max(0.0, local_top - row_bottom_z - 0.32)
            trough_influence = min(1.0, trough_raise / max(groove_depth * 0.76, 0.001))
            interrupted_cross = cross_raise * (1.0 - trough_influence) ** 9.60
            interrupted_trough = trough_raise * (0.20 if cross_raise > 0.04 else 0.13)
            raise_z = min(groove_depth * 0.50, max(interrupted_cross, interrupted_trough), max_raise)
            bottom_row.append(len(vertices))
            vertices.append((x_pos, y_pos, row_bottom_z + raise_z))
        top_rows.append(top_row)
        bottom_rows.append(bottom_row)

    faces: list[tuple[int, ...]] = []
    face_indices: list[int] = []
    shadow_index = 1 if shadow_material is not None else 0
    deep_index = 2 if shadow_material is not None else 0
    for row in range(len(y_values) - 1):
        y_mid = (y_values[row] + y_values[row + 1]) / 2.0
        mid_top_half_width = top_half_width_at(y_mid)
        mid_bottom_half_width = bottom_half_width_at(y_mid)
        for col in range(len(x_samples) - 1):
            x_mid_norm = (x_samples[col] + x_samples[col + 1]) / 2.0
            x_top_mid = x_mid_norm * mid_top_half_width
            x_bottom_mid = x_mid_norm * mid_bottom_half_width
            top_face = (top_rows[row][col], top_rows[row][col + 1], top_rows[row + 1][col + 1], top_rows[row + 1][col])
            faces.append(top_face)
            if in_top_channel_norm(x_mid_norm, y_mid):
                face_indices.append(deep_index if top_channel_strength_norm(x_mid_norm, y_mid) > 0.18 else shadow_index)
            else:
                face_indices.append(0)
            bottom_face = (
                bottom_rows[row][col + 1],
                bottom_rows[row][col],
                bottom_rows[row + 1][col],
                bottom_rows[row + 1][col + 1],
            )
            faces.append(bottom_face)
            cross_cut = y_mid >= pad_start_y and cross_raise_at(y_mid) > 0.04
            trough_cut = y_mid >= pad_start_y and trough_raise_at(x_bottom_mid, y_mid) > 0.04
            if trough_cut and cross_cut:
                face_indices.append(0)
            elif cross_cut:
                face_indices.append(shadow_index if abs(x_bottom_mid) > trough_half_width * 1.18 else 0)
            elif trough_cut:
                face_indices.append(deep_index)
            else:
                face_indices.append(0)
    for row in range(len(y_values) - 1):
        faces.append((top_rows[row][0], top_rows[row + 1][0], bottom_rows[row + 1][0], bottom_rows[row][0]))
        face_indices.append(0)
        last = len(x_samples) - 1
        faces.append((top_rows[row + 1][last], top_rows[row][last], bottom_rows[row][last], bottom_rows[row + 1][last]))
        face_indices.append(0)
    for col in range(len(x_samples) - 1):
        faces.append((top_rows[0][col + 1], top_rows[0][col], bottom_rows[0][col], bottom_rows[0][col + 1]))
        front_x_mid = ((x_samples[col] + x_samples[col + 1]) / 2.0) * bottom_half_width_at(root_y)
        face_indices.append(0)
        faces.append((top_rows[-1][col], top_rows[-1][col + 1], bottom_rows[-1][col + 1], bottom_rows[-1][col]))
        face_indices.append(0)

    materials = [material]
    if shadow_material is not None:
        materials.append(shadow_material)
        materials.append(DEEP_SHADOW_GREEN)
    body = add_multimaterial_mesh(name, vertices, faces, materials, face_indices)
    clean_mesh(body)
    weighted = body.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    bpy.ops.object.modifier_apply(modifier=weighted.name)
    body.select_set(False)
    return body


def make_compact_rectangular_source_arm_body(
    name: str,
    *,
    extrusion_size: float,
    clamp_len: float,
    jaw_len: float,
    pad_len: float,
    wall_thickness: float,
    groove_depth: float,
    clearance: float,
    material: MaterialSpec,
    shadow_material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Arm body with a short compact root and the source's rectangular jaw plan.

    This keeps the front/root projection from being governed by the full arm
    shell while preserving the top-view rectangle produced by the SCAD hull of
    the extrusion shell and pad boxes.
    """
    root_y = clearance
    tip_y = clearance + jaw_len
    pad_start_y = clearance + jaw_len - pad_len
    bottom_z = -clamp_len / 2.0
    top_root_z = clamp_len / 2.0
    top_pad_z = bottom_z + wall_thickness

    full_half_width = extrusion_size / 2.0
    compact_half_width = extrusion_size * 0.410
    width_start_y = root_y + extrusion_size * 0.08
    width_end_y = root_y + extrusion_size * 0.55
    top_slope_start_y = root_y + extrusion_size * 0.44
    top_slope_end_y = pad_start_y

    slot_y0 = pad_start_y + extrusion_size * 0.08
    slot_y1 = tip_y - extrusion_size * 0.11
    slot_centers = (-0.365, 0.365)
    slot_half_width = 0.125
    slot_center_half_width = 0.018

    pitch = max(extrusion_size / 4.0, groove_depth * 2.0)
    cross_half_y = min(pitch * 0.16, groove_depth * 0.38)
    trough_half_width = groove_depth * 0.34

    x_marks = (
        -1.0,
        -0.68,
        slot_centers[0] - slot_half_width,
        slot_centers[0] - slot_center_half_width,
        slot_centers[0],
        slot_centers[0] + slot_center_half_width,
        slot_centers[0] + slot_half_width,
        -0.12,
        -0.06,
        0.0,
        0.06,
        0.12,
        slot_centers[1] - slot_half_width,
        slot_centers[1] - slot_center_half_width,
        slot_centers[1],
        slot_centers[1] + slot_center_half_width,
        slot_centers[1] + slot_half_width,
        0.68,
        1.0,
    )
    edge = 0.006
    y_marks = [
        root_y,
        root_y + extrusion_size * 0.16,
        width_start_y - edge,
        width_start_y,
        width_end_y,
        width_end_y + edge,
        top_slope_start_y,
        pad_start_y - edge,
        pad_start_y,
        slot_y0 - edge,
        slot_y0,
        slot_y0 + edge,
        slot_y1 - edge,
        slot_y1,
        slot_y1 + edge,
        tip_y,
    ]
    groove_y = pad_start_y + pitch
    while groove_y <= tip_y - pitch * 0.35:
        y_marks.extend((groove_y - cross_half_y, groove_y, groove_y + cross_half_y))
        groove_y += pitch
    y_values = sorted({round(max(root_y, min(tip_y, value)), 4) for value in y_marks})

    def smoothstep(value: float) -> float:
        value = max(0.0, min(1.0, value))
        return value * value * (3.0 - 2.0 * value)

    def top_half_width_at(y_pos: float) -> float:
        if y_pos <= width_start_y:
            return compact_half_width
        if y_pos >= width_end_y:
            return full_half_width
        t = smoothstep((y_pos - width_start_y) / max(width_end_y - width_start_y, 0.001))
        return compact_half_width + (full_half_width - compact_half_width) * t

    def bottom_half_width_at(y_pos: float) -> float:
        if y_pos < pad_start_y:
            return compact_half_width
        if y_pos >= pad_start_y + extrusion_size * 0.16:
            return full_half_width
        t = smoothstep((y_pos - pad_start_y) / max(extrusion_size * 0.16, 0.001))
        return compact_half_width + (full_half_width - compact_half_width) * t

    def top_z_at(y_pos: float) -> float:
        if y_pos <= top_slope_start_y:
            return top_root_z
        if y_pos >= top_slope_end_y:
            return top_pad_z
        t = smoothstep((y_pos - top_slope_start_y) / max(top_slope_end_y - top_slope_start_y, 0.001))
        return top_root_z + (top_pad_z - top_root_z) * t

    def in_top_slot_norm(x_norm: float, y_pos: float) -> bool:
        if y_pos < slot_y0 or y_pos > slot_y1:
            return False
        return any(abs(x_norm - center) <= slot_half_width for center in slot_centers)

    def top_z_for(x_norm: float, y_pos: float) -> float:
        local_top = top_z_at(y_pos)
        if not in_top_slot_norm(x_norm, y_pos):
            return local_top
        nearest_center = min(slot_centers, key=lambda center: abs(x_norm - center))
        distance = abs(x_norm - nearest_center)
        edge_t = max(0.0, min(1.0, 1.0 - distance / max(slot_half_width, 0.001)))
        end_t = min(
            1.0,
            max(0.0, (y_pos - slot_y0) / max(groove_depth * 0.70, 0.001)),
            max(0.0, (slot_y1 - y_pos) / max(groove_depth * 0.70, 0.001)),
        )
        cut_depth = min(groove_depth * 1.18, local_top - bottom_z - 0.56)
        return max(bottom_z + wall_thickness * 0.34, local_top - cut_depth * edge_t * end_t)

    def cross_strength_at(y_pos: float) -> float:
        if y_pos < pad_start_y + pitch * 0.30:
            return 0.0
        local = y_pos - pad_start_y
        nearest_center = round(local / pitch) * pitch
        distance = abs(local - nearest_center)
        if distance >= cross_half_y:
            return 0.0
        return 1.0 - distance / max(cross_half_y, 0.001)

    def trough_strength_at(x_pos: float) -> float:
        distance = abs(x_pos)
        if distance >= trough_half_width:
            return 0.0
        return 1.0 - distance / max(trough_half_width, 0.001)

    def bottom_z_for(x_pos: float, y_pos: float) -> float:
        if y_pos < pad_start_y:
            return bottom_z
        cross = cross_strength_at(y_pos)
        trough = trough_strength_at(x_pos)
        in_trough = abs(x_pos) <= trough_half_width * 1.05
        cross_raise = groove_depth * 0.42 * cross * (1.0 - trough * 0.68)
        trough_raise = groove_depth * 0.30 * trough * (1.0 - cross * 0.58)
        if cross > 0.0 and trough > 0.0:
            raise_z = max(cross_raise, trough_raise, groove_depth * 0.10 * min(cross, trough))
        else:
            raise_z = max(cross_raise, trough_raise)
        return bottom_z + raise_z

    vertices: list[tuple[float, float, float]] = []
    top_rows: list[list[int]] = []
    bottom_rows: list[list[int]] = []
    for y_pos in y_values:
        top_half_width = top_half_width_at(y_pos)
        bottom_half_width = bottom_half_width_at(y_pos)
        top_row: list[int] = []
        bottom_row: list[int] = []
        for x_norm in x_marks:
            x_pos = x_norm * top_half_width
            top_row.append(len(vertices))
            vertices.append((x_pos, y_pos, top_z_for(x_norm, y_pos)))
        for x_norm in x_marks:
            x_pos = x_norm * bottom_half_width
            bottom_row.append(len(vertices))
            vertices.append((x_pos, y_pos, bottom_z_for(x_pos, y_pos)))
        top_rows.append(top_row)
        bottom_rows.append(bottom_row)

    shadow_index = 1 if shadow_material is not None else 0
    deep_index = 2 if shadow_material is not None else 0
    faces: list[tuple[int, ...]] = []
    face_indices: list[int] = []
    for row in range(len(y_values) - 1):
        y_mid = (y_values[row] + y_values[row + 1]) / 2.0
        bottom_mid_half_width = bottom_half_width_at(y_mid)
        for col in range(len(x_marks) - 1):
            x_norm_mid = (x_marks[col] + x_marks[col + 1]) / 2.0
            x_mid = x_norm_mid * bottom_mid_half_width

            faces.append((top_rows[row][col], top_rows[row][col + 1], top_rows[row + 1][col + 1], top_rows[row + 1][col]))
            face_indices.append(shadow_index if in_top_slot_norm(x_norm_mid, y_mid) else 0)

            faces.append((bottom_rows[row][col + 1], bottom_rows[row][col], bottom_rows[row + 1][col], bottom_rows[row + 1][col + 1]))
            if y_mid >= pad_start_y:
                cross = cross_strength_at(y_mid)
                trough = trough_strength_at(x_mid)
                in_trough = abs(x_mid) <= trough_half_width * 1.10
                if cross > 0.20 and trough > 0.18:
                    face_indices.append(deep_index if in_trough else shadow_index)
                elif cross > 0.20 and not in_trough:
                    face_indices.append(shadow_index)
                elif trough > 0.20 and cross < 0.10:
                    face_indices.append(deep_index)
                else:
                    face_indices.append(0)
            else:
                face_indices.append(0)

    for row in range(len(y_values) - 1):
        faces.append((top_rows[row][0], top_rows[row + 1][0], bottom_rows[row + 1][0], bottom_rows[row][0]))
        face_indices.append(0)
        last = len(x_marks) - 1
        faces.append((top_rows[row + 1][last], top_rows[row][last], bottom_rows[row][last], bottom_rows[row + 1][last]))
        face_indices.append(0)

    for col in range(len(x_marks) - 1):
        faces.append((top_rows[0][col + 1], top_rows[0][col], bottom_rows[0][col], bottom_rows[0][col + 1]))
        face_indices.append(0)
        faces.append((top_rows[-1][col], top_rows[-1][col + 1], bottom_rows[-1][col + 1], bottom_rows[-1][col]))
        face_indices.append(0)

    materials = [material]
    if shadow_material is not None:
        materials.append(shadow_material)
        materials.append(DEEP_SHADOW_GREEN)
    body = add_multimaterial_mesh(name, vertices, faces, materials, face_indices)
    clean_mesh(body)
    weighted = body.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    bpy.ops.object.modifier_apply(modifier=weighted.name)
    body.select_set(False)
    return body


def make_rear_started_rectangular_source_arm_body(
    name: str,
    *,
    extrusion_size: float,
    clamp_len: float,
    jaw_len: float,
    pad_len: float,
    wall_thickness: float,
    groove_depth: float,
    clearance: float,
    start_y: float,
    material: MaterialSpec,
    shadow_material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Clean long arm body whose front silhouette is owned by groove_clamp().

    The source arm() hull projects as a rectangular jaw in plan but the nearest
    front evidence is the compact groove slider.  This helper starts the large
    rectangular body behind that throat, keeps the plan rectangular, and uses a
    bounded mesh equivalent for the two top cuts plus the underside cross-field.
    """
    root_y = clearance + start_y
    tip_y = clearance + jaw_len
    pad_start_y = clearance + jaw_len - pad_len
    bottom_z = -clamp_len / 2.0
    top_root_z = clamp_len / 2.0
    top_pad_z = bottom_z + wall_thickness

    compact_half_width = extrusion_size * 0.210
    slot_y0 = pad_start_y + extrusion_size * 0.08
    slot_y1 = tip_y - extrusion_size * 0.10
    slot_edges_norm = (
        (-0.60, -0.23),
        (0.23, 0.60),
    )
    slot_wall = 0.055
    top_drop = min(groove_depth * 1.48, wall_thickness - 0.54)

    pitch = max(extrusion_size / 4.0, groove_depth * 2.0)
    cross_half_y = min(pitch * 0.20, groove_depth * 0.47)
    trough_half_width = groove_depth * 0.43

    trough_norm = trough_half_width / compact_half_width
    x_marks = [
        -1.0,
        -0.74,
        slot_edges_norm[0][0] - slot_wall,
        slot_edges_norm[0][0] + slot_wall,
        slot_edges_norm[0][1] - slot_wall,
        slot_edges_norm[0][1] + slot_wall,
        -trough_norm - slot_wall,
        -trough_norm + slot_wall,
        0.0,
        trough_norm - slot_wall,
        trough_norm + slot_wall,
        slot_edges_norm[1][0] - slot_wall,
        slot_edges_norm[1][0] + slot_wall,
        slot_edges_norm[1][1] - slot_wall,
        slot_edges_norm[1][1] + slot_wall,
        0.74,
        1.0,
    ]
    y_marks = [
        root_y,
        root_y + 0.10,
        min(pad_start_y - 0.10, root_y + extrusion_size * 0.70),
        pad_start_y,
        slot_y0 - 0.10,
        slot_y0 + 0.10,
        slot_y1 - 0.10,
        slot_y1 + 0.10,
        tip_y,
    ]
    groove_y = pad_start_y
    while groove_y <= tip_y + 0.001:
        y_marks.extend((groove_y - cross_half_y, groove_y, groove_y + cross_half_y))
        groove_y += pitch
    y_values = sorted({round(max(root_y, min(tip_y, value)), 4) for value in y_marks})

    def top_z_at(y_pos: float) -> float:
        _ = y_pos
        return top_pad_z - groove_depth * 0.42

    def smoothstep(value: float) -> float:
        value = max(0.0, min(1.0, value))
        return value * value * (3.0 - 2.0 * value)

    def lower_half_width_at(y_pos: float) -> float:
        _ = y_pos
        return compact_half_width

    def top_half_width_at(y_pos: float) -> float:
        _ = y_pos
        return compact_half_width

    def in_top_slot(x_norm: float, y_pos: float) -> bool:
        if y_pos <= slot_y0 or y_pos >= slot_y1:
            return False
        return any(left + slot_wall <= x_norm <= right - slot_wall for left, right in slot_edges_norm)

    def top_slot_overlay_masks_body(x_norm: float, y_pos: float) -> bool:
        _ = x_norm, y_pos
        return False

    def top_z_for(x_norm: float, y_pos: float) -> float:
        local_top = top_z_at(y_pos)
        if in_top_slot(x_norm, y_pos):
            return max(bottom_z + 0.42, local_top - top_drop)
        return local_top

    def cross_strength_at(y_pos: float) -> float:
        if y_pos < pad_start_y:
            return 0.0
        local = y_pos - pad_start_y
        nearest_center = round(local / pitch) * pitch
        distance = abs(local - nearest_center)
        if distance >= cross_half_y:
            return 0.0
        return 1.0 - distance / max(cross_half_y, 0.001)

    def trough_strength_at(x_pos: float) -> float:
        distance = abs(x_pos)
        if distance >= trough_half_width:
            return 0.0
        return 1.0 - distance / max(trough_half_width, 0.001)

    def bottom_z_for(x_pos: float, y_pos: float) -> float:
        if y_pos < pad_start_y:
            return bottom_z
        cross = cross_strength_at(y_pos)
        trough = trough_strength_at(x_pos)
        interrupted_trough = trough * groove_depth * 0.42 * (1.0 - cross * 0.72)
        side_tooth = cross * groove_depth * 0.58 * (1.0 - trough * 0.38)
        relief = max(interrupted_trough, side_tooth)
        if cross > 0.0 and trough > 0.0:
            relief = max(relief, groove_depth * 0.18 * min(cross, trough))
        return bottom_z + relief

    vertices: list[tuple[float, float, float]] = []
    top_rows: list[list[int]] = []
    bottom_rows: list[list[int]] = []
    for y_pos in y_values:
        lower_half_width = lower_half_width_at(y_pos)
        top_half_width = top_half_width_at(y_pos)
        top_row: list[int] = []
        bottom_row: list[int] = []
        for x_norm in x_marks:
            x_pos = x_norm * top_half_width
            top_row.append(len(vertices))
            vertices.append((x_pos, y_pos, top_z_for(x_norm, y_pos)))
        for x_norm in x_marks:
            x_pos = x_norm * lower_half_width
            bottom_row.append(len(vertices))
            vertices.append((x_pos, y_pos, bottom_z_for(x_pos, y_pos)))
        top_rows.append(top_row)
        bottom_rows.append(bottom_row)

    shadow_index = 1 if shadow_material is not None else 0
    deep_index = 2 if shadow_material is not None else 0
    faces: list[tuple[int, ...]] = []
    face_indices: list[int] = []
    for row in range(len(y_values) - 1):
        y_mid = (y_values[row] + y_values[row + 1]) / 2.0
        lower_half_width = lower_half_width_at(y_mid)
        for col in range(len(x_marks) - 1):
            x_norm_mid = (x_marks[col] + x_marks[col + 1]) / 2.0
            x_mid = x_norm_mid * lower_half_width
            faces.append((top_rows[row][col], top_rows[row][col + 1], top_rows[row + 1][col + 1], top_rows[row + 1][col]))
            face_indices.append(0 if top_slot_overlay_masks_body(x_norm_mid, y_mid) else shadow_index if in_top_slot(x_norm_mid, y_mid) else 0)

            faces.append((bottom_rows[row][col + 1], bottom_rows[row][col], bottom_rows[row + 1][col], bottom_rows[row + 1][col + 1]))
            if y_mid >= pad_start_y:
                cross = cross_strength_at(y_mid)
                trough = trough_strength_at(x_mid)
                if cross > 0.22 and trough > 0.22:
                    face_indices.append(shadow_index)
                elif cross > 0.22:
                    face_indices.append(shadow_index)
                elif trough > 0.24:
                    face_indices.append(deep_index)
                else:
                    face_indices.append(0)
            else:
                face_indices.append(0)

    for row in range(len(y_values) - 1):
        faces.append((top_rows[row][0], top_rows[row + 1][0], bottom_rows[row + 1][0], bottom_rows[row][0]))
        face_indices.append(0)
        last = len(x_marks) - 1
        faces.append((top_rows[row + 1][last], top_rows[row][last], bottom_rows[row][last], bottom_rows[row + 1][last]))
        face_indices.append(0)

    for col in range(len(x_marks) - 1):
        faces.append((top_rows[0][col + 1], top_rows[0][col], bottom_rows[0][col], bottom_rows[0][col + 1]))
        face_indices.append(0)
        faces.append((top_rows[-1][col], top_rows[-1][col + 1], bottom_rows[-1][col + 1], bottom_rows[-1][col]))
        face_indices.append(0)

    materials = [material]
    if shadow_material is not None:
        materials.append(shadow_material)
        materials.append(DEEP_SHADOW_GREEN)
    body = add_multimaterial_mesh(name, vertices, faces, materials, face_indices)
    clean_mesh(body)
    weighted = body.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    bpy.ops.object.modifier_apply(modifier=weighted.name)
    body.select_set(False)
    return body


def make_direct_boolean_jaw_body(
    name: str,
    *,
    extrusion_size: float,
    clamp_len: float,
    jaw_len: float,
    pad_len: float,
    wall_thickness: float,
    groove_depth: float,
    clearance: float,
    material: MaterialSpec,
) -> bpy.types.Object:
    """Source hull arm body; visible groove faces are handled by the body/cut assembly."""
    return make_literal_source_arm_body(
        name,
        extrusion_size=extrusion_size,
        clamp_len=clamp_len,
        jaw_len=jaw_len,
        pad_len=pad_len,
        wall_thickness=wall_thickness,
        groove_depth=groove_depth,
        clearance=clearance,
        material=material,
        shadow_material=SHADOW_GREEN,
    )


def make_source_cut_shadow_inserts(
    name: str,
    *,
    extrusion_size: float,
    clamp_len: float,
    jaw_len: float,
    pad_len: float,
    wall_thickness: float,
    groove_depth: float,
    clearance: float,
    material: MaterialSpec,
    deep_material: MaterialSpec,
) -> list[bpy.types.Object]:
    """Dark recess surfaces placed inside the literal source cube cuts."""
    matrix = source_arm_to_target_matrix(clearance=clearance, clamp_len=clamp_len)
    inserts: list[bpy.types.Object] = []

    slot_start_x = extrusion_size + 1.0
    slot_end_x = jaw_len - 2.0
    slot_len = slot_end_x - slot_start_x
    slot_center_x = (slot_start_x + slot_end_x) / 2.0
    for side, source_y in (("left", -extrusion_size / 5.0), ("right", extrusion_size / 5.0)):
        floor = add_transformed_box(
            f"{name}_{side}_top_slot_deep_floor",
            (slot_len, extrusion_size * 0.145, 0.07),
            (slot_center_x, source_y, clamp_len / 2.0 - groove_depth * 0.92),
            matrix,
            deep_material,
        )
        inserts.append(floor)

    pad_center_x = jaw_len - pad_len / 2.0
    center_trough = add_transformed_box(
        f"{name}_continuous_underside_v_shadow",
        (pad_len + extrusion_size * 0.02, groove_depth * 0.34, 0.055),
        (pad_center_x, 0.0, clamp_len / 2.0 - groove_depth * 0.52),
        matrix,
        deep_material,
    )
    inserts.append(center_trough)

    pitch = max(extrusion_size / 4.0, groove_depth * 2.0)
    groove_i = 0.0
    groove_index = 0
    while groove_i <= pad_len + 0.001:
        local_x = groove_i - pad_len / 2.0
        cross = add_transformed_box(
            f"{name}_underside_transverse_shadow_{groove_index:02d}",
            (groove_depth * 0.42, extrusion_size * 0.74, 0.05),
            (pad_center_x + local_x, 0.0, clamp_len / 2.0 - groove_depth * 0.34),
            matrix,
            material,
        )
        inserts.append(cross)
        groove_i += pitch
        groove_index += 1
    for obj in inserts:
        clean_mesh(obj)
    return inserts


def make_integrated_underside_cut_field(
    name: str,
    *,
    pad_start_y: float,
    pad_len: float,
    half_width: float,
    bottom_z: float,
    pitch: float,
    groove_depth: float,
    shadow_material: MaterialSpec,
    deep_material: MaterialSpec,
) -> bpy.types.Object:
    """Dark recess faces for the body-owned underside tooth and trough cuts."""
    trough_half = groove_depth * 0.58
    cross_half = min(pitch * 0.18, groove_depth * 0.46)

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    face_indices: list[int] = []

    def add_quad(
        coords: Sequence[tuple[float, float, float]],
        material_index: int,
    ) -> None:
        start = len(vertices)
        vertices.extend(coords)
        faces.append((start + 3, start + 2, start + 1, start))
        face_indices.append(material_index)

    trough_edge_z = bottom_z - 0.006
    trough_center_z = bottom_z - 0.190
    cross_edge_z = bottom_z - 0.010
    cross_center_z = bottom_z - 0.120
    y0 = pad_start_y
    y1 = pad_start_y + pad_len
    add_quad(
        (
            (-trough_half, y0, trough_edge_z),
            (0.0, y0, trough_center_z),
            (0.0, y1, trough_center_z),
            (-trough_half, y1, trough_edge_z),
        ),
        1,
    )
    add_quad(
        (
            (0.0, y0, trough_center_z),
            (trough_half, y0, trough_edge_z),
            (trough_half, y1, trough_edge_z),
            (0.0, y1, trough_center_z),
        ),
        1,
    )
    groove_y = pad_start_y
    while groove_y <= pad_start_y + pad_len + 0.001:
        cut_y0 = max(pad_start_y, groove_y - cross_half * 0.94)
        cut_y1 = min(pad_start_y + pad_len, groove_y + cross_half * 0.94)
        add_quad(
            (
                (-half_width, cut_y0, cross_edge_z),
                (-trough_half, cut_y0, cross_center_z),
                (-trough_half, cut_y1, cross_center_z),
                (-half_width, cut_y1, cross_edge_z),
            ),
            0,
        )
        add_quad(
            (
                (trough_half, cut_y0, cross_center_z),
                (half_width, cut_y0, cross_edge_z),
                (half_width, cut_y1, cross_edge_z),
                (trough_half, cut_y1, cross_center_z),
            ),
            0,
        )
        groove_y += pitch

    obj = add_multimaterial_mesh(name, vertices, faces, [shadow_material, deep_material], face_indices)
    clean_mesh(obj)
    return obj


def make_longitudinal_v_groove_faces(
    name: str,
    *,
    center_x: float,
    y_start: float,
    y_end: float,
    top_z: float,
    width: float,
    depth: float,
    shadow_material: MaterialSpec,
    deep_material: MaterialSpec,
) -> bpy.types.Object:
    """Visible sloped faces for a narrow source-like longitudinal V cut."""
    half_width = width / 2.0
    center_half = width * 0.010
    edge_z = top_z + 0.008
    center_z = top_z - depth
    vertices = [
        (center_x - half_width, y_start, edge_z),
        (center_x - center_half, y_start, center_z),
        (center_x - center_half, y_end, center_z),
        (center_x - half_width, y_end, edge_z),
        (center_x + center_half, y_start, center_z),
        (center_x + half_width, y_start, edge_z),
        (center_x + half_width, y_end, edge_z),
        (center_x + center_half, y_end, center_z),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    ]
    obj = add_multimaterial_mesh(name, vertices, faces, [shadow_material], [0, 0])
    clean_mesh(obj)
    return obj


def make_rectangular_pad_top_surface(
    name: str,
    *,
    half_width: float,
    y_start: float,
    y_end: float,
    top_z: float,
    top_z_start: float | None = None,
    slope_y_end: float | None = None,
    slot_centers: Sequence[float],
    slot_width: float,
    slot_y_start: float,
    slot_y_end: float,
    slot_depth: float,
    material: MaterialSpec,
    shadow_material: MaterialSpec,
) -> bpy.types.Object:
    """Zero-thickness top pad surface with grooves cut into the visible face."""
    x_marks = [-half_width]
    for center in slot_centers:
        half_slot = slot_width / 2.0
        center_half = slot_width * 0.018
        x_marks.extend(
            (
                center - half_slot,
                center - center_half,
                center + center_half,
                center + half_slot,
            )
        )
    x_marks.append(half_width)
    x_marks = sorted({round(value, 4) for value in x_marks})
    y_seed = [y_start, slot_y_start, slot_y_end, y_end]
    if slope_y_end is not None:
        y_seed.append(slope_y_end)
    y_marks = sorted({round(max(y_start, min(y_end, value)), 4) for value in y_seed})

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    face_indices: list[int] = []

    def in_slot(x_mid: float, y_mid: float) -> bool:
        if y_mid <= slot_y_start or y_mid >= slot_y_end:
            return False
        return any(abs(x_mid - center) < slot_width / 2.0 for center in slot_centers)

    def base_z_at(y_pos: float) -> float:
        if top_z_start is None or slope_y_end is None:
            return top_z
        if y_pos <= y_start:
            return top_z_start
        if y_pos >= slope_y_end:
            return top_z
        t = (y_pos - y_start) / max(slope_y_end - y_start, 0.001)
        return top_z_start + (top_z - top_z_start) * t

    def z_at(x_pos: float, y_pos: float) -> float:
        local_top = base_z_at(y_pos)
        if y_pos <= slot_y_start or y_pos >= slot_y_end:
            return local_top
        for center in slot_centers:
            half_slot = slot_width / 2.0
            dist = abs(x_pos - center)
            if dist < half_slot:
                return local_top - slot_depth * (1.0 - dist / max(half_slot, 0.001))
        return local_top

    for y0, y1 in zip(y_marks[:-1], y_marks[1:]):
        y_mid = (y0 + y1) / 2.0
        for x0, x1 in zip(x_marks[:-1], x_marks[1:]):
            x_mid = (x0 + x1) / 2.0
            start = len(vertices)
            vertices.extend(
                (
                    (x0, y0, z_at(x0, y0)),
                    (x1, y0, z_at(x1, y0)),
                    (x1, y1, z_at(x1, y1)),
                    (x0, y1, z_at(x0, y1)),
                )
            )
            faces.append((start, start + 1, start + 2, start + 3))
            face_indices.append(1 if in_slot(x_mid, y_mid) else 0)

    obj = add_multimaterial_mesh(name, vertices, faces, [material, shadow_material], face_indices)
    clean_mesh(obj)
    return obj


def make_continuous_sloped_arm_top_shell(
    name: str,
    *,
    half_width: float,
    y_start: float,
    y_slope_end: float,
    y_end: float,
    top_z_start: float,
    top_z_end: float,
    thickness: float,
    slot_centers: Sequence[float],
    slot_width: float,
    slot_y_start: float,
    slot_y_end: float,
    slot_depth: float,
    material: MaterialSpec,
    shadow_material: MaterialSpec,
) -> bpy.types.Object:
    """Thin volume that keeps the source arm plan rectangular while the side profile slopes."""
    x_marks = [-half_width]
    for center in slot_centers:
        half_slot = slot_width / 2.0
        center_half = max(slot_width * 0.018, 0.018)
        x_marks.extend(
            (
                center - half_slot,
                center - center_half,
                center + center_half,
                center + half_slot,
            )
        )
    x_marks.append(half_width)
    x_marks = sorted({round(value, 4) for value in x_marks})

    y_marks = sorted(
        {
            round(max(y_start, min(y_end, value)), 4)
            for value in (
                y_start,
                y_slope_end,
                slot_y_start - 0.01,
                slot_y_start,
                slot_y_start + 0.01,
                slot_y_end - 0.01,
                slot_y_end,
                slot_y_end + 0.01,
                y_end,
            )
        }
    )

    def top_z_base(y_pos: float) -> float:
        if y_pos >= y_slope_end:
            return top_z_end
        t = (y_pos - y_start) / max(y_slope_end - y_start, 0.001)
        return top_z_start + (top_z_end - top_z_start) * max(0.0, min(1.0, t))

    def in_slot(x_pos: float, y_pos: float) -> bool:
        if y_pos <= slot_y_start or y_pos >= slot_y_end:
            return False
        return any(abs(x_pos - center) <= slot_width / 2.0 for center in slot_centers)

    def top_z_at(x_pos: float, y_pos: float) -> float:
        local_top = top_z_base(y_pos)
        if y_pos <= slot_y_start or y_pos >= slot_y_end:
            return local_top
        for center in slot_centers:
            half_slot = slot_width / 2.0
            distance = abs(x_pos - center)
            if distance <= half_slot:
                return local_top - slot_depth * (1.0 - distance / max(half_slot, 0.001))
        return local_top

    vertices: list[tuple[float, float, float]] = []
    top_rows: list[list[int]] = []
    bottom_rows: list[list[int]] = []
    for y_pos in y_marks:
        top_row: list[int] = []
        bottom_row: list[int] = []
        for x_pos in x_marks:
            top_row.append(len(vertices))
            vertices.append((x_pos, y_pos, top_z_at(x_pos, y_pos)))
        for x_pos in x_marks:
            local_bottom = min(top_z_base(y_pos) - thickness, top_z_at(x_pos, y_pos) - thickness * 0.22)
            bottom_row.append(len(vertices))
            vertices.append((x_pos, y_pos, local_bottom))
        top_rows.append(top_row)
        bottom_rows.append(bottom_row)

    faces: list[tuple[int, ...]] = []
    face_indices: list[int] = []
    for row in range(len(y_marks) - 1):
        y_mid = (y_marks[row] + y_marks[row + 1]) / 2.0
        for col in range(len(x_marks) - 1):
            x_mid = (x_marks[col] + x_marks[col + 1]) / 2.0
            faces.append((top_rows[row][col], top_rows[row][col + 1], top_rows[row + 1][col + 1], top_rows[row + 1][col]))
            face_indices.append(1 if in_slot(x_mid, y_mid) else 0)
            faces.append((bottom_rows[row][col + 1], bottom_rows[row][col], bottom_rows[row + 1][col], bottom_rows[row + 1][col + 1]))
            face_indices.append(0)

    last_x = len(x_marks) - 1
    for row in range(len(y_marks) - 1):
        faces.append((top_rows[row][0], top_rows[row + 1][0], bottom_rows[row + 1][0], bottom_rows[row][0]))
        face_indices.append(0)
        faces.append((top_rows[row + 1][last_x], top_rows[row][last_x], bottom_rows[row][last_x], bottom_rows[row + 1][last_x]))
        face_indices.append(0)

    for col in range(len(x_marks) - 1):
        faces.append((top_rows[0][col + 1], top_rows[0][col], bottom_rows[0][col], bottom_rows[0][col + 1]))
        face_indices.append(0)
        faces.append((top_rows[-1][col], top_rows[-1][col + 1], bottom_rows[-1][col + 1], bottom_rows[-1][col]))
        face_indices.append(0)

    obj = add_multimaterial_mesh(name, vertices, faces, [material, shadow_material], face_indices)
    clean_mesh(obj)
    weighted = obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=weighted.name)
    obj.select_set(False)
    return obj


def make_flat_cam_lever_hull(
    name: str,
    *,
    root_center: tuple[float, float, float],
    lever_len: float,
    cam_dia: float,
    material: MaterialSpec,
    vertices: int = 72,
) -> bpy.types.Object:
    """Source-style two-hull handle with a flatter main plate and lower bevel."""
    root = Vector(root_center)
    end = root + Vector((0.0, lever_len, 0.0))
    main = add_vertical_cylinder_hull(
        f"{name}_upper_flat_plate",
        (
            ((root.x, root.y, root.z), cam_dia * 0.430, cam_dia * 0.150),
            ((end.x, end.y, root.z), cam_dia * 0.215, cam_dia * 0.150),
        ),
        vertices=vertices,
        material=material,
    )
    bevel = add_vertical_conical_hull(
        f"{name}_lower_bevel_plate",
        (
            ((root.x, root.y, root.z - cam_dia * 0.165), cam_dia * 0.240, cam_dia * 0.405, cam_dia * 0.205),
            ((end.x, end.y, root.z - cam_dia * 0.165), cam_dia * 0.092, cam_dia * 0.210, cam_dia * 0.205),
        ),
        vertices=vertices,
        material=material,
    )
    obj = boolean_union(name, (main, bevel))
    clean_mesh(obj)
    weighted = obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=weighted.name)
    obj.select_set(False)
    return obj


def make_compact_root_front_profile(
    name: str,
    *,
    width: float,
    height: float,
    y_front: float,
    thickness: float,
    material: MaterialSpec,
    shadow_material: MaterialSpec,
) -> list[bpy.types.Object]:
    """Deprecated overlay hook; the active root is the actual groove_clamp() prism."""
    _ = (name, width, height, y_front, thickness, material, shadow_material)
    return []


def make_serrated_jaw_body(
    name: str,
    *,
    length: float,
    root_y: float,
    root_half_width: float,
    full_half_width: float,
    width_transition: float,
    top_z_root: float,
    top_z_tip: float,
    bottom_z: float,
    pad_start_y: float,
    groove_pitch: float,
    cross_depth: float,
    longitudinal_depth: float,
    channel_start_y: float,
    channel_end_y: float,
    channel_depth: float,
    channel_floor_z: float | None = None,
    material: MaterialSpec,
    shadow_material: MaterialSpec,
) -> bpy.types.Object:
    """Wedge jaw mesh with rectangular top recesses and rotated-cube-like underside cuts."""
    tip_y = root_y + length
    x_base = (
        -1.0,
        -0.67,
        -0.60,
        -0.30,
        -0.26,
        0.0,
        0.26,
        0.30,
        0.60,
        0.67,
        1.0,
    )
    channel_edge = 0.004
    y_values: list[float] = [
        root_y,
        root_y + 10.0,
        channel_start_y - channel_edge,
        channel_start_y,
        channel_start_y + channel_edge,
        channel_end_y - channel_edge,
        channel_end_y,
        channel_end_y + channel_edge,
    ]
    cross_half_width = min(groove_pitch * 0.34, max(cross_depth * 1.08, 1.45))
    cross_flat_half_width = cross_half_width * 0.22
    y = pad_start_y
    while y <= tip_y + 0.001:
        for offset in (
            -cross_half_width,
            0.0,
            cross_half_width,
        ):
            y_values.append(y + offset)
        y += groove_pitch
    y_values.append(tip_y)
    y_values = sorted({round(max(root_y, min(tip_y, value)), 4) for value in y_values})

    def half_width_at(y_pos: float) -> float:
        if y_pos <= root_y:
            return root_half_width
        if y_pos >= root_y + width_transition:
            return full_half_width
        t = (y_pos - root_y) / max(width_transition, 0.001)
        t = t * t * (3.0 - 2.0 * t)
        return root_half_width + (full_half_width - root_half_width) * t

    def top_z_at(y_pos: float) -> float:
        shoulder_y = root_y + 10.0
        if y_pos <= shoulder_y:
            return top_z_root
        t = (y_pos - shoulder_y) / max(tip_y - shoulder_y, 0.001)
        return top_z_root + (top_z_tip - top_z_root) * t

    def cross_raise_at(y_pos: float) -> float:
        if y_pos < pad_start_y:
            return 0.0
        local = y_pos - pad_start_y
        nearest_center = round(local / groove_pitch) * groove_pitch
        distance = abs(local - nearest_center)
        if distance >= cross_half_width:
            return 0.0
        if distance <= cross_flat_half_width:
            return cross_depth
        t = (distance - cross_flat_half_width) / max(cross_half_width - cross_flat_half_width, 0.001)
        return (1.0 - t) * cross_depth

    def trough_raise_at(x_norm: float) -> float:
        normalized = abs(x_norm) / 0.20
        return max(0.0, 1.0 - normalized) * longitudinal_depth

    def in_top_channel(x_norm: float, y_pos: float) -> bool:
        if y_pos < channel_start_y or y_pos > channel_end_y:
            return False
        return (-0.60 <= x_norm <= -0.30) or (0.30 <= x_norm <= 0.60)

    def top_channel_edge(x_norm: float, y_pos: float) -> bool:
        if y_pos < channel_start_y or y_pos > channel_end_y:
            return False
        return x_norm in {-0.60, -0.30, 0.30, 0.60}

    def top_drop_at(x_norm: float, y_pos: float) -> float:
        if not in_top_channel(x_norm, y_pos):
            return 0.0
        return channel_depth

    def top_cut_z_at(x_norm: float, y_pos: float, local_top_z: float) -> float:
        if not in_top_channel(x_norm, y_pos):
            return local_top_z
        available_depth = max(0.0, local_top_z - bottom_z - 0.42)
        floor_z = local_top_z - min(top_drop_at(x_norm, y_pos), available_depth)
        if channel_floor_z is not None:
            floor_z = max(bottom_z + 0.42, min(floor_z, channel_floor_z))
        return floor_z

    vertices: list[tuple[float, float, float]] = []
    top_rows: list[list[int]] = []
    bottom_rows: list[list[int]] = []
    for y_pos in y_values:
        half_width = half_width_at(y_pos)
        x_values = [value * half_width for value in x_base]
        top_row: list[int] = []
        bottom_row: list[int] = []
        for x_norm, x_pos in zip(x_base, x_values, strict=True):
            top_row.append(len(vertices))
            local_top_z = top_z_at(y_pos)
            vertices.append((x_pos, y_pos, top_cut_z_at(x_norm, y_pos, local_top_z)))
        cross_raise = cross_raise_at(y_pos)
        for x_norm, x_pos in zip(x_base, x_values, strict=True):
            bottom_row.append(len(vertices))
            trough = trough_raise_at(x_norm)
            z_raise = cross_raise if cross_raise > 0.0 else trough
            vertices.append((x_pos, y_pos, bottom_z + z_raise))
        top_rows.append(top_row)
        bottom_rows.append(bottom_row)

    faces: list[tuple[int, ...]] = []
    face_indices: list[int] = []
    channel_floor_faces: list[tuple[int, ...]] = []
    y_index = {round(value, 4): index for index, value in enumerate(y_values)}
    x_index = {round(value, 4): index for index, value in enumerate(x_base)}
    channel_start_index = y_index[round(channel_start_y, 4)]
    channel_end_index = y_index[round(channel_end_y, 4)]
    for x_left_norm, x_right_norm in ((-0.60, -0.30), (0.30, 0.60)):
        x_left_index = x_index[round(x_left_norm, 4)]
        x_right_index = x_index[round(x_right_norm, 4)]
        channel_floor_faces.append(
            (
                top_rows[channel_start_index][x_left_index],
                top_rows[channel_start_index][x_right_index],
                top_rows[channel_end_index][x_right_index],
                top_rows[channel_end_index][x_left_index],
            )
        )
    for row in range(len(y_values) - 1):
        for col in range(len(x_base) - 1):
            avg_x_norm = (x_base[col] + x_base[col + 1]) / 2.0
            avg_y = (y_values[row] + y_values[row + 1]) / 2.0
            if not in_top_channel(avg_x_norm, avg_y):
                faces.append((top_rows[row][col], top_rows[row][col + 1], top_rows[row + 1][col + 1], top_rows[row + 1][col]))
                face_indices.append(0)
            bottom_face = (
                bottom_rows[row][col + 1],
                bottom_rows[row][col],
                bottom_rows[row + 1][col],
                bottom_rows[row + 1][col + 1],
            )
            faces.append(bottom_face)
            long_raise = trough_raise_at(avg_x_norm)
            cross_raise = cross_raise_at((y_values[row] + y_values[row + 1]) / 2.0)
            if cross_raise > 0.08 or (long_raise > longitudinal_depth * 0.70 and cross_raise < 0.04):
                face_indices.append(1)
            else:
                face_indices.append(0)

    for floor_face in channel_floor_faces:
        faces.append(floor_face)
        face_indices.append(2)

    for row in range(len(y_values) - 1):
        faces.append((top_rows[row][0], top_rows[row + 1][0], bottom_rows[row + 1][0], bottom_rows[row][0]))
        face_indices.append(0)
        last = len(x_base) - 1
        faces.append((top_rows[row + 1][last], top_rows[row][last], bottom_rows[row][last], bottom_rows[row + 1][last]))
        face_indices.append(0)

    for col in range(len(x_base) - 1):
        faces.append((top_rows[0][col + 1], top_rows[0][col], bottom_rows[0][col], bottom_rows[0][col + 1]))
        face_indices.append(0)
        faces.append((top_rows[-1][col], top_rows[-1][col + 1], bottom_rows[-1][col + 1], bottom_rows[-1][col]))
        face_indices.append(0)

    obj = add_multimaterial_mesh(name, vertices, faces, [material, shadow_material, DEEP_SHADOW_GREEN], face_indices)
    clean_mesh(obj)
    weighted = obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=weighted.name)
    obj.select_set(False)
    return obj


def make_sloped_slot_floor_overlay(
    name: str,
    *,
    x_min: float,
    x_max: float,
    y_start: float,
    y_end: float,
    top_z_root: float,
    top_z_tip: float,
    bottom_z: float,
    channel_depth: float,
    root_y: float,
    tip_y: float,
    material: MaterialSpec,
) -> bpy.types.Object:
    """Clean shadow floor for the two long cube-cut top relief slots."""

    def top_z_at(y_pos: float) -> float:
        shoulder_y = root_y + 10.0
        if y_pos <= shoulder_y:
            return top_z_root
        t = (y_pos - shoulder_y) / max(tip_y - shoulder_y, 0.001)
        return top_z_root + (top_z_tip - top_z_root) * t

    def floor_z_at(y_pos: float) -> float:
        local_top_z = top_z_at(y_pos)
        available_depth = max(0.0, local_top_z - bottom_z - 0.42)
        return local_top_z - min(channel_depth, available_depth) + 0.16

    z0 = floor_z_at(y_start)
    z1 = floor_z_at(y_end)
    bottom_offset = 0.05
    vertices = [
        (x_min, y_start, z0 - bottom_offset),
        (x_max, y_start, z0 - bottom_offset),
        (x_max, y_end, z1 - bottom_offset),
        (x_min, y_end, z1 - bottom_offset),
        (x_min, y_start, z0),
        (x_max, y_start, z0),
        (x_max, y_end, z1),
        (x_min, y_end, z1),
    ]
    faces = [
        (4, 5, 6, 7),
        (0, 3, 2, 1),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
    ]
    obj = add_multimaterial_mesh(name, vertices, faces, [material])
    clean_mesh(obj)
    return obj


def cylinder_ring_points(
    *,
    center: Vector,
    radius: float,
    height: float,
    vertices: int,
) -> list[Vector]:
    half_h = height / 2.0
    points: list[Vector] = []
    for z in (-half_h, half_h):
        for index in range(vertices):
            angle = 2.0 * math.pi * index / vertices
            points.append(center + Vector((math.cos(angle) * radius, math.sin(angle) * radius, z)))
    return points


def cylinder_ring_points_axis(
    *,
    center: Vector,
    radius: float,
    depth: float,
    axis: str,
    vertices: int,
) -> list[Vector]:
    half_depth = depth / 2.0
    axis_key = axis.upper()
    points: list[Vector] = []
    for side in (-half_depth, half_depth):
        for index in range(vertices):
            angle = 2.0 * math.pi * index / vertices
            ca = math.cos(angle) * radius
            sa = math.sin(angle) * radius
            if axis_key == "X":
                points.append(center + Vector((side, ca, sa)))
            elif axis_key == "Y":
                points.append(center + Vector((ca, side, sa)))
            elif axis_key == "Z":
                points.append(center + Vector((ca, sa, side)))
            else:
                raise ValueError(f"Unsupported axis {axis!r}")
    return points


def cone_ring_points(
    *,
    center: Vector,
    radius_bottom: float,
    radius_top: float,
    height: float,
    vertices: int,
) -> list[Vector]:
    half_h = height / 2.0
    points: list[Vector] = []
    for z, radius in ((-half_h, radius_bottom), (half_h, radius_top)):
        for index in range(vertices):
            angle = 2.0 * math.pi * index / vertices
            points.append(center + Vector((math.cos(angle) * radius, math.sin(angle) * radius, z)))
    return points


def add_vertical_cylinder_hull(
    name: str,
    cylinders: Sequence[tuple[tuple[float, float, float], float, float]],
    *,
    vertices: int = 40,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Hull several vertical cylinders, matching the source's cylinder-hull idiom."""
    points: list[Vector] = []
    for center, radius, height in cylinders:
        points.extend(
            cylinder_ring_points(
                center=Vector(center),
                radius=radius,
                height=height,
                vertices=vertices,
            )
        )
    obj = add_convex_hull_mesh(name, points, material)
    if material:
        assign_material(obj, material)
    return shade_smooth(obj)


def add_axis_cylinder_hull(
    name: str,
    cylinders: Sequence[tuple[tuple[float, float, float], float, float]],
    *,
    axis: str,
    vertices: int = 48,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Hull cylinders along an arbitrary axis for source-style swept cutters."""
    points: list[Vector] = []
    for center, radius, depth in cylinders:
        points.extend(
            cylinder_ring_points_axis(
                center=Vector(center),
                radius=radius,
                depth=depth,
                axis=axis,
                vertices=vertices,
            )
        )
    obj = add_convex_hull_mesh(name, points, material)
    if material:
        assign_material(obj, material)
    return shade_smooth(obj)


def add_vertical_conical_hull(
    name: str,
    cones: Sequence[tuple[tuple[float, float, float], float, float, float]],
    *,
    vertices: int = 40,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Hull vertical cones, preserving the source cam lever's tapered lower bevel."""
    points: list[Vector] = []
    for center, radius_bottom, radius_top, height in cones:
        points.extend(
            cone_ring_points(
                center=Vector(center),
                radius_bottom=radius_bottom,
                radius_top=radius_top,
                height=height,
                vertices=vertices,
            )
        )
    obj = add_convex_hull_mesh(name, points, material)
    if material:
        assign_material(obj, material)
    return shade_smooth(obj)


# ============================================================
# Procedural assembly
# ============================================================

"""Readable assembly logic for the quick-release t-slot clamp model."""






EXTRUSION_SIZE = 20.0
CLAMP_LEN = 30.0
JAW_LEN = 100.0
GROOVE_DEPTH = 2.25
PAD_RATIO = 0.5
PAD_LEN = JAW_LEN * PAD_RATIO
WALL_THICKNESS = GROOVE_DEPTH + 2.0
GROOVE_MIN = 5.15
GROOVE_MAX = 11.75
GROOVE_LIP = 1.55
CAM_DIA = min(EXTRUSION_SIZE * 0.8, CLAMP_LEN / 2.0)
CAM_RADIUS = CAM_DIA / 2.0
CAM_Z_OFFSET = CAM_DIA / 2.0
CAM_LOCK_OFFSET = 1.0
CLEARANCE = 0.25
EPSILON = 0.01
LEVER_LEN = CLAMP_LEN + EXTRUSION_SIZE / 4.0

ROOT_Y = 0.0
TIP_Y = JAW_LEN
PAD_START_Y = JAW_LEN - PAD_LEN
CAM_TRANSLATE = Vector((EXTRUSION_SIZE * 0.82, 0.0, -CLAMP_LEN / 2.0 + CAM_DIA / 2.0))


def build_jaw_boolean_body() -> bpy.types.Object:
    """Continuous source arm hull clipped only at the compact front throat."""
    jaw = make_compact_rectangular_source_arm_body(
        "continuous_source_arm_body_owned_grooves_and_teeth",
        extrusion_size=EXTRUSION_SIZE,
        clamp_len=CLAMP_LEN,
        jaw_len=JAW_LEN,
        pad_len=PAD_LEN,
        wall_thickness=WALL_THICKNESS,
        groove_depth=GROOVE_DEPTH,
        clearance=CLEARANCE,
        material=BODY_YELLOW,
        shadow_material=SHADOW_GREEN,
    )
    cutters: list[bpy.types.Object] = [
        add_axis_cylinder_hull(
            "arm_owned_three_lobe_cam_clearance",
            [
                ((0.0, CAM_Z_OFFSET, 0.0), CAM_RADIUS + CLEARANCE * 0.36, EXTRUSION_SIZE * 2.05),
                ((0.0, CAM_Z_OFFSET - CAM_LOCK_OFFSET, 0.0), CAM_RADIUS + CLEARANCE * 0.36, EXTRUSION_SIZE * 2.05),
                (
                    (0.0, CAM_Z_OFFSET - CAM_LOCK_OFFSET - CAM_DIA * 0.82, CAM_RADIUS * 0.68),
                    CAM_RADIUS + CLEARANCE * 0.36,
                    EXTRUSION_SIZE * 2.05,
                ),
            ],
            axis="X",
            vertices=144,
        ),
    ]
    boolean_difference(jaw, cutters, solver="FLOAT")
    clean_mesh(jaw)
    for polygon in jaw.data.polygons:
        center = polygon.center
        if center.y <= CLEARANCE + EXTRUSION_SIZE * 2.55:
            polygon.material_index = 0
        if (
            CLEARANCE + PAD_START_Y - 0.75 <= center.y <= CLEARANCE + JAW_LEN + 0.75
            and center.z < -CLAMP_LEN / 2.0 + 0.72
        ):
            polygon.material_index = 2 if abs(center.x) < GROOVE_DEPTH * 0.84 else 1
    add_bevel_modifier(jaw, 0.008, 1)
    return jaw


def build_arm_top_shell() -> bpy.types.Object:
    """Rectangular top volume unioned to the sloped arm body, with owned groove cuts."""
    raise RuntimeError("The current pass uses body-owned top cuts instead of a separate shell.")


def build_front_throat_profile() -> list[bpy.types.Object]:
    """Deprecated overlay hook; the unified groove/arm body now owns the front."""
    return []


def build_root_logo_cap() -> bpy.types.Object:
    """Small compact logo cap unioned into the jaw root before the final cam cut."""
    cap = add_box(
        "integrated_compact_rectangular_logo_root_cap",
        (GROOVE_MAX * 0.92, EXTRUSION_SIZE * 0.18, WALL_THICKNESS * 0.12),
        (0.0, EXTRUSION_SIZE * 0.18, CLAMP_LEN / 2.0 - WALL_THICKNESS * 0.06),
        BODY_YELLOW,
    )
    add_bevel_modifier(cap, 0.012, 1)
    return cap


def build_top_slot_floor_inserts() -> list[bpy.types.Object]:
    """Continuous source-like V faces for the two pad groove cuts."""
    return []


def build_underside_tooth_shadow_inserts() -> list[bpy.types.Object]:
    """The body mesh now owns the underside trough and transverse tooth relief."""
    return []


def build_side_underside_tooth_faces() -> list[bpy.types.Object]:
    """Side-visible serrations come from the body-owned underside mesh."""
    return []


def build_root_groove_clamp() -> list[bpy.types.Object]:
    """Literal groove_clamp() cross-section used as part of the unified jaw root."""
    depth = EXTRUSION_SIZE / 2.0 - GROOVE_LIP / 2.0
    groove = add_source_groove_clamp(
        "compact_source_groove_clamp_tongue",
        length=CLAMP_LEN,
        groove_max=GROOVE_MAX,
        groove_min=GROOVE_MIN,
        groove_lip=GROOVE_LIP,
        groove_depth=5.95,
        extrusion_size=EXTRUSION_SIZE,
        extension_depth=depth * 0.42,
        material=BODY_YELLOW,
        shadow_material=SHADOW_GREEN,
    )
    clean_mesh(groove)
    add_bevel_modifier(groove, 0.015, 1)
    throat_shadow = add_box(
        "actual_groove_clamp_front_throat_shadow",
        (GROOVE_MIN * 0.96, 0.060, CLAMP_LEN * 0.62),
        (0.0, -5.99, -CLAMP_LEN * 0.01),
        SHADOW_GREEN,
    )
    return [groove, throat_shadow]


def build_cam_rider() -> bpy.types.Object:
    rider = add_cylinder(
        "compact_source_cam_rider_cylinder",
        (CAM_RADIUS + CAM_LOCK_OFFSET * 0.70) * 2.0,
        GROOVE_MIN,
        (0.0, CAM_Z_OFFSET, 0.0),
        rotation=(0.0, math.pi / 2.0, 0.0),
        vertices=144,
        material=BODY_YELLOW,
    )
    add_bevel_modifier(rider, 0.012, 1)
    return rider


def build_integrated_jaw_root() -> list[bpy.types.Object]:
    """Union source arm, groove_clamp, rider, and cap before the shared cam bite."""
    groove_parts = build_root_groove_clamp()
    jaw_body = build_jaw_boolean_body()
    rider = build_cam_rider()
    cap = build_root_logo_cap()
    integrated = boolean_union(
        "single_connected_jaw_root_groove_arm_rider_body",
        [
            groove_parts[0],
            jaw_body,
            rider,
            cap,
        ],
    )
    final_cutters = [
        add_boolean_cylinder_cutter(
            "shared_outer_cam_shaft_clearance_cut",
            radius=CAM_RADIUS + CLEARANCE * 0.40,
            depth=EXTRUSION_SIZE * 2.12,
            location=(0.0, CAM_Z_OFFSET - CAM_LOCK_OFFSET * 1.5, 0.0),
            axis="X",
            vertices=160,
        ),
        add_box(
            "front_left_shoulder_relief_cut",
            (EXTRUSION_SIZE * 0.22, EXTRUSION_SIZE * 0.62, CLAMP_LEN * 1.08),
            (-EXTRUSION_SIZE * 0.42, EXTRUSION_SIZE * 0.22, -CLAMP_LEN * 0.02),
        ),
        add_box(
            "front_right_shoulder_relief_cut",
            (EXTRUSION_SIZE * 0.22, EXTRUSION_SIZE * 0.62, CLAMP_LEN * 1.08),
            (EXTRUSION_SIZE * 0.42, EXTRUSION_SIZE * 0.22, -CLAMP_LEN * 0.02),
        ),
        add_box(
            "front_center_tslot_relief_cut",
            (GROOVE_MIN * 1.28, EXTRUSION_SIZE * 0.70, CLAMP_LEN * 0.80),
            (0.0, EXTRUSION_SIZE * 0.16, -CLAMP_LEN * 0.03),
        ),
    ]
    boolean_difference(integrated, final_cutters, solver="FLOAT")
    clean_mesh(integrated)
    add_bevel_modifier(integrated, 0.006, 1)
    parts = [
        integrated,
        *build_front_throat_profile(),
        groove_parts[1],
    ]
    parts.extend(build_top_slot_floor_inserts())
    parts.extend(build_underside_tooth_shadow_inserts())
    parts.extend(build_side_underside_tooth_faces())
    for obj in parts:
        clean_mesh(obj)
    return parts


def build_logo_mark() -> list[bpy.types.Object]:
    """Small readable stand-in for the source's engraved root logo."""
    objects: list[bpy.types.Object] = []
    logo_z = 15.04
    logo_y = 5.0
    logo_x = 0.0
    objects.extend(
        make_rounded_rect_frame(
            "engraved_logo_outer_frame",
            width=5.9,
            height=4.25,
            stroke=0.36,
            thickness=0.055,
            location=(logo_x, logo_y, logo_z),
            material=SHADOW_GREEN,
        )
    )
    for offset in (-1.55, 0.0, 1.55):
        objects.append(
            add_capsule_prism(
                f"engraved_logo_arch_{offset:+.2f}",
                2.25,
                0.45,
                0.06,
                (logo_x + offset, logo_y + 0.25, logo_z + 0.01),
                "Y",
                SHADOW_GREEN,
                vertices=18,
            )
        )
        objects.append(
            add_box(
                f"engraved_logo_open_bottom_{offset:+.2f}",
                (0.68, 0.85, 0.16),
                (logo_x + offset, logo_y - 0.75, logo_z + 0.02),
                BODY_YELLOW,
            )
        )
    objects.append(add_box("engraved_logo_lower_bar", (5.1, 0.32, 0.055), (0.45, logo_y - 1.55, logo_z + 0.01), SHADOW_GREEN))
    objects.append(add_cylinder("engraved_logo_left_dot", 0.34, 0.055, (-2.0, logo_y - 1.12, logo_z + 0.01), vertices=18, material=SHADOW_GREEN))
    objects.append(add_cylinder("engraved_logo_right_dot", 0.34, 0.055, (1.0, logo_y - 1.12, logo_z + 0.01), vertices=18, material=SHADOW_GREEN))
    return objects


def build_nested_offset_cam_groove_cutter() -> bpy.types.Object:
    """SCAD cam_lever() offset-groove cutter: outer cylinder minus inner core/cone."""
    base = CAM_TRANSLATE + Vector((0.0, CAM_LOCK_OFFSET, EXTRUSION_SIZE / 2.0 - GROOVE_MIN / 2.0))
    cutter = add_cylinder(
        "nested_offset_groove_outer_cutter",
        CAM_DIA * 1.80,
        GROOVE_MIN + CAM_DIA + 1.10 * CLEARANCE,
        tuple(base + Vector((0.0, 0.0, GROOVE_MIN / 2.0 + CAM_DIA / 2.0 + CLEARANCE))),
        vertices=112,
    )
    inner_core = add_cylinder(
        "nested_offset_groove_inner_core_saved",
        (CAM_RADIUS - CAM_LOCK_OFFSET) * 2.0,
        EXTRUSION_SIZE,
        tuple(base),
        vertices=112,
    )
    flared_cone = add_cylinder(
        "nested_offset_groove_top_flare_saved",
        CAM_DIA,
        CAM_DIA,
        tuple(base + Vector((0.0, 0.0, CAM_DIA / 2.0 + GROOVE_MIN - 0.01))),
        vertices=112,
        radius_bottom=CAM_RADIUS - CAM_LOCK_OFFSET,
        radius_top=CAM_DIA * 1.5,
    )
    boolean_difference(cutter, [inner_core, flared_cone], solver="EXACT")
    clean_mesh(cutter)
    return cutter


def build_cam_lever() -> list[bpy.types.Object]:
    """Source cam_lever(): one flat hull, small stop nub, and source offset-groove cut."""
    shaft = add_cylinder(
        "source_scale_clean_cam_shaft",
        CAM_DIA * 0.925,
        EXTRUSION_SIZE * 0.90,
        tuple(CAM_TRANSLATE + Vector((0.0, 0.0, EXTRUSION_SIZE / 2.0))),
        vertices=160,
        material=BODY_YELLOW,
    )
    lever_block_z = CAM_TRANSLATE.z - CAM_DIA / 4.0 + EPSILON
    handle_center_z = lever_block_z + CAM_DIA / 8.0
    lever = make_flat_cam_lever_hull(
        "single_flat_source_cam_lever_hull",
        root_center=(CAM_TRANSLATE.x, CAM_TRANSLATE.y + 0.32, handle_center_z - 0.78),
        lever_len=LEVER_LEN * 0.91,
        cam_dia=CAM_DIA * 0.76,
        material=BODY_YELLOW,
        vertices=96,
    )
    nub = add_cylinder(
        "source_cam_lever_stop_nub",
        CAM_DIA * 0.255,
        CAM_DIA * 0.155,
        (CAM_TRANSLATE.x, CAM_TRANSLATE.y + LEVER_LEN * 0.89, CAM_TRANSLATE.z + CAM_DIA * 0.130),
        vertices=96,
        material=BODY_YELLOW,
    )
    cam = boolean_union("source_scale_cam_lever_union", [shaft, lever, nub])
    offset_groove = build_nested_offset_cam_groove_cutter()
    boolean_difference(cam, [offset_groove], solver="EXACT")
    clean_mesh(cam)
    add_bevel_modifier(cam, 0.018, 1)
    return [cam]


def build_model() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    objects.extend(build_integrated_jaw_root())
    objects.extend(build_logo_mark())
    objects.extend(build_cam_lever())
    for obj in objects:
        obj.select_set(False)
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

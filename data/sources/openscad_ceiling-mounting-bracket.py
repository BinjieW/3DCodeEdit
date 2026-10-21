from __future__ import annotations
import bmesh
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the ceiling bracket model."""




CUTTER_PREFIX = "boolean_cutter_"
SOURCE_YELLOW = (0.86, 0.73, 0.08, 1.0)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        if "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = color
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = color
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 1.5
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.55
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.35
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.35
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
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
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material is not None:
        assign_material(obj, material)
    return obj


def add_cylinder(
    name: str,
    radius: float,
    height: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 24,
    material: bpy.types.Material | None = None,
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
    if material is not None:
        assign_material(obj, material)
    return obj


def add_cylinder_x(
    name: str,
    radius: float,
    height: float,
    location: tuple[float, float, float],
    vertices: int = 24,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    return add_cylinder(
        name=name,
        radius=radius,
        height=height,
        location=location,
        rotation=(0.0, math.pi / 2.0, 0.0),
        vertices=vertices,
        material=material,
    )


def add_tube_x(
    name: str,
    outer_radius: float,
    inner_radius: float,
    height: float,
    location: tuple[float, float, float],
    vertices: int = 24,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    x_min = location[0] - height / 2.0
    x_max = location[0] + height / 2.0
    verts = []
    for x in (x_min, x_max):
        for radius in (outer_radius, inner_radius):
            for index in range(vertices):
                angle = 2.0 * math.pi * index / vertices
                verts.append(
                    (
                        x,
                        location[1] + radius * math.cos(angle),
                        location[2] + radius * math.sin(angle),
                    )
                )

    def idx(x_side: int, radius_side: int, index: int) -> int:
        return x_side * vertices * 2 + radius_side * vertices + index % vertices

    faces = []
    for index in range(vertices):
        next_index = (index + 1) % vertices
        faces.append((idx(0, 0, index), idx(0, 0, next_index), idx(1, 0, next_index), idx(1, 0, index)))
        faces.append((idx(0, 1, next_index), idx(0, 1, index), idx(1, 1, index), idx(1, 1, next_index)))
        faces.append((idx(1, 0, index), idx(1, 0, next_index), idx(1, 1, next_index), idx(1, 1, index)))
        faces.append((idx(0, 0, next_index), idx(0, 0, index), idx(0, 1, index), idx(0, 1, next_index)))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material is not None:
        assign_material(obj, material)
    return obj


def join_objects(name: str, objects: list[bpy.types.Object]) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    joined = bpy.context.object
    joined.name = name
    joined.data.name = f"{name}_mesh"
    recalculate_normals(joined)
    return joined


def recalculate_normals(obj: bpy.types.Object) -> bpy.types.Object:
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def apply_boolean(target: bpy.types.Object, cutter: bpy.types.Object, operation: str) -> bpy.types.Object:
    modifier = target.modifiers.new(name=f"{operation.lower()}_{cutter.name}", type="BOOLEAN")
    modifier.operation = operation
    modifier.object = cutter
    modifier.solver = "EXACT"
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    target.select_set(False)
    bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def union_object(target: bpy.types.Object, addition: bpy.types.Object) -> bpy.types.Object:
    return apply_boolean(target, addition, "UNION")


def subtract_box(
    target: bpy.types.Object,
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
) -> bpy.types.Object:
    cutter = add_box(f"{CUTTER_PREFIX}{name}", size=size, location=location)
    return apply_boolean(target, cutter, "DIFFERENCE")


def subtract_cylinder_x(
    target: bpy.types.Object,
    name: str,
    radius: float,
    height: float,
    location: tuple[float, float, float],
    vertices: int = 24,
) -> bpy.types.Object:
    cutter = add_cylinder_x(
        f"{CUTTER_PREFIX}{name}",
        radius=radius,
        height=height,
        location=location,
        vertices=vertices,
    )
    return apply_boolean(target, cutter, "DIFFERENCE")


def shade_flat(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_flat()
    obj.select_set(False)
    return obj


# ============================================================
# Procedural assembly
# ============================================================

"""Readable assembly of the ceiling mounting bracket."""




THING_WIDTH = 23.0
THING_HEIGHT = 5.0
FINGER_DEPTH = 3.0
STANDOFF_HEIGHT = 15.0
WALL_THICKNESS = 2.0
BRACKET_HEIGHT = 6.0
MOUNTING_SCREW_RADIUS = 2.5

OUTER_DEPTH = 2.0 * WALL_THICKNESS + THING_HEIGHT + max(0.0, STANDOFF_HEIGHT)
OUTER_WIDTH = THING_WIDTH + 2.0 * WALL_THICKNESS
MOUNTING_SCREW_HEIGHT = max(
    BRACKET_HEIGHT / 2.0,
    WALL_THICKNESS + MOUNTING_SCREW_RADIUS,
)


def centered_box_from_scad(
    name: str,
    corner: tuple[float, float, float],
    size: tuple[float, float, float],
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    location = (
        corner[0] + size[0] / 2.0,
        corner[1] + size[1] / 2.0,
        corner[2] + size[2] / 2.0,
    )
    return add_box(name=name, size=size, location=location, material=material)


def scad_box_center(
    corner: tuple[float, float, float],
    size: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        corner[0] + size[0] / 2.0,
        corner[1] + size[1] / 2.0,
        corner[2] + size[2] / 2.0,
    )


def build_bracket_body(material: bpy.types.Material) -> bpy.types.Object:
    """Build the source CSG body before the four OpenSCAD subtractions."""
    body = centered_box_from_scad(
        "outer_bracket_block",
        corner=(0.0, 0.0, 0.0),
        size=(OUTER_DEPTH, OUTER_WIDTH, BRACKET_HEIGHT),
        material=material,
    )
    boss = add_cylinder_x(
        "integrated_rear_screw_boss",
        radius=MOUNTING_SCREW_RADIUS + WALL_THICKNESS,
        height=WALL_THICKNESS,
        location=(WALL_THICKNESS / 2.0, OUTER_WIDTH / 2.0, MOUNTING_SCREW_HEIGHT),
        vertices=24,
        material=material,
    )
    body = union_object(body, boss)
    body.name = "ceiling_mounting_bracket_model"
    body.data.name = "ceiling_mounting_bracket_model_mesh"
    return body


def cut_mounting_clearances(body: bpy.types.Object) -> bpy.types.Object:
    """Apply the same subtractive cuts as the OpenSCAD source."""
    mounted_thing_slot_size = (THING_HEIGHT, THING_WIDTH, BRACKET_HEIGHT + 2.0)
    subtract_box(
        body,
        name="mounted_thing_slot",
        size=mounted_thing_slot_size,
        location=scad_box_center(
            corner=(
                WALL_THICKNESS + max(0.0, STANDOFF_HEIGHT),
                WALL_THICKNESS,
                -1.0,
            ),
            size=mounted_thing_slot_size,
        ),
    )
    core_size = (OUTER_DEPTH, THING_WIDTH - 2.0 * FINGER_DEPTH, BRACKET_HEIGHT + 2.0)
    subtract_box(
        body,
        name="bracket_core",
        size=core_size,
        location=scad_box_center(
            corner=(WALL_THICKNESS, WALL_THICKNESS + FINGER_DEPTH, -1.0),
            size=core_size,
        ),
    )
    subtract_cylinder_x(
        body,
        name="mounting_screw_hole",
        radius=MOUNTING_SCREW_RADIUS,
        height=WALL_THICKNESS + 2.0,
        location=(WALL_THICKNESS / 2.0, OUTER_WIDTH / 2.0, MOUNTING_SCREW_HEIGHT),
        vertices=24,
    )
    upper_relief_size = (
        max(0.0, STANDOFF_HEIGHT - WALL_THICKNESS),
        THING_WIDTH,
        BRACKET_HEIGHT,
    )
    subtract_box(
        body,
        name="upper_standoff_relief",
        size=upper_relief_size,
        location=scad_box_center(
            corner=(
                WALL_THICKNESS,
                WALL_THICKNESS,
                WALL_THICKNESS,
            ),
            size=upper_relief_size,
        ),
    )
    return body


def build_model() -> list:
    material = make_material("source_preview_yellow", SOURCE_YELLOW)
    bracket = build_bracket_body(material)
    cut_mounting_clearances(bracket)
    assign_material(bracket, material)
    shade_flat(bracket)
    return [bracket]


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

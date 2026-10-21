from __future__ import annotations
from dataclasses import dataclass
from mathutils import Matrix, Vector
import bmesh
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the couch cup holder model."""





@dataclass(frozen=True)
class CouchCupHolderSpec:
    """Default parameters from the source OpenSCAD customizer model."""

    cup_radius: float = 50.0
    cup_depth: float = 90.0
    wall_thickness: float = 5.0
    handle_depth: float = 105.0
    handle_thickness: float = 5.0
    clip_length: float = 30.0
    clip_angle_degrees: float = 5.0
    fn: int = 100

    @property
    def outer_cup_radius(self) -> float:
        return self.cup_radius + self.wall_thickness

    @property
    def outer_cup_depth(self) -> float:
        return self.cup_depth + self.wall_thickness

    @property
    def strap_width(self) -> float:
        return self.cup_radius / 2.0

    @property
    def strap_x_min(self) -> float:
        return -self.cup_radius / 4.0

    @property
    def strap_x_max(self) -> float:
        return self.cup_radius / 4.0

    @property
    def strap_y_max(self) -> float:
        return self.handle_depth + self.cup_radius + self.wall_thickness

    @property
    def strap_z_min(self) -> float:
        return self.cup_depth

    @property
    def strap_z_max(self) -> float:
        return self.cup_depth + self.handle_thickness

    @property
    def clip_rotation_x(self) -> float:
        return math.radians(-self.clip_angle_degrees - 180.0)


SOURCE_YELLOW = (0.94, 0.76, 0.04, 1.0)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.58,
) -> bpy.types.Material:
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.32
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.32
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = color
        elif "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = color
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 0.20
    return material


def source_materials() -> dict[str, bpy.types.Material]:
    return {
        "yellow": make_material("openscad_preview_yellow", SOURCE_YELLOW),
    }


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
    return obj


def recalculate_normals(obj: bpy.types.Object) -> bpy.types.Object:
    if obj.type != "MESH":
        return obj
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def add_cylinder_z(
    name: str,
    radius: float,
    height: float,
    z_min: float = 0.0,
    vertices: int = 100,
    material: bpy.types.Material | None = None,
    end_fill_type: str = "NGON",
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=height,
        end_fill_type=end_fill_type,
        location=(0.0, 0.0, z_min + height / 2.0),
    )
    obj = bpy.context.object
    obj.name = name
    if material is not None:
        assign_material(obj, material)
    return obj


def add_box_from_min_corner(
    name: str,
    min_corner: tuple[float, float, float],
    size: tuple[float, float, float],
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    location = (
        min_corner[0] + size[0] / 2.0,
        min_corner[1] + size[1] / 2.0,
        min_corner[2] + size[2] / 2.0,
    )
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material is not None:
        assign_material(obj, material)
    return obj


def _box_vertices_from_min_corner(
    min_corner: Vector,
    size: tuple[float, float, float],
) -> list[Vector]:
    sx, sy, sz = size
    return [
        min_corner + Vector((0.0, 0.0, 0.0)),
        min_corner + Vector((sx, 0.0, 0.0)),
        min_corner + Vector((sx, sy, 0.0)),
        min_corner + Vector((0.0, sy, 0.0)),
        min_corner + Vector((0.0, 0.0, sz)),
        min_corner + Vector((sx, 0.0, sz)),
        min_corner + Vector((sx, sy, sz)),
        min_corner + Vector((0.0, sy, sz)),
    ]


def add_scad_rotated_box_from_min_corner(
    name: str,
    min_corner: tuple[float, float, float],
    size: tuple[float, float, float],
    rotation_x: float,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Build a cube whose points are rotated around its OpenSCAD min corner."""
    origin = Vector(min_corner)
    rotation = Matrix.Rotation(rotation_x, 4, "X")
    vertices = [origin + rotation @ (vertex - origin) for vertex in _box_vertices_from_min_corner(origin, size)]
    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata([tuple(vertex) for vertex in vertices], [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material is not None:
        assign_material(obj, material)
    recalculate_normals(obj)
    return obj


def apply_boolean(
    target: bpy.types.Object,
    tool: bpy.types.Object,
    operation: str,
    remove_tool: bool = True,
) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    modifier = target.modifiers.new(name=f"{operation.lower()}_{tool.name}", type="BOOLEAN")
    modifier.operation = operation
    modifier.object = tool
    if hasattr(modifier, "solver"):
        modifier.solver = "EXACT"
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    target.select_set(False)
    if remove_tool:
        bpy.data.objects.remove(tool, do_unlink=True)
    recalculate_normals(target)
    return target


def boolean_difference(target: bpy.types.Object, cutter: bpy.types.Object) -> bpy.types.Object:
    return apply_boolean(target, cutter, "DIFFERENCE")


def boolean_union(target: bpy.types.Object, addition: bpy.types.Object) -> bpy.types.Object:
    return apply_boolean(target, addition, "UNION")


# ============================================================
# Procedural assembly
# ============================================================

"""Assembly logic for the couch cup holder model."""





def add_inner_cup_cutter(spec: CouchCupHolderSpec, name: str) -> bpy.types.Object:
    """Match the source inner_cup() module used to cut both cup and strap."""
    return add_cylinder_z(
        name=name,
        radius=spec.cup_radius,
        height=spec.cup_depth + 1.0,
        z_min=spec.wall_thickness,
        vertices=spec.fn,
    )


def build_cup_holder(
    spec: CouchCupHolderSpec,
    material: bpy.types.Material,
) -> bpy.types.Object:
    outer_cup = add_cylinder_z(
        name="outer_cup_shell",
        radius=spec.outer_cup_radius,
        height=spec.outer_cup_depth,
        z_min=0.0,
        vertices=spec.fn,
        material=material,
    )
    inner_cup = add_inner_cup_cutter(spec, "inner_cup_boolean_cutter")
    return boolean_difference(outer_cup, inner_cup)


def build_handle(
    spec: CouchCupHolderSpec,
    material: bpy.types.Material,
) -> bpy.types.Object:
    strap = add_box_from_min_corner(
        name="straight_top_handle",
        min_corner=(spec.strap_x_min, 0.0, spec.strap_z_min),
        size=(spec.strap_width, spec.strap_y_max, spec.handle_thickness),
        material=material,
    )
    inner_cup = add_inner_cup_cutter(spec, "handle_inner_cup_boolean_cutter")
    return boolean_difference(strap, inner_cup)


def build_rear_clip(
    spec: CouchCupHolderSpec,
    material: bpy.types.Material,
) -> bpy.types.Object:
    return add_scad_rotated_box_from_min_corner(
        name="rear_downturned_clip",
        min_corner=(spec.strap_x_min, spec.strap_y_max, spec.cup_depth),
        size=(spec.strap_width, spec.handle_thickness, spec.clip_length),
        rotation_x=spec.clip_rotation_x,
        material=material,
    )


def build_model() -> list[bpy.types.Object]:
    """Recreate the source union of cup_holder(), handle(), and clip()."""
    spec = CouchCupHolderSpec()
    materials = source_materials()
    yellow = materials["yellow"]

    cup = build_cup_holder(spec, yellow)
    handle = build_handle(spec, yellow)
    clip = build_rear_clip(spec, yellow)

    model = boolean_union(cup, handle)
    model = boolean_union(model, clip)
    model.name = "couch_cup_holder"
    model.data.name = "couch_cup_holder_mesh"
    return [model]


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

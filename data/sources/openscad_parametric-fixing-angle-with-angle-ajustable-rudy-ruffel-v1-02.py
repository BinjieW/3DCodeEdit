from __future__ import annotations
from dataclasses import dataclass
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the fixing-angle model."""





def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.55,
) -> bpy.types.Material:
    """Create a simple material suitable for geometry review renders."""
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
    return material


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
    if material:
        obj.data.materials.append(material)
    return obj


def add_box_from_min_corner(
    name: str,
    min_corner: tuple[float, float, float],
    size: tuple[float, float, float],
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Add an OpenSCAD-style cube whose local coordinates are baked in world space."""
    location = (
        min_corner[0] + size[0] / 2.0,
        min_corner[1] + size[1] / 2.0,
        min_corner[2] + size[2] / 2.0,
    )
    return add_box(name=name, size=size, location=location, material=material)


def add_cylinder(
    name: str,
    radius: float,
    height: float,
    location: tuple[float, float, float],
    axis: str = "Z",
    vertices: int = 96,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    rotation = {
        "X": (0.0, math.radians(90.0), 0.0),
        "Y": (math.radians(90.0), 0.0, 0.0),
        "Z": (0.0, 0.0, 0.0),
    }[axis.upper()]
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=height,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if material:
        obj.data.materials.append(material)
    return obj


def apply_boolean_difference(target: bpy.types.Object, cutters: list[bpy.types.Object]) -> None:
    """Subtract cutters from target and remove the temporary cutter objects."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = target
    target.select_set(True)

    for cutter in cutters:
        modifier = target.modifiers.new(name=f"cut_{cutter.name}", type="BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        modifier.solver = "EXACT"
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter, do_unlink=True)

    target.select_set(False)


def rotate_mesh_about_origin(obj: bpy.types.Object, y_degrees: float) -> None:
    """Bake a world-origin Y rotation into an object with baked local coordinates."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.rotation_euler[1] = math.radians(y_degrees)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    obj.select_set(False)


def join_meshes(name: str, objects: list[bpy.types.Object]) -> bpy.types.Object:
    """Join visible component meshes into one object while preserving baked geometry."""
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    joined = bpy.context.object
    joined.name = name
    joined.data.name = f"{name}_mesh"
    return joined


# ============================================================
# Procedural assembly
# ============================================================

"""Readable composition for the parametric OpenSCAD fixing angle."""





@dataclass(frozen=True)
class FixingAngleParameters:
    """Default source parameters from the selected OpenSCAD model."""

    width: float = 15.0
    height: float = 10.0
    angle_degrees: float = 60.0
    screw_rows: int = 1
    screw_columns: int = 1
    screw_diameter: float = 3.0
    screw_head_diameter: float = 6.0
    screw_head_depth: float = 3.0
    wall_thickness_override: float = 0.0
    screw_resolution: int = 100
    head_resolution: int = 100

    @property
    def wall_thickness(self) -> float:
        return self.width / 3.0 if self.wall_thickness_override == 0 else self.wall_thickness_override

    @property
    def screw_radius(self) -> float:
        return self.screw_diameter / 2.0

    @property
    def screw_head_radius(self) -> float:
        return self.screw_head_diameter / 2.0

    @property
    def screw_x_or_z_center(self) -> float:
        return self.wall_thickness + ((self.width - self.wall_thickness) / 2.0)

    @property
    def screw_y_center(self) -> float:
        return -self.height / 2.0


def _horizontal_plate(params: FixingAngleParameters, material) -> object:
    """Build Base1() from the SCAD, including its through-hole and head recess."""
    cutter_length = params.width * 2.0
    plate = add_box_from_min_corner(
        name="horizontal_foot_plate",
        min_corner=(0.0, -params.height, 0.0),
        size=(params.width, params.height, params.wall_thickness),
        material=material,
    )

    through_hole = add_cylinder(
        name="horizontal_screw_through_hole",
        radius=params.screw_radius,
        height=cutter_length,
        location=(params.screw_x_or_z_center, params.screw_y_center, 0.0),
        axis="Z",
        vertices=params.screw_resolution,
    )
    head_start_z = params.wall_thickness - params.screw_head_depth
    head_recess = add_cylinder(
        name="horizontal_head_counterbore",
        radius=params.screw_head_radius,
        height=cutter_length,
        location=(
            params.screw_x_or_z_center,
            params.screw_y_center,
            head_start_z + cutter_length / 2.0,
        ),
        axis="Z",
        vertices=params.head_resolution,
    )
    apply_boolean_difference(plate, [through_hole, head_recess])
    return plate


def _upright_plate_before_rotation(params: FixingAngleParameters, material) -> object:
    """Build Base2() and Cut() in source coordinates before the SCAD world rotation."""
    cutter_length = params.width * 2.0
    plate = add_box_from_min_corner(
        name="upright_plate_unrotated",
        min_corner=(0.0, -params.height, 0.0),
        size=(params.wall_thickness, params.height, params.width),
        material=material,
    )

    through_hole = add_cylinder(
        name="upright_screw_through_hole",
        radius=params.screw_radius,
        height=cutter_length,
        location=(0.0, params.screw_y_center, params.screw_x_or_z_center),
        axis="X",
        vertices=params.screw_resolution,
    )
    head_start_x = params.width * 2.0 + params.wall_thickness - params.screw_head_depth
    head_recess = add_cylinder(
        name="upright_head_counterbore",
        radius=params.screw_head_radius,
        height=cutter_length,
        location=(
            head_start_x - cutter_length / 2.0,
            params.screw_y_center,
            params.screw_x_or_z_center,
        ),
        axis="X",
        vertices=params.head_resolution,
    )
    apply_boolean_difference(plate, [through_hole, head_recess])
    return plate


def build_model() -> list:
    """Construct the complete two-plate fixing angle with recessed screw openings."""
    params = FixingAngleParameters()
    yellow_plastic = make_material("source_yellow_plastic", (1.0, 0.78, 0.02, 1.0))

    horizontal_plate = _horizontal_plate(params, yellow_plastic)
    upright_plate = _upright_plate_before_rotation(params, yellow_plastic)
    rotate_mesh_about_origin(upright_plate, y_degrees=params.angle_degrees - 90.0)

    bracket = join_meshes("parametric_fixing_angle", [horizontal_plate, upright_plate])
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

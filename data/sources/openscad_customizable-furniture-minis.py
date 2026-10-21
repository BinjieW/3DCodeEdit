from __future__ import annotations
import bpy


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the furniture-minis loveseat."""



SOURCE_SCALE = 25.4 / 50.0
SOFT_RADIUS_SOURCE = 3.0 * SOURCE_SCALE
SHARP_RADIUS_SOURCE = 0.2


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    existing = bpy.data.materials.get(name)
    if existing:
        return existing

    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.58
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material | None) -> bpy.types.Object:
    if material is None:
        return obj
    obj.data.materials.clear()
    obj.data.materials.append(material)
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
    return assign_material(obj, material)


def _apply_modifier(obj: bpy.types.Object, modifier_name: str) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier_name)


def add_rounded_cuboid(
    name: str,
    size: tuple[float, float, float],
    origin: tuple[float, float, float],
    radius: float,
    material: bpy.types.Material | None = None,
    bevel_segments: int = 4,
) -> bpy.types.Object:
    """Create a bounded rounded box matching the OpenSCAD cuboid hull helper."""
    location = (
        origin[0] + size[0] / 2.0,
        origin[1] + size[1] / 2.0,
        origin[2] + size[2] / 2.0,
    )
    obj = add_box(name=name, size=size, location=location, material=material)

    max_radius = max(min(size) / 2.0 - 1e-6, 0.0)
    bevel = obj.modifiers.new(name="scad_sphere_hull_rounding", type="BEVEL")
    bevel.width = min(radius, max_radius)
    bevel.segments = bevel_segments
    bevel.profile = 0.5
    if hasattr(bevel, "affect"):
        bevel.affect = "EDGES"
    _apply_modifier(obj, bevel.name)

    bpy.ops.object.shade_flat()
    obj.data.update()
    return obj


def add_scad_rounded_cuboid(
    name: str,
    depth: float,
    length: float,
    height: float,
    radius: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Build `cuboid(depth,length,height,r)` after the enclosing `scale(s)`."""
    scaled_size = (
        depth * SOURCE_SCALE,
        length * SOURCE_SCALE,
        height * SOURCE_SCALE,
    )
    scaled_origin = (
        origin[0] * SOURCE_SCALE,
        origin[1] * SOURCE_SCALE,
        origin[2] * SOURCE_SCALE,
    )
    return add_rounded_cuboid(
        name=name,
        size=scaled_size,
        origin=scaled_origin,
        radius=radius * SOURCE_SCALE,
        material=material,
    )


def add_cylinder(
    name: str,
    diameter: float,
    height: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=diameter / 2.0, depth=height, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    return assign_material(obj, material)


# ============================================================
# Procedural assembly
# ============================================================

"""Assembly logic for the stock small loveseat selection."""





SOFA_DEPTH = 36.0
SOFA_LENGTH = 54.0
SOFA_HEIGHT = 34.0


def build_sofa(
    depth: float = SOFA_DEPTH,
    length: float = SOFA_LENGTH,
    height: float = SOFA_HEIGHT,
) -> list[bpy.types.Object]:
    """Recreate `sofa(depth=36,length=54,height=34)` from the source SCAD."""
    material = make_material(
        "openscad_preview_yellow",
        (1.0, 0.78, 0.02, 1.0),
    )

    back_depth = depth / 4.0
    arm_width = depth / 4.0

    parts = [
        add_scad_rounded_cuboid(
            name="sofa_back_full_height_depth_quarter",
            depth=back_depth,
            length=length,
            height=height,
            radius=SOFT_RADIUS_SOURCE,
            material=material,
        ),
        add_scad_rounded_cuboid(
            name="sofa_left_arm_lower",
            depth=depth,
            length=arm_width,
            height=height * 0.6,
            radius=SOFT_RADIUS_SOURCE,
            material=material,
        ),
        add_scad_rounded_cuboid(
            name="sofa_right_arm_lower",
            depth=depth,
            length=arm_width,
            height=height * 0.6,
            radius=SOFT_RADIUS_SOURCE,
            origin=(0.0, length - arm_width, 0.0),
            material=material,
        ),
        add_scad_rounded_cuboid(
            name="sofa_full_width_seat_cushion",
            depth=depth,
            length=length,
            height=height * 0.4,
            radius=SOFT_RADIUS_SOURCE,
            material=material,
        ),
    ]

    return parts


def build_model() -> list[bpy.types.Object]:
    return build_sofa()


# ============================================================
# Scene setup and rendering
# ============================================================

"""Minimal render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

from __future__ import annotations
import bpy


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the Greek temple model."""



SOURCE_GOLD = (1.0, 0.76, 0.04, 1.0)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.5,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material | None) -> bpy.types.Object:
    if material is not None:
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


def add_box_from_min_corner(
    name: str,
    min_corner: tuple[float, float, float],
    size: tuple[float, float, float],
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Match OpenSCAD's translated/scaled cube(1), which starts at the origin."""
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
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
    material: bpy.types.Material | None = None,
    smooth: bool = True,
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
    assign_material(obj, material)
    if smooth:
        shade_smooth(obj)
    return obj


def add_cylinder_from_bottom(
    name: str,
    radius: float,
    height: float,
    bottom_center: tuple[float, float, float],
    vertices: int = 64,
    material: bpy.types.Material | None = None,
    smooth: bool = True,
) -> bpy.types.Object:
    location = (bottom_center[0], bottom_center[1], bottom_center[2] + height / 2.0)
    return add_cylinder(
        name=name,
        radius=radius,
        height=height,
        location=location,
        vertices=vertices,
        material=material,
        smooth=smooth,
    )


def add_scaled_sphere(
    name: str,
    radius: float,
    scale: tuple[float, float, float],
    location: tuple[float, float, float],
    material: bpy.types.Material | None = None,
    segments: int = 64,
    ring_count: int = 24,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=ring_count,
        radius=radius,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign_material(obj, material)
    shade_smooth(obj)
    return obj


def add_open_scad_top_cap(
    name: str,
    center: tuple[float, float, float],
    radius_xy: float,
    cylinder_height: float,
    sphere_z_radius: float,
    material: bpy.types.Material | None = None,
) -> list[bpy.types.Object]:
    """Build the source PTopBot top cap for avoid_extreme_overhangs=-1."""
    cap_sphere = add_scaled_sphere(
        f"{name}_flattened_spherical_cap",
        radius=1.0,
        scale=(radius_xy, radius_xy, sphere_z_radius),
        location=center,
        material=material,
    )
    cap_cylinder = add_cylinder_from_bottom(
        f"{name}_short_round_cap_band",
        radius=radius_xy,
        height=cylinder_height,
        bottom_center=(center[0], center[1], center[2]),
        material=material,
        smooth=True,
    )
    return [cap_sphere, cap_cylinder]


def shade_smooth(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)
    return obj


# ============================================================
# Procedural assembly
# ============================================================

"""Procedural assembly for the customizable Greek temple model."""




PILLARS_PER_ROW = 7
FLOOR_WIDTH = 12.0
FLOOR_LENGTH = 24.0
PILLAR_HEIGHT = 8.0
PILLAR_WIDTH = 1.2
PILLAR_RADIUS = PILLAR_WIDTH / 2.0
PILLAR_ROW_X = (2.0, 8.0)
PILLAR_START_Y = 2.0
PILLAR_BOTTOM_Z = 1.0
PILLAR_DISTANCE = (18.0 - 2.0 * (PILLAR_WIDTH - 1.2)) / (PILLARS_PER_ROW - 1)
CAP_RADIUS = PILLAR_WIDTH / 12.0 * 10.0
CAP_SPHERE_Z_RADIUS = 0.075 * 10.0
CAP_CYLINDER_HEIGHT = 0.5


def build_base(material: bpy.types.Material) -> list[bpy.types.Object]:
    """Recreate the three overlapping translated/scaled cube tiers from the SCAD."""
    return [
        add_box_from_min_corner(
            "lowest_widest_base_slab",
            min_corner=(-2.0 + FLOOR_WIDTH / 12.0, -2.0 + FLOOR_LENGTH / 24.0, 0.0),
            size=(FLOOR_WIDTH, FLOOR_LENGTH, 0.5),
            material=material,
        ),
        add_box_from_min_corner(
            "middle_stepped_base_slab",
            min_corner=(
                -1.5 / (11.0 / 12.0) + FLOOR_WIDTH / 11.0,
                -1.5 / (23.0 / 24.0) + FLOOR_LENGTH / 23.0,
                0.0,
            ),
            size=(FLOOR_WIDTH * 11.0 / 12.0, FLOOR_LENGTH * 11.5 / 12.0, 1.0),
            material=material,
        ),
        add_box_from_min_corner(
            "upper_temple_floor_slab",
            min_corner=(
                -1.0 / (10.0 / 12.0) + FLOOR_WIDTH / 10.0,
                -1.0 / (22.0 / 24.0) + FLOOR_LENGTH / 22.0,
                0.0,
            ),
            size=(FLOOR_WIDTH * 10.0 / 12.0, FLOOR_LENGTH * 11.0 / 12.0, 1.5),
            material=material,
        ),
    ]


def pillar_y_positions() -> list[float]:
    return [PILLAR_START_Y + PILLAR_DISTANCE * index for index in range(PILLARS_PER_ROW)]


def build_pillar(name: str, x: float, y: float, material: bpy.types.Material) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    objects.append(
        add_cylinder_from_bottom(
            f"{name}_plain_cylindrical_shaft",
            radius=PILLAR_RADIUS,
            height=PILLAR_HEIGHT,
            bottom_center=(x, y, PILLAR_BOTTOM_Z),
            vertices=72,
            material=material,
            smooth=True,
        )
    )
    top_center_z = PILLAR_BOTTOM_Z + PILLAR_HEIGHT - 0.25
    objects.extend(
        add_open_scad_top_cap(
            f"{name}_source_top_cap",
            center=(x, y, top_center_z),
            radius_xy=CAP_RADIUS,
            cylinder_height=CAP_CYLINDER_HEIGHT,
            sphere_z_radius=CAP_SPHERE_Z_RADIUS,
            material=material,
        )
    )
    return objects


def build_pillars(material: bpy.types.Material) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    for row_index, x in enumerate(PILLAR_ROW_X):
        row_name = "left" if row_index == 0 else "right"
        for pillar_index, y in enumerate(pillar_y_positions()):
            objects.extend(build_pillar(f"{row_name}_row_pillar_{pillar_index + 1:02d}", x, y, material))
    return objects


def build_roof(material: bpy.types.Material) -> list[bpy.types.Object]:
    bottom_z = PILLAR_HEIGHT + 1.25
    return [
        add_box_from_min_corner(
            "lowest_widest_roof_slab",
            min_corner=(1.0, 1.0, bottom_z),
            size=(8.0, 20.0, 0.5),
            material=material,
        ),
        add_box_from_min_corner(
            "middle_inset_roof_slab",
            min_corner=(1.5, 1.0, bottom_z + 0.5),
            size=(7.0, 20.0, 0.5),
            material=material,
        ),
        add_box_from_min_corner(
            "top_narrow_roof_slab",
            min_corner=(2.0, 1.0, bottom_z + 1.0),
            size=(6.0, 20.0, 0.5),
            material=material,
        ),
    ]


def build_model() -> list:
    """Build the source temple from readable source-derived dimensions."""
    gold = make_material("openscad_default_gold", SOURCE_GOLD)
    objects: list[bpy.types.Object] = []
    objects.extend(build_base(gold))
    objects.extend(build_pillars(gold))
    objects.extend(build_roof(gold))
    return objects


# ============================================================
# Scene setup and rendering
# ============================================================

"""Minimal render entrypoint for current_version."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

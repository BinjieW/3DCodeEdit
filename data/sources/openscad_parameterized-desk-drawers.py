from __future__ import annotations
from dataclasses import dataclass
import bpy


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the desk drawer model."""




Color = tuple[float, float, float, float]


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(
    name: str,
    color: Color,
    *,
    roughness: float = 0.55,
) -> bpy.types.Material:
    """Create a simple render material, including alpha when requested."""
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = color
    material.use_nodes = True

    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Alpha"].default_value = color[3]
        bsdf.inputs["Roughness"].default_value = roughness

    if color[3] < 1.0:
        material.blend_method = "BLEND"
        material.use_screen_refraction = True
        material.show_transparent_back = True
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


def add_scad_cube(
    name: str,
    size: tuple[float, float, float],
    corner: tuple[float, float, float],
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Add a cube using OpenSCAD's min-corner placement convention."""
    location = (
        corner[0] + size[0] / 2.0,
        corner[1] + size[1] / 2.0,
        corner[2] + size[2] / 2.0,
    )
    return add_box(name=name, size=size, location=location, material=material)


def add_x_extruded_yz_profile(
    name: str,
    x_min: float,
    x_width: float,
    yz_profile: list[tuple[float, float]],
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Create a prism by extruding a Y/Z profile along X."""
    if len(yz_profile) < 3:
        raise ValueError("An extruded profile needs at least three points")

    x_max = x_min + x_width
    vertices = [(x_min, y, z) for y, z in yz_profile]
    vertices.extend((x_max, y, z) for y, z in yz_profile)

    count = len(yz_profile)
    faces: list[tuple[int, ...]] = [
        tuple(reversed(range(count))),
        tuple(range(count, count * 2)),
    ]
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, count + next_index, count + index))

    mesh = bpy.data.meshes.new(name=f"{name}Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return assign_material(obj, material)


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
    return assign_material(obj, material)


# ============================================================
# Procedural assembly
# ============================================================

"""Assembly logic for the parameterized desk drawers model."""






@dataclass(frozen=True)
class DeskDrawerParams:
    drawer_width_inner: float = 170.0
    drawer_height_inner: float = 25.0
    drawer_depth_inner: float = 90.0
    drawer_wall: float = 2.0
    drawer_count: int = 4
    casing_wall: float = 3.8
    print_tolerance: float = 0.5

    @property
    def casing_height(self) -> float:
        return self.drawer_height_inner + self.drawer_wall + self.casing_wall + self.print_tolerance

    @property
    def casing_width(self) -> float:
        return self.drawer_width_inner + self.casing_wall * 3.0 + self.print_tolerance * 2.0

    @property
    def casing_depth(self) -> float:
        return self.drawer_depth_inner + self.casing_wall + self.print_tolerance

    @property
    def total_depth(self) -> float:
        return self.casing_depth + self.casing_wall

    @property
    def total_height(self) -> float:
        return self.casing_wall + self.casing_height * self.drawer_count + self.casing_wall

    @property
    def drawer_outer_width(self) -> float:
        return self.drawer_width_inner + self.drawer_wall * 2.0

    @property
    def drawer_outer_depth(self) -> float:
        return self.drawer_depth_inner + self.drawer_wall * 2.0

    @property
    def drawer_outer_height(self) -> float:
        return self.drawer_height_inner + self.drawer_wall


def _materials() -> dict[str, bpy.types.Material]:
    return {
        "casing": make_material("opaque warm yellow casing", (1.0, 0.76, 0.02, 1.0)),
        "drawer": make_material("transparent grey drawer preview", (0.72, 0.72, 0.66, 0.38)),
        "pull": make_material("slightly darker transparent drawer pulls", (0.58, 0.58, 0.54, 0.48)),
    }


def _bay_z(params: DeskDrawerParams, bay_index: int) -> float:
    return params.casing_wall + params.casing_height * bay_index


def _add_casing_bay(
    params: DeskDrawerParams,
    materials: dict[str, bpy.types.Material],
    bay_index: int,
) -> list[bpy.types.Object]:
    """Rebuild the OpenSCAD casing() module at one stacked Z offset."""
    ct = params.casing_wall
    dt = params.drawer_wall
    pt = params.print_tolerance
    dw = params.drawer_width_inner
    dh = params.drawer_height_inner
    dd = params.drawer_depth_inner
    z = _bay_z(params, bay_index)
    casing_material = materials["casing"]

    objects: list[bpy.types.Object] = []

    objects.append(
        add_scad_cube(
            f"bay_{bay_index + 1}_left_side_wall",
            (ct, params.casing_depth, params.casing_height),
            (0.0, 0.0, z),
            casing_material,
        )
    )
    objects.append(
        add_scad_cube(
            f"bay_{bay_index + 1}_left_inner_top_slide",
            (ct, dd + ct - dt, ct - pt),
            (ct, dt + pt, z + dh + dt + pt * 2.0),
            casing_material,
        )
    )
    objects.append(
        add_scad_cube(
            f"bay_{bay_index + 1}_left_rear_return",
            (ct * 2.0 - pt, ct, params.casing_height),
            (0.0, params.casing_depth, z),
            casing_material,
        )
    )

    right_x = dw + ct * 2.0 + pt * 2.0
    objects.append(
        add_scad_cube(
            f"bay_{bay_index + 1}_right_side_wall",
            (ct, params.casing_depth, params.casing_height),
            (right_x, 0.0, z),
            casing_material,
        )
    )
    objects.append(
        add_scad_cube(
            f"bay_{bay_index + 1}_right_inner_top_slide",
            (ct, dd + ct - dt, ct - pt),
            (right_x - ct, dt + pt, z + dh + dt + pt * 2.0),
            casing_material,
        )
    )
    objects.append(
        add_scad_cube(
            f"bay_{bay_index + 1}_right_rear_return",
            (ct * 2.0 - pt, ct, params.casing_height),
            (right_x - ct + pt, params.casing_depth, z),
            casing_material,
        )
    )

    back_z = z + dh - dt - pt - (params.casing_height - ct) / 2.0
    objects.append(
        add_scad_cube(
            f"bay_{bay_index + 1}_rear_stabilizer_bar",
            (dw - ct + pt * 4.0, ct, ct * 2.0),
            (ct * 2.0 - pt, params.casing_depth, back_z),
            casing_material,
        )
    )
    return objects


def _add_drawer_preview(
    params: DeskDrawerParams,
    materials: dict[str, bpy.types.Material],
    bay_index: int,
) -> list[bpy.types.Object]:
    """Rebuild the previewed %drawer() module as translucent panels."""
    dt = params.drawer_wall
    ct = params.casing_wall
    pt = params.print_tolerance
    w = params.drawer_outer_width
    d = params.drawer_outer_depth
    h = params.drawer_outer_height

    origin_x = ct + pt
    origin_y = 0.0
    origin_z = _bay_z(params, bay_index)
    drawer_material = materials["drawer"]
    pull_material = materials["pull"]

    objects: list[bpy.types.Object] = []
    objects.append(
        add_scad_cube(
            f"bay_{bay_index + 1}_drawer_bottom_panel",
            (w, d, dt),
            (origin_x, origin_y, origin_z),
            drawer_material,
        )
    )
    objects.append(
        add_scad_cube(
            f"bay_{bay_index + 1}_drawer_left_wall",
            (dt, d, h),
            (origin_x, origin_y, origin_z),
            drawer_material,
        )
    )
    objects.append(
        add_scad_cube(
            f"bay_{bay_index + 1}_drawer_right_wall",
            (dt, d, h),
            (origin_x + w - dt, origin_y, origin_z),
            drawer_material,
        )
    )
    objects.append(
        add_scad_cube(
            f"bay_{bay_index + 1}_drawer_back_wall",
            (params.drawer_width_inner, dt, h),
            (origin_x + dt, origin_y + d - dt, origin_z),
            drawer_material,
        )
    )
    objects.append(
        add_scad_cube(
            f"bay_{bay_index + 1}_drawer_front_panel",
            (w, dt, h + ct),
            (origin_x, origin_y, origin_z),
            drawer_material,
        )
    )

    handle_width = w / 32.0
    handle_height = (h + ct) / 2.0
    diagonal = handle_height * 0.7071067811865476
    cut_y = -handle_height * 0.5
    x_min = origin_x + (w - handle_width) / 2.0
    z_offset = origin_z + h / 6.0

    # Matches the rotated cube pull in source_model.scad, with the lower-front
    # corner clipped by the translated subtraction cube.
    pull_profile = [
        (origin_y + 0.0, z_offset + 0.0),
        (origin_y + diagonal, z_offset + diagonal),
        (origin_y + 0.0, z_offset + diagonal * 2.0),
        (origin_y + cut_y, z_offset + diagonal * 2.0 + cut_y),
        (origin_y + cut_y, z_offset - cut_y),
    ]
    objects.append(
        add_x_extruded_yz_profile(
            f"bay_{bay_index + 1}_center_angled_pull",
            x_min,
            handle_width,
            pull_profile,
            pull_material,
        )
    )
    return objects


def build_model() -> list:
    """Build the selected casing print with OpenSCAD-matching dimensions."""
    params = DeskDrawerParams()
    materials = _materials()
    casing_material = materials["casing"]

    objects: list[bpy.types.Object] = []
    objects.append(
        add_scad_cube(
            "bottom_full_width_slab",
            (params.casing_width, params.total_depth, params.casing_wall),
            (0.0, 0.0, 0.0),
            casing_material,
        )
    )
    objects.append(
        add_scad_cube(
            "top_full_width_slab",
            (params.casing_width, params.total_depth, params.casing_wall),
            (0.0, 0.0, params.casing_wall + params.casing_height * params.drawer_count),
            casing_material,
        )
    )

    for bay_index in range(params.drawer_count):
        objects.extend(_add_drawer_preview(params, materials, bay_index))
        objects.extend(_add_casing_bay(params, materials, bay_index))

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

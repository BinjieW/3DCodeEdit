from __future__ import annotations
from dataclasses import dataclass
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the modular building facade."""





Color = tuple[float, float, float, float]


@dataclass(frozen=True)
class FacadeParams:
    columns: int = 3
    rows: int = 2
    x_init: float = 5.0
    y_init: float = 6.0
    x_space: float = 10.0
    y_space: float = 15.5
    wall_depth: float = 2.0
    window_width: float = 6.4
    window_height: float = 10.0
    window_cut_depth: float = 4.0
    muntin_width: float = 0.7
    muntin_depth: float = 0.6
    arch_ratio: float = 0.15
    roof_height: float = 4.0
    roof_extra_depth: float = 2.0
    crenel_width: float = 1.0

    @property
    def wall_width(self) -> float:
        return self.x_init * 2.0 + self.x_space * (self.columns - 1) + self.window_width

    @property
    def wall_height(self) -> float:
        return self.y_init * 2.0 + self.y_space * (self.rows - 1) + self.window_height

    @property
    def total_height(self) -> float:
        return self.wall_height + self.roof_height

    @property
    def arch_height(self) -> float:
        return self.window_height * self.arch_ratio

    @property
    def roof_depth(self) -> float:
        return self.wall_depth + self.roof_extra_depth

    @property
    def x_door_gap(self) -> float:
        jamb_width = 1.0
        return (self.x_space + self.window_width) / 2.0 - self.window_width - jamb_width / 2.0


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(name: str, color: Color) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.58
        bsdf.inputs["Metallic"].default_value = 0.0
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> bpy.types.Object:
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
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material:
        assign_material(obj, material)
    return obj


def add_elliptical_cylinder(
    name: str,
    radius_x: float,
    radius_y: float,
    depth: float,
    center: tuple[float, float, float],
    material: bpy.types.Material | None = None,
    vertices: int = 72,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=1.0,
        depth=depth,
        location=center,
        rotation=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = (radius_x, radius_y, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material:
        assign_material(obj, material)
    return obj


def add_arched_opening_cutter(
    name: str,
    x_left: float,
    y_bottom: float,
    width: float,
    height: float,
    arch_height: float,
    depth: float,
) -> list[bpy.types.Object]:
    z_center = 0.0
    rect = add_box(
        f"{name}_rect_cutter",
        size=(width, height, depth),
        location=(x_left + width / 2.0, y_bottom + height / 2.0, z_center),
    )
    arch = add_elliptical_cylinder(
        f"{name}_arch_cutter",
        radius_x=width / 2.0,
        radius_y=arch_height,
        depth=depth,
        center=(x_left + width / 2.0, y_bottom + height, z_center),
    )
    return [rect, arch]


def boolean_difference(target: bpy.types.Object, cutters: list[bpy.types.Object]) -> None:
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    for cutter in cutters:
        modifier = target.modifiers.new(name=f"cut_{cutter.name}", type="BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter, do_unlink=True)


def set_origin_centered(model_width: float, model_height: float, objects: list[bpy.types.Object]) -> None:
    offset_x = -model_width / 2.0
    offset_y = -model_height / 2.0
    for obj in objects:
        obj.location.x += offset_x
        obj.location.y += offset_y


# ============================================================
# Procedural assembly
# ============================================================

"""High-level assembly for the modular building facade model."""





def _window_positions(params: FacadeParams) -> list[tuple[float, float]]:
    return [
        (params.x_init + col * params.x_space, params.y_init + row * params.y_space)
        for row in range(params.rows)
        for col in range(params.columns)
    ]


def _build_wall_with_openings(params: FacadeParams, wall_mat: bpy.types.Material) -> bpy.types.Object:
    wall = add_box(
        "source_facade_slab_through_roof_height_with_arch_cutouts",
        size=(params.wall_width, params.total_height, params.wall_depth),
        location=(params.wall_width / 2.0, params.total_height / 2.0, -params.wall_depth / 2.0),
        material=wall_mat,
    )

    cutters: list[bpy.types.Object] = []
    for index, (x_left, y_bottom) in enumerate(_window_positions(params), start=1):
        cutters.extend(
            add_arched_opening_cutter(
                f"window_{index:02d}",
                x_left=x_left,
                y_bottom=y_bottom,
                width=params.window_width,
                height=params.window_height,
                arch_height=params.arch_height,
                depth=params.window_cut_depth + params.wall_depth,
            )
        )

    door_x = params.x_init + params.x_space * (params.columns // 2)
    cutters.extend(
        add_arched_opening_cutter(
            "center_door",
            x_left=door_x,
            y_bottom=params.y_init,
            width=params.window_width,
            height=params.window_height,
            arch_height=params.arch_height,
            depth=params.window_cut_depth + params.wall_depth,
        )
    )
    cutters.append(
        add_box(
            "door_lower_rect_cutter",
            size=(params.window_width, params.y_init, params.window_cut_depth + params.wall_depth),
            location=(door_x + params.window_width / 2.0, params.y_init / 2.0, 0.0),
        )
    )

    boolean_difference(wall, cutters)
    return wall


def _build_roof(params: FacadeParams, wall_mat: bpy.types.Material) -> bpy.types.Object:
    roof = add_box(
        "source_roof_band_front_half_crenel_cutouts",
        size=(params.wall_width, params.roof_height, params.roof_depth),
        location=(params.wall_width / 2.0, params.wall_height + params.roof_height / 2.0, 0.0),
        material=wall_mat,
    )

    count = int((params.wall_width - params.x_init * 2.0) / (params.crenel_width * 2.0))
    base_span = (count + 1) * params.crenel_width * 2.0 - params.crenel_width
    x_start = (params.wall_width - base_span) / 2.0
    cutters = []
    for i in range(count + 1):
        cutters.append(
            add_box(
                f"roof_crenel_slot_{i:02d}",
                size=(params.crenel_width, params.roof_height * 0.6, params.roof_extra_depth + 0.08),
                location=(
                    x_start + i * params.crenel_width * 2.0 + params.crenel_width / 2.0,
                    params.wall_height + params.roof_height * 0.6 / 2.0,
                    params.roof_extra_depth / 2.0 + 0.02,
                ),
            )
        )
    boolean_difference(roof, cutters)
    return roof


def _add_muntins(params: FacadeParams, trim_mat: bpy.types.Material) -> list[bpy.types.Object]:
    pieces: list[bpy.types.Object] = []
    muntin_z = -params.wall_depth + params.muntin_depth / 2.0
    for index, (x_left, y_bottom) in enumerate(_window_positions(params), start=1):
        x_center = x_left + params.window_width / 2.0
        pieces.append(
            add_box(
                f"window_{index:02d}_horizontal_muntin",
                size=(params.window_width, params.muntin_width, params.muntin_depth),
                location=(x_center, y_bottom + params.window_height / 2.0, muntin_z),
                material=trim_mat,
            )
        )
        pieces.append(
            add_box(
                f"window_{index:02d}_arched_top_bar",
                size=(params.window_width, params.arch_height, params.muntin_depth),
                location=(x_center, y_bottom + params.window_height + params.arch_height / 2.0, muntin_z),
                material=trim_mat,
            )
        )
    return pieces


def _add_door_fill(params: FacadeParams, door_mat: bpy.types.Material) -> bpy.types.Object:
    door_x = params.x_init + params.x_space * (params.columns // 2)
    return add_box(
        "green_center_door_lower_panel",
        size=(params.window_width, params.y_init, params.muntin_depth),
        location=(door_x + params.window_width / 2.0, params.y_init / 2.0, -params.wall_depth + params.muntin_depth / 2.0),
        material=door_mat,
    )


def build_model() -> list[bpy.types.Object]:
    params = FacadeParams()
    wall_mat = make_material("source_yellow_wall", (0.98, 0.82, 0.05, 1.0))
    trim_mat = make_material("source_green_muntins", (0.36, 0.58, 0.20, 1.0))
    door_mat = make_material("source_green_door_panel", (0.43, 0.67, 0.22, 1.0))

    objects: list[bpy.types.Object] = []
    objects.append(_build_wall_with_openings(params, wall_mat))
    objects.append(_build_roof(params, wall_mat))
    objects.extend(_add_muntins(params, trim_mat))
    objects.append(_add_door_fill(params, door_mat))

    set_origin_centered(params.wall_width, params.total_height, objects)
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

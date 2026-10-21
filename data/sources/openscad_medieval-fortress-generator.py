from __future__ import annotations
from dataclasses import dataclass
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the seed-42 fortress model."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.75


MATERIALS = {
    "stone": MaterialSpec("source light grey stone", (0.66, 0.67, 0.66, 1.0)),
    "stone_dark": MaterialSpec("source shaded stone facets", (0.28, 0.30, 0.30, 1.0)),
    "stone_light": MaterialSpec("source pale roof caps", (0.80, 0.80, 0.78, 1.0)),
    "roof": MaterialSpec("source orange red roof", (1.0, 0.20, 0.02, 1.0)),
    "roof_dark": MaterialSpec("source dark red roof shade", (0.36, 0.04, 0.01, 1.0)),
    "opening": MaterialSpec("deep gate and courtyard shadow", (0.025, 0.025, 0.023, 1.0)),
    "trim": MaterialSpec("gate arch golden trim", (0.92, 0.76, 0.0, 1.0)),
}


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def get_material(key: str) -> bpy.types.Material:
    spec = MATERIALS[key]
    material = bpy.data.materials.get(spec.name)
    if material is None:
        material = bpy.data.materials.new(spec.name)
        material.diffuse_color = spec.color
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            base_color = bsdf.inputs.get("Base Color")
            roughness = bsdf.inputs.get("Roughness")
            if base_color is not None:
                base_color.default_value = spec.color
            if roughness is not None:
                roughness.default_value = spec.roughness
    return material


def assign_material(obj: bpy.types.Object, material_key: str) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(get_material(material_key))
    for polygon in obj.data.polygons:
        polygon.material_index = 0
    return obj


def shade_flat(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_flat()
    obj.select_set(False)
    return obj


def shade_smooth(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)
    return obj


def apply_boolean_difference(target: bpy.types.Object, cutter: bpy.types.Object, name: str) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    modifier = target.modifiers.new(name=name, type="BOOLEAN")
    modifier.operation = "DIFFERENCE"
    modifier.object = cutter
    if hasattr(modifier, "solver"):
        modifier.solver = "EXACT"
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    target.select_set(False)
    bpy.data.objects.remove(cutter, do_unlink=True)
    return shade_flat(target)


def correction_angle(sides: int) -> float:
    if sides == 4:
        return math.radians(45.0)
    if sides == 5:
        return math.radians(18.0)
    if sides == 8:
        return math.radians(22.5)
    if sides == 12:
        return math.radians(15.0)
    return 0.0


def inradius(diameter: float, sides: int) -> float:
    return diameter * math.cos(math.pi / sides)


def polygon_points(diameter: float, sides: int, rotation_z: float = 0.0) -> list[tuple[float, float]]:
    radius = diameter / 2.0
    return [
        (
            radius * math.cos(rotation_z + 2.0 * math.pi * i / sides),
            radius * math.sin(rotation_z + 2.0 * math.pi * i / sides),
        )
        for i in range(sides)
    ]


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: str = "stone",
    rotation_z: float = 0.0,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=(0.0, 0.0, rotation_z))
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return assign_material(obj, material)


def add_polygon_prism(
    name: str,
    diameter: float,
    height: float,
    base_z: float,
    xy: tuple[float, float],
    sides: int,
    material: str = "stone",
    rotation_z: float | None = None,
    top_scale: float = 1.0,
) -> bpy.types.Object:
    if rotation_z is None:
        rotation_z = correction_angle(sides)
    bpy.ops.mesh.primitive_cone_add(
        vertices=sides,
        radius1=diameter / 2.0,
        radius2=diameter * top_scale / 2.0,
        depth=height,
        end_fill_type="NGON",
        location=(xy[0], xy[1], base_z + height / 2.0),
        rotation=(0.0, 0.0, rotation_z),
    )
    obj = bpy.context.object
    obj.name = name
    assign_material(obj, material)
    return shade_flat(obj)


def add_polygon_ring(
    name: str,
    outer_diameter: float,
    inner_diameter: float,
    height: float,
    base_z: float,
    sides: int,
    xy: tuple[float, float] = (0.0, 0.0),
    material: str = "stone",
    rotation_z: float | None = None,
) -> bpy.types.Object:
    """Create the hollow extruded polygon used by the curtain wall."""
    if rotation_z is None:
        rotation_z = correction_angle(sides)

    outer = polygon_points(outer_diameter, sides, rotation_z)
    inner = polygon_points(inner_diameter, sides, rotation_z)
    verts: list[tuple[float, float, float]] = []
    for z in (base_z, base_z + height):
        verts.extend((xy[0] + x, xy[1] + y, z) for x, y in outer)
        verts.extend((xy[0] + x, xy[1] + y, z) for x, y in inner)

    bottom_outer = 0
    bottom_inner = sides
    top_outer = sides * 2
    top_inner = sides * 3
    faces: list[tuple[int, ...]] = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.append((bottom_outer + i, bottom_outer + j, top_outer + j, top_outer + i))
        faces.append((bottom_inner + j, bottom_inner + i, top_inner + i, top_inner + j))
        faces.append((top_outer + i, top_outer + j, top_inner + j, top_inner + i))
        faces.append((bottom_outer + j, bottom_outer + i, bottom_inner + i, bottom_inner + j))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    return shade_flat(obj)


def add_tapered_polygon_ring(
    name: str,
    outer_bottom_diameter: float,
    outer_top_diameter: float,
    inner_bottom_diameter: float,
    inner_top_diameter: float,
    height: float,
    base_z: float,
    sides: int,
    xy: tuple[float, float] = (0.0, 0.0),
    material: str = "stone",
    rotation_z: float | None = None,
) -> bpy.types.Object:
    """Create a hollow polygon frustum for the flared wall-walk lower roof band."""
    if rotation_z is None:
        rotation_z = correction_angle(sides)

    outer_bottom = polygon_points(outer_bottom_diameter, sides, rotation_z)
    inner_bottom = polygon_points(inner_bottom_diameter, sides, rotation_z)
    outer_top = polygon_points(outer_top_diameter, sides, rotation_z)
    inner_top = polygon_points(inner_top_diameter, sides, rotation_z)

    verts: list[tuple[float, float, float]] = []
    verts.extend((xy[0] + x, xy[1] + y, base_z) for x, y in outer_bottom)
    verts.extend((xy[0] + x, xy[1] + y, base_z) for x, y in inner_bottom)
    verts.extend((xy[0] + x, xy[1] + y, base_z + height) for x, y in outer_top)
    verts.extend((xy[0] + x, xy[1] + y, base_z + height) for x, y in inner_top)

    bottom_outer = 0
    bottom_inner = sides
    top_outer = sides * 2
    top_inner = sides * 3
    faces: list[tuple[int, ...]] = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.append((bottom_outer + i, bottom_outer + j, top_outer + j, top_outer + i))
        faces.append((bottom_inner + j, bottom_inner + i, top_inner + i, top_inner + j))
        faces.append((top_outer + i, top_outer + j, top_inner + j, top_inner + i))
        faces.append((bottom_outer + j, bottom_outer + i, bottom_inner + i, bottom_inner + j))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    return shade_flat(obj)


def _add_circular_slotted_parapet_segments(
    name: str,
    outer_diameter: float,
    inner_diameter: float,
    height: float,
    base_z: float,
    xy: tuple[float, float],
    material: str,
    rotation_z: float,
    element_max_width: float,
    slot_fraction: float,
) -> list[bpy.types.Object]:
    radial_depth = max(0.25, (outer_diameter - inner_diameter) / 2.0)
    mid_radius = inner_diameter / 2.0 + radial_depth / 2.0
    count = max(12, int(math.pi * outer_diameter / max(0.4, element_max_width)))
    cell_arc = math.tau * mid_radius / count
    solid_width = max(0.18, cell_arc * (1.0 - slot_fraction))
    objects: list[bpy.types.Object] = []

    for i in range(count):
        angle = rotation_z + (i + 0.5) * math.tau / count
        objects.append(
            add_box(
                f"{name}_solid_{i:02d}",
                size=(solid_width, radial_depth * 1.15, height),
                location=(
                    xy[0] + math.cos(angle) * mid_radius,
                    xy[1] + math.sin(angle) * mid_radius,
                    base_z + height / 2.0,
                ),
                material=material,
                rotation_z=angle + math.pi / 2.0,
            )
        )
    return objects


def _add_polygon_slotted_parapet_segments(
    name: str,
    outer_diameter: float,
    inner_diameter: float,
    height: float,
    base_z: float,
    sides: int,
    xy: tuple[float, float],
    material: str,
    rotation_z: float,
    element_max_width: float,
    slot_fraction: float,
) -> list[bpy.types.Object]:
    outer = polygon_points(outer_diameter, sides, rotation_z)
    inner = polygon_points(inner_diameter, sides, rotation_z)
    objects: list[bpy.types.Object] = []

    for side_index in range(sides):
        next_index = (side_index + 1) % sides
        ox1, oy1 = outer[side_index]
        ox2, oy2 = outer[next_index]
        ix1, iy1 = inner[side_index]
        ix2, iy2 = inner[next_index]
        edge_x = ox2 - ox1
        edge_y = oy2 - oy1
        side_length = math.hypot(edge_x, edge_y)
        tangent_angle = math.atan2(edge_y, edge_x)
        tangent = (edge_x / side_length, edge_y / side_length)

        outer_mid = ((ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0)
        inner_mid = ((ix1 + ix2) / 2.0, (iy1 + iy2) / 2.0)
        center_mid = ((outer_mid[0] + inner_mid[0]) / 2.0, (outer_mid[1] + inner_mid[1]) / 2.0)
        radial_depth = max(0.25, math.hypot(outer_mid[0] - inner_mid[0], outer_mid[1] - inner_mid[1]))

        element_width = min(side_length, max(0.4, element_max_width))
        elements_per_facet = max(1, int(side_length / element_width))
        cell_width = side_length / elements_per_facet
        slot_width = cell_width * slot_fraction
        slot_centers = [
            -0.5 * (elements_per_facet - 1) * cell_width + cell_index * cell_width
            for cell_index in range(elements_per_facet)
        ]

        intervals: list[tuple[float, float]] = []
        cursor = -side_length / 2.0
        for slot_center in slot_centers:
            slot_start = max(-side_length / 2.0, slot_center - slot_width / 2.0)
            slot_end = min(side_length / 2.0, slot_center + slot_width / 2.0)
            if slot_start > cursor:
                intervals.append((cursor, slot_start))
            cursor = max(cursor, slot_end)
        if cursor < side_length / 2.0:
            intervals.append((cursor, side_length / 2.0))

        for interval_index, (start, end) in enumerate(intervals):
            length = end - start
            if length <= 0.05:
                continue
            offset = (start + end) / 2.0
            objects.append(
                add_box(
                    f"{name}_side_{side_index:02d}_solid_{interval_index:02d}",
                    size=(length, radial_depth * 1.08, height),
                    location=(
                        xy[0] + center_mid[0] + tangent[0] * offset,
                        xy[1] + center_mid[1] + tangent[1] * offset,
                        base_z + height / 2.0,
                    ),
                    material=material,
                    rotation_z=tangent_angle,
                )
            )

    return objects


def add_slotted_parapet_segments(
    name: str,
    outer_diameter: float,
    inner_diameter: float,
    height: float,
    base_z: float,
    sides: int,
    xy: tuple[float, float] = (0.0, 0.0),
    material: str = "stone_light",
    rotation_z: float | None = None,
    element_max_width: float = 3.0,
    slot_fraction: float = 0.16,
) -> list[bpy.types.Object]:
    """Model the SCAD archer-slot cutters as narrow physical gaps between parapet solids."""
    if rotation_z is None:
        rotation_z = correction_angle(sides)
    if sides > 12:
        return _add_circular_slotted_parapet_segments(
            name,
            outer_diameter,
            inner_diameter,
            height,
            base_z,
            xy,
            material,
            rotation_z,
            element_max_width,
            slot_fraction,
        )
    return _add_polygon_slotted_parapet_segments(
        name,
        outer_diameter,
        inner_diameter,
        height,
        base_z,
        sides,
        xy,
        material,
        rotation_z,
        element_max_width,
        slot_fraction,
    )


def add_radial_slot_panels(
    name: str,
    diameter: float,
    base_z: float,
    xy: tuple[float, float],
    count: int,
    material: str = "opening",
    slot_width: float = 0.55,
    slot_height: float = 0.85,
    slot_depth: float = 0.06,
    rotation_z: float = 0.0,
) -> list[bpy.types.Object]:
    radius = diameter / 2.0 + slot_depth / 2.0
    objects: list[bpy.types.Object] = []
    for i in range(count):
        angle = rotation_z + 2.0 * math.pi * i / count
        objects.append(
            add_box(
                f"{name}_slot_{i:02d}",
                size=(slot_width, slot_depth, slot_height),
                location=(
                    xy[0] + math.cos(angle) * radius,
                    xy[1] + math.sin(angle) * radius,
                    base_z + slot_height / 2.0,
                ),
                material=material,
                rotation_z=angle + math.pi / 2.0,
            )
        )
    return objects


def add_flat_roof(
    name: str,
    diameter: float,
    base_z: float,
    xy: tuple[float, float],
    sides: int,
    material: str = "stone_light",
    rotation_z: float | None = None,
    floor_height: float = 3.0,
) -> list[bpy.types.Object]:
    bottom_h = 1.8 * 0.5 * floor_height
    total_h = 1.8 * floor_height
    top_h = total_h - bottom_h
    top_diameter = diameter + 1.8 * 0.5 * floor_height
    recess_h = max(0.25, 0.3 * top_h)
    slot_base_offset = 0.5 * top_h
    lower_ring_h = max(0.08, slot_base_offset - recess_h)
    slot_h = max(0.12, top_h - slot_base_offset)
    parapet_base = base_z + bottom_h - 0.01
    objects = [
        add_polygon_prism(
            f"{name}_flared_base",
            diameter=diameter,
            height=bottom_h,
            base_z=base_z,
            xy=xy,
            sides=sides,
            material=material,
            rotation_z=rotation_z,
            top_scale=top_diameter / diameter,
        ),
        add_polygon_prism(
            f"{name}_shallow_recess_floor",
            diameter=top_diameter,
            height=recess_h,
            base_z=parapet_base,
            xy=xy,
            sides=sides,
            material=material,
            rotation_z=rotation_z,
            top_scale=1.0,
        ),
        add_polygon_ring(
            f"{name}_continuous_lower_parapet_ring",
            outer_diameter=top_diameter,
            inner_diameter=diameter,
            height=lower_ring_h + 0.01,
            base_z=parapet_base + recess_h - 0.005,
            sides=sides,
            xy=xy,
            material=material,
            rotation_z=rotation_z,
        ),
    ]
    objects.extend(
        add_slotted_parapet_segments(
            f"{name}_parapet",
            outer_diameter=top_diameter,
            inner_diameter=diameter,
            height=slot_h,
            base_z=parapet_base + slot_base_offset - 0.005,
            xy=xy,
            sides=sides,
            material=material,
            rotation_z=rotation_z or 0.0,
            element_max_width=floor_height,
            slot_fraction=0.16,
        )
    )
    return objects


def add_pointed_roof(
    name: str,
    diameter: float,
    height: float,
    base_z: float,
    xy: tuple[float, float],
    sides: int,
    rotation_z: float | None = None,
) -> list[bpy.types.Object]:
    inverted_h = height * 0.04
    roof_base = diameter + 2.0 * inverted_h
    objects = [
        add_polygon_prism(
            f"{name}_flared_eave",
            diameter=diameter,
            height=inverted_h,
            base_z=base_z,
            xy=xy,
            sides=sides,
            material="roof",
            rotation_z=rotation_z,
            top_scale=roof_base / diameter,
        ),
        add_polygon_prism(
            f"{name}_spire",
            diameter=roof_base,
            height=height - inverted_h,
            base_z=base_z + inverted_h - 0.01,
            xy=xy,
            sides=sides,
            material="roof",
            rotation_z=rotation_z,
            top_scale=0.05,
        ),
    ]
    return objects


def add_flat_plus_point_roof(
    name: str,
    diameter: float,
    roof_height: float,
    base_z: float,
    xy: tuple[float, float],
    sides: int,
    rotation_z: float | None = None,
    floor_height: float = 3.0,
) -> list[bpy.types.Object]:
    objects = add_flat_roof(name, diameter, base_z, xy, sides, rotation_z=rotation_z, floor_height=floor_height)
    upper_diameter = 0.7 * diameter
    upper_body_base = base_z + 0.5 * floor_height
    upper_body_height = 2.0 * floor_height
    objects.append(
        add_polygon_prism(
            f"{name}_small_upper_drum",
            diameter=upper_diameter,
            height=upper_body_height,
            base_z=upper_body_base,
            xy=xy,
            sides=sides,
            material="stone",
            rotation_z=rotation_z,
        )
    )
    objects.extend(
        add_pointed_roof(
            f"{name}_small_spire",
            diameter=upper_diameter,
            height=0.8 * roof_height,
            base_z=base_z + 2.5 * floor_height - 0.02,
            xy=xy,
            sides=sides,
            rotation_z=rotation_z,
        )
    )
    return objects


def add_crenellations(
    name: str,
    diameter: float,
    base_z: float,
    xy: tuple[float, float],
    count: int,
    material: str = "stone_light",
    merlon_width: float = 1.2,
    merlon_depth: float = 1.0,
    merlon_height: float = 1.4,
    rotation_z: float = 0.0,
) -> list[bpy.types.Object]:
    radius = diameter / 2.0
    objects: list[bpy.types.Object] = []
    for i in range(count):
        angle = rotation_z + 2.0 * math.pi * i / count
        block = add_box(
            f"{name}_merlon_{i:02d}",
            size=(merlon_width, merlon_depth, merlon_height),
            location=(
                xy[0] + math.cos(angle) * radius,
                xy[1] + math.sin(angle) * radius,
                base_z + merlon_height / 2.0,
            ),
            material=material,
            rotation_z=angle,
        )
        objects.append(block)
    return objects


def add_wall_walk_battlements(
    name: str,
    outer_diameter: float,
    sides: int,
    base_z: float,
    material: str = "stone_light",
    rotation_z: float | None = None,
    inner_diameter: float | None = None,
) -> list[bpy.types.Object]:
    if rotation_z is None:
        rotation_z = correction_angle(sides)
    if inner_diameter is None:
        inner_diameter = outer_diameter * 0.82
    floor_height = 3.0
    bottom_h = 1.8 * 0.5 * floor_height
    top_h = 1.8 * floor_height - bottom_h
    top_diameter = outer_diameter + 1.8 * 0.5 * floor_height
    recess_h = max(0.25, 0.3 * top_h)
    slot_base_offset = 0.5 * top_h
    lower_ring_h = max(0.08, slot_base_offset - recess_h)
    slot_h = max(0.12, top_h - slot_base_offset)
    parapet_base = base_z + bottom_h - 0.01
    objects: list[bpy.types.Object] = [
        add_tapered_polygon_ring(
            f"{name}_flared_wall_walk_base",
            outer_bottom_diameter=outer_diameter,
            outer_top_diameter=top_diameter,
            inner_bottom_diameter=inner_diameter,
            inner_top_diameter=inner_diameter,
            height=bottom_h,
            base_z=base_z,
            sides=sides,
            material=material,
            rotation_z=rotation_z,
        ),
        add_polygon_ring(
            f"{name}_wide_lower_wall_walk_shelf",
            outer_diameter=top_diameter,
            inner_diameter=inner_diameter,
            height=recess_h,
            base_z=parapet_base,
            sides=sides,
            material=material,
            rotation_z=rotation_z,
        ),
        add_polygon_ring(
            f"{name}_continuous_lower_wall_walk_parapet_ring",
            outer_diameter=top_diameter,
            inner_diameter=outer_diameter,
            height=lower_ring_h + 0.01,
            base_z=parapet_base + recess_h - 0.005,
            sides=sides,
            material=material,
            rotation_z=rotation_z,
        ),
    ]
    objects.extend(
        add_slotted_parapet_segments(
            f"{name}_wall_walk_parapet",
            outer_diameter=top_diameter,
            inner_diameter=outer_diameter,
            height=slot_h,
            base_z=parapet_base + slot_base_offset - 0.005,
            xy=(0.0, 0.0),
            sides=sides,
            material=material,
            rotation_z=rotation_z,
            element_max_width=floor_height,
            slot_fraction=0.14,
        )
    )
    return objects


def add_vertical_facet_shadows(
    name: str,
    diameter: float,
    height: float,
    base_z: float,
    xy: tuple[float, float],
    count: int,
    material: str = "stone_dark",
    rotation_z: float = 0.0,
    width: float = 0.9,
    depth: float = 0.18,
) -> list[bpy.types.Object]:
    radius = diameter / 2.0 + depth / 2.0
    strips: list[bpy.types.Object] = []
    for i in range(count):
        angle = rotation_z + 2.0 * math.pi * i / count
        strips.append(
            add_box(
                f"{name}_dark_facet_{i:02d}",
                size=(depth, width, height),
                location=(
                    xy[0] + math.cos(angle) * radius,
                    xy[1] + math.sin(angle) * radius,
                    base_z + height / 2.0,
                ),
                material=material,
                rotation_z=angle,
            )
        )
    return strips


def add_hanging_turret_base(
    name: str,
    diameter: float,
    height: float,
    base_z: float,
    xy: tuple[float, float],
    sides: int,
    material: str = "stone",
    rotation_z: float | None = None,
    toward_xy: tuple[float, float] | None = None,
) -> bpy.types.Object:
    if rotation_z is None:
        rotation_z = correction_angle(sides)
    if toward_xy is None:
        toward = (0.0, -1.0)
    else:
        dx = toward_xy[0] - xy[0]
        dy = toward_xy[1] - xy[1]
        length = math.hypot(dx, dy) or 1.0
        toward = (dx / length, dy / length)

    local_x = (-toward[1], toward[0])
    away_from_parent = (-toward[0], -toward[1])
    levels = (
        (0.0, 1.0, 0.0),
        (1.0, 0.1, 0.45 * diameter),
    )
    verts: list[tuple[float, float, float]] = []
    for t, scale, center_offset_toward_parent in levels:
        radius = diameter * scale / 2.0
        center_x = xy[0] + toward[0] * center_offset_toward_parent
        center_y = xy[1] + toward[1] * center_offset_toward_parent
        z = base_z - height * t + 0.02
        for i in range(sides):
            angle = rotation_z + math.tau * i / sides
            px = math.cos(angle) * radius
            py = math.sin(angle) * radius
            verts.append(
                (
                    center_x + local_x[0] * px + away_from_parent[0] * py,
                    center_y + local_x[1] * px + away_from_parent[1] * py,
                    z,
                )
            )

    faces: list[tuple[int, ...]] = [tuple(range(sides))]
    bottom_start = sides
    faces.append(tuple(range(bottom_start + sides - 1, bottom_start - 1, -1)))
    for i in range(sides):
        j = (i + 1) % sides
        faces.append((i, j, bottom_start + j, bottom_start + i))

    mesh = bpy.data.meshes.new(f"{name}_tapered_hanging_base_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"{name}_tapered_hanging_base", mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    return shade_flat(obj)


def arch_profile(width: float, height: float, segments: int = 10) -> list[tuple[float, float]]:
    """Approximate OpenSCAD special_rectangle: straight sides with a pointed Gothic cap."""
    half_w = width / 2.0
    shoulder = max(height - half_w, height * 0.55)
    pts = [(-half_w, 0.0), (half_w, 0.0), (half_w, shoulder)]
    for i in range(1, segments):
        t = i / segments
        x = half_w * (1.0 - t)
        z = shoulder + half_w * t
        pts.append((x, z))
    pts.append((0.0, height))
    for i in range(1, segments):
        t = i / segments
        x = -half_w * t
        z = height - half_w * t
        pts.append((x, z))
    pts.append((-half_w, shoulder))
    return pts


def add_arch_panel(
    name: str,
    width: float,
    height: float,
    depth: float,
    location: tuple[float, float, float],
    material: str,
    rotation_z: float = 0.0,
) -> bpy.types.Object:
    pts = arch_profile(width, height)
    verts = [(x, -depth / 2.0, z) for x, z in pts] + [(x, depth / 2.0, z) for x, z in pts]
    n = len(pts)
    faces: list[tuple[int, ...]] = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    for i in range(n):
        faces.append((i, (i + 1) % n, (i + 1) % n + n, i + n))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler[2] = rotation_z
    assign_material(obj, material)
    return shade_flat(obj)


def add_arch_rim(
    name: str,
    outer_width: float,
    outer_height: float,
    inner_width: float,
    inner_height: float,
    depth: float,
    location: tuple[float, float, float],
    material: str,
    rotation_z: float = 0.0,
) -> bpy.types.Object:
    outer = arch_profile(outer_width, outer_height)
    inner = arch_profile(inner_width, inner_height)
    if len(outer) != len(inner):
        raise ValueError("arch rim profiles must have matching point counts")

    verts: list[tuple[float, float, float]] = []
    verts.extend((x, -depth / 2.0, z) for x, z in outer)
    verts.extend((x, -depth / 2.0, z) for x, z in inner)
    verts.extend((x, depth / 2.0, z) for x, z in outer)
    verts.extend((x, depth / 2.0, z) for x, z in inner)

    n = len(outer)
    outer_front = 0
    inner_front = n
    outer_back = n * 2
    inner_back = n * 3
    faces: list[tuple[int, ...]] = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((outer_front + i, outer_front + j, inner_front + j, inner_front + i))
        faces.append((outer_back + j, outer_back + i, inner_back + i, inner_back + j))
        faces.append((outer_front + j, outer_front + i, outer_back + i, outer_back + j))
        faces.append((inner_front + i, inner_front + j, inner_back + j, inner_back + i))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler[2] = rotation_z
    assign_material(obj, material)
    return shade_flat(obj)


def add_arch_tunnel_liner(
    name: str,
    width: float,
    height: float,
    depth: float,
    location: tuple[float, float, float],
    material: str,
    rotation_z: float = 0.0,
) -> bpy.types.Object:
    pts = arch_profile(width, height)
    verts = [(x, -depth / 2.0, z) for x, z in pts] + [(x, depth / 2.0, z) for x, z in pts]
    n = len(pts)
    faces: list[tuple[int, ...]] = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler[2] = rotation_z
    assign_material(obj, material)
    return shade_flat(obj)


# ============================================================
# Procedural assembly
# ============================================================

"""High-level assembly for the medieval fortress generator seed 42."""






WALL_DIAMETER = 80.0
WALL_SIDES = 5
WALL_HEIGHT = 35.0
WALL_THICKNESS = 6.0
FLOOR_HEIGHT = 3.0
GATE_WIDTH = 3.0 * FLOOR_HEIGHT
GATE_HEIGHT = 5.0 * FLOOR_HEIGHT
TOWER_WIDTH = 23.0
TOWER_HEIGHT = 40.0
ROOF_POINTED_RATIO = 1.8
WALL_ROTATION = correction_angle(WALL_SIDES)
TOWER_RING_ROTATION = math.radians(36.0)


@dataclass(frozen=True)
class TowerSeed:
    name: str
    index: int
    sides: int
    roof_type: int
    child_count: int
    child_sides: int | None = None
    child_roof_type: int | None = None
    child_angle_deg: float = 0.0
    child_z: float = 0.0


TOWER_SEEDS = [
    TowerSeed("front_six_sided_flat_tower", 0, 6, 2, 0),
    TowerSeed("right_rear_flat_plus_point_tower", 1, 12, 3, 1, 6, 2, 141.826, 25.5698),
    TowerSeed("rear_octagonal_flat_tower", 2, 8, 2, 1, 8, 2, 147.035, 31.6732),
    TowerSeed("left_rear_round_flat_plus_point_tower", 3, 60, 3, 1, 8, 2, -74.9387, 31.9412),
    TowerSeed("left_front_octagonal_flat_tower", 4, 8, 2, 1, 12, 3, 90.4434, 34.3545),
]


def add_roof_for_type(
    name: str,
    roof_type: int,
    diameter: float,
    body_height: float,
    xy: tuple[float, float],
    sides: int,
    rotation_z: float,
) -> list[bpy.types.Object]:
    roof_base = body_height - 0.01
    roof_height = diameter * ROOF_POINTED_RATIO
    if roof_type == 1:
        return add_pointed_roof(name, diameter, roof_height, roof_base, xy, sides, rotation_z)
    if roof_type == 2:
        return add_flat_roof(name, diameter, roof_base, xy, sides, rotation_z=rotation_z, floor_height=FLOOR_HEIGHT)
    if roof_type == 3:
        return add_flat_plus_point_roof(
            name,
            diameter,
            roof_height,
            roof_base,
            xy,
            sides,
            rotation_z=rotation_z,
            floor_height=FLOOR_HEIGHT,
        )
    raise ValueError(f"Unsupported roof type: {roof_type}")


def add_tower(
    seed: TowerSeed,
    xy: tuple[float, float],
    diameter: float = TOWER_WIDTH,
    body_height: float = TOWER_HEIGHT,
    base_z: float = 0.0,
    main_tower: bool = True,
    local_rotation_z: float = 0.0,
) -> list[bpy.types.Object]:
    """Build one SCAD turret_round plus optional generated child turret."""
    objects: list[bpy.types.Object] = []
    rotation_z = correction_angle(seed.sides)
    body = add_polygon_prism(
        f"{seed.name}_body",
        diameter=diameter,
        height=body_height,
        base_z=base_z,
        xy=xy,
        sides=seed.sides,
        material="stone",
        rotation_z=rotation_z,
    )
    objects.append(body)
    objects.extend(add_roof_for_type(f"{seed.name}_roof", seed.roof_type, diameter, base_z + body_height, xy, seed.sides, rotation_z))

    if main_tower and seed.child_count and seed.child_sides and seed.child_roof_type:
        child_width = max(diameter * 0.36, 0.8 * FLOOR_HEIGHT)
        child_height = body_height * 0.49
        parent_inradius = inradius(diameter, seed.sides)
        child_angle = local_rotation_z + math.radians(seed.child_angle_deg) + math.pi / 2.0
        child_xy = (
            xy[0] + math.cos(child_angle) * parent_inradius * 0.55,
            xy[1] + math.sin(child_angle) * parent_inradius * 0.55,
        )
        child_seed = TowerSeed(
            name=f"{seed.name}_child_turret",
            index=seed.index,
            sides=seed.child_sides,
            roof_type=seed.child_roof_type,
            child_count=0,
        )
        objects.append(
            add_hanging_turret_base(
                child_seed.name,
                diameter=child_width,
                height=child_width * 2.0,
                base_z=seed.child_z,
                xy=child_xy,
                sides=seed.child_sides,
                material="stone",
                rotation_z=correction_angle(seed.child_sides),
                toward_xy=xy,
            )
        )
        objects.extend(
            add_tower(
                child_seed,
                xy=child_xy,
                diameter=child_width,
                body_height=child_height,
                base_z=seed.child_z,
                main_tower=False,
                local_rotation_z=child_angle,
            )
        )

    return objects


def tower_position(index: int) -> tuple[float, float]:
    angle = TOWER_RING_ROTATION + index * math.tau / len(TOWER_SEEDS) + math.pi / len(TOWER_SEEDS)
    radius = 0.97 * WALL_DIAMETER / 2.0
    return (-math.sin(angle) * radius, math.cos(angle) * radius)


def add_curtain_wall() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    objects.append(
        add_polygon_ring(
            "hollow_pentagonal_curtain_wall",
            outer_diameter=WALL_DIAMETER,
            inner_diameter=WALL_DIAMETER - 2.0 * WALL_THICKNESS,
            height=WALL_HEIGHT,
            base_z=0.0,
            sides=WALL_SIDES,
            material="stone",
            rotation_z=WALL_ROTATION,
        )
    )
    objects.extend(
        add_wall_walk_battlements(
            "pentagonal_curtain_wall",
            outer_diameter=WALL_DIAMETER,
            inner_diameter=WALL_DIAMETER - 2.0 * WALL_THICKNESS,
            sides=WALL_SIDES,
            base_z=WALL_HEIGHT,
            material="stone_light",
            rotation_z=WALL_ROTATION,
        )
    )

    # Thin dark interior panels preserve the visibly hollow courtyard in non-boolean BPY form.
    inner_span = WALL_DIAMETER - 2.0 * WALL_THICKNESS
    for side_index in range(WALL_SIDES):
        angle = WALL_ROTATION + 2.0 * math.pi * (side_index + 0.5) / WALL_SIDES
        objects.append(
            add_box(
                f"courtyard_dark_inner_wall_{side_index}",
                size=(inner_span * 0.42, 0.16, WALL_HEIGHT * 0.92),
                location=(
                    math.cos(angle) * inradius(inner_span, WALL_SIDES) / 2.0,
                    math.sin(angle) * inradius(inner_span, WALL_SIDES) / 2.0,
                    WALL_HEIGHT * 0.46,
                ),
                material="opening",
                rotation_z=angle + math.pi / 2.0,
            )
        )
    return objects


def add_gatehouse() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    gate_tower_width = 3.0 * GATE_WIDTH
    gate_y = -inradius(WALL_DIAMETER, WALL_SIDES) / 2.0
    gate_height = 1.4 * GATE_HEIGHT
    gate_rotation = correction_angle(4)

    gatehouse_body = add_polygon_prism(
        "front_square_gatehouse_body",
        diameter=gate_tower_width,
        height=gate_height,
        base_z=0.0,
        xy=(0.0, gate_y),
        sides=4,
        material="stone",
        rotation_z=gate_rotation,
    )
    cutter = add_arch_panel(
        "front_gate_boolean_arch_cutter",
        width=GATE_WIDTH,
        height=GATE_HEIGHT,
        depth=gate_tower_width * 2.4,
        location=(0.0, gate_y, -0.02),
        material="opening",
    )
    apply_boolean_difference(gatehouse_body, cutter, "real_archway_through_gatehouse")
    objects.append(gatehouse_body)
    objects.extend(
        add_flat_roof(
            "front_square_gatehouse_roof",
            diameter=gate_tower_width,
            base_z=gate_height - 0.01,
            xy=(0.0, gate_y),
            sides=4,
            rotation_z=gate_rotation,
            floor_height=FLOOR_HEIGHT,
        )
    )

    # Source boolean subtracts the inner black arch and leaves a yellow/gold rim from the gate tower.
    front_y = gate_y - inradius(gate_tower_width, 4) / 2.0 - 0.22
    objects.append(
        add_arch_rim(
            "front_gate_outer_gold_pointed_arch",
            outer_width=GATE_WIDTH + 1.7,
            outer_height=GATE_HEIGHT + 1.9,
            inner_width=GATE_WIDTH,
            inner_height=GATE_HEIGHT,
            depth=0.36,
            location=(0.0, front_y - 0.10, 0.0),
            material="trim",
        )
    )
    objects.append(
        add_arch_tunnel_liner(
            "front_gate_dark_through_opening_liner",
            width=GATE_WIDTH,
            height=GATE_HEIGHT,
            depth=gate_tower_width * 1.06,
            location=(0.0, gate_y, 0.01),
            material="opening",
        )
    )
    return objects


def add_major_towers() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    for seed in TOWER_SEEDS:
        tower_rotation = TOWER_RING_ROTATION + seed.index * math.tau / len(TOWER_SEEDS) + math.pi / len(TOWER_SEEDS)
        objects.extend(add_tower(seed, tower_position(seed.index), local_rotation_z=tower_rotation))
    return objects


def build_model() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    objects.extend(add_curtain_wall())
    objects.extend(add_gatehouse())
    objects.extend(add_major_towers())
    return objects


# ============================================================
# Scene setup and rendering
# ============================================================

"""Minimal render entrypoint for the current model version."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

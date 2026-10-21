from __future__ import annotations
from dataclasses import dataclass
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
from typing import Iterable
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable geometry primitives for the slotted drawer-box model."""





@dataclass(frozen=True)
class DrawerBoxDimensions:
    """Numerical defaults derived from the source OpenSCAD parameters."""

    num_drawers: int = 3
    drawer_height: float = 20.0
    drawer_depth: float = 62.0
    drawer_width: float = 84.0
    clearance: float = 0.5
    wall_thick: float = 4.0
    floor_thick: float = 4.0
    drawer_corner_rad: float = 2.0
    bump_radius: float = 1.5
    ear_radius: float = 10.0
    ear_thickness: float = 0.6

    @property
    def wall_width(self) -> float:
        return self.wall_thick + 0.001

    @property
    def box_width(self) -> float:
        return self.drawer_width + self.clearance * 2.0 + self.wall_width * 2.0

    @property
    def box_depth(self) -> float:
        return self.drawer_depth + self.clearance + self.wall_width

    @property
    def box_height(self) -> float:
        return self.num_drawers * self.layer_pitch + self.wall_width

    @property
    def layer_pitch(self) -> float:
        return self.drawer_height + self.clearance * 2.0 + self.wall_width

    @property
    def opening_width(self) -> float:
        return self.drawer_height + self.clearance * 2.0

    @property
    def side_slot_radius(self) -> float:
        return self.drawer_height * 0.5 * 0.7

    @property
    def world_width(self) -> float:
        return self.box_width

    @property
    def world_depth(self) -> float:
        # Source local Z is the drawer stack. After rotate([270,0,0]) it is world Y.
        return self.box_height

    @property
    def world_height(self) -> float:
        # Source local Y is drawer depth. After rotate([270,0,0]) it is world Z.
        return self.box_depth

    @property
    def x_min(self) -> float:
        return -self.box_width * 0.5

    @property
    def x_max(self) -> float:
        return self.box_width * 0.5

    @property
    def x_inner_min(self) -> float:
        return self.x_min + self.wall_width

    @property
    def x_inner_max(self) -> float:
        return self.x_max - self.wall_width

    @property
    def y_front(self) -> float:
        return -self.box_height * 0.5

    @property
    def y_back(self) -> float:
        return self.box_height * 0.5

    @property
    def z_bottom(self) -> float:
        return -self.box_depth * 0.5

    @property
    def z_top(self) -> float:
        return self.box_depth * 0.5

    @property
    def channel_floor_top_z(self) -> float:
        # Main drawer-channel subtraction ends at source local y = DRAWER_DEPTH + CLEARANCE.
        return self.box_depth * 0.5 - (self.drawer_depth + self.clearance)

    def source_to_world(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Map source positive drawerbox coordinates after translate + rotate([270,0,0]).

        The OpenSCAD module first translates the box by
        [-BOX_WIDTH/2, -BOX_DEPTH/2, 0]. The final rotate([270,0,0]) maps
        local (x, y, z) to (x, z, -y). Centering by the source body gives:
        world_x = x - BOX_WIDTH/2
        world_y = z - BOX_HEIGHT/2
        world_z = BOX_DEPTH/2 - y
        """

        return (
            x - self.box_width * 0.5,
            z - self.box_height * 0.5,
            self.box_depth * 0.5 - y,
        )

    def channel_y_bounds(self, index: int) -> tuple[float, float]:
        z0 = self.wall_width + index * self.layer_pitch
        return (
            self.source_to_world(0.0, 0.0, z0)[1],
            self.source_to_world(0.0, 0.0, z0 + self.opening_width)[1],
        )

    def channel_center_y(self, index: int) -> float:
        y_min, y_max = self.channel_y_bounds(index)
        return (y_min + y_max) * 0.5

    def divider_center_y(self, index: int) -> float:
        local_z = self.wall_width + self.opening_width + self.wall_width * 0.5 + index * self.layer_pitch
        return self.source_to_world(0.0, 0.0, local_z)[1]

    def side_slot_profile(self) -> tuple[float, float]:
        # From the first SCAD hull centers after translate([WALL_WIDTH,-1,z0]).
        upper_z = self.source_to_world(0.0, self.drawer_height * 0.5 + self.wall_thick * 0.5 - 1.0, 0.0)[2]
        lower_z = self.source_to_world(0.0, self.drawer_depth - self.drawer_height * 0.5 + self.wall_thick - 1.0, 0.0)[2]
        return (upper_z + lower_z) * 0.5, abs(upper_z - lower_z)

    def floor_slot_profile(self) -> tuple[float, float]:
        # From the second SCAD hull center x positions after rotate([90,90,0]).
        left_x = self.source_to_world(self.wall_width + self.drawer_height * 0.5, 0.0, 0.0)[0]
        right_x = self.source_to_world(
            self.wall_width + self.drawer_width - self.drawer_height * 0.5 + self.clearance * 2.0,
            0.0,
            0.0,
        )[0]
        return (left_x + right_x) * 0.5, abs(right_x - left_x)

    def bump_base_locations(self) -> list[tuple[float, float, float]]:
        locations: list[tuple[float, float, float]] = []
        for index in range(self.num_drawers):
            z = index * self.layer_pitch + self.wall_width - 0.1
            for x in (self.box_width * 0.25, self.box_width * 0.75):
                locations.append(self.source_to_world(x, self.drawer_corner_rad * 2.0, z))
        return locations

    def mouse_ear_centers(self) -> list[tuple[str, tuple[float, float, float]]]:
        result: list[tuple[str, tuple[float, float, float]]] = []
        source_points = [
            ("front_left", -self.ear_radius * 0.5 + 4.0, self.box_depth, -self.ear_radius * 0.5 + 4.0),
            ("front_right", self.box_width + self.ear_radius * 0.5 - 4.0, self.box_depth, -self.ear_radius * 0.5 + 4.0),
            (
                "rear_left",
                -self.ear_radius * 0.5 + 4.0,
                self.box_depth,
                self.box_height + self.ear_radius * 0.5 - 4.0,
            ),
            (
                "rear_right",
                self.box_width + self.ear_radius * 0.5 - 4.0,
                self.box_depth,
                self.box_height + self.ear_radius * 0.5 - 4.0,
            ),
        ]
        for name, sx, sy, sz in source_points:
            x, y, _ = self.source_to_world(sx, sy, sz)
            z = self.z_bottom + self.ear_thickness * 0.5
            result.append((name, (x, y, z)))
        return result


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.58
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
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material:
        assign_material(obj, material)
    return obj


def add_cylinder(
    name: str,
    radius: float,
    depth: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    if material:
        assign_material(obj, material)
    return obj


def add_edge_roundover(obj: bpy.types.Object, radius: float, segments: int = 2) -> bpy.types.Object:
    bevel = obj.modifiers.new(name="small_edge_roundover", type="BEVEL")
    bevel.width = radius
    bevel.segments = segments
    bevel.affect = "EDGES"
    bevel.profile = 0.5
    obj.modifiers.new(name="weighted_corner_normals", type="WEIGHTED_NORMAL")
    return obj


def add_plate_with_capsule_holes(
    name: str,
    extrusion_axis: str,
    plane_axes: tuple[str, str],
    extrusion_center: float,
    thickness: float,
    outer_loop: list[tuple[float, float]],
    hole_loops: list[list[tuple[float, float]]],
    material: bpy.types.Material,
    cut_material: bpy.types.Material | None = None,
    positive_face_material: bpy.types.Material | None = None,
    negative_face_material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Create a single extruded plate mesh with real capsule holes."""

    axis_lookup = {"X": 0, "Y": 1, "Z": 2}
    extrusion_index = axis_lookup[extrusion_axis]
    plane_indices = (axis_lookup[plane_axes[0]], axis_lookup[plane_axes[1]])

    loops = [_with_area_sign(outer_loop, want_positive=True)]
    loops.extend(_with_area_sign(loop, want_positive=False) for loop in hole_loops)
    flat_points = [point for loop in loops for point in loop]
    poly_vectors = [[Vector((u, v, 0.0)) for u, v in loop] for loop in loops]
    triangles = tessellate_polygon(poly_vectors)

    vertices: list[tuple[float, float, float]] = []
    for offset in (thickness * 0.5, -thickness * 0.5):
        for u, v in flat_points:
            coords = [0.0, 0.0, 0.0]
            coords[extrusion_index] = extrusion_center + offset
            coords[plane_indices[0]] = u
            coords[plane_indices[1]] = v
            vertices.append(tuple(coords))

    face_material_names: list[str] = []
    faces: list[tuple[int, ...]] = []
    point_count = len(flat_points)

    for tri in triangles:
        faces.append(tuple(tri))
        face_material_names.append("positive")
    for tri in triangles:
        faces.append(tuple(point_count + index for index in reversed(tri)))
        face_material_names.append("negative")

    start = 0
    for loop_index, loop in enumerate(loops):
        loop_len = len(loop)
        is_hole = loop_index > 0
        for index in range(loop_len):
            a = start + index
            b = start + (index + 1) % loop_len
            if is_hole:
                faces.append((a, b, point_count + b, point_count + a))
                face_material_names.append("cut")
            else:
                faces.append((point_count + a, point_count + b, b, a))
                face_material_names.append("shell")
        start += loop_len

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    materials: list[bpy.types.Material] = [material]
    for candidate in (cut_material, positive_face_material, negative_face_material):
        if candidate and candidate.name not in {existing.name for existing in materials}:
            materials.append(candidate)
    for mat in materials:
        obj.data.materials.append(mat)

    material_index_by_role = {
        "shell": materials.index(material),
        "cut": materials.index(cut_material) if cut_material in materials else materials.index(material),
        "positive": materials.index(positive_face_material) if positive_face_material in materials else materials.index(material),
        "negative": materials.index(negative_face_material) if negative_face_material in materials else materials.index(material),
    }
    for polygon, role in zip(obj.data.polygons, face_material_names):
        polygon.material_index = material_index_by_role[role]

    obj.data.update()
    return obj


def capsule_loop(
    center_u: float,
    center_v: float,
    straight_length: float,
    radius: float,
    axis: str,
    segments: int = 24,
) -> list[tuple[float, float]]:
    """Return a CCW capsule loop in generic (u, v) plane coordinates."""

    half_straight = max(0.0, straight_length * 0.5)
    points: list[tuple[float, float]] = []
    if axis == "V":
        top = center_v + half_straight
        bottom = center_v - half_straight
        points.append((center_u + radius, bottom))
        points.append((center_u + radius, top))
        for step in range(1, segments + 1):
            angle = step * math.pi / segments
            points.append((center_u + math.cos(angle) * radius, top + math.sin(angle) * radius))
        points.append((center_u - radius, bottom))
        for step in range(1, segments + 1):
            angle = math.pi + step * math.pi / segments
            points.append((center_u + math.cos(angle) * radius, bottom + math.sin(angle) * radius))
    elif axis == "U":
        right = center_u + half_straight
        left = center_u - half_straight
        points.append((left, center_v - radius))
        points.append((right, center_v - radius))
        for step in range(1, segments + 1):
            angle = -math.pi * 0.5 + step * math.pi / segments
            points.append((right + math.cos(angle) * radius, center_v + math.sin(angle) * radius))
        points.append((left, center_v + radius))
        for step in range(1, segments + 1):
            angle = math.pi * 0.5 + step * math.pi / segments
            points.append((left + math.cos(angle) * radius, center_v + math.sin(angle) * radius))
    else:
        raise ValueError(f"Unsupported capsule axis: {axis}")
    return points


def add_hemisphere_cap_y(
    name: str,
    radius: float,
    location: tuple[float, float, float],
    axis_scale: float,
    material: bpy.types.Material,
    segments: int = 32,
    rings: int = 8,
) -> bpy.types.Object:
    """Create the source half-sphere bump as a cap protruding along +Y."""

    vertices: list[tuple[float, float, float]] = []
    for ring in range(rings + 1):
        theta = ring * (math.pi * 0.5) / rings
        ring_radius = math.cos(theta) * radius
        y = math.sin(theta) * radius * axis_scale
        for segment in range(segments):
            phi = segment * math.tau / segments
            vertices.append(
                (
                    location[0] + math.cos(phi) * ring_radius,
                    location[1] + y,
                    location[2] + math.sin(phi) * ring_radius,
                )
            )

    faces: list[tuple[int, ...]] = []
    for ring in range(rings):
        base = ring * segments
        next_base = (ring + 1) * segments
        for segment in range(segments):
            nxt = (segment + 1) % segments
            faces.append((base + segment, base + nxt, next_base + nxt, next_base + segment))
    faces.append(tuple(range(segments - 1, -1, -1)))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def shade_smooth(objects: Iterable[bpy.types.Object]) -> None:
    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        try:
            bpy.ops.object.shade_smooth()
        finally:
            obj.select_set(False)


def _with_area_sign(points: list[tuple[float, float]], want_positive: bool) -> list[tuple[float, float]]:
    area = _signed_area(points)
    if (area >= 0.0) == want_positive:
        return points
    return list(reversed(points))


def _signed_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return area * 0.5


# ============================================================
# Procedural assembly
# ============================================================

"""Readable assembly for the default `part="drawerbox"` source model."""





def build_model() -> list[bpy.types.Object]:
    dims = DrawerBoxDimensions()
    materials = _make_materials()

    objects: list[bpy.types.Object] = []
    objects.extend(_build_housing_shell(dims, materials["shell"], materials["interior"]))
    objects.extend(_build_cut_face_liners(dims, materials["interior"]))
    objects.extend(_build_mouse_ears(dims, materials["shell"]))
    objects.extend(_build_bump_caps(dims, materials["shell"]))

    shade_smooth(obj for obj in objects if obj.name.startswith(("source_bump", "mouse_ear")))
    return objects


def _make_materials() -> dict[str, bpy.types.Material]:
    return {
        "shell": make_material("source_warm_yellow_shell", (0.95, 0.72, 0.02, 1.0)),
        "interior": make_material("source_olive_green_cut_faces", (0.43, 0.61, 0.18, 1.0)),
    }


def _build_housing_shell(
    dims: DrawerBoxDimensions,
    shell_mat: bpy.types.Material,
    interior_mat: bpy.types.Material,
) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    objects.append(_build_bottom_floor_plate(dims, shell_mat, interior_mat))

    for side_name, x_center in [
        ("left", dims.x_min + dims.wall_width * 0.5),
        ("right", dims.x_max - dims.wall_width * 0.5),
    ]:
        objects.append(_build_side_wall_plate(side_name, x_center, dims, shell_mat, interior_mat))

    for name, y_center in [
        ("front_plain_wall", dims.y_front + dims.wall_width * 0.5),
        ("rear_plain_wall", dims.y_back - dims.wall_width * 0.5),
    ]:
        objects.append(
            add_box(
                name,
                (dims.world_width, dims.wall_width, dims.world_height),
                (0.0, y_center, 0.0),
                material=shell_mat,
            )
        )

    for divider_index in range(dims.num_drawers - 1):
        objects.append(
            add_box(
                f"solid_internal_divider_between_drawers_{divider_index + 1}",
                (dims.world_width, dims.wall_width, dims.world_height),
                (0.0, dims.divider_center_y(divider_index), 0.0),
                material=shell_mat,
            )
        )

    return objects


def _build_bottom_floor_plate(
    dims: DrawerBoxDimensions,
    shell_mat: bpy.types.Material,
    interior_mat: bpy.types.Material,
) -> bpy.types.Object:
    slot_center_x, slot_straight_length = dims.floor_slot_profile()
    holes = [
        capsule_loop(
            center_u=slot_center_x,
            center_v=dims.channel_center_y(index),
            straight_length=slot_straight_length,
            radius=dims.side_slot_radius,
            axis="U",
            segments=28,
        )
        for index in range(dims.num_drawers)
    ]
    return add_plate_with_capsule_holes(
        "bottom_floor_with_three_source_capsule_slots",
        extrusion_axis="Z",
        plane_axes=("X", "Y"),
        extrusion_center=dims.z_bottom + dims.floor_thick * 0.5,
        thickness=dims.floor_thick,
        outer_loop=[
            (dims.x_min, dims.y_front),
            (dims.x_max, dims.y_front),
            (dims.x_max, dims.y_back),
            (dims.x_min, dims.y_back),
        ],
        hole_loops=holes,
        material=shell_mat,
        cut_material=interior_mat,
        positive_face_material=interior_mat,
        negative_face_material=shell_mat,
    )


def _build_side_wall_plate(
    side_name: str,
    x_center: float,
    dims: DrawerBoxDimensions,
    shell_mat: bpy.types.Material,
    interior_mat: bpy.types.Material,
) -> bpy.types.Object:
    slot_center_z, slot_straight_length = dims.side_slot_profile()
    holes = [
        capsule_loop(
            center_u=dims.channel_center_y(index),
            center_v=slot_center_z,
            straight_length=slot_straight_length,
            radius=dims.side_slot_radius,
            axis="V",
            segments=28,
        )
        for index in range(dims.num_drawers)
    ]
    return add_plate_with_capsule_holes(
        f"{side_name}_side_wall_single_plate_with_three_true_capsule_openings",
        extrusion_axis="X",
        plane_axes=("Y", "Z"),
        extrusion_center=x_center,
        thickness=dims.wall_width,
        outer_loop=[
            (dims.y_front, dims.z_bottom),
            (dims.y_back, dims.z_bottom),
            (dims.y_back, dims.z_top),
            (dims.y_front, dims.z_top),
        ],
        hole_loops=holes,
        material=shell_mat,
        cut_material=interior_mat,
        positive_face_material=interior_mat if side_name == "left" else shell_mat,
        negative_face_material=shell_mat if side_name == "left" else interior_mat,
    )


def _build_cut_face_liners(
    dims: DrawerBoxDimensions,
    interior_mat: bpy.types.Material,
) -> list[bpy.types.Object]:
    """Add only thin green surfaces where the SCAD boolean cut exposes interior faces."""

    objects: list[bpy.types.Object] = []
    channel_z_center = (dims.channel_floor_top_z + dims.z_top) * 0.5
    channel_z_height = dims.z_top - dims.channel_floor_top_z
    inner_width = dims.x_inner_max - dims.x_inner_min

    for index in range(dims.num_drawers):
        y0, y1 = dims.channel_y_bounds(index)
        y_center = (y0 + y1) * 0.5
        objects.append(
            add_box(
                f"green_front_cut_face_channel_{index + 1}",
                (inner_width, 0.06, channel_z_height),
                (0.0, y0 + 0.03, channel_z_center),
                material=interior_mat,
            )
        )
        objects.append(
            add_box(
                f"green_rear_cut_face_channel_{index + 1}",
                (inner_width, 0.06, channel_z_height),
                (0.0, y1 - 0.03, channel_z_center),
                material=interior_mat,
            )
        )

    return objects


def _build_mouse_ears(
    dims: DrawerBoxDimensions,
    shell_mat: bpy.types.Material,
) -> list[bpy.types.Object]:
    return [
        add_cylinder(
            f"mouse_ear_{name}",
            radius=dims.ear_radius,
            depth=dims.ear_thickness,
            location=location,
            vertices=96,
            material=shell_mat,
        )
        for name, location in dims.mouse_ear_centers()
    ]


def _build_bump_caps(
    dims: DrawerBoxDimensions,
    shell_mat: bpy.types.Material,
) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    for bump_index, location in enumerate(dims.bump_base_locations(), start=1):
        objects.append(
            add_hemisphere_cap_y(
                f"source_bump_half_sphere_{bump_index}",
                radius=dims.bump_radius,
                location=location,
                axis_scale=0.9,
                material=shell_mat,
                segments=32,
                rings=8,
            )
        )
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

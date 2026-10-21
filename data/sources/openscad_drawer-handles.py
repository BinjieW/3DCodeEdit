from __future__ import annotations
from dataclasses import dataclass
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the drawer handle model."""





Point2D = tuple[float, float]


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.52


@dataclass(frozen=True)
class DrawerHandleSpec:
    handle_length: float = 100.0
    handle_height: float = 30.0
    handle_thickness: float = 10.0
    handle_proportion: float = 2.0
    handle_bent_radius: float = 10.0
    base_diameter: float = 18.0
    base_height: float = 3.0
    base_proportion: float = 1.5
    filet_radius: float = 3.0
    hole_diameter: float = 3.0
    hole_depth: float = 15.0
    bend_segments: int = 30
    profile_segments: int = 28

    @property
    def handle_radius(self) -> float:
        return self.handle_thickness / 2.0

    @property
    def handle_extra_width(self) -> float:
        return max(0.0, self.handle_thickness * (self.handle_proportion - 1.0))

    @property
    def handle_half_y(self) -> float:
        return self.handle_extra_width / 2.0

    @property
    def base_radius(self) -> float:
        return self.base_diameter / 2.0

    @property
    def base_half_y(self) -> float:
        return max(0.0, self.base_diameter * (self.base_proportion - 1.0) / 2.0)

    @property
    def riser_x(self) -> float:
        return self.handle_length / 2.0

    @property
    def vertical_top_z(self) -> float:
        return self.handle_height - self.handle_bent_radius

    @property
    def straight_bar_end_x(self) -> float:
        return self.riser_x - self.handle_bent_radius


OPENSCAD_YELLOW = MaterialSpec("openscad_preview_yellow", (1.0, 0.82, 0.03, 1.0))
FILET_GREEN = MaterialSpec("openscad_preview_filet_green", (0.54, 0.78, 0.26, 1.0))


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def get_material(spec: MaterialSpec) -> bpy.types.Material:
    material = bpy.data.materials.get(spec.name) or bpy.data.materials.new(spec.name)
    material.diffuse_color = spec.color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = spec.color
        bsdf.inputs["Roughness"].default_value = spec.roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = 0.0
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.35
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.35
    return material


def assign_material(obj: bpy.types.Object, spec: MaterialSpec) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(get_material(spec))
    return obj


def make_mesh_object(
    name: str,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, ...]],
    material: MaterialSpec,
    smooth: bool,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign_material(obj, material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = smooth
    recalculate_normals(obj)
    if smooth:
        add_weighted_normals(obj)
    return obj


def recalculate_normals(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return obj


def add_weighted_normals(obj: bpy.types.Object) -> bpy.types.Object:
    modifier = obj.modifiers.new("weighted source-like normals", "WEIGHTED_NORMAL")
    modifier.keep_sharp = True
    modifier.weight = 50
    return obj


def stadium_outline(radius: float, half_y: float, segments_per_arc: int) -> list[Point2D]:
    """Return a rounded capsule outline in local (transverse, Y) coordinates."""
    segments = max(6, segments_per_arc)
    points: list[Point2D] = []
    for index in range(segments + 1):
        angle = index * math.pi / segments
        points.append((radius * math.cos(angle), half_y + radius * math.sin(angle)))
    for index in range(segments + 1):
        angle = math.pi + index * math.pi / segments
        points.append((radius * math.cos(angle), -half_y + radius * math.sin(angle)))
    return points


def transform_swept_profile(
    center_x: float,
    center_z: float,
    angle: float,
    profile: list[Point2D],
) -> list[tuple[float, float, float]]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return [
        (center_x + transverse * cos_a, y, center_z + transverse * sin_a)
        for transverse, y in profile
    ]


def add_swept_capsule_handle(
    name: str,
    spec: DrawerHandleSpec,
    material: MaterialSpec = OPENSCAD_YELLOW,
) -> bpy.types.Object:
    """Build the source-style handle by lofting the flattened capsule slice path."""
    profile = stadium_outline(spec.handle_radius, spec.handle_half_y, spec.profile_segments)
    stations: list[tuple[float, float, float]] = []

    vertical_segments = 5
    for index in range(vertical_segments + 1):
        z = spec.base_height + (spec.vertical_top_z - spec.base_height) * index / vertical_segments
        stations.append((-spec.riser_x, z, math.pi))

    for index in range(1, spec.bend_segments + 1):
        phi = (math.pi / 2.0) * index / spec.bend_segments
        x = -spec.riser_x + spec.handle_bent_radius * (1.0 - math.cos(phi))
        z = spec.vertical_top_z + spec.handle_bent_radius * math.sin(phi)
        stations.append((x, z, math.pi - phi))

    straight_segments = 24
    left_bar_x = -spec.straight_bar_end_x
    right_bar_x = spec.straight_bar_end_x
    for index in range(1, straight_segments + 1):
        x = left_bar_x + (right_bar_x - left_bar_x) * index / straight_segments
        stations.append((x, spec.handle_height, math.pi / 2.0))

    for index in range(1, spec.bend_segments + 1):
        phi = (math.pi / 2.0) * (1.0 - index / spec.bend_segments)
        x = spec.riser_x - spec.handle_bent_radius * (1.0 - math.cos(phi))
        z = spec.vertical_top_z + spec.handle_bent_radius * math.sin(phi)
        stations.append((x, z, phi))

    for index in range(1, vertical_segments + 1):
        z = spec.vertical_top_z - (spec.vertical_top_z - spec.base_height) * index / vertical_segments
        stations.append((spec.riser_x, z, 0.0))

    ring_size = len(profile)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for center_x, center_z, angle in stations:
        vertices.extend(transform_swept_profile(center_x, center_z, angle, profile))

    for station_index in range(len(stations) - 1):
        offset = station_index * ring_size
        next_offset = (station_index + 1) * ring_size
        for point_index in range(ring_size):
            next_point = (point_index + 1) % ring_size
            faces.append(
                (
                    offset + point_index,
                    next_offset + point_index,
                    next_offset + next_point,
                    offset + next_point,
                )
            )

    faces.append(tuple(reversed(range(ring_size))))
    end_offset = (len(stations) - 1) * ring_size
    faces.append(tuple(end_offset + index for index in range(ring_size)))

    return make_mesh_object(name, vertices, faces, material, smooth=True)


def add_capsule_prism(
    name: str,
    center: tuple[float, float],
    radius: float,
    half_y: float,
    z_min: float,
    height: float,
    material: MaterialSpec = OPENSCAD_YELLOW,
    segments_per_arc: int = 45,
) -> bpy.types.Object:
    outline = stadium_outline(radius, half_y, segments_per_arc)
    count = len(outline)
    cx, cy = center
    bottom = [(cx + x, cy + y, z_min) for x, y in outline]
    top = [(cx + x, cy + y, z_min + height) for x, y in outline]
    vertices = bottom + top
    faces: list[tuple[int, ...]] = [
        tuple(reversed(range(count))),
        tuple(range(count, count * 2)),
    ]
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, next_index + count, index + count))
    return make_mesh_object(name, vertices, faces, material, smooth=False)


def cut_vertical_screw_holes(
    targets: list[bpy.types.Object],
    spec: DrawerHandleSpec,
) -> list[bpy.types.Object]:
    """Cut the source screw holes through the bases and lower risers."""
    for x in (-spec.riser_x, spec.riser_x):
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64,
            radius=spec.hole_diameter / 2.0,
            depth=spec.hole_depth + 0.4,
            location=(x, 0.0, spec.hole_depth / 2.0 - 0.1),
        )
        cutter = bpy.context.object
        cutter.name = f"screw_hole_cutter_x_{x:g}"
        for target in targets:
            apply_difference(target, cutter)
        bpy.data.objects.remove(cutter, do_unlink=True)
    return targets


def apply_difference(target: bpy.types.Object, cutter: bpy.types.Object) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    had_weighted_normals = any(modifier.type == "WEIGHTED_NORMAL" for modifier in target.modifiers)
    for modifier in list(target.modifiers):
        if modifier.type == "WEIGHTED_NORMAL":
            target.modifiers.remove(modifier)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    modifier = target.modifiers.new(f"subtract_{cutter.name}", "BOOLEAN")
    modifier.operation = "DIFFERENCE"
    modifier.object = cutter
    if hasattr(modifier, "solver"):
        modifier.solver = "EXACT"
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    target.select_set(False)
    recalculate_normals(target)
    if had_weighted_normals:
        add_weighted_normals(target)
    return target


def add_base_plate(
    name: str,
    x: float,
    spec: DrawerHandleSpec,
    material: MaterialSpec = OPENSCAD_YELLOW,
) -> bpy.types.Object:
    base = add_capsule_prism(
        name=name,
        center=(x, 0.0),
        radius=spec.base_radius,
        half_y=spec.base_half_y,
        z_min=0.0,
        height=spec.base_height,
        material=material,
    )
    return base


def add_fillet_ring(
    name: str,
    x: float,
    spec: DrawerHandleSpec,
    material: MaterialSpec = FILET_GREEN,
) -> bpy.types.Object:
    """Approximate the SCAD filet modules as a rounded capsule skirt around a riser."""
    profile_count = spec.profile_segments
    fillet_steps = 8
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    for step in range(fillet_steps + 1):
        theta = (math.pi / 2.0) * step / fillet_steps
        offset = spec.filet_radius * math.sin(theta)
        z = spec.base_height + spec.filet_radius * math.cos(theta)
        outline = stadium_outline(
            spec.handle_radius + offset,
            spec.handle_half_y,
            profile_count,
        )
        vertices.extend((x + local_x, local_y, z) for local_x, local_y in outline)

    ring_size = len(stadium_outline(spec.handle_radius, spec.handle_half_y, profile_count))
    for step in range(fillet_steps):
        offset = step * ring_size
        next_offset = (step + 1) * ring_size
        for index in range(ring_size):
            next_index = (index + 1) % ring_size
            faces.append((offset + index, next_offset + index, next_offset + next_index, offset + next_index))

    return make_mesh_object(name, vertices, faces, material, smooth=True)


# ============================================================
# Procedural assembly
# ============================================================

"""Readable assembly for the drawer handle model."""





def build_model() -> list[bpy.types.Object]:
    """Compose the model from source-equivalent dimensions."""
    spec = DrawerHandleSpec()
    objects: list[bpy.types.Object] = []

    left_base = add_base_plate("left_elongated_base_with_screw_hole", -spec.riser_x, spec)
    right_base = add_base_plate("right_elongated_base_with_screw_hole", spec.riser_x, spec)
    objects.extend([left_base, right_base])

    objects.append(add_fillet_ring("left_rounded_base_filet", -spec.riser_x, spec))
    objects.append(add_fillet_ring("right_rounded_base_filet", spec.riser_x, spec))

    handle_body = add_swept_capsule_handle("flattened_capsule_handle_body", spec)
    objects.append(handle_body)

    cut_vertical_screw_holes([left_base, right_base, handle_body], spec)
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

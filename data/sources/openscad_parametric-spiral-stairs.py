from __future__ import annotations
from dataclasses import dataclass
import bmesh
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for this iteration."""





def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(name: str, rgba: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
    material.diffuse_color = rgba
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = 0.55
        bsdf.inputs["Alpha"].default_value = rgba[3]
    material.use_nodes = True
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)
    if hasattr(obj.data, "polygons"):
        for polygon in obj.data.polygons:
            polygon.material_index = 0


def _finish_mesh_object(
    name: str,
    mesh: bpy.types.Mesh,
    material: bpy.types.Material | None = None,
    shade_smooth: bool = False,
) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material is not None:
        assign_material(obj, material)
    if shade_smooth:
        for polygon in mesh.polygons:
            polygon.use_smooth = True
    return obj


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def add_cylinder(
    name: str,
    diameter: float,
    height: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
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
    return obj


def add_column(
    name: str,
    radius: float,
    height: float,
    material: bpy.types.Material,
    vertices: int = 48,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=height,
        location=(0.0, 0.0, height / 2.0),
    )
    obj = bpy.context.object
    obj.name = name
    assign_material(obj, material)
    bpy.ops.object.shade_smooth()
    return obj


def add_sector_prism(
    name: str,
    radius: float,
    angle_degrees: float,
    height: float,
    material: bpy.types.Material,
    segments: int = 24,
    angular_offset_degrees: float = -90.0,
) -> bpy.types.Object:
    """Create the SCAD step footprint: triangle(y=r) intersected with cylinder(r=r).

    The OpenSCAD triangle apex sits at the origin and the cylinder clips the far
    corners, leaving a circular sector bounded by two radial sides and an arc.
    """

    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    half_height = height / 2.0
    angle_radians = math.radians(angle_degrees)
    offset_radians = math.radians(angular_offset_degrees)

    center_bottom = bm.verts.new((0.0, 0.0, -half_height))
    center_top = bm.verts.new((0.0, 0.0, half_height))

    bottom_ring: list[bmesh.types.BMVert] = []
    top_ring: list[bmesh.types.BMVert] = []
    for index in range(segments + 1):
        fraction = index / segments
        theta = offset_radians + fraction * angle_radians
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        bottom_ring.append(bm.verts.new((x, y, -half_height)))
        top_ring.append(bm.verts.new((x, y, half_height)))

    bm.verts.ensure_lookup_table()

    for index in range(segments):
        b0 = bottom_ring[index]
        b1 = bottom_ring[index + 1]
        t0 = top_ring[index]
        t1 = top_ring[index + 1]

        bm.faces.new((center_bottom, b0, b1))
        bm.faces.new((center_top, t1, t0))
        bm.faces.new((b0, b1, t1, t0))

    bm.faces.new((center_bottom, bottom_ring[0], top_ring[0], center_top))
    bm.faces.new((center_bottom, center_top, top_ring[-1], bottom_ring[-1]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    return _finish_mesh_object(name, mesh, material=material, shade_smooth=False)


# ============================================================
# Procedural assembly
# ============================================================

"""Assembly logic for the current model version."""






@dataclass(frozen=True)
class SpiralStairSpec:
    revolutions: float = 3.5
    radius: float = 100.0
    segments_per_turn: int = 13
    step_height: float = 40.0
    column_radius: float = 10.0
    sector_segments: int = 24

    @property
    def tread_count(self) -> int:
        return int(self.revolutions * self.segments_per_turn)

    @property
    def step_angle(self) -> float:
        return 360.0 / self.segments_per_turn

    @property
    def step_offset(self) -> float:
        return (self.step_height + 0.01) / 2.0

    @property
    def column_height(self) -> float:
        return self.revolutions * self.step_height * self.segments_per_turn / 2.0 + 0.01 * (
            self.revolutions * self.segments_per_turn
        )


def _materials() -> dict[str, bpy.types.Material]:
    return {
        "red": make_material("spiral_step_red", (0.92, 0.05, 0.05, 1.0)),
        "orange": make_material("spiral_step_orange", (0.96, 0.58, 0.02, 1.0)),
        "yellow": make_material("spiral_step_yellow", (0.95, 0.92, 0.02, 1.0)),
        "green": make_material("spiral_step_green", (0.02, 0.55, 0.05, 1.0)),
        "blue": make_material("spiral_step_blue", (0.04, 0.09, 0.92, 1.0)),
        "indigo": make_material("spiral_step_indigo", (0.30, 0.03, 0.55, 1.0)),
        "violet": make_material("spiral_step_violet", (0.76, 0.48, 0.88, 1.0)),
        "silver": make_material("spiral_column_silver", (0.72, 0.72, 0.74, 1.0)),
    }


def _step_material_name(index: int) -> str:
    palette = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
    return palette[index % len(palette)]


def build_model(spec: SpiralStairSpec = SpiralStairSpec()) -> list[bpy.types.Object]:
    """Build the spiral staircase as repeated colored sector treads and a center column."""

    materials = _materials()
    objects: list[bpy.types.Object] = []

    for index in range(spec.tread_count):
        step = add_sector_prism(
            name=f"spiral_tread_{index + 1:02d}",
            radius=spec.radius,
            angle_degrees=spec.step_angle,
            height=spec.step_height,
            material=materials[_step_material_name(index)],
            segments=spec.sector_segments,
            angular_offset_degrees=-90.0,
        )
        step.rotation_euler[2] = math.radians(spec.step_angle * index)
        step.location.z = spec.step_offset * index + spec.step_height / 2.0
        objects.append(step)

    column = add_column(
        name="central_column",
        radius=spec.column_radius,
        height=spec.column_height,
        material=materials["silver"],
    )
    objects.append(column)

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

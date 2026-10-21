from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the drawer-pull model."""





Vector3 = tuple[float, float, float]
Profile = list[tuple[float, float]]


@dataclass(frozen=True)
class PullDimensions:
    stem_height: float = 20.0
    stem_radius_bottom: float = 9.0
    stem_taper: float = 0.8
    screw_radius: float = 1.875
    screw_depth: float = 6.0
    ornament_radius: float = 15.0
    ornament_vertical_scale: float = 0.2
    ornament_indent_offset: float = 14.0

    @property
    def stem_radius_top(self) -> float:
        return self.stem_radius_bottom * self.stem_taper

    @property
    def cap_center_z(self) -> float:
        contact_z = math.sqrt(self.ornament_radius**2 - self.stem_radius_top**2)
        return self.stem_height + self.ornament_vertical_scale * contact_z

    @property
    def cap_thickness(self) -> float:
        return 2.0 * self.ornament_radius * self.ornament_vertical_scale


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.55,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = roughness
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> bpy.types.Object:
    obj.data.materials.append(material)
    return obj


def shade_smooth(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)
    return obj


def add_cylinder(
    name: str,
    radius: float,
    height: float,
    location: Vector3,
    vertices: int = 48,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=height,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    if material:
        assign_material(obj, material)
    return obj


def add_revolved_profile(
    name: str,
    profile: Iterable[tuple[float, float]],
    segments: int = 72,
    material: bpy.types.Material | None = None,
    smooth: bool = False,
) -> bpy.types.Object:
    points = list(profile)
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []

    for segment in range(segments):
        angle = 2.0 * math.pi * segment / segments
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        for radius, z in points:
            verts.append((radius * cos_a, radius * sin_a, z))

    profile_count = len(points)
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        base = segment * profile_count
        next_base = next_segment * profile_count
        for idx in range(profile_count - 1):
            faces.append((base + idx, next_base + idx, next_base + idx + 1, base + idx + 1))

    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material:
        assign_material(obj, material)
    if smooth:
        shade_smooth(obj)
    return obj


def profile_from_sphere(
    radius: float,
    vertical_scale: float,
    center_z: float,
    local_z_values: Iterable[float],
) -> Profile:
    return [
        (math.sqrt(max(radius**2 - local_z**2, 0.0)), center_z + vertical_scale * local_z)
        for local_z in local_z_values
    ]


def add_chamfered_tapered_stem(
    name: str,
    dimensions: PullDimensions,
    material: bpy.types.Material,
    segments: int = 48,
) -> bpy.types.Object:
    bottom = dimensions.stem_radius_bottom
    top = dimensions.stem_radius_top
    height = dimensions.stem_height
    bore = dimensions.screw_radius
    bore_depth = min(dimensions.screw_depth, height - 0.5)

    profile: Profile = [
        (bore, 0.0),
        (bottom - 1.1, 0.0),
        (bottom - 0.15, 0.35),
        (bottom, 1.0),
        (top, height - 0.8),
        (top - 0.15, height - 0.25),
        (top - 1.0, height),
        (0.0, height),
        (0.0, bore_depth),
        (bore, bore_depth),
        (bore, 0.0),
    ]
    return add_revolved_profile(name, profile, segments=segments, material=material, smooth=False)


def add_ufo_ornament(
    name: str,
    dimensions: PullDimensions,
    material: bpy.types.Material,
    segments: int = 36,
) -> bpy.types.Object:
    radius = dimensions.ornament_radius
    scale_z = dimensions.ornament_vertical_scale
    center_z = dimensions.cap_center_z

    # The source is a flattened sphere with a shifted sphere removed from the
    # top and an inner "bump" sphere restored. Their intersection occurs at
    # local z ~= 7, which creates the visible circular crease.
    outer_until_groove = [-15.0, -14.0, -12.0, -9.5, -7.0, -4.0, -1.0, 2.0, 4.5, 6.0, 7.0]
    profile = profile_from_sphere(radius, scale_z, center_z, outer_until_groove)

    bump_radius = radius - 1.0
    bump_from_groove = [7.0, 8.5, 10.0, 11.4, 12.6, 13.5, 14.0]
    profile.extend(profile_from_sphere(bump_radius, scale_z, center_z, bump_from_groove))
    return add_revolved_profile(name, profile, segments=segments, material=material, smooth=False)


def add_top_recess_shadow(
    name: str,
    dimensions: PullDimensions,
    material: bpy.types.Material,
    segments: int = 36,
) -> bpy.types.Object:
    center_z = dimensions.cap_center_z
    scale_z = dimensions.ornament_vertical_scale
    profile: Profile = [
        (12.05, center_z + scale_z * 7.02),
        (13.25, center_z + scale_z * 7.02),
        (13.00, center_z + scale_z * 7.20),
        (12.25, center_z + scale_z * 7.20),
    ]
    return add_revolved_profile(name, profile, segments=segments, material=material, smooth=False)


def add_screw_bore_marker(
    name: str,
    dimensions: PullDimensions,
    material: bpy.types.Material,
    segments: int = 24,
) -> bpy.types.Object:
    height = 0.045
    ceiling_z = min(dimensions.screw_depth, dimensions.stem_height - 0.5)
    obj = add_cylinder(
        name=name,
        radius=dimensions.screw_radius * 0.97,
        height=height,
        location=(0.0, 0.0, ceiling_z + height / 2.0),
        vertices=segments,
        material=material,
    )
    return shade_smooth(obj)


def add_bottom_bore_wall(
    name: str,
    dimensions: PullDimensions,
    material: bpy.types.Material,
    segments: int = 24,
) -> bpy.types.Object:
    bore = dimensions.screw_radius
    profile: Profile = [
        (bore, -0.02),
        (bore + 0.08, -0.02),
        (bore + 0.08, 0.55),
        (bore, 0.55),
    ]
    return add_revolved_profile(name, profile, segments=segments, material=material, smooth=True)


# ============================================================
# Procedural assembly
# ============================================================

"""Assembly logic for the default UFO drawer pull."""





def build_model() -> list[bpy.types.Object]:
    dims = PullDimensions()
    yellow = make_material("openscad_warm_yellow", (1.0, 0.82, 0.02, 1.0), roughness=0.5)
    shadow_green = make_material("bore_shadow_green", (0.34, 0.46, 0.18, 1.0), roughness=0.75)
    groove_shadow = make_material("top_recess_shadow", (0.50, 0.56, 0.18, 1.0), roughness=0.75)
    bore_depth = make_material("blind_bore_floor", (0.18, 0.22, 0.12, 1.0), roughness=0.8)

    objects = [
        add_chamfered_tapered_stem("tapered_chamfered_stem", dims, yellow),
        add_ufo_ornament("flattened_ufo_cap_with_top_bump_recess", dims, yellow),
        add_top_recess_shadow("visible_top_annular_recess", dims, groove_shadow),
        add_bottom_bore_wall("thin_visible_screw_bore_wall", dims, shadow_green),
        add_screw_bore_marker("flat_blind_screw_bore_floor", dims, bore_depth),
    ]
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

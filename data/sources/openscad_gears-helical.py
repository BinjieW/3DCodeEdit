from __future__ import annotations
from dataclasses import dataclass
from mathutils import Matrix, Vector
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Low-level geometry helpers for the helical gear model."""





TAU = math.tau
SCAD_FA_DEG = 12.0
SCAD_FS = 2.0


@dataclass(frozen=True)
class GearParameters:
    """Source OpenSCAD parameters, normalized into reusable dimensions."""

    teeth_num: int = 30
    gear_height: float = 7.0
    pitch_diameter: float = 45.0
    shaft_diameter: float = 5.0
    addendum: float = 2.0
    dedendum: float = 2.0
    tooth_width: float = 2.0
    helix_angle_deg: float = 30.0
    circle_num: int = 7
    circle_diameter: float = 7.0
    extrude_out_height: float = 3.0
    extrude_out_width: float = 7.0
    extrude_in_height: float = 3.0

    @property
    def total_height(self) -> float:
        return self.gear_height * 2.0

    @property
    def root_radius(self) -> float:
        return self.pitch_diameter / 2.0 - self.dedendum

    @property
    def outer_radius(self) -> float:
        return self.pitch_diameter / 2.0 + self.addendum

    @property
    def tooth_center_radius(self) -> float:
        # Mirrors helical(): translate by rootRad - dedendum before rotating each tooth.
        return self.root_radius - self.dedendum

    @property
    def hole_ring_radius(self) -> float:
        # Mirrors circle(): rootRad / 2 + shaftDiam / 2.
        return self.root_radius / 2.0 + self.shaft_diameter / 2.0

    @property
    def hub_radius(self) -> float:
        return (self.extrude_out_width + self.shaft_diameter) / 2.0

    @property
    def top_recess_radius(self) -> float:
        return self.pitch_diameter / 2.0 - self.dedendum * 2.0

    @property
    def tooth_height(self) -> float:
        return self.addendum + self.dedendum * 2.0

    @property
    def tooth_blank_depth(self) -> float:
        return self.gear_height * 1.5

    @property
    def lower_tooth_z_offset(self) -> float:
        return -self.gear_height + 0.1

    @property
    def root_body_z_min(self) -> float:
        # circle() calls solid(gearHeight - 0.1 * doubleHelical, pitchDiam).
        return -(self.gear_height - 0.1) + 0.1

    @property
    def root_body_height(self) -> float:
        height_gear = self.gear_height - 0.1
        return height_gear + (height_gear - 0.1)

    @property
    def hub_z_min(self) -> float:
        return -self.gear_height + 0.1

    @property
    def hub_height(self) -> float:
        return self.gear_height - self.extrude_in_height + self.extrude_out_height + self.gear_height - 0.1

    @property
    def recess_z_min(self) -> float:
        return self.gear_height - self.extrude_in_height + 0.1

    @property
    def helical_root_cut_z_min(self) -> float:
        # helical() is translated upward by gearHeight / 2 in gear().
        return -self.gear_height / 2.0 - 1.0 + self.gear_height / 2.0

    @property
    def helical_root_cut_height(self) -> float:
        return self.gear_height + 2.0

    @property
    def shaft_cut_z_min(self) -> float:
        return -self.gear_height

    @property
    def shaft_cut_height(self) -> float:
        # Matches the source expression, including romeHeight even though rome=0.
        return self.gear_height + self.extrude_out_height + self.gear_height + self.gear_height


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.55,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
    return material


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


def scad_fragment_count(radius: float, fn: int = 0, fa: float = SCAD_FA_DEG, fs: float = SCAD_FS) -> int:
    """Approximate OpenSCAD's default cylinder fragment count."""

    if fn > 0:
        return max(fn, 3)
    return int(math.ceil(max(min(360.0 / fa, radius * TAU / fs), 5.0)))


def add_scad_cylinder(
    name: str,
    radius: float,
    height: float,
    z_min: float,
    material: bpy.types.Material | None = None,
    vertices: int | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices if vertices is not None else scad_fragment_count(radius),
        radius=radius,
        depth=height,
        location=(0.0, 0.0, z_min + height / 2.0),
    )
    obj = bpy.context.object
    obj.name = name
    if material:
        obj.data.materials.append(material)
    return shade_flat(obj)


def add_cylinder(
    name: str,
    radius: float,
    height: float,
    z_min: float,
    material: bpy.types.Material | None = None,
    vertices: int = 96,
) -> bpy.types.Object:
    return add_scad_cylinder(name, radius, height, z_min, material, vertices)


def add_cube(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    rotation: tuple[float, float, float],
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = size
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material:
        obj.data.materials.append(material)
    return shade_flat(obj)


def apply_boolean_difference(
    target: bpy.types.Object,
    cutters: list[bpy.types.Object],
    name: str,
    keep_cutters: bool = False,
) -> bpy.types.Object:
    """Apply exact difference booleans from each cutter to target."""

    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    for index, cutter in enumerate(cutters):
        modifier = target.modifiers.new(f"{name}_{index:02d}", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        modifier.solver = "EXACT"
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        if not keep_cutters:
            bpy.data.objects.remove(cutter, do_unlink=True)
    target.select_set(False)
    return target


def _apply_all_transforms(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.select_set(False)
    return obj


def _mesh_data_copy(obj: bpy.types.Object) -> tuple[list[Vector], list[tuple[int, ...]]]:
    vertices = [vertex.co.copy() for vertex in obj.data.vertices]
    faces = [tuple(vertex for vertex in polygon.vertices) for polygon in obj.data.polygons]
    return vertices, faces


def _make_tooth_half_blank(
    name: str,
    params: GearParameters,
    direction: int,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Build rotateTooth(): rotated 2 x 6 x 10.5 box clipped by upper/lower boxes."""

    y_angle = math.radians(direction * -params.helix_angle_deg)
    blank = add_cube(
        name,
        size=(params.tooth_width, params.tooth_height, params.tooth_blank_depth),
        location=(0.0, 0.0, 0.0),
        rotation=(0.0, y_angle, 0.0),
        material=material,
    )
    _apply_all_transforms(blank)

    cutter_height = params.dedendum + params.gear_height / 2.0
    cutter_size = (params.gear_height, params.tooth_height + 0.5, cutter_height)
    upper = add_cube(
        f"{name}_upper_cutter",
        size=cutter_size,
        location=(0.0, 0.0, params.gear_height * 0.75 + params.dedendum / 2.0),
        rotation=(0.0, 0.0, 0.0),
    )
    lower = add_cube(
        f"{name}_lower_cutter",
        size=cutter_size,
        location=(0.0, 0.0, -params.gear_height * 0.75 - params.dedendum / 2.0),
        rotation=(0.0, 0.0, 0.0),
    )
    _apply_all_transforms(upper)
    _apply_all_transforms(lower)
    apply_boolean_difference(blank, [upper, lower], f"{name}_crop")
    return blank


def _tooth_half_template(
    params: GearParameters,
    direction: int,
    material: bpy.types.Material | None = None,
) -> tuple[list[Vector], list[tuple[int, ...]]]:
    blank = _make_tooth_half_blank(f"tooth_half_template_{direction:+d}", params, direction, material)
    vertices, faces = _mesh_data_copy(blank)
    bpy.data.objects.remove(blank, do_unlink=True)
    return vertices, faces


def create_scad_box_teeth(
    params: GearParameters,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Create the complete herringbone rim from SCAD-like cropped box teeth."""

    upper_vertices, upper_faces = _tooth_half_template(params, direction=1, material=material)
    lower_vertices, lower_faces = _tooth_half_template(params, direction=-1, material=material)
    all_vertices: list[tuple[float, float, float]] = []
    all_faces: list[tuple[int, ...]] = []

    def make_half_object(
        name: str,
        template_vertices: list[Vector],
        template_faces: list[tuple[int, ...]],
        matrix: Matrix,
    ) -> bpy.types.Object:
        vertices: list[tuple[float, float, float]] = []
        for vertex in template_vertices:
            transformed = matrix @ vertex
            vertices.append((transformed.x, transformed.y, transformed.z))
        mesh = bpy.data.meshes.new(f"{name}Mesh")
        mesh.from_pydata(vertices, [], template_faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        return obj

    def append_clipped_half(obj: bpy.types.Object) -> None:
        cutter = add_scad_cylinder(
            f"{obj.name}_root_radius_cutter",
            radius=params.root_radius,
            height=params.helical_root_cut_height,
            z_min=params.helical_root_cut_z_min,
        )
        apply_boolean_difference(obj, [cutter], f"{obj.name}_inner_root_cut")
        base_index = len(all_vertices)
        all_vertices.extend((vertex.co.x, vertex.co.y, vertex.co.z) for vertex in obj.data.vertices)
        all_faces.extend(tuple(base_index + index for index in polygon.vertices) for polygon in obj.data.polygons)
        bpy.data.objects.remove(obj, do_unlink=True)

    gear_lift = Matrix.Translation((0.0, 0.0, params.gear_height / 2.0))
    upper_shift = Matrix.Translation((0.0, params.tooth_height / 2.0, 0.0))
    lower_shift = Matrix.Translation((0.0, params.tooth_height / 2.0, params.lower_tooth_z_offset))

    for index in range(1, params.teeth_num + 1):
        angle = TAU * index / params.teeth_num
        radial = Matrix.Translation(
            (
                math.sin(angle) * params.tooth_center_radius,
                math.cos(angle) * params.tooth_center_radius,
                0.0,
            )
        )
        orient = Matrix.Rotation(-angle, 4, "Z")
        append_clipped_half(
            make_half_object(
                f"upper_tooth_half_{index:02d}",
                upper_vertices,
                upper_faces,
                gear_lift @ radial @ orient @ upper_shift,
            )
        )
        append_clipped_half(
            make_half_object(
                f"lower_tooth_half_{index:02d}",
                lower_vertices,
                lower_faces,
                gear_lift @ radial @ orient @ lower_shift,
            )
        )

    mesh = bpy.data.meshes.new("scad_box_herringbone_teethMesh")
    mesh.from_pydata(all_vertices, [], all_faces)
    mesh.update()
    teeth = bpy.data.objects.new("scad_box_herringbone_teeth", mesh)
    bpy.context.scene.collection.objects.link(teeth)
    if material:
        teeth.data.materials.append(material)
    return shade_flat(teeth)


def create_radial_hole_cutters(
    params: GearParameters,
    z_min: float,
    height: float,
    vertices: int | None = None,
) -> list[bpy.types.Object]:
    cutters = []
    for index in range(params.circle_num):
        angle = TAU * index / params.circle_num
        cutter = add_scad_cylinder(
            f"lightening_hole_cutter_{index + 1:02d}",
            radius=params.circle_diameter / 2.0,
            height=height,
            z_min=z_min,
            vertices=vertices,
        )
        cutter.location.x = math.sin(angle) * params.hole_ring_radius
        cutter.location.y = math.cos(angle) * params.hole_ring_radius
        cutters.append(cutter)
    return cutters


def create_annular_recess_cutter(params: GearParameters) -> bpy.types.Object:
    """Build the source extrudeIn cutter: outer cylinder minus hub keepout."""

    outer = add_scad_cylinder(
        "top_annular_recess_outer_cutter",
        radius=params.top_recess_radius,
        height=params.extrude_in_height,
        z_min=params.recess_z_min,
    )
    inner = add_scad_cylinder(
        "top_annular_recess_inner_keepout",
        radius=params.hub_radius,
        height=params.gear_height + params.extrude_out_height,
        z_min=0.0,
    )
    apply_boolean_difference(outer, [inner], "annular_recess_keepout")
    return outer


def create_shaft_bore_cutter(params: GearParameters, name: str = "shaft_bore_cutter") -> bpy.types.Object:
    return add_scad_cylinder(
        name,
        radius=params.shaft_diameter / 2.0,
        height=params.shaft_cut_height,
        z_min=params.shaft_cut_z_min,
    )


def assign_recess_floor_material(
    obj: bpy.types.Object,
    material: bpy.types.Material,
    params: GearParameters,
    z_tolerance: float = 0.03,
) -> None:
    """Color the real recessed top faces without adding overlay geometry."""

    if material.name not in [slot.material.name for slot in obj.material_slots if slot.material]:
        obj.data.materials.append(material)
    material_index = list(obj.data.materials).index(material)

    vertices = obj.data.vertices
    for polygon in obj.data.polygons:
        if polygon.normal.z < 0.8:
            continue
        coords = [obj.matrix_world @ vertices[index].co for index in polygon.vertices]
        center = sum(coords, Vector()) / len(coords)
        radius = math.hypot(center.x, center.y)
        if (
            abs(center.z - params.recess_z_min) <= z_tolerance
            and params.hub_radius - 0.08 <= radius <= params.top_recess_radius + 0.08
        ):
            polygon.material_index = material_index


def create_tube(
    name: str,
    outer_radius: float,
    inner_radius: float,
    height: float,
    z_min: float,
    material: bpy.types.Material | None = None,
    vertices: int | None = None,
) -> bpy.types.Object:
    outer = add_scad_cylinder(name, outer_radius, height, z_min, material, vertices)
    inner = add_scad_cylinder(f"{name}_inner_cutter", inner_radius, height + 0.4, z_min - 0.2, vertices=vertices)
    apply_boolean_difference(outer, [inner], f"{name}_hole")
    return outer


# ============================================================
# Procedural assembly
# ============================================================

"""High-level assembly for the double-helical gear model."""





def build_model() -> list[bpy.types.Object]:
    """Compose the gear from source-like primitives and boolean operations."""

    params = GearParameters()

    brass = make_material("warm_yellow_gear", (0.95, 0.73, 0.08, 1.0))
    olive = make_material("green_recessed_faces", (0.47, 0.64, 0.22, 1.0))

    objects: list[bpy.types.Object] = []

    # Source circle(): a low-facet solid web with seven through-cylinders removed.
    root = add_scad_cylinder(
        "circle_center_root_web",
        radius=params.root_radius,
        height=params.root_body_height,
        z_min=params.root_body_z_min,
        material=brass,
    )
    hole_cutters = create_radial_hole_cutters(
        params,
        z_min=-params.gear_height,
        height=params.total_height,
    )
    apply_boolean_difference(root, hole_cutters, "circle_lightening_holes")
    objects.append(root)

    # Source extrudeOut: central boss merged into the gear body, later cut by shaft/recess.
    hub = add_scad_cylinder(
        "extrude_out_center_hub",
        radius=params.hub_radius,
        height=params.hub_height,
        z_min=params.hub_z_min,
        material=brass,
    )
    objects.append(hub)

    # Source tooth()/helical(): cropped rotated box halves, radially repeated and
    # inner-cut by the root cylinder. Keeping this as one mesh preserves the hard
    # rectangular facets without the previous polar wedge smoothing.
    teeth = create_scad_box_teeth(params, material=brass)
    objects.append(teeth)

    recess_cutters = [create_annular_recess_cutter(params), create_annular_recess_cutter(params)]
    shaft_cutters = [create_shaft_bore_cutter(params, "root_shaft_bore_cutter"), create_shaft_bore_cutter(params, "hub_shaft_bore_cutter")]

    apply_boolean_difference(root, [recess_cutters[0], shaft_cutters[0]], "root_final_cuts")
    apply_boolean_difference(hub, [recess_cutters[1], shaft_cutters[1]], "hub_final_cuts")
    assign_recess_floor_material(root, olive, params)

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

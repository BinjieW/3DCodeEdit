from __future__ import annotations
from dataclasses import dataclass
from mathutils import Euler, Matrix, Vector
import bmesh
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the bird-house model."""





@dataclass(frozen=True)
class BirdHouseSpec:
    """Source dimensions for the default Chickadee preset."""

    height_x: float = 203.2
    width_y: float = 101.6
    length_z: float = 101.6
    opening_diameter: float = 31.75
    opening_height_x: float = 152.4
    outer_wall_inset: float = 7.0
    flange_length_x: float = 80.0
    flange_depth_y: float = 50.8
    flange_height_z: float = 10.0
    perch_radius: float = 5.0
    perch_height: float = 50.0
    roof_thickness_x: float = 15.0
    roof_extra_height_z: float = 40.0
    roof_cut_angle_degrees: float = 7.0
    ventilation_radius: float = 5.0
    label_text: str = "TNH"
    label_size: float = 24.0
    label_extrude: float = 2.0

    @property
    def opening_radius(self) -> float:
        return self.opening_diameter / 2.0

    @property
    def body_min_x(self) -> float:
        return -self.height_x / 2.0 - 7.0

    @property
    def body_max_x(self) -> float:
        return self.body_min_x + self.height_x

    @property
    def body_min_y(self) -> float:
        return -self.width_y / 2.0

    @property
    def body_max_y(self) -> float:
        return self.width_y / 2.0

    @property
    def body_min_z(self) -> float:
        return 0.0

    @property
    def body_max_z(self) -> float:
        return self.length_z

    @property
    def source_to_world(self) -> Vector:
        return Vector((self.body_min_x, self.body_min_y, 0.0))

    def world(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        point = Vector((x, y, z)) + self.source_to_world
        return (point.x, point.y, point.z)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.56,
) -> bpy.types.Material:
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = color
        elif "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = color
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 0.25
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.28
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.28
    return material


def birdhouse_materials() -> dict[str, bpy.types.Material]:
    return {
        "turquoise": make_material("openscad_turquoise", (0.06, 0.78, 0.72, 1.0)),
        "turquoise_dark": make_material("turquoise_dark_cut_faces", (0.01, 0.16, 0.15, 1.0), roughness=0.72),
        "label": make_material("raised_label_turquoise", (0.05, 0.66, 0.61, 1.0)),
    }


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> bpy.types.Object:
    obj.data.materials.clear()
    obj.data.materials.append(material)
    if hasattr(obj.data, "polygons"):
        for polygon in obj.data.polygons:
            polygon.material_index = 0
    return obj


def recalculate_normals(obj: bpy.types.Object) -> bpy.types.Object:
    if obj.type != "MESH":
        return obj
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return obj


def shade_smooth(obj: bpy.types.Object, auto_smooth_angle: float | None = None) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)
    if auto_smooth_angle is not None:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        obj.data.update()
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
    if material is not None:
        assign_material(obj, material)
    return obj


def add_box_from_bounds(
    name: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    return add_box(
        name=name,
        size=(xmax - xmin, ymax - ymin, zmax - zmin),
        location=((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0),
        material=material,
    )


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
        end_fill_type="NGON",
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    if material is not None:
        assign_material(obj, material)
    if vertices > 24:
        shade_smooth(obj)
    return obj


def add_cylinder_y(
    name: str,
    radius: float,
    depth: float,
    location: tuple[float, float, float],
    vertices: int = 64,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    return add_cylinder(
        name=name,
        radius=radius,
        depth=depth,
        location=location,
        rotation=(math.pi / 2.0, 0.0, 0.0),
        vertices=vertices,
        material=material,
    )


def add_cylinder_z(
    name: str,
    radius: float,
    depth: float,
    location: tuple[float, float, float],
    vertices: int = 64,
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    return add_cylinder(name, radius, depth, location, vertices=vertices, material=material)


def apply_boolean(
    target: bpy.types.Object,
    cutter: bpy.types.Object,
    operation: str = "DIFFERENCE",
    keep_cutter: bool = False,
) -> bpy.types.Object:
    modifier = target.modifiers.new(f"{operation.lower()}_{cutter.name}", "BOOLEAN")
    modifier.operation = operation
    modifier.object = cutter
    if hasattr(modifier, "solver"):
        modifier.solver = "EXACT"
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    target.select_set(False)
    if not keep_cutter:
        bpy.data.objects.remove(cutter, do_unlink=True)
    recalculate_normals(target)
    return target


def join_objects(name: str, objects: list[bpy.types.Object]) -> bpy.types.Object:
    if not objects:
        raise ValueError("join_objects requires at least one object")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    joined = bpy.context.object
    joined.name = name
    joined.data.name = f"{name}_mesh"
    return recalculate_normals(joined)


def remove_doubles(obj: bpy.types.Object, distance: float = 0.0001) -> bpy.types.Object:
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=distance)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def _transform_point(point: tuple[float, float, float], matrix: Matrix) -> tuple[float, float, float]:
    transformed = matrix @ Vector(point)
    return (transformed.x, transformed.y, transformed.z)


def scad_rotation_matrix(rotation_degrees: tuple[float, float, float]) -> Matrix:
    return Euler(tuple(math.radians(value) for value in rotation_degrees), "XYZ").to_matrix().to_4x4()


def add_scad_box(
    name: str,
    spec: BirdHouseSpec,
    size: tuple[float, float, float],
    source_origin: tuple[float, float, float],
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    """Create an OpenSCAD-style cube: corner-origin, optionally rotated, then translated."""

    sx, sy, sz = size
    local_vertices = [
        (0.0, 0.0, 0.0),
        (sx, 0.0, 0.0),
        (sx, sy, 0.0),
        (0.0, sy, 0.0),
        (0.0, 0.0, sz),
        (sx, 0.0, sz),
        (sx, sy, sz),
        (0.0, sy, sz),
    ]
    transform = (
        Matrix.Translation(spec.source_to_world)
        @ Matrix.Translation(Vector(source_origin))
        @ scad_rotation_matrix(rotation_degrees)
    )
    vertices = [_transform_point(vertex, transform) for vertex in local_vertices]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material is not None:
        assign_material(obj, material)
    return recalculate_normals(obj)


def add_rotated_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    rotation: tuple[float, float, float],
    material: bpy.types.Material | None = None,
) -> bpy.types.Object:
    obj = add_box(name, size, location, material)
    obj.rotation_euler = rotation
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    obj.select_set(False)
    return obj


def add_outer_body_mesh(
    name: str,
    spec: BirdHouseSpec,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """Create the body and apply the SCAD trim cubes as boolean cuts."""

    body = add_box_from_bounds(
        name,
        spec.body_min_x,
        spec.body_max_x,
        spec.body_min_y,
        spec.body_max_y,
        spec.body_min_z,
        spec.body_max_z,
        material,
    )
    trim_cutters = [
        add_scad_box(
            "cutter_body_back_side_chamfer",
            spec,
            (spec.height_x + 20.0, 10.0, 10.0),
            (-20.0, spec.width_y, spec.length_z - 6.0),
            (45.0, 0.0, 0.0),
        ),
        add_scad_box(
            "cutter_body_front_side_chamfer",
            spec,
            (spec.height_x + 20.0, 10.0, 10.0),
            (-20.0, 0.0, spec.length_z - 6.0),
            (45.0, 0.0, 0.0),
        ),
        add_scad_box(
            "cutter_body_left_top_chamfer",
            spec,
            (10.0, spec.width_y + 20.0, 10.0),
            (-7.0, 0.0, spec.length_z + 1.0),
            (0.0, 45.0, 0.0),
        ),
    ]
    for cutter in trim_cutters:
        apply_boolean(body, cutter)
    return body


def add_hollow_body(
    name: str,
    spec: BirdHouseSpec,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """Boolean the SCAD-like hollow interior from the chamfered outer body."""

    body = add_outer_body_mesh(name, spec, material)
    inner = add_box_from_bounds(
        "cutter_body_hollow",
        spec.body_min_x + spec.outer_wall_inset,
        spec.body_min_x + spec.outer_wall_inset + spec.height_x * 0.9,
        spec.body_min_y + spec.outer_wall_inset,
        spec.body_min_y + spec.outer_wall_inset + spec.width_y * 0.9,
        spec.body_min_z + spec.outer_wall_inset,
        spec.body_min_z + spec.outer_wall_inset + spec.length_z * 0.9,
    )
    apply_boolean(body, inner)
    return body


def add_top_flange_mesh(
    name: str,
    spec: BirdHouseSpec,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """Create the raised entrance flange with the SCAD trim cuts."""

    flange = add_box_from_bounds(
        name,
        spec.body_min_x + spec.opening_height_x - 50.0,
        spec.body_min_x + spec.opening_height_x - 50.0 + spec.flange_length_x,
        spec.body_min_y + spec.width_y / 4.0,
        spec.body_min_y + spec.width_y * 3.0 / 4.0,
        spec.body_max_z,
        spec.body_max_z + spec.flange_height_z,
        material,
    )
    trim_cutters = [
        add_scad_box(
            "cutter_flange_right_chamfer",
            spec,
            (10.0, spec.width_y + 20.0, 10.0),
            (spec.opening_height_x + 25.0, 0.0, spec.length_z + 11.0),
            (0.0, 45.0, 0.0),
        ),
        add_scad_box(
            "cutter_flange_left_chamfer",
            spec,
            (10.0, spec.width_y + 20.0, 10.0),
            (spec.opening_height_x - 59.0, 0.0, spec.length_z + 11.0),
            (0.0, 45.0, 0.0),
        ),
        add_scad_box(
            "cutter_flange_front_chamfer",
            spec,
            (90.0, 10.0, 10.0),
            (spec.opening_height_x - 55.0, spec.width_y / 4.0, spec.length_z + 6.0),
            (45.0, 0.0, 0.0),
        ),
        add_scad_box(
            "cutter_flange_back_chamfer",
            spec,
            (90.0, 10.0, 10.0),
            (spec.opening_height_x - 55.0, spec.width_y * 3.0 / 4.0, spec.length_z + 6.0),
            (45.0, 0.0, 0.0),
        ),
    ]
    for cutter in trim_cutters:
        apply_boolean(flange, cutter)
    return flange


def add_roof_wall_mesh(
    name: str,
    spec: BirdHouseSpec,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """Build the rear roof wall using the source cube-minus-slanted-cube operation."""

    roof = add_scad_box(
        name,
        spec,
        (spec.roof_thickness_x, spec.width_y, spec.length_z + spec.roof_extra_height_z),
        (spec.height_x, 0.0, 0.0),
        material=material,
    )
    cutter = add_scad_box(
        "cutter_slanted_roof_plane",
        spec,
        (100.0, spec.width_y + 30.0, spec.length_z + spec.roof_extra_height_z),
        (spec.height_x + 22.0, -10.0, 10.0),
        (0.0, -spec.roof_cut_angle_degrees, 0.0),
    )
    return apply_boolean(roof, cutter)


def cut_entrance_and_vents(target: bpy.types.Object, spec: BirdHouseSpec) -> bpy.types.Object:
    entrance = add_cylinder_z(
        "cutter_entrance_opening",
        spec.opening_radius,
        48.0,
        spec.world(spec.opening_height_x, spec.width_y / 2.0, spec.length_z - 25.0 + 20.0),
        vertices=96,
    )
    apply_boolean(target, entrance)

    for index, dz in enumerate((20.0, 0.0, -20.0), start=1):
        cutter = add_cylinder_y(
            f"cutter_ventilation_hole_{index}",
            spec.ventilation_radius,
            300.0,
            spec.world(spec.height_x - 20.0, spec.width_y / 2.0, spec.length_z / 2.0 + dz),
            vertices=48,
        )
        apply_boolean(target, cutter)
    return target


def add_dark_hole_backers(spec: BirdHouseSpec, materials: dict[str, bpy.types.Material]) -> list[bpy.types.Object]:
    """Add tiny dark disks behind the openings so the holes read clearly in Cycles."""

    objects: list[bpy.types.Object] = []
    dark = materials["turquoise_dark"]
    objects.append(
        add_cylinder_z(
            "entrance_backer",
            radius=spec.opening_radius * 0.92,
            depth=2.0,
            location=spec.world(spec.opening_height_x + 0.6, spec.width_y / 2.0, spec.length_z - 5.5),
            vertices=96,
            material=dark,
        )
    )
    for index, dz in enumerate((20.0, 0.0, -20.0), start=1):
        objects.append(
            add_cylinder_y(
                f"vent_backer_{index}",
                radius=spec.ventilation_radius * 0.82,
                depth=2.0,
                location=spec.world(spec.height_x - 0.6, spec.width_y / 2.0, spec.length_z / 2.0 + dz),
                vertices=48,
                material=dark,
            )
        )
    return objects


def add_perch(spec: BirdHouseSpec, material: bpy.types.Material) -> bpy.types.Object:
    return add_cylinder_z(
        "vertical_perch",
        radius=spec.perch_radius,
        depth=spec.perch_height,
        location=spec.world(spec.opening_height_x - 40.0, spec.width_y / 2.0, spec.length_z + spec.perch_height / 2.0),
        vertices=48,
        material=material,
    )


def add_raised_text_label(spec: BirdHouseSpec, material: bpy.types.Material) -> list[bpy.types.Object]:
    """Approximate Write.scad's top label with Blender text converted to mesh."""

    objects: list[bpy.types.Object] = []
    if not spec.label_text:
        return objects

    bpy.ops.object.text_add(
        location=spec.world(15.0, spec.width_y / 2.0 + 30.0, spec.length_z + 0.35),
        rotation=(0.0, 0.0, -math.pi / 2.0),
    )
    text = bpy.context.object
    text.name = "raised_TNH_label"
    text.data.body = spec.label_text
    text.data.align_x = "CENTER"
    text.data.align_y = "CENTER"
    text.data.size = spec.label_size
    text.data.extrude = spec.label_extrude / 2.0
    text.data.resolution_u = 8
    text.scale = (0.85, 0.85, 0.85)
    assign_material(text, material)
    bpy.ops.object.convert(target="MESH")
    text_mesh = bpy.context.object
    text_mesh.name = "raised_TNH_label"
    assign_material(text_mesh, material)
    objects.append(text_mesh)
    return objects


# ============================================================
# Procedural assembly
# ============================================================

"""High-level assembly for the bird-house model."""





def _set_origin_metadata(obj: bpy.types.Object) -> None:
    obj["source_model"] = "bird_house_customizer.scad"
    obj["source_variant"] = "Prebuilt bird_type=0 Chickadee"


def build_model() -> list[bpy.types.Object]:
    """Build the default prebuilt birdhouse from readable source dimensions."""

    spec = BirdHouseSpec()
    materials = birdhouse_materials()

    body = add_hollow_body("hollow_rectangular_body_with_trim", spec, materials["turquoise"])
    flange = add_top_flange_mesh("raised_entrance_flange", spec, materials["turquoise"])
    shell = join_objects("turquoise_body_and_top_flange", [body, flange])
    cut_entrance_and_vents(shell, spec)

    roof_wall = add_roof_wall_mesh("slanted_rear_roof_wall", spec, materials["turquoise"])

    perch = add_perch(spec, materials["turquoise"])
    hole_backers = add_dark_hole_backers(spec, materials)

    labels = add_raised_text_label(spec, materials["label"])

    objects = [shell, roof_wall, perch, *hole_backers, *labels]
    for obj in objects:
        _set_origin_metadata(obj)
        recalculate_normals(obj)
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

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable SCAD-like primitive builders for the twin-slot shelf bracket."""





@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.56


OPENSCAD_GOLD = MaterialSpec("openscad_preview_gold", (1.0, 0.72, 0.02, 1.0))
CUTTER_NAME_PREFIX = "boolean_cutter_"


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def get_material(spec: MaterialSpec) -> bpy.types.Material:
    material = bpy.data.materials.get(spec.name) or bpy.data.materials.new(spec.name)
    material.diffuse_color = spec.color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = spec.color
        bsdf.inputs["Roughness"].default_value = spec.roughness
    return material


def assign_material(obj: bpy.types.Object, spec: MaterialSpec | None) -> bpy.types.Object:
    if spec is not None:
        obj.data.materials.clear()
        obj.data.materials.append(get_material(spec))
    return obj


def _link_mesh_object(
    name: str,
    vertices: Sequence[tuple[float, float, float]],
    faces: Sequence[Sequence[int]],
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(list(vertices), [], [tuple(face) for face in faces])
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return assign_material(obj, material)


def add_scad_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Create an OpenSCAD cube(size, center=false)."""
    sx, sy, sz = size
    lx, ly, lz = location
    vertices = [
        (lx, ly, lz),
        (lx + sx, ly, lz),
        (lx + sx, ly + sy, lz),
        (lx, ly + sy, lz),
        (lx, ly, lz + sz),
        (lx + sx, ly, lz + sz),
        (lx + sx, ly + sy, lz + sz),
        (lx, ly + sy, lz + sz),
    ]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return _link_mesh_object(name, vertices, faces, material)


def _signed_area(points: Sequence[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return area * 0.5


def add_linear_extruded_polygon(
    name: str,
    points: Sequence[tuple[float, float]],
    height: float,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Create a linear_extrude(height) polygon(points) in source XY, along +Z."""
    if len(points) < 3:
        raise ValueError("A polygon extrusion needs at least three points")

    boundary = list(points)
    if _signed_area(boundary) < 0.0:
        boundary.reverse()

    lx, ly, lz = location
    vertices = (
        [(lx + x, ly + y, lz) for x, y in boundary]
        + [(lx + x, ly + y, lz + height) for x, y in boundary]
    )
    point_count = len(boundary)

    triangles = tessellate_polygon([[Vector((x, y, 0.0)) for x, y in boundary]])
    faces: list[tuple[int, ...]] = []
    for triangle in triangles:
        faces.append(tuple(reversed(triangle)))
        faces.append(tuple(point_count + index for index in triangle))

    for index in range(point_count):
        next_index = (index + 1) % point_count
        faces.append((index, next_index, next_index + point_count, index + point_count))

    return _link_mesh_object(name, vertices, faces, material)


def add_cylinder_between_x(
    name: str,
    radius: float,
    x_start: float,
    x_end: float,
    center_y: float,
    center_z: float,
    vertices: int = 64,
    material: MaterialSpec | None = None,
) -> bpy.types.Object:
    """Create a cylinder whose axis runs along source X."""
    depth = x_end - x_start
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=((x_start + x_end) * 0.5, center_y, center_z),
        rotation=(0.0, math.radians(90.0), 0.0),
    )
    obj = bpy.context.object
    obj.name = name
    assign_material(obj, material)
    bpy.ops.object.shade_smooth()
    return obj


def apply_boolean_difference(
    target: bpy.types.Object,
    cutters: Sequence[bpy.types.Object],
) -> bpy.types.Object:
    """Apply exact boolean differences and remove the temporary cutter meshes."""
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    for cutter in cutters:
        modifier = target.modifiers.new(name=f"cut_{cutter.name}", type="BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        if hasattr(modifier, "solver"):
            modifier.solver = "EXACT"
        bpy.context.view_layer.objects.active = target
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
    target.select_set(False)
    return target


def apply_boolean_union(
    target: bpy.types.Object,
    additions: Sequence[bpy.types.Object],
) -> bpy.types.Object:
    """Apply exact boolean unions and remove the temporary addition meshes."""
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    for addition in additions:
        modifier = target.modifiers.new(name=f"union_{addition.name}", type="BOOLEAN")
        modifier.operation = "UNION"
        modifier.object = addition
        if hasattr(modifier, "solver"):
            modifier.solver = "EXACT"
        bpy.context.view_layer.objects.active = target
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(addition, do_unlink=True)
    target.select_set(False)
    return target


# ============================================================
# Procedural assembly
# ============================================================

"""Readable assembly of the twin-slot shelf bracket model."""






@dataclass(frozen=True)
class BracketParameters:
    key_thick: float = 2.50
    slot_gap: float = 6.0
    key_width: float = 18.0
    backbone: float = 10.0
    bracket_l: float = 110.0
    bracket_h: float = 48.0
    screw_holes: bool = True
    screw_hole: float = 3.4
    access_hole: float = 6.0
    fixing_t: float = 5.0
    cylinder_segments: int = 60

    @property
    def arm_l(self) -> float:
        return self.bracket_l - self.backbone

    @property
    def arm_t(self) -> float:
        return self.slot_gap + self.key_thick * 2.0


KEY_PROFILE_POINTS: tuple[tuple[float, float], ...] = (
    (16.0, 14.0),
    (18.0, 13.0),
    (26.0, 13.0),
    (26.0, 8.0),
    (42.0, 6.0),
    (42.0, 13.0),
    (48.0, 13.0),
    (48.0, 22.0),
    (0.0, 22.0),
    (0.0, 5.0),
    (2.0, 2.0),
    (5.0, 0.0),
    (15.0, 0.0),
    (16.0, 1.0),
    (16.0, 9.0),
    (15.0, 10.0),
    (12.0, 10.0),
    (12.0, 14.0),
)


def _arm_profile(params: BracketParameters) -> tuple[tuple[float, float], ...]:
    return (
        (params.bracket_h, 0.0),
        (5.0, params.arm_l),
        (0.0, params.arm_l),
        (0.0, 0.0),
    )


def _build_positive_solids(params: BracketParameters) -> list[bpy.types.Object]:
    parts: list[bpy.types.Object] = []
    parts.append(
        add_linear_extruded_polygon(
            name="lower_twin_slot_key_profile",
            points=KEY_PROFILE_POINTS,
            height=params.key_thick,
            material=OPENSCAD_GOLD,
        )
    )
    parts.append(
        add_linear_extruded_polygon(
            name="upper_twin_slot_key_profile",
            points=KEY_PROFILE_POINTS,
            height=params.key_thick,
            location=(0.0, 0.0, params.slot_gap + params.key_thick),
            material=OPENSCAD_GOLD,
        )
    )
    parts.append(
        add_scad_box(
            name="vertical_backbone_between_keys",
            size=(params.bracket_h, params.backbone, params.arm_t),
            location=(0.0, params.key_width, 0.0),
            material=OPENSCAD_GOLD,
        )
    )
    parts.append(
        add_linear_extruded_polygon(
            name="long_tapered_shelf_arm",
            points=_arm_profile(params),
            height=params.arm_t,
            location=(0.0, params.key_width + params.backbone, 0.0),
            material=OPENSCAD_GOLD,
        )
    )
    return parts


def _build_hole_cutters(params: BracketParameters) -> list[bpy.types.Object]:
    if not params.screw_holes:
        return []

    cutters: list[bpy.types.Object] = []
    center_z = params.key_thick + params.slot_gap / 2.0
    for index, fraction in enumerate((0.25, 0.75), start=1):
        y_offset = params.key_width + params.backbone + fraction * params.arm_l
        cutters.append(
            add_cylinder_between_x(
                name=f"{CUTTER_NAME_PREFIX}screw_core_{index}",
                radius=params.screw_hole / 2.0,
                x_start=-1.0,
                x_end=119.0,
                center_y=y_offset,
                center_z=center_z,
                vertices=params.cylinder_segments,
            )
        )
        cutters.append(
            add_cylinder_between_x(
                name=f"{CUTTER_NAME_PREFIX}screwdriver_access_{index}",
                radius=params.access_hole / 2.0,
                x_start=params.fixing_t,
                x_end=params.fixing_t + 120.0,
                center_y=y_offset,
                center_z=center_z,
                vertices=params.cylinder_segments,
            )
        )
    return cutters


def build_model() -> list[bpy.types.Object]:
    params = BracketParameters()
    positive_solids = _build_positive_solids(params)
    bracket = positive_solids[0]
    bracket.name = "twin_slot_shelf_bracket_positive_union"
    apply_boolean_union(bracket, positive_solids[1:])
    apply_boolean_difference(bracket, _build_hole_cutters(params))
    bracket.name = "twin_slot_shelf_bracket"
    bracket.data.name = "twin_slot_shelf_bracket_mesh"
    return [bracket]


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

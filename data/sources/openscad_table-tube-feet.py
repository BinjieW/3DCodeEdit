from __future__ import annotations
from dataclasses import dataclass
import bpy
import math


# ============================================================
# Primitive geometry helpers
# ============================================================

"""Reusable primitive builders for the table tube feet model."""





def make_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.55,
) -> bpy.types.Material:
    """Create or replace a simple diffuse/principled material."""
    existing = bpy.data.materials.get(name)
    if existing:
        bpy.data.materials.remove(existing)

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
    material.diffuse_color = color
    return material


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def create_open_tube_foot(
    name: str,
    outside_diameter: float,
    inside_diameter: float,
    height: float,
    bottom_thickness: float,
    location: tuple[float, float, float],
    material: bpy.types.Material | None = None,
    segments: int = 360,
) -> bpy.types.Object:
    """Build one hollow cylindrical cup matching the OpenSCAD difference.

    The inner bore starts above the closed bottom, leaving a visible bottom floor
    and an open top with an annular rim.
    """
    if inside_diameter >= outside_diameter:
        raise ValueError("inside_diameter must be smaller than outside_diameter")
    if not 0.0 < bottom_thickness < height:
        raise ValueError("bottom_thickness must be between 0 and height")
    if segments < 24:
        raise ValueError("segments should be high enough to read as a round tube")

    outer_radius = outside_diameter / 2.0
    inner_radius = inside_diameter / 2.0

    vertices: list[tuple[float, float, float]] = []
    for radius, z in (
        (outer_radius, 0.0),
        (outer_radius, height),
        (inner_radius, bottom_thickness),
        (inner_radius, height),
    ):
        for index in range(segments):
            angle = math.tau * index / segments
            vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))

    def ring(ring_index: int, segment: int) -> int:
        return ring_index * segments + (segment % segments)

    faces: list[tuple[int, ...]] = []
    bottom_center_index = len(vertices)
    vertices.append((0.0, 0.0, 0.0))
    inner_floor_center_index = len(vertices)
    vertices.append((0.0, 0.0, bottom_thickness))

    for index in range(segments):
        next_index = (index + 1) % segments

        # Outer cylindrical wall runs the full source height.
        faces.append((ring(0, index), ring(0, next_index), ring(1, next_index), ring(1, index)))

        # Inner cylindrical wall begins at the bottom thickness and remains open.
        faces.append((ring(2, next_index), ring(2, index), ring(3, index), ring(3, next_index)))

        # Bottom underside and the visible inner floor.
        faces.append((bottom_center_index, ring(0, next_index), ring(0, index)))
        faces.append((inner_floor_center_index, ring(2, index), ring(2, next_index)))

        # Top annular rim is the remaining top face after subtracting the bore.
        faces.append((ring(3, index), ring(1, index), ring(1, next_index), ring(3, next_index)))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    for polygon_index, polygon in enumerate(mesh.polygons):
        polygon.use_smooth = polygon_index % 5 in {0, 1}

    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.collection.objects.link(obj)

    if material:
        obj.data.materials.append(material)

    return obj


# ============================================================
# Procedural assembly
# ============================================================

"""High-level assembly for the table tube feet model."""





@dataclass(frozen=True)
class TubeFootDimensions:
    wall_and_bottom_thickness: float = 3.0
    height: float = 20.0
    inside_diameter: float = 26.2
    count: int = 4
    y_step: float = 5.0

    @property
    def outside_diameter(self) -> float:
        return self.inside_diameter + self.wall_and_bottom_thickness * 2.0

    @property
    def outside_radius(self) -> float:
        return self.outside_diameter / 2.0


def source_loop_positions(dimensions: TubeFootDimensions) -> list[tuple[float, float, float]]:
    """Return the exact centers from the OpenSCAD loop for the four feet."""
    start = -dimensions.count / 2.0 + 1.0
    stop = dimensions.count / 2.0
    x_values = [start + index for index in range(int(stop - start) + 1)]
    return [
        (
            x_value * dimensions.outside_diameter - dimensions.outside_radius,
            x_value * dimensions.y_step,
            0.0,
        )
        for x_value in x_values
    ]


def build_model() -> list:
    """Build the four identical open tube feet."""
    dimensions = TubeFootDimensions()
    foot_material = make_material(
        "source_yellow_plastic",
        (0.82, 0.68, 0.03, 1.0),
    )

    feet = []
    for index, location in enumerate(source_loop_positions(dimensions), start=1):
        feet.append(
            create_open_tube_foot(
                name=f"tube_foot_{index:02d}",
                outside_diameter=dimensions.outside_diameter,
                inside_diameter=dimensions.inside_diameter,
                height=dimensions.height,
                bottom_thickness=dimensions.wall_and_bottom_thickness,
                location=location,
                material=foot_material,
            )
        )
    return feet


# ============================================================
# Scene setup and rendering
# ============================================================

"""Render entrypoint for the current iteration."""



def main() -> None:
    clear_scene()
    build_model()


if __name__ == "__main__":
    main()

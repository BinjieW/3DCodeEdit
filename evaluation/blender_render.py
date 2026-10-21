"""Runs inside Blender: render an original/modified script pair from six matched views.

    blender --background --python evaluation/blender_render.py -- --task task.json

Both scenes are framed by one shared transform fitted to the union of their bounds,
so size and position changes stay visible across the pair. Writes
original_<view>.png, modified_<view>.png and frame.json into the task's output_dir.
"""
import argparse
import json
import math
import os
import sys
import tempfile
import traceback

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

RENDERABLE_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT"}


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def get_scene_bbox():
    """Evaluated world-space bounds of visible renderable geometry, and whether any exists."""
    bbox_min = [math.inf] * 3
    bbox_max = [-math.inf] * 3
    found = False
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bpy.context.view_layer.update()
    for obj in bpy.context.scene.objects:
        if obj.type not in RENDERABLE_TYPES or obj.hide_render:
            continue
        evaluated_obj = obj.evaluated_get(depsgraph)
        if not hasattr(evaluated_obj, "bound_box"):
            continue
        found = True
        for coord in evaluated_obj.bound_box:
            wc = evaluated_obj.matrix_world @ Vector(coord)
            for i in range(3):
                bbox_min[i] = min(bbox_min[i], wc[i])
                bbox_max[i] = max(bbox_max[i], wc[i])
    if not found:
        return (Vector((0, 0, 0)), Vector((1, 1, 1))), False
    return (Vector(bbox_min), Vector(bbox_max)), True


def apply_shared_frame(center, scale):
    """Apply one world-space framing transform to every pre-render scene object."""
    norm_mat = Matrix.Scale(scale, 4) @ Matrix.Translation(-center)
    objects = [obj for obj in bpy.context.scene.objects if obj.type not in {"CAMERA", "LIGHT"}]
    world_matrices = [(obj, obj.matrix_world.copy()) for obj in objects]

    def hierarchy_depth(obj):
        depth = 0
        parent = obj.parent
        while parent is not None:
            depth += 1
            parent = parent.parent
        return depth

    # Parents move first; each child's saved world matrix is then restored relative
    # to the already-transformed parent without applying the frame twice.
    for obj, world_matrix in sorted(world_matrices, key=lambda item: hierarchy_depth(item[0])):
        obj.matrix_world = norm_mat @ world_matrix
    bpy.context.view_layer.update()


def _bbox_corners(bbox_min, bbox_max):
    return [Vector((x, y, z))
            for x in (bbox_min.x, bbox_max.x)
            for y in (bbox_min.y, bbox_max.y)
            for z in (bbox_min.z, bbox_max.z)]


def _frame_fits(center, scale, bbox_min, bbox_max, views, margin):
    """Check whether a framed bbox stays inside every camera's safe image area."""
    scene = bpy.context.scene
    camera = scene.camera
    points = [(point - center) * scale for point in _bbox_corners(bbox_min, bbox_max)]
    for azimuth, elevation in views.values():
        set_camera(azimuth, elevation)
        bpy.context.view_layer.update()
        for point in points:
            projected = world_to_camera_view(scene, camera, point)
            if projected.z <= camera.data.clip_start:
                return False
            if not (margin <= projected.x <= 1.0 - margin and margin <= projected.y <= 1.0 - margin):
                return False
    return True


def compute_union_frame(original_bbox, modified_bbox, views, target_coverage=0.95):
    """Fit the original/modified union into all views with one transform (binary search on scale)."""
    original_min, original_max = original_bbox
    modified_min, modified_max = modified_bbox
    union_min = Vector(tuple(min(original_min[i], modified_min[i]) for i in range(3)))
    union_max = Vector(tuple(max(original_max[i], modified_max[i]) for i in range(3)))
    center = (union_min + union_max) / 2
    max_dim = max(union_max[i] - union_min[i] for i in range(3))
    if max_dim < 1e-6:
        max_dim = 1.0

    margin = (1.0 - target_coverage) / 2.0
    low, high = 0.0, 2.0 / max_dim
    for _ in range(36):
        candidate = (low + high) / 2.0
        if _frame_fits(center, candidate, union_min, union_max, views, margin):
            low = candidate
        else:
            high = candidate
    scale = low * 0.999  # stay slightly inside the boundary to avoid numerical clipping
    return {
        "center": [float(v) for v in center],
        "scale": float(scale),
        "target_coverage": float(target_coverage),
        "margin": float(margin),
        "original_bbox": {"min": [float(v) for v in original_min], "max": [float(v) for v in original_max]},
        "modified_bbox": {"min": [float(v) for v in modified_min], "max": [float(v) for v in modified_max]},
        "union_bbox": {"min": [float(v) for v in union_min], "max": [float(v) for v in union_max]},
    }


def setup_lighting():
    for name, rot, energy in [
        ("Key", (0.8, 0, -0.8), 4.0),
        ("Fill", (0.8, 0, 2.4), 2.5),
        ("Rim", (-0.8, 0, -3.9), 3.5),
        ("Bottom", (3.14, 0, 0), 1.5),
    ]:
        light_data = bpy.data.lights.new(name=name, type="SUN")
        light_data.energy = energy
        light_data.use_shadow = False
        light_obj = bpy.data.objects.new(name, light_data)
        light_obj.rotation_euler = rot
        bpy.context.scene.collection.objects.link(light_obj)


def fix_materials():
    """Sync diffuse_color into Principled BSDF Base Color for Cycles rendering."""
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bc = bsdf.inputs["Base Color"]
            if not bc.links:  # only if no texture is connected
                c = mat.diffuse_color
                bc.default_value = (c[0], c[1], c[2], 1.0)


def setup_render_settings(render_cfg):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = render_cfg.get("samples", 64)
    scene.render.resolution_x = render_cfg.get("resolution", 512)
    scene.render.resolution_y = render_cfg.get("resolution", 512)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.18, 0.18, 0.18, 1.0)
        bg.inputs["Strength"].default_value = 1.0


def set_camera(azimuth_deg, elevation_deg, distance=2.0):
    """Place the camera at the given azimuth/elevation (degrees), looking at the origin."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cam = bpy.context.scene.camera
    cam.location = Vector((distance * math.cos(el) * math.sin(az),
                           -distance * math.cos(el) * math.cos(az),
                           distance * math.sin(el)))
    direction = Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera():
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_data.lens = 50
    cam_data.sensor_width = 36
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj


def execute_scene_script(script_path):
    """Reset Blender and execute one scene script."""
    clear_scene()
    import bmesh
    from mathutils import Euler
    exec_globals = {
        "__name__": "__main__",
        "__file__": script_path,
        "__builtins__": __builtins__,
        "bpy": bpy,
        "math": math,
        "bmesh": bmesh,
        "Vector": Vector,
        "Matrix": Matrix,
        "Euler": Euler,
    }
    with open(script_path) as f:
        exec(f.read(), exec_globals)
    bpy.context.view_layer.update()


def render_views(output_dir, prefix, views):
    for view_name, (azimuth, elevation) in views.items():
        set_camera(azimuth, elevation)
        bpy.context.view_layer.update()
        bpy.context.scene.render.filepath = os.path.join(output_dir, f"{prefix}_{view_name}.png")
        bpy.ops.render.render(write_still=True)


def render_pair(task):
    output_dir = task["output_dir"]
    views = task["views"]
    render_cfg = task.get("render", {})
    handle, original_blend = tempfile.mkstemp(prefix="3dcodeedit_original_", suffix=".blend")
    os.close(handle)
    os.remove(original_blend)
    try:
        # Keep the original scene so it can be rendered once the union frame is known.
        execute_scene_script(task["original_script"])
        fix_materials()
        original_bbox, original_has_geometry = get_scene_bbox()
        bpy.ops.wm.save_as_mainfile(filepath=original_blend, copy=True)

        execute_scene_script(task["modified_script"])
        fix_materials()
        modified_bbox, modified_has_geometry = get_scene_bbox()

        setup_camera()
        setup_render_settings(render_cfg)
        frame = compute_union_frame(original_bbox, modified_bbox, views, task.get("target_coverage", 0.95))
        center, scale = Vector(frame["center"]), frame["scale"]
        apply_shared_frame(center, scale)
        setup_lighting()
        render_views(output_dir, "modified", views)

        # Reload the exact original scene and apply the same transform and cameras.
        bpy.ops.wm.open_mainfile(filepath=original_blend)
        apply_shared_frame(center, scale)
        setup_camera()
        setup_lighting()
        setup_render_settings(render_cfg)
        render_views(output_dir, "original", views)

        frame.update(views=views, original_has_geometry=original_has_geometry,
                     modified_has_geometry=modified_has_geometry,
                     camera={"distance": 2.0, "lens": 50.0, "sensor_width": 36.0})
        with open(os.path.join(output_dir, "frame.json"), "w") as f:
            json.dump(frame, f, indent=2)
    finally:
        if os.path.exists(original_blend):
            os.remove(original_blend)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    task_path = parser.parse_args(argv).task
    with open(task_path) as f:
        task = json.load(f)
    for path in reversed(task.get("python_paths", [])):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    render_pair(task)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

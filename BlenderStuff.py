import bpy
import bmesh
import mathutils
import numpy as np
import json

def main():

    scene = bpy.context.scene

    obj = bpy.data.objects.get("Cube")
    obj.select_set(True)
    bpy.ops.object.delete()

    # bpy.data.objects["Light"].data.energy = 2000

    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    bpy.data.lights["Sun"].energy = 5  # Harnessing the full unmatched power of the sun

    object_loader("lane_objects.json", bpy.data.objects.get("Camera"))
    # object_loader("lane_objects.json", bpy.data.objects.get("Camera"))
    for ob in scene.objects:
        print(ob)
    render_scene(scene, "test_lanes.png")

def object_loader(json_filepath: str, camera):
    with open(json_filepath, 'r') as fp:
        data = json.load(fp)

    cam_location = data["camera_pose"][0:3]
    cam_euler = data["camera_pose"][3:]
    camera.location = mathutils.Vector(cam_location)
    camera.rotation_euler = mathutils.Euler(cam_euler)

    for obj in data["objects"]:
        if obj["type"] == "lane_line":
            obj_handler(mathutils.Vector(obj["pose"][0:3]), mathutils.Euler(obj["pose"][3:]))

    # for obj in data["objects"]:
    #     if obj["type"] == "box":
    #         obj_handler(mathutils.Vector(obj["pose"][0:3]), mathutils.Euler(obj["pose"][3:]))

def render_scene(scene, file_path):
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = file_path
    bpy.ops.render.render(write_still = 1)

def obj_handler(pose_vector, euler_vector):

    with bpy.data.libraries.load("Assets/Lane_Line.blend", link=False) as (data_from, data_to):
        data_to.objects = [ name for name in data_from.objects if name not in ["Light", "Camera"]]

    for obj in data_to.objects:
        print(obj.name)
        if obj.name not in ["Light", "Camera"]:
            bpy.context.collection.objects.link(obj)  # Link object to current scene
            obj.location = pose_vector
            obj.rotation_euler = euler_vector
    bpy.context.view_layer.update()

if __name__ == "__main__":
    main()
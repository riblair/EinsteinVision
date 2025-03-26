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
    print(bpy.data.objects.get("Camera").location)
    print(bpy.data.objects.get("Camera").rotation_euler)

    object_loader("objects.json", bpy.data.objects.get("Camera"))
    for ob in scene.objects:
        print(ob)
    render_scene(scene, "test_image.png")

def object_loader(json_filepath: str, camera):
    with open(json_filepath, 'r') as fp:
        data = json.load(fp)

    cam_location = data["camera_pose"][0:3]
    cam_euler = data["camera_pose"][3:]
    camera.location = mathutils.Vector(cam_location)
    camera.rotation_euler = mathutils.Euler(cam_euler)

    for obj in data["objects"]:
        if obj["type"] == "box":
            box_handler(mathutils.Vector(obj["pose"][0:3]), mathutils.Euler(obj["pose"][3:]))

def render_scene(scene, file_path):
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = file_path
    bpy.ops.render.render(write_still = 1)

def box_handler(pose_vector, euler_vector):
    mesh = bpy.data.meshes.new('cube')
    ob = bpy.data.objects.new('cube', mesh)
    ob.location = pose_vector
    ob.rotation_euler = euler_vector
    bpy.context.collection.objects.link(ob)
    bpy.context.view_layer.update()
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(mesh)
    bm.free()

if __name__ == "__main__":
    main()
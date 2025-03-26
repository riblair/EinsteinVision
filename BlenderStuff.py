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

    bpy.data.objects["Light"].data.energy = 2000
    # print(a)
    # exit(1)

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
    # mesh = bpy.ops.import_mesh.stl("INVOKE_DEFAULT", filepath="Assets/Dustbin.blend")
    # # speed = bpy.ops.import_curve.svg("Assets/Speed_Limit_blank_sign.svg") # for meshes
    # # exit(1)
    # ob = bpy.data.objects.new('cube', mesh)

    with bpy.data.libraries.load("Assets/Vehicles/Bicycle.blend", link=False) as (data_from, data_to):
        data_to.objects = [ name for name in data_from.objects if name not in ["Light", "Camera"]]

    for obj in data_to.objects:
        print(obj.name)
        if obj.name not in ["Light", "Camera"]:
            bpy.context.collection.objects.link(obj)  # Link object to current scene
            obj.location = pose_vector
            obj.rotation_euler = euler_vector
        # print(type(data_from))
        # print(type(data_to))
        # for ob in data_from.objects:
        #     print(type(ob))
            # ob.location = pose_vector
            # ob.rotation_euler = euler_vector
            # bpy.context.collection.objects.link(ob)
        # data_to.objects = data_from.objects
    # print(data_to.objects)
    # for ob in bpy.context.scene.objects:
    #     print(ob)
    # exit(1)
    
    bpy.context.view_layer.update()
    # bm = bmesh.new()
    # bmesh.ops.create_cube(bm, size=1.0)
    # bm.to_mesh(mesh)
    # bm.free()

if __name__ == "__main__":
    main()
import bpy
import bmesh
import cv2
import mathutils
import os
import numpy as np
import json

def main():
    render_images("scenes.json", "Output/")

def render_scene(scene, file_path):
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = file_path
    bpy.ops.render.render(write_still = 1)

def obj_handler(pose_vector, euler_vector, blend_file):
    with bpy.data.libraries.load(blend_file, link=False) as (data_from, data_to):
        data_to.objects = [ name for name in data_from.objects if name not in ["Light", "Camera"]]

    for obj in data_to.objects:
        print(obj.name)
        if obj.name not in ["Light", "Camera"]:
            bpy.context.collection.objects.link(obj)  # Link object to current scene
            obj.location = pose_vector
            obj.rotation_euler = euler_vector
    bpy.context.view_layer.update()

def reset_scene():
    bpy.ops.object.select_all(action='DESELECT')
    obj_dict = bpy.data.objects
    kept_objs = ["Camera", "Sun", "Light"]
    # for ob in bpy.context.scene.objects:
    #     print(ob)
    for key in obj_dict:
        if key.name not in kept_objs:
            # print(key.name)
            obj = bpy.data.objects.get(key.name)
            if obj is not None:
                obj.select_set(True)
                bpy.ops.object.delete()

def render_images(json_filepath, output_dir):
    scene = bpy.context.scene

    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    bpy.data.lights["Sun"].energy = 5  # Harnessing the full unmatched power of the sun

    with open(json_filepath, 'r') as fp:
        data = json.load(fp)

    cam_location = data["camera_pose"][0:3]
    cam_euler = data["camera_pose"][3:]
    camera = bpy.data.objects.get("Camera")
    camera.location = mathutils.Vector(cam_location)
    camera.rotation_euler = mathutils.Euler(cam_euler)
    bpy.context.view_layer.update()

    scene_list = data["Scenes"]
    for i in range(len(scene_list)):
        reset_scene()
        objects = scene_list[i]['objects']
        for obj in objects:
            if obj["type"] == "lane_line":
                obj_handler(mathutils.Vector(obj["pose"][0:3]), 
                            mathutils.Euler(obj["pose"][3:]), 
                            "Assets/Lane_Line.blend")
        render_scene(scene, f"{output_dir}image_{scene_list[i]['scene_num']}.png")

def directory_to_video(output_dir):
    # frame_list = []
    filenames = os.listdir(output_dir)
    filenames = [output_dir+filename for filename in filenames if filename[-4:] == ".png"]
    # print(filenames)

    filenames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0])) 
    output = cv2.VideoWriter( 
        f"{output_dir}video.avi", cv2.VideoWriter_fourcc(*'X264'), 36, (1080, 1920)) 

    for filename in filenames:
        image = cv2.imread(filename)
        print(filename)
        print(image.shape)
        output.write(image)
    output.release()

if __name__ == "__main__":
    main()
    # directory_to_video("Output/")
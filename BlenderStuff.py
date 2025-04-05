import bpy
import bmesh
import cv2
import mathutils
import os
import numpy as np
import json

ASSETS = {
    "car": "Assets/Vehicles/SedanAndHatchback.blend",
    "truck": "Assets/Vehicles/Truck.blend",
    "stop sign": "Assets/StopSign.blend",
    "person": "Assets/Pedestrain.blend",
    "traffic light": "Assets/TrafficSignal.blend",
    "motorcycle": "Assets/Vehicles/Motorcycle.blend",
    "bicycle": "Assets/Vehicles/Bicycle.blend",
    "lane_line": "Assets/Lane_Line.blend"
}

def main():
    render_images("scenes.json", "Output/", "Videos/scene1_front.mp4", 469)

def setup_scene():
    scene = bpy.context.scene
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960
    scene.render.resolution_percentage = 100 

    # Make sure a world is assigned
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")

    world = bpy.context.scene.world

    # Use nodes (required for background color in Cycles and Eevee)
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = (0.1, 0.1, 0.1, 1.0)

    bpy.ops.object.light_add(type='SUN', location=(0, -2, 10))
    bpy.data.lights["Sun"].energy = 10  # Harnessing the full unmatched power of the sun
    scene.unit_settings.system = 'METRIC'
    bpy.context.scene.world.mist_settings.use_mist = False
    return scene

def render_scene(scene, file_path):
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = file_path
    bpy.ops.render.render(write_still = 1)

def render_combined_frame(frame, raw_filename, combined_filename):
    rendered_scene = cv2.imread(raw_filename)
    combined_scene = np.hstack((frame, rendered_scene))
    cv2.imwrite(combined_filename, combined_scene)

def obj_handler(pose_vector, euler_vector, blend_file):
    with bpy.data.libraries.load(blend_file, link=False) as (data_from, data_to):
        data_to.objects = [ name for name in data_from.objects if name not in ["Light", "Camera"]]

    for obj in data_to.objects:
        print(obj.name)
        if obj.name not in ["Light", "Camera"]:
            bpy.context.collection.objects.link(obj)  # Link object to current scene
            obj.location = pose_vector + obj.location
            # obj.rotation_euler = euler_vector
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

def render_images(json_filepath, output_dir, video_file, start_frame):

    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    scene = setup_scene()
    with open(json_filepath, 'r') as fp:
        data = json.load(fp)

    cam_location = data["camera_pose"][0:3]
    cam_euler = data["camera_pose"][3:]
    camera = bpy.data.objects.get("Camera")
    # camera.matrix_basis = mathutils.Matrix.Translation(tuple(cam_location))
    camera.location = mathutils.Vector(cam_location)
    camera.rotation_euler = mathutils.Euler(cam_euler)
    bpy.context.view_layer.update()

    scene_list = data["Scenes"]
    for i in range(len(scene_list)):
        ret, frame = cap.read() 
        reset_scene()
        objects = scene_list[i]['objects']
        for obj in objects:
            if obj["type"] not in ASSETS:
                continue
            obj_handler(mathutils.Vector(obj["pose"][0:3]), 
                        mathutils.Euler(obj["pose"][3:]), 
                        ASSETS[obj["type"]])
            
        raw_filename = f"{output_dir}image_{scene_list[i]['scene_num']:05}.png"
        combined_filename = f"{output_dir}Video/image_combined_{scene_list[i]['scene_num']:05}.png"
        render_scene(scene, raw_filename)
        render_combined_frame(frame, raw_filename, combined_filename)

def directory_to_video(output_dir):
    os.system("ffmpeg -framerate 36 -pattern_type glob -i 'Output/Video/image_combined_*.png' -c:v libx264 -pix_fmt yuv420p Output/Video/output.mp4")

if __name__ == "__main__":
    main()
    directory_to_video("Output/Video/")
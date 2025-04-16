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
    "person": "Assets/Human.blend",
    "traffic light": "Assets/TrafficSignal.blend",
    "motorcycle": "Assets/Vehicles/Motorcycle.blend",
    "bicycle": "Assets/Vehicles/Bicycle.blend",
    "lane_line": "Assets/Lane_Line.blend",
    "warning": "Assets/SpeedLimitSign.blend"
}

# TODO, make this chunkable and run rendering in seperate processes...
def main():
    render_images("Scenes/scenes.json", "Output/", "Videos/scene5_front.mp4", 860)
    # directory_to_video("Outputs10_2/Video/", "out10_2.mp4")


def setup_scene():
    scene = bpy.context.scene
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960
    scene.render.resolution_percentage = 100 

    for obj in list(bpy.data.objects):
        if obj.name != "Camera":
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


    absolute_path = os.path.abspath("Assets/Background.blend")
    # Loading world w/ background
    bpy.ops.wm.append(
    filepath=absolute_path,
    directory=absolute_path + "/World",
    filename="World")

    # loading camera, sun, ground plane
    bpy.ops.wm.append(
    filepath=absolute_path,
    directory=absolute_path + "/Collection/",
    filename="Collection")

    for obj in list(bpy.data.objects):
        if obj.name == "Camera.001":
            bpy.data.objects.remove(obj, do_unlink=True)
    scene.world = bpy.data.worlds[1]
    scene.unit_settings.system = 'METRIC'
    bpy.context.scene.world.mist_settings.use_mist = False
    return scene

def render_scene(scene, file_path):
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = file_path
    bpy.ops.render.render(write_still = 1)
    bpy.data.images.remove(bpy.data.images['Render Result']) # to prevent memory backup

def render_combined_frame(frame, raw_filename, combined_filename):
    rendered_scene = cv2.imread(raw_filename)
    combined_scene = np.hstack((frame, rendered_scene))
    cv2.imwrite(combined_filename, combined_scene)

def load_objects(blend_file):
    with bpy.data.libraries.load(blend_file, link=False) as (data_from, data_to):
        data_to.objects = [ name for name in data_from.objects if name not in ["Light", "Camera", "Sun"]]
    return data_to.objects

def append_human_model(blend_path):

    absolute_path = os.path.abspath(blend_path)

    pre_objects = set(bpy.data.objects)
    bpy.ops.wm.append(
    filepath=f"{absolute_path}",
    directory=f"{absolute_path}/Collection/",
    filename="Body")

    post_objects = set(bpy.data.objects)
    new_objects = post_objects - pre_objects
    return list(new_objects)

# bpy.ops.object.make_local(type='ALL')
# FOR Rigid Bodies only where all objects need to move and rotate by the same angles
def update_pose(objects, pose):
    position = mathutils.Vector(pose[0:3])
    for obj in objects:
        bpy.context.collection.objects.link(obj)  # Link object to current scene
        obj.location = position + obj.location
        rotation = obj.rotation_euler
        rotation.rotate_axis('X', pose[3])
        rotation.rotate_axis('Y', pose[4])
        rotation.rotate_axis('Z', pose[5])
        obj.rotation_euler = rotation

def handle_obj(pose_vector, blend_file):
    objects = load_objects(blend_file)
    update_pose(objects, pose_vector)

def handle_traffic_light(pose_vector, blend_file, color):
    objects = load_objects(blend_file)
    update_pose(objects, pose_vector)
    for obj in objects:
        if color not in obj.name and not "Traffic_signal" in obj.name:
            obj.hide_render = True # hide all non-lit lights

def handle_road_sign(pose_vector, blend_file, text):
    objects = load_objects(blend_file)
    update_pose(objects, pose_vector)
    for obj in objects:
        if "Text" in obj.name:
            obj.data.body = text

def handle_pedestrian(pose_vector, blend_file, joint_dict):
    objects = append_human_model(blend_file)
    for obj in objects:
        if "Pelvis" in obj.name:
            update_pose([obj], pose_vector)
        elif "RightShoulder" in obj.name:
            obj.rotation_euler.rotate_axis('Y', joint_dict["RightShoulder"])
        elif "RightElbow" in obj.name:
            obj.rotation_euler.rotate_axis('Z', joint_dict["RightElbow"])
        elif "RightHip" in obj.name:
            obj.rotation_euler.rotate_axis('X', joint_dict["RightHip"])
        elif "RightKnee" in obj.name:
            obj.rotation_euler.rotate_axis('X', joint_dict["RightKnee"])
        elif "LeftShoulder" in obj.name:
            obj.rotation_euler.rotate_axis('Y', joint_dict["LeftShoulder"])
        elif "LeftElbow" in obj.name:
            obj.rotation_euler.rotate_axis('Z', joint_dict["LeftElbow"])
        elif "LeftHip" in obj.name:
            obj.rotation_euler.rotate_axis('X', joint_dict["LeftHip"])
        elif "LeftKnee" in obj.name:
            obj.rotation_euler.rotate_axis('X', joint_dict["LeftKnee"])

def handle_vehicle(pose_vector, blend_file, direction, lights, is_parked):
    objects = load_objects(blend_file)
    # if (pose_vector[5] - 0 > 0.01): # if we have a non-zero yaw
    if direction == "front":
        pose_vector[5] = np.pi
    elif direction == "left": 
        pose_vector[5] = np.pi/2
    elif direction == "right": 
        pose_vector[5] = -np.pi/2
    update_pose(objects, pose_vector)

    vehicle_color = (0.8, 0.8, 0.8, 1) if not is_parked else (0.1, 0.1, 0.1, 1) 

    for obj in objects:
        if "Front_Left_Light" in obj.name:
            obj.hide_render = False if lights[0] else True
        elif "Front_Right_Light" in obj.name:
            obj.hide_render = False if lights[1] else True
        elif "Back_Left_Light" in obj.name:
            obj.hide_render = False if lights[2] else True
        elif "Back_Right_Light" in obj.name:
            obj.hide_render = False if lights[3] else True
        elif "Vehicle" in obj.name:
            obj.active_material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = vehicle_color

def reset_scene():
    bpy.ops.object.select_all(action='DESELECT')
    kept_objs = ["Camera", "Sun", "Ground"]
    for obj in list(bpy.data.objects):
        if obj.name not in kept_objs:
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    for datablock in bpy.data.objects:
        if datablock.users == 0:
            bpy.data.objects.remove(datablock)
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in bpy.data.materials:
        if material.users == 0:
            bpy.data.materials.remove(material)
    for light in bpy.data.lights:
        if light.users == 0:
            bpy.data.lights.remove(light)
    for image in bpy.data.images:
        if image.name in ["Render Result", "Viewer Node"] or image.users == 0:
            bpy.data.images.remove(image)

# Fixed some issues, but memory still slowly increases. 
def render_images(json_filepath, output_dir, video_file, start_frame):

    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    scene = setup_scene()
    bpy.context.view_layer.update()
    with open(json_filepath, 'r') as fp:
        data = json.load(fp)

    cam_location = data["camera_pose"][0:3]
    cam_euler = data["camera_pose"][3:]
    camera = bpy.data.objects.get("Camera")
    
    camera.location = mathutils.Vector(cam_location)
    camera.rotation_euler = mathutils.Euler(cam_euler)

    scene_list = data["Scenes"]
    for i in range(len(scene_list)):
        ret, frame = cap.read() 
        reset_scene()
        objects = scene_list[i]['objects']
        for obj in objects:
            if obj["type"] not in ASSETS:
                continue
            if obj["type"] == "traffic light":
                handle_traffic_light(obj["pose"], ASSETS[obj["type"]], obj["color"])
            elif obj["type"] == "warning":
                handle_road_sign(obj["pose"], ASSETS[obj["type"]], obj["speed_limit"])
            elif obj["type"] == "person":
                handle_pedestrian(obj["pose"], ASSETS[obj["type"]], obj["joint_dict"])
            elif obj["type"] == "car" or obj["type"] == "truck":
                handle_vehicle(obj["pose"], ASSETS[obj["type"]], obj["direction"], obj["light_array"], obj["is_parked"])
            else:
                handle_obj(obj["pose"], ASSETS[obj["type"]])
            bpy.context.view_layer.update()
            
        raw_filename = f"{output_dir}image_{scene_list[i]['scene_num']:05}.png"
        combined_filename = f"{output_dir}Video/image_combined_{scene_list[i]['scene_num']:05}.png"
        render_scene(scene, raw_filename)
        render_combined_frame(frame, raw_filename, combined_filename)
    cap.release()

def directory_to_video(output_dir, video_name):
    os.system(f"ffmpeg -framerate 36 -y -pattern_type glob -i '{output_dir}image_combined_*.png' -c:v libx264 -pix_fmt yuv420p {output_dir}{video_name}")

if __name__ == "__main__":
    main()

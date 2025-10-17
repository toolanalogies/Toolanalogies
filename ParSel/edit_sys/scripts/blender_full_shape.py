# In case the script is rerun without restarting blender, we need to delete the geolipi modules from sys.modules

import sys
delete_key = []
for key, value in sys.modules.items():
    if "geolipi" in key:
        delete_key.append(key)
    if "edit_sys" in key:
        delete_key.append(key)

for key in delete_key:
    del sys.modules[key]
import os
import math
import random
from pathlib import Path
import numpy as np
import _pickle as cPickle

import bpy
import mathutils

GEOLIPI_PATH = "/home/aditya/projects/iccv_23/repos/geolipi"
EDIT_SYS_PATH = "/home/aditya/projects/llm_vpi/edit_sys/"
MODE = "PARTNET"
# TEASER

## Method Figure
#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45/objs_def/"
#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45_more/objs_def/"
#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45_less/objs_def/"

#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45_prop_0/objs_def/"
#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45_ft/objs_def/"
#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45_ft2/objs_def/"
#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45_ft2_wrong/objs_def/"
#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45_ft_wrong/objs_def/"
#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45_rotate/objs_def/"
#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45_rotate_wrong/objs_def/"
#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45_scale_wrong/objs_def/"
#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45_translate/objs_def/"
#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45_translate_wrong/objs_def/"
#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45_translate_2/objs_def/"
#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45_prop_2/objs_def/"
#save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/42012_45_prop_3/objs_def/"

save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/45028_50/objs_def/"
save_dir = "/media/aditya/OS/data/edit_vpi/outputs/objs/LLMPrompterHuman_5_qualitative/45028_50_orig/objs_def/"
processed_file = "/media/aditya/OS/data/edit_vpi/dataset_1/shapes/42012/42012.pkl"



print(GEOLIPI_PATH)
sys.path.insert(0, GEOLIPI_PATH)
print(GEOLIPI_PATH)
sys.path.insert(0, EDIT_SYS_PATH)

from geolipi.geometry_nodes.bl_utils import clean_up, init_camera, init_lights, set_render_settings
from geolipi.geometry_nodes.utils import BASE_COLORS
from geolipi.geometry_nodes.materials import (
    create_material_tree,
    create_simple_material_tree,
    create_edge_material_tree,
    create_monotone_material,
)


def load_object_from_dir(file_path):
    bpy.ops.wm.obj_import(filepath=file_path)
    return bpy.context.selected_objects[0]

# Blender setup
clean_up()
init_camera()
# init_lights()
set_render_settings(resolution=1024, transparent_mode=True)

# set camera
camera = bpy.data.objects['Camera']
# Position, Focal length, and shift
origin_point = mathutils.Vector((0, 0, 0))
theta = 60 
theta = theta * math.pi/180
size = math.sqrt(20)
camera_x = -size * math.cos(theta)
camera_y = -size * math.sin(theta)
camera.location = (camera_x, camera_y, 1)
direction = origin_point - camera.location
rotQuat = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rotQuat.to_euler()
# Camera update the focal length.
camera.data.lens = 75
camera.data.shift_y = -0.02
#camera.data.lens = 178
#camera.data.shift_x = 0.535
#camera.data.shift_y = -0.356


# lighting
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0
bpy.context.scene.view_settings.view_transform = 'Filmic'
bpy.context.scene.view_settings.look = 'Medium High Contrast'

bpy.ops.object.light_add(type='SUN', align='WORLD', location=(-1.5, -1.0, 0.5), scale=(1, 1, 1))
light = bpy.data.objects["Sun"]
light.data.energy = 5# 25
light.data.angle = 0.69
direction = origin_point - light.location
rotQuat = direction.to_track_quat('-Z', 'Y')
light.rotation_euler = rotQuat.to_euler()


bpy.ops.object.light_add(type='POINT', align='WORLD', location=(3.35, 0, 0.5), scale=(1, 1, 1))
light = bpy.data.objects["Point"]
light.data.energy = 1000
light.data.color = (0.799926, 0.950844, 1)

## Render related

bpy.context.scene.use_nodes = True
bpy.context.scene.view_layers["ViewLayer"].use_pass_ambient_occlusion = True
tree = bpy.context.scene.node_tree
inp = tree.nodes["Render Layers"]
composite = tree.nodes["Composite"]
denoise = tree.nodes.new(type="CompositorNodeDenoise")
tree.links.new(inp.outputs["Denoising Normal"], denoise.inputs["Normal"])
tree.links.new(inp.outputs["Denoising Albedo"], denoise.inputs["Albedo"])
tree.links.new(denoise.outputs["Image"], composite.inputs["Image"])
# link inp image to compositor
# link inp ao to compositor
mixer = tree.nodes.new(type="CompositorNodeMixRGB")
mixer.blend_type = 'MULTIPLY'




mixer.inputs[0].default_value = 1.0

tree.links.new(inp.outputs["Image"], mixer.inputs[1])
tree.links.new(inp.outputs["AO"], mixer.inputs[2])
tree.links.new(mixer.outputs["Image"], denoise.inputs["Image"])

# Next load all the smaller objs:
if MODE == "COMPAT":
    all_objs_files = [x for x in os.listdir(save_dir) if '.glb' in x]
elif MODE == "PARTNET":
    all_objs_files = [x for x in os.listdir(save_dir) if x.endswith(".obj") ]
all_objs_files = sorted(all_objs_files)
#all_objs_files = all_objs_files[::-1]

all_objs_files = [os.path.join(save_dir, x) for x in all_objs_files]

all_processed_objs = []
mat_id = 0
material_type = "base"
# set seed
#random.seed(69)
#random.shuffle(BASE_COLORS)
#m = 3
#colors = np.concatenate([BASE_COLORS[m:], BASE_COLORS[:m]], 0)
colors = BASE_COLORS
ADD_BBOX = False
# material_type = "edge"
# From the pkl file load the points and add them
ball_material = create_material_tree("ball_mat", [1.0, 0, 0])
model, clean_relations, intersections, category = cPickle.load(open(processed_file, "rb"))

#not_valid_inds = []
#for ind, part in enumerate(model):
#    if len(part['children']) > 0:
#        not_valid_inds.append(ind)
#for inter in intersections:
#    points = inter[2]
#    if inter[0] in not_valid_inds or inter[1] in not_valid_inds:
#        continue
#    for point in points:
#        # flip y and z
#        point = (point[0], -point[2], point[1])
#        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.010, location=point)
#        bpy.context.active_object.active_material = ball_material


for ind, cur_file in enumerate(all_objs_files):
    # cur_obj = load_object_from_dir(os.path.join(save_dir, cur_file))
    if MODE == "COMPAT":
        bpy.ops.import_scene.gltf(filepath=cur_file)
        cur_obj = bpy.context.selected_objects[0]
        all_processed_objs.extend(list(cur_obj.children))
    # break
    elif MODE == "PARTNET":

        cur_obj = load_object_from_dir(os.path.join(save_dir, cur_file))
        # bpy.context.active_object.rotation_euler[0] = math.radians(180)
        # bpy.context.active_object.rotation_euler[2] = math.radians(180)
        # bpy.context.active_object.location[2] += 0.25
        all_processed_objs.append(cur_obj)
        # add graph and material
        mod = cur_obj.modifiers.new("concat_nodes", "NODES")
        node_group = bpy.data.node_groups.new("concat_tree", "GeometryNodeTree")
        node_group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        output_node = node_group.nodes.new("NodeGroupOutput")
        output_node.is_active_output = True
        output_node.select = False
        mod.node_group = node_group
        input_node = node_group.nodes.new("NodeGroupInput")
        node_group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
        node_group.links.new(material_node.outputs["Geometry"], output_node.inputs["Geometry"])
        if ADD_BBOX:
            bound_box = node_group.nodes.new(type="GeometryNodeBoundBox")
            node_group.links.new(input_node.outputs["Geometry"], bound_box.inputs["Geometry"])
            node_group.links.new(bound_box.outputs["Bounding Box"], material_node.inputs["Geometry"])
        else:
            node_group.links.new(input_node.outputs["Geometry"], material_node.inputs["Geometry"])

        mat_name = f"Material_{mat_id}"
        color = colors[mat_id % len(colors)]
        # TODO: Better color management.
        mat_name = f"Material_{mat_id}"
        color = colors[mat_id % len(colors)]
        if material_type == "base":
            material = create_material_tree(mat_name, color)
        elif material_type == "edge":
            material = create_edge_material_tree(mat_name, color, edge_thickness=0.005)

        material_node.inputs["Material"].default_value = material
        mat_id += 1
        
#        mod_2 = cur_obj.modifiers.new("wireframe", 'WIREFRAME')
#        mod_2.thickness = 0.0075


   
def object_bbox(obj):
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    min_coords = np.min(np.array(bbox_corners), axis=0)
    max_coords = np.max(np.array(bbox_corners), axis=0)
    return min_coords, max_coords

min_list = []
max_list = []
for obj in all_processed_objs:
    min_coords, max_coords = object_bbox(obj)
    min_list.append(min_coords)
    max_list.append(max_coords)

min_coords = np.min(np.array(min_list), axis=0)
max_coords = np.max(np.array(max_list), axis=0)


center = (min_coords + max_coords) / 2

for obj in all_processed_objs:
    obj.location[0] -= center[0]
    obj.location[1] -= center[1]


# Insert a plane and set its z to be min
bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, -0.0), scale=(50, 50, 1))
# from all_processed_objs, get the min z
plane = bpy.context.active_object
plane.scale[0] = 50
plane.scale[1] = 50

min_z = min_coords[2]
plane.location[2] = min_z - 0.05
plane.visible_camera = False
plane.hide_render = True



#  Render
# save_loc = os.path.join(GEOLIPI_PATH, "assets")
# Path(save_loc).mkdir(parents=True, exist_ok=True)
# save_template = f"{save_loc}/{selected_ind}"
# save_file_name = f"{save_template}.png"
# bpy.context.scene.render.filepath = save_file_name
# bpy.ops.render.render(write_still = True)

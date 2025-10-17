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

import bpy
import mathutils

GEOLIPI_PATH = "/home/aditya/projects/iccv_23/repos/geolipi"
EDIT_SYS_PATH = "/home/aditya/projects/llm_vpi/edit_sys/"

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
from scripts.local_config import (DEFAULT_OUTPUT_DIR, DATA_DIR, METHOD_MARKER, DATASET_INDEX, REPETITIONS)

method_marker = f"{METHOD_MARKER}_{REPETITIONS}"
save_dir = os.path.join(DEFAULT_OUTPUT_DIR,"objs", method_marker, f"{DATASET_INDEX}")
###  Files
category = 'bench'

def load_object_from_dir(file_path):
    bpy.ops.wm.obj_import(filepath=file_path)
    return bpy.context.selected_objects[0]

# Blender setup
clean_up()
init_camera()
init_lights()
set_render_settings(resolution=512, transparent_mode=True)

origin_point = mathutils.Vector((0, 0, 0))
# set camera
theta = 45 
theta = theta * math.pi/180
size = math.sqrt(8)
camera_x = size * math.cos(theta)
camera_y = size * math.sin(theta)
camera = bpy.data.objects['Camera']
camera.location = (camera_x, camera_y, 1.5)
direction = origin_point - camera.location
rotQuat = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rotQuat.to_euler()

# Generate Geometry Node Graph:
random.shuffle(BASE_COLORS)
colors = BASE_COLORS


# Next load all the smaller objs:
all_objs_files = [x for x in os.listdir(save_dir) if x.endswith(".obj") and x != f"{category}_recon.obj"]
all_processed_files = [x for x in all_objs_files if "processed" in x]

all_processed_objs = []
mat_id = 0
material_type = "base"

for ind, cur_file in enumerate(all_processed_files):
    cur_obj = load_object_from_dir(os.path.join(save_dir, cur_file))
    bpy.context.active_object.rotation_euler[0] = math.radians(180)
    bpy.context.active_object.rotation_euler[2] = math.radians(180)
    bpy.context.active_object.location[2] += 0.25
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
    node_group.links.new(input_node.outputs["Geometry"], material_node.inputs["Geometry"])
    node_group.links.new(material_node.outputs["Geometry"], output_node.inputs["Geometry"])

    mat_name = f"Material_{mat_id}"
    color = colors[mat_id % len(colors)]
    # TODO: Better color management.
    mat_name = f"Material_{mat_id}"
    color = colors[mat_id % len(colors)]
    if material_type == "base":
        material = create_material_tree(mat_name, color)

    material_node.inputs["Material"].default_value = material
    mat_id += 1
    # break

    

#  Render
# save_loc = os.path.join(GEOLIPI_PATH, "assets")
# Path(save_loc).mkdir(parents=True, exist_ok=True)
# save_template = f"{save_loc}/{selected_ind}"
# save_file_name = f"{save_template}.png"
# bpy.context.scene.render.filepath = save_file_name
# bpy.ops.render.render(write_still = True)
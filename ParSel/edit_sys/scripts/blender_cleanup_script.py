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

from scripts.local_config import (DEFAULT_OUTPUT_DIR, DATA_DIR, METHOD_MARKER, REPETITIONS, DATASET_INDEX)

method_marker = f"{METHOD_MARKER}_{REPETITIONS}"
save_dir = os.path.join(DEFAULT_OUTPUT_DIR,"objs", method_marker, f"{DATASET_INDEX}")
###  Files


def load_object_from_dir(file_path):
    bpy.ops.wm.obj_import(filepath=file_path)
    return bpy.context.selected_objects[0]

# Blender setup
clean_up()
category = "bench"
recon_obj_file = f"{category}_recon.obj"
recon_file = os.path.join(save_dir, recon_obj_file)

# blender load obj
recon_obj = load_object_from_dir(recon_file)
remesh_mod = recon_obj.modifiers.new("remesh", "REMESH")
remesh_mod.mode = "SMOOTH"
remesh_mod.octree_depth = 10
remesh_mod.scale = 0.99
remesh_mod.use_remove_disconnected = False
bpy.ops.object.modifier_apply(modifier="remesh")

decimate_mod = recon_obj.modifiers.new("decimate", "DECIMATE")
decimate_mod.ratio = 0.1
bpy.ops.object.modifier_apply(modifier="decimate")


# Next load all the smaller objs:
all_objs_files = [x for x in os.listdir(save_dir) if x.endswith(".obj") and x != recon_obj_file]
all_deformed_files = [x for x in all_objs_files if "deformed" in x]

all_deformed_objs = []
for cur_file in all_deformed_files:
    cur_obj = load_object_from_dir(os.path.join(save_dir, cur_file))
    all_deformed_objs.append(cur_obj)

# create the concat object
bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, -0.0), scale=(1, 1, 1))
concat_obj = bpy.context.active_object
# change name
concat_obj.name = "concat_obj"
# set as active
mod = concat_obj.modifiers.new("concat_nodes", "NODES")
node_group = bpy.data.node_groups.new("concat_tree", "GeometryNodeTree")
node_group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
output_node = node_group.nodes.new("NodeGroupOutput")
output_node.is_active_output = True
output_node.select = False
mod.node_group = node_group
join_node = node_group.nodes.new(type="GeometryNodeJoinGeometry")
node_group.links.new(join_node.outputs["Geometry"], output_node.inputs["Geometry"])
inner_obj_list = []
for i in range(len(all_deformed_objs) - 1):
    obj_node = node_group.nodes.new("GeometryNodeObjectInfo")
    node_group.links.new(obj_node.outputs["Geometry"], join_node.inputs[0])
    inner_obj_list.append(obj_node)

# Create the divider graph and object
# Instantiate a plane
def create_processor_object():
    bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, -0.0), scale=(1, 1, 1))
    processor_obj = bpy.context.active_object
    mod = processor_obj.modifiers.new("concat_nodes", "NODES")
    node_group = bpy.data.node_groups.new("concat_tree", "GeometryNodeTree")
    node_group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    output_node = node_group.nodes.new("NodeGroupOutput")
    output_node.is_active_output = True
    output_node.select = False
    mod.node_group = node_group
    main_obj_node = node_group.nodes.new("GeometryNodeObjectInfo")
    main_obj_node.inputs[0].default_value = recon_obj
    concat_obj_node = node_group.nodes.new("GeometryNodeObjectInfo")
    concat_obj_node.inputs[0].default_value = concat_obj
    reminder_obj_node = node_group.nodes.new("GeometryNodeObjectInfo")
    reminder_obj_node.inputs[0].default_value = all_deformed_objs[-1]
    concat_proximity_node = node_group.nodes.new("GeometryNodeProximity")
    node_group.links.new(concat_obj_node.outputs["Geometry"], concat_proximity_node.inputs["Geometry"])
    reminder_proximity_node = node_group.nodes.new("GeometryNodeProximity")
    node_group.links.new(reminder_obj_node.outputs["Geometry"], reminder_proximity_node.inputs["Geometry"])
    #math node
    math_node = node_group.nodes.new("ShaderNodeMath")
    math_node.operation = "LESS_THAN"
    node_group.links.new(concat_proximity_node.outputs["Distance"], math_node.inputs[0])
    node_group.links.new(reminder_proximity_node.outputs["Distance"], math_node.inputs[1])

    # Create Delete geom
    deleter = node_group.nodes.new("GeometryNodeDeleteGeometry")
    node_group.links.new(math_node.outputs["Value"], deleter.inputs[1])
    node_group.links.new(main_obj_node.outputs["Geometry"], deleter.inputs[0])
    node_group.links.new(deleter.outputs["Geometry"], output_node.inputs["Geometry"])
    return processor_obj, reminder_obj_node

for cur_ind, cur_obj in enumerate(all_deformed_objs):
    counter = 0
    for other_ind, other_obj in enumerate(all_deformed_objs):
        if other_ind == cur_ind:
            continue
        else:
            inner_obj_list[counter].inputs[0].default_value = other_obj
            counter += 1
    # set to reminder
    processor_obj, reminder_obj_node = create_processor_object()
    reminder_obj_node.inputs[0].default_value = cur_obj
    # Now extract this and save this
    bpy.context.view_layer.objects.active= processor_obj
    bpy.ops.object.convert(target='MESH')
    save_file = os.path.join(save_dir, f"processed_{all_deformed_files[cur_ind]}")
    bpy.ops.wm.obj_export(filepath=save_file, export_selected_objects=True)

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a OBJ/STL/FBX into USD format.

The OBJ file format is a simple data-format that represents 3D geometry alone — namely, the position
of each vertex, the UV position of each texture coordinate vertex, vertex normals, and the faces that
make each polygon defined as a list of vertices, and texture vertices.

An STL file describes a raw, unstructured triangulated surface by the unit normal and vertices (ordered
by the right-hand rule) of the triangles using a three-dimensional Cartesian coordinate system.

FBX files are a type of 3D model file created using the Autodesk FBX software. They can be designed and
modified in various modeling applications, such as Maya, 3ds Max, and Blender. Moreover, FBX files typically
contain mesh, material, texture, and skeletal animation data.
Link: https://www.autodesk.com/products/fbx/overview


This script uses the asset converter extension from Isaac Sim (``omni.kit.asset_converter``) to convert a
OBJ/STL/FBX asset into USD format. It is designed as a convenience script for command-line use.


positional arguments:
  input               The path to the input mesh (.OBJ/.STL/.FBX) file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                    Show this help message and exit
  --make-instanceable,          Make the asset instanceable for efficient cloning. (default: False)
  --collision-approximation     The method used for approximating collision mesh. Defaults to convexDecomposition.
                                Set to \"none\" to not add a collision mesh to the converted mesh. (default: convexDecomposition)
  --mass                        The mass (in kg) to assign to the converted asset. (default: None)

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# # add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a mesh file into USD format.")
parser.add_argument("--input", type=str, help="The path to the input dataset.")
# parser.add_argument("input", type=str, help="The path to the input mesh file.")
# parser.add_argument("output", type=str, help="The path to store the USD file.")
# parser.add_argument(
#     "--make-instanceable",
#     action="store_true",
#     default=False,
#     help="Make the asset instanceable for efficient cloning.",
# )
# parser.add_argument(
#     "--collision-approximation",
#     type=str,
#     default="convexDecomposition",
#     choices=["convexDecomposition", "convexHull", "none"],
#     help=(
#         'The method used for approximating collision mesh. Set to "none" '
#         "to not add a collision mesh to the converted mesh."
#     ),
# )
# parser.add_argument(
#     "--mass",
#     type=float,
#     default=None,
#     help="The mass (in kg) to assign to the converted asset. If not provided, then no mass is added.",
# )
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import json

import carb
import omni.isaac.core.utils.stage as stage_utils
import omni.kit.app

from omni.isaac.lab.sim.converters import MeshConverter, MeshConverterCfg
from omni.isaac.lab.sim.schemas import schemas_cfg
from omni.isaac.lab.utils.assets import check_file_path
from omni.isaac.lab.utils.dict import print_dict

import os

def get_all_obj_files(dataset_folder):
    obj_file_paths = []
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith('.obj'):
                full_path = os.path.join(root, file)
                obj_file_paths.append(full_path)
    return obj_file_paths

def write_json_next_to_obj(obj_file_path, data):
    """
    Write a JSON file next to the .obj file.
    
    Arguments:
    - obj_file_path: The full path to the .obj file.
    - data: A dictionary or object that can be serialized to JSON.
    """
    # Extract directory and base name
    directory = os.path.dirname(obj_file_path)
    base_name = os.path.splitext(os.path.basename(obj_file_path))[0]
    
    # Construct the JSON file path. 
    # e.g., if obj file is foo.obj, JSON file will be foo.json in the same folder.
    json_file_path = os.path.join(directory, f"{base_name}.json")
    
    # Write the JSON file
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)  # indent=4 for pretty-print
    
    print(f"JSON written to: {json_file_path}")

def read_json_next_to_obj(obj_path):
    """
    Given an .obj path, read the corresponding .json file in the same folder.
    Returns the JSON data as a dict or None if the file doesn't exist.
    """
    directory = os.path.dirname(obj_path)
    base_name = os.path.splitext(os.path.basename(obj_path))[0]  # e.g., "model" from "model.obj"
    json_file_path = os.path.join(directory, f"{base_name}.json")

    if os.path.isfile(json_file_path):
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        return data
    else:
        print(f"No JSON file found for {obj_path}")
        return None


def main():

    # Usage:
    dataset_path = args_cli.input
    all_obj_files = get_all_obj_files(dataset_path)
    print(all_obj_files)

    for obj_path in all_obj_files:
        # Create some data to store; this could be any Python dictionary 
        # (string keys, serializable values, etc.).
        # You can modify or compute this data as you need.
        # Split the path into (root, ext). Example: ("/path/to/some_model", ".obj")
        root, ext = os.path.splitext(obj_path)

        # usd_dir_name = os.path.dirname(root) + "/usd/" + os.path.basename(root)
        usd_dir_name = root

        # Now just append .usd instead
        usd_path = usd_dir_name + ".usd"
        data_to_store = {
            "mass": 1.0,
            "obj_file_name": os.path.basename(obj_path),
            "asset_path": obj_path,
            "collision_approximation": "convexDecomposition",
            "make_instanceable": False,
            "usd_path": usd_path,
            # ... other metadata or parameters ...
        }
        
        # Write the JSON file next to the obj file
        write_json_next_to_obj(obj_path, data_to_store)

        json_data = read_json_next_to_obj(obj_path)

        # # check valid file path
        mesh_path = json_data["asset_path"]
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.abspath(mesh_path)
        if not check_file_path(mesh_path):
            raise ValueError(f"Invalid mesh file path: {mesh_path}")

        # create destination path
        dest_path = json_data["usd_path"]
        if not os.path.isabs(dest_path):
            dest_path = os.path.abspath(dest_path)

        print(dest_path)
        print(os.path.dirname(dest_path))
        print(os.path.basename(dest_path))

        # Mass properties
        if json_data["mass"] is not None:
            mass_props = schemas_cfg.MassPropertiesCfg(mass=json_data["mass"])
            rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
        else:
            mass_props = None
            rigid_props = None

        # Collision properties
        collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=json_data["collision_approximation"] != "none")
        print(json_data["collision_approximation"])

        # Create Mesh converter config
        mesh_converter_cfg = MeshConverterCfg(
            mass_props=mass_props,
            rigid_props=rigid_props,
            collision_props=collision_props,
            asset_path=mesh_path,
            force_usd_conversion=True,
            usd_dir=os.path.dirname(dest_path),
            usd_file_name=os.path.basename(dest_path),
            make_instanceable=json_data["make_instanceable"],
            collision_approximation=json_data["collision_approximation"],
        )

        # Print info
        print("-" * 80)
        print("-" * 80)
        print(f"Input Mesh file: {mesh_path}")
        print("Mesh importer config:")
        print_dict(mesh_converter_cfg.to_dict(), nesting=0)
        print("-" * 80)
        print("-" * 80)

        # Create Mesh converter and import the file
        mesh_converter = MeshConverter(mesh_converter_cfg)
        # print output
        print("Mesh importer output:")
        print(f"Generated USD file: {mesh_converter.usd_path}")
        print("-" * 80)
        print("-" * 80)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
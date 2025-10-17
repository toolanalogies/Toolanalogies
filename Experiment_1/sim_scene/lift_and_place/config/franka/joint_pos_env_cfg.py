# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sensors import CameraCfg, TiledCameraCfg

from omni.isaac.lab_tasks.manager_based.manipulation.lift_and_place import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.lift_and_place.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip

import omni.isaac.lab.sim as sim_utils

from dataclasses import MISSING

from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, DeformableObjectCfg

import os
import json



from pathlib import Path


import torch


import numpy as np
import open3d as o3d
import glob
import os
import copy


def get_usd_lists(data_folder):
    """
    1. Recursively walk through `data_folder`.
    2. Whenever we find exactly one `.usd` in the same folder, 
       we pair them up.
    3. Return two parallel lists:
         usd_list[i]  -> path to the .usd file

    """
    usd_list = []

    for root, dirs, files in os.walk(data_folder):
        print("root")
        print(root)
        print("dirs")
        print(dirs)
        print("files")
        print(files)
        # Collect all .usd in this folder
        usd_files = [f for f in files if f.endswith(".usd") and not f.endswith("non_metric.usd")]

        print("usd_files")
        print(usd_files)

        # For simplicity, assume there is exactly ONE .usd 
        # If you expect more, adapt the logic.
        if len(usd_files) == 1:
            usd_file_path = os.path.join(root, usd_files[0])
            usd_list.append(usd_file_path)
        print("usd_list")
        print(usd_list)

    usd_list.sort()
    return usd_list

@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):

    dataset_config_path: str = "path to edit_config.json"
    range_list: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # with open(dataset_config_path, 'r') as f:
    #     shp_data = json.load(f)
    # shape_folder_path = shp_data["data_path"]

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.seed = 10
        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.98, -0.05, 0.005], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"path to hockey_ball.usd in the dataset",
                scale=(100., 100., 100.),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                semantic_tags=[("class", "cube")],
                mass_props=sim_utils.MassPropertiesCfg(mass=0.050)
            ),
        )

        asset_list = []

        with open(self.dataset_config_path, 'r') as f:
            shp_data = json.load(f)
        shape_folder_path = shp_data["data_path"]
        print(shape_folder_path)
        usd_list = get_usd_lists(shape_folder_path)
        print("USD LIST")
        print(usd_list)


        for usd in usd_list:
            asset_list.append(
                sim_utils.UsdFileCfg(
                    usd_path=usd,
                    scale=[110., 150., 110.], 
                )
            )

        self.scene.object2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object2",
            spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=asset_list,
            semantic_tags=[("class", "stick")],
            random_choice=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.330),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.65, -0.4, 0.2], rot=[ 0.7071068, -0.7071068, 0, 0 ]),#pos = [0.55, -0.55, 0.2], rot = [ 0.7071068, 0.7071068, 0, 0 ])#pos=[1.05, -0.55, 0.2], rot=[0.5, 0, 0, -0.8660254]),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

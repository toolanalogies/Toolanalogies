# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--shape_data_path", type=str, default=None, help="The path to shape dataset.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

import omni.isaac.lab.sim as sim_utils
import omni.usd

from pxr import Gf, Sdf
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg

import h5py

from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab_tasks.manager_based.manipulation.lift_and_place import mdp
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

import json
import os

from omni.isaac.lab.utils.math import transform_points, unproject_depth

from transform_utils import transform_points, get_properties_list, get_obj_list, point_transform_3, point_transform_2, inverse_transform_3, point_transform_to_ee, tip_x_close, tip_y_close, on_pose, read_json_from_path, write_json_to_path
from helper_simple import get_points_from_properties, controller, check_group, define_pcl_markers, define_pose_markers, randomize_shape_color, grasp_tips_points, get_points_from_list, get_points_from_mesh

def apply_yup_to_zup(points):
    """Rotate (x,y,z) from OBJ's Y-up to Isaac's Z-up frame."""
    # (x,y,z) -> (x,z,-y)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return torch.stack([x, -z, y], dim=-1)

def recenter(points, mesh_center):
    return points - torch.tensor(mesh_center, device=points.device)

def main():

    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.scene.replicate_physics = False
    env_cfg.viewer.eye = (2.5, 0.5, 0.5)
    env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_line)
    env_cfg.sim.render_interval = 6

    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # obj_list = get_obj_list(args_cli.shape_data_path)
    props_list = get_properties_list(args_cli.shape_data_path)
    # print('props_list')
    # print(props_list)
    props_list_dict = {"Props_list": props_list}
    base_handle, base_tips = get_points_from_properties(env, props_list)

    pcl_visualizer = define_pcl_markers()
    test_pcl_visualizer = define_pose_markers()
    grasp_visualizer = define_pose_markers()
    grasp_visualizer_move = define_pose_markers()
    tips_visualizer = define_pose_markers()

    with torch.inference_mode():
        for i in range(50):
            env.step(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=env.unwrapped.device).unsqueeze(0))
            # pcl_visualizer.visualize(pcl_tensor.reshape(-1, 3))
            # test_pcl_visualizer.visualize(test_pcl.reshape(-1, 3))

    # transformed_pcl = point_transform_2(pcl_tensor, env)

    # grasp_pcl, tips_pcl = grasp_tips_points(transformed_pcl, env)
    base_handle = apply_yup_to_zup(base_handle)
    base_tips = apply_yup_to_zup(base_tips)
    grasp_pcl = point_transform_3(base_handle, env)
    tips_pcl = point_transform_3(base_tips, env)
    base_grasp_pcl = inverse_transform_3(grasp_pcl, env)
    base_tips_pcl = inverse_transform_3(tips_pcl, env)
    n_envs = args_cli.num_envs
    
    grasped_z = torch.zeros(n_envs, device=env.unwrapped.device, dtype=torch.bool)
    grasped_z_move = torch.zeros(n_envs, device=env.unwrapped.device, dtype=torch.bool)
    grasped = torch.zeros(n_envs, device=env.unwrapped.device, dtype=torch.bool)
    hand_closed = torch.zeros(n_envs, device=env.unwrapped.device, dtype=torch.bool)
    reached_x = torch.zeros(n_envs, device=env.unwrapped.device, dtype=torch.bool)
    reached_y = torch.zeros(n_envs, device=env.unwrapped.device, dtype=torch.bool)
    task_done = torch.zeros(n_envs, device=env.unwrapped.device, dtype=torch.bool)
    grasped_z_old = torch.zeros(n_envs, device=env.unwrapped.device, dtype=torch.bool)
    grasped_old = torch.zeros(n_envs, device=env.unwrapped.device, dtype=torch.bool)
    reached_x_old = torch.zeros(n_envs, device=env.unwrapped.device, dtype=torch.bool)
    reached_y_old = torch.zeros(n_envs, device=env.unwrapped.device, dtype=torch.bool)
    task_done_old = torch.zeros(n_envs, device=env.unwrapped.device, dtype=torch.bool)

    grasp_pcl = point_transform_3(base_grasp_pcl, env)
    tips_pcl = point_transform_3(base_tips_pcl, env)
    grasp_pcl_z = grasp_pcl.clone()
    grasp_pcl_z[:,:,2] = grasp_pcl[:,:,2] + 0.15
    grasp_pcl_z_move = grasp_pcl.clone()
    grasp_pcl_z_move[:,:,2] = grasp_pcl[:,:,2] + 0.05
    grasp_pcl_z_move[:,:,1] = grasp_pcl[:,:,1] - 0.04
    # simulate environment
    counter = 0
    counter_list = [0] * n_envs
    while simulation_app.is_running():
        # run everything in inference mode
        
        with torch.inference_mode():
            counter += 1
            tips_pcl = point_transform_3(base_tips_pcl, env)
            # tips_pcl = transform_points(base_tips_pcl, env, obj_name="object2").unsqueeze(1)
            #
            # grasp_visualizer.visualize(grasp_pcl.reshape(-1, 3))
            # tips_visualizer.visualize(tips_pcl.reshape(-1, 3))
            # pcl_visualizer.visualize(transformed_pcl.reshape(-1, 3))
            # grasp_visualizer_move.visualize(grasp_pcl_z_move.reshape(-1, 3))
            # print('Shapes')
            # print('Grasp_pcl', grasp_pcl.shape)
            # print('Tips_pcl', tips_pcl.shape)
            # print('Grasp_pcl_z', grasp_pcl_z.shape)
            # print('Grasp_pcl', grasp_pcl)
            # print('Tips_pcl', tips_pcl)
            # print('Grasp_pcl_z', grasp_pcl_z)

            grasped_z = torch.logical_or(on_pose(grasp_pcl_z, env), grasped_z)
            grasped = torch.logical_or(on_pose(grasp_pcl, env), grasped)
            grasped_z_move = torch.logical_or(on_pose(grasp_pcl_z_move, env), grasped_z_move)
            reached_x = torch.logical_or(tip_x_close(tips_pcl, env), reached_x)
            reached_y = torch.logical_or(tip_y_close(tips_pcl, env), reached_y)
            # print('grasped_z', grasped_z)
            # print('grasped', grasped)
            # print('reached_x', reached_x)
            # print('reached_y', reached_y)

            actions, modes = controller(n_envs, hand_closed, grasp_pcl_z, grasped_z, grasp_pcl_z_move, grasped_z_move, grasped, reached_x, reached_y, task_done, grasp_pcl, tips_pcl, env)
            for l in range(n_envs):
                if actions[l][-1] == -1:
                    counter_list[l] += 1
            #For each index in counter_list, if counter_list[index] > 20, set hand_closed[index] to True
            for m in range(n_envs):
                if counter_list[m] > 30:
                    hand_closed[m] = True
            #Clone the first action 3 times to test
            actions2  = actions[0].unsqueeze(0).repeat(n_envs, 1)

            observation, reward, done, truncated, _ = env.step(actions)

            task_done = torch.logical_or(done, task_done)
            print('task_done', task_done)

            print("Counter", counter)
            if all(task_done):
                # Write task_done to a json file
                task_done_list = task_done.tolist()
                task_done_dict = {"task_done": task_done_list}
                write_json_to_path(task_done_dict, os.path.join(args_cli.shape_data_path, "task_done.json"))
                env.close()
                simulation_app.close()

            if counter > 1500:
                # Write task_done to a json file
                task_done_list = task_done.tolist()
                task_done_dict = {"task_done": task_done_list}
                write_json_to_path(task_done_dict, os.path.join(args_cli.shape_data_path, "task_done.json"))
                env.close()
                simulation_app.close()
        ## Use if Camera point cloud is needed

        # if "rgb" in env.env.scene.sensors['camera'].data.output.keys():
        #     stick_segmentation = torch.where(env.env.scene.sensors['camera'].data.output["semantic_segmentation"] == 2, True, False)
        #     segmented_depth = torch.where(stick_segmentation, env.env.scene.sensors['camera'].data.output["depth"], torch.zeros_like(env.env.scene.sensors['camera'].data.output["depth"]))

        #     points_3d_cam = unproject_depth(
        #         segmented_depth, env.env.scene.sensors['camera'].data.intrinsic_matrices
        #     )
        #     points_3d_world = transform_points(points_3d_cam, env.env.scene.sensors['camera'].data.pos_w, env.env.scene.sensors['camera'].data.quat_w_ros)

        ## 

            # if points_3d_world.size()[0] > 0:
                
            #     pcl_visualizer.visualize(translations=points_3d_world.reshape(-1, 3))
            #breakpoint()

    # close the simulator
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()



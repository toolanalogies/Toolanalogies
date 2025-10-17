from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.isaac.lab.sim as sim_utils
import omni.usd
from pxr import Gf, Sdf
import random
import torch
import open3d as o3d
import numpy as np

from transform_utils import quaternion_mul, get_obj_list, point_transform_3, point_transform_2, inverse_transform_3, point_transform_to_ee, tip_x_close, tip_y_close, on_pose, read_json_from_path, write_json_to_path


def check_group(f, num: int):
    """Print the data from different keys in stored dictionary."""
    # print name of the group first
    for subs in f:
        if isinstance(subs, str):
            print("\t" * num, subs, ":", type(f[subs]))
            check_group(f[subs], num + 1)
    # print attributes of the group
    print("\t" * num, "attributes", ":")
    for attr in f.attrs:
        print("\t" * (num + 1), attr, ":", type(f.attrs[attr]), ":", f.attrs[attr])

def define_pcl_markers() -> VisualizationMarkers:

    """Define markers with various different shapes."""

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pointcloud_markers",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.005,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ),
        },
    )

    return VisualizationMarkers(marker_cfg)

def define_pose_markers() -> VisualizationMarkers:

    """Define markers with various different shapes."""

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pointcloud_markers",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.15, 0.15, 0.15),
            ),
        },
    )

    return VisualizationMarkers(marker_cfg)

def randomize_shape_color(prim_path_expr: str):

    """Randomize the color of the geometry."""
    # acquire stage
    stage = omni.usd.get_context().get_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
            # Note: Just need to acquire the right attribute about the property you want to set
            # Here is an example on setting color randomly
            color_spec = prim_spec.GetAttributeAtPath(prim_path + "/geometry/material/Shader.inputs:diffuseColor")
            color_spec.default = Gf.Vec3f(random.random(), random.random(), random.random())

def grasp_tips_points(points, env):

    origins = env.env.scene.env_origins
    B = origins.size(0)  # number of environments
    t = 0.5  # threshold distance
    M = 150
    # Step 1: Compute distances.
    # origins: [B, 3] -> [B, 1, 3] for broadcasting
    diff = points - origins.unsqueeze(1)  # [B, 2000, 3]
    distances = diff.norm(dim=2)          # [B, 2000]

    # Step 2: Create a boolean mask.
    grasp_values, grasp_indices = distances.topk(M, dim=1, largest=False)  # smallest M distances

    # dist_indices: [B, M] gives you the indices of the closest M points per environment.
    # Gather points at these indices:
    grasp_points = torch.gather(points, 1, grasp_indices.unsqueeze(-1).expand(B, M, 3))

    distances_x = points[:,:,1] - origins.unsqueeze(1)[:,:,1] #closest in y

    tip_values, tip_indices = distances_x.topk(300, dim=1, largest=True)  # smallest M distances

    # dist_indices: [B, M] gives you the indices of the closest M points per environment.
    # Gather points at these indices:
    tip_points = torch.gather(points, 1, tip_indices.unsqueeze(-1).expand(B, 300, 3))


    return grasp_points.mean(dim = 1), tip_points.mean(dim = 1)

def get_points_from_list(env, obj_path_list: list):
    
        points_list = []
        for obj_path in obj_path_list:
            points = torch.tensor(np.array(get_points_from_mesh(obj_path).points), device=env.unwrapped.device)
            # scale points with 0.6 in x and y axis
            points[:, 0] = points[:, 0] * 100.
            points[:, 1] = points[:, 1] * 150.
            # scale points with 0.8 in z axis
            points[:, 2] = points[:, 2] * 100.
            points_list.append(points)

        points_tensor = torch.stack(points_list, dim=0)
        return points_tensor

def get_points_from_properties(env, props_list):
    """
    Convert all handle/tip points to tensors on the correct device.
    Returns two tensors of shape [B, 3] for handles and tips.
    """
    device = env.unwrapped.device
    handles = []
    tips = []
    for props in props_list:
        handles.append(torch.tensor(props["handle_point"], device=device))
        tips.append(torch.tensor(props["tip_point"], device=device))
    handle_tensor = torch.stack(handles, dim=0)  # [B,3]
    tip_tensor = torch.stack(tips, dim=0)        # [B,3]
    return handle_tensor, tip_tensor

def get_points_from_mesh(mesh_path: str):

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertices():
        print("Mesh has no vertices.")
    elif not mesh.has_triangles():
        print("Mesh has no triangles.")
    else:
        print("Mesh loaded successfully!")
    print("Mesh has %d vertices." % len(mesh.vertices))
    print("Mesh has %d triangles." % len(mesh.triangles))
    # mesh.paint_uniform_color([1, 0.0, 0])
    points = mesh.sample_points_poisson_disk(number_of_points=2000)
    R = points.get_rotation_matrix_from_xyz((0.5 * np.pi, 0, 0))
    points = points.rotate(R, center=(0,0,0))
    points.scale(0.01, center=(0,0,0))
    #points.scale(0.8, center=(0,0,0))

    #points.rotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 1.5708))
    #points.translate([0. , 0. , -0.5])

    # o3d.visualization.draw_geometries([points])
    return points

def controller(n_envs, hand_closed_b, grasp_pcl_z, grasped_z_b, grasp_pcl_z_move, grasped_z_move_b, grasped_b, reached_x_b, reached_y_b , task_done_b, grasp_pcl, tips_pcl, env):
    target_q = torch.tensor([[0., 1., 0., 0.]], device=env.unwrapped.device)
    action_list = list()
    action_mode_list = list()
    # grasp_pcl_z[:,:,2] = grasp_pcl[:,:,2] + 0.4
    # grasp_pcl_z[:,:,1] = grasp_pcl[:,:,1] + 0.4
    # grasp_pcl_z[:,:,0] = grasp_pcl[:,:,0] + 0.2
    action = torch.tensor([0.04, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], device=env.unwrapped.device).unsqueeze(0).squeeze()
    for i in range(n_envs):
        grasped_z = grasped_z_b[i]
        grasped_z_move = grasped_z_move_b[i]
        grasped = grasped_b[i]
        hand_closed = hand_closed_b[i]
        reached_x = reached_x_b[i]
        reached_y = reached_y_b[i]
        task_done = task_done_b[i]
        # print('grasped_z', grasped_z)
        # print('grasped', grasped)
        # print('reached_x', reached_x)
        # print('reached_y', reached_y)
        # print('task_done', task_done)
        if not grasped_z:
            pos_distances, rot_distances = point_transform_to_ee(grasp_pcl_z, env, target_q)
            g = torch.ones(len(rot_distances), 1, device=env.unwrapped.device, dtype=torch.float32)
            grasp_actions = torch.cat((torch.clamp(pos_distances.squeeze(), -0.04, 0.04), torch.clamp(rot_distances, -0.08, 0.08), g), dim=1)
            ##JUST GET TF and TRANSFORM THE POINTS
            action = grasp_actions[i].squeeze()
            action_mode_list.append('grasped_z')
        elif not grasped:

            # sample actions from -1 to 1
            # actions = torch.tensor(action_list.pop(0), device=env.unwrapped.device)
            # actions = actions.unsqueeze(0)
            actions2 = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            # All the points are transformed to the world's frame 
            # transformed_pcl = point_transform_2(pcl_tensor, env)
            # pcl_visualizer.visualize(transformed_pcl.reshape(-1, 3))
            
            pos_distances, rot_distances = point_transform_to_ee(grasp_pcl, env, target_q)
            g = torch.ones(len(rot_distances), 1, device=env.unwrapped.device, dtype=torch.float32)
            grasp_actions = torch.cat((torch.clamp(pos_distances.squeeze(), -0.04, 0.04), torch.clamp(rot_distances, -0.08, 0.08), g), dim=1)
            ##JUST GET TF and TRANSFORM THE POINTS
            action = grasp_actions[i].squeeze()
            action_mode_list.append('grasped')

        elif not hand_closed:
            action = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], device=env.unwrapped.device).unsqueeze(0).squeeze()
            action_mode_list.append('hand_closed')

        elif not grasped_z_move:
            pos_distances, rot_distances = point_transform_to_ee(grasp_pcl_z_move, env, target_q)
            g = -1 * torch.ones(len(rot_distances), 1, device=env.unwrapped.device, dtype=torch.float32)
            grasp_actions = torch.cat((torch.clamp(pos_distances.squeeze(), -0.04, 0.04), torch.clamp(rot_distances, -0.08, 0.08), g), dim=1)
            ##JUST GET TF and TRANSFORM THE POINTS
            action = grasp_actions[i].squeeze()
            action_mode_list.append('grasped_z')
        
        elif not reached_x:
            action = torch.tensor([0.04, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], device=env.unwrapped.device).unsqueeze(0).squeeze()
            action_mode_list.append('reached_x')

        elif not reached_y:
            action = torch.tensor([0.0, 0.04, 0.0, 0.0, 0.0, 0.0, -1.0], device=env.unwrapped.device).unsqueeze(0).squeeze()
            action_mode_list.append('reached_y')

        elif not task_done:
            quat1 = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=env.unwrapped.device)
            quat2 = torch.tensor([[0.9238795, 0.0, 0.0, -0.3826834]], device=env.unwrapped.device)
            target_q = quaternion_mul(quat1, quat2)
            pos_distances, rot_distances = point_transform_to_ee(grasp_pcl, env, target_q)
            pos_command = torch.tensor([-0.04, -0.04, 0.000], device=env.unwrapped.device).unsqueeze(0).expand(rot_distances.size(0), -1)
            rot_command = torch.tensor([0.0, 0.00, 0.12], device=env.unwrapped.device).unsqueeze(0).expand(rot_distances.size(0), -1)
            g = torch.ones(len(rot_distances), 1, device=env.unwrapped.device, dtype=torch.float32) * -1
            #action = torch.cat((pos_command, torch.clamp(rot_distances, -0.08, 0.08), g), dim=1)[i].squeeze()
            action = torch.cat((pos_command, rot_command, g), dim=1)[i].squeeze()
            action_mode_list.append('task_done')

        action_list.append(action)
    return torch.stack(action_list, dim=0), action_mode_list

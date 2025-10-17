import torch

import os
import json

def get_obj_list(data_folder):
    """
    1. Recursively walk through `data_folder`.
    2. Whenever we find exactly one `.usd` and one `.txt` in the same folder, 
       we pair them up.
    3. Return two parallel lists:
         usd_list[i]  -> path to the .usd file
         txt_list[i]  -> contents (or path) of the matching .txt file
    """
    obj_list = []

    for root, dirs, files in os.walk(data_folder):

        # Collect all .usd and .txt files in this folder
        obj_files = [f for f in files if f.endswith(".obj")]
        #txt_files = [f for f in files if f.endswith(".txt")]

        # For simplicity, assume there is exactly ONE .usd and ONE .txt per folder.
        # If you expect more, adapt the logic.
        if len(obj_files) == 1:
            pbj_file_path = os.path.join(root, obj_files[0])
            obj_list.append(pbj_file_path)
    obj_list.sort()
    return obj_list

def get_properties_list(data_folder):
    """
    Walk through `data_folder` and collect all properties_value JSONs.
    Returns a list of dicts with {path, handle_point, tip_point}.
    """
    props_list = []
    for root, dirs, files in os.walk(data_folder):
        for f in files:
            if f.startswith("properties_value") and f.endswith(".json"):
                json_path = os.path.join(root, f)
                with open(json_path, "r") as jf:
                    data = json.load(jf)
                props_list.append({
                    "path": json_path,
                    "part": data.get("part"),
                    "handle_point": data.get("handle_point"),
                    "tip_point": data.get("tip_point")
                })
    props_list.sort(key=lambda x: x["path"])
    return props_list

def read_json_from_path(json_path):
    """
    Reads a JSON file from a known path and returns the data as a dictionary.
    """
    if not os.path.isfile(json_path):
        print(f"File {json_path} does not exist.")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def write_json_to_path(data, json_path):
    # Ensure the directory exists before writing
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Write data to the JSON file
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"JSON file written to {json_path}")


def quaternion_to_rotation_vector_batch(quats: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of quaternions to axis-angle rotation vectors.
    Input:
        quats: tensor of shape (B, 4) in the format (w, x, y, z)
    Output:
        rot_vecs: tensor of shape (B, 3) representing rotation vectors (axis * angle)
    """
    # Ensure normalization if needed (optional if your quaternions are already normalized)
    # quats = quats / quats.norm(p=2, dim=-1, keepdim=True)

    w = quats[:, 0]
    x = quats[:, 1]
    y = quats[:, 2]
    z = quats[:, 3]

    # Compute the angle
    # Clamp w to [-1,1] to avoid numerical issues outside the domain of arccos due to floating-point errors
    w_clamped = torch.clamp(w, -1.0, 1.0)
    theta = 2.0 * torch.acos(w_clamped)

    # For the axis, we need sin(theta/2). Using sin(theta/2) = sqrt(1 - w^2)
    # Actually, sqrt(1 - w^2) = sin(theta/2) directly
    sin_half_theta = torch.sqrt(1.0 - w_clamped * w_clamped)

    # To avoid division by zero, define a small epsilon
    eps = 1e-8
    # If sin_half_theta is very small, angle is near 0 => rotation vector ~ 0
    # Otherwise, axis = (x, y, z)/sin_half_theta
    valid = sin_half_theta > eps
    axis = torch.zeros_like(quats[:, 1:4])
    axis[valid] = quats[valid, 1:4] / sin_half_theta[valid].unsqueeze(-1)
    
    # Rotation vector = theta * axis
    rot_vecs = axis * theta.unsqueeze(-1)

    return rot_vecs

def quaternion_difference_batch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # Normalize to unit quaternions if they represent rotations
    q1 = q1 / q1.norm(dim=1, keepdim=True)
    q2 = q2 / q2.norm(dim=1, keepdim=True)

    # Inverse of q2
    q2_inv = torch.cat([q2[:, :1], -q2[:, 1:]], dim=1)

    # Quaternion multiplication function
    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
        w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack([w, x, y, z], dim=1)

    # Compute q_diff
    q_diff = quaternion_multiply(q1, q2_inv)
    return q_diff

def rotation_matrices_to_quaternions_batch(R: torch.Tensor) -> torch.Tensor:
    # R: (B, 3, 3)
    # Compute trace for each batch element
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # To avoid numerical issues, ensure the trace argument of sqrt is non-negative
    w = 0.5 * torch.sqrt(torch.clamp_min(1.0 + trace, 1e-8))

    # Compute quaternion components
    # These are vectorized across the batch dimension:
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4.0 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4.0 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4.0 * w)

    # Stack into a quaternion tensor (B, 4)
    q = torch.stack([w, x, y, z], dim=1)
    return q

def quaternion_to_rotation_matrix_batch(quaternions: torch.Tensor) -> torch.Tensor:
    # quaternions: (B,4)
    # Normalize if needed
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)
    w, x, y, z = quaternions.unbind(dim=1)

    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    # Rotation matrices: (B,3,3)
    rot = torch.stack([
        1 - 2*(y2 + z2), 2*(xy - wz),       2*(xz + wy),
        2*(xy + wz),     1 - 2*(x2 + z2),   2*(yz - wx),
        2*(xz - wy),     2*(yz + wx),       1 - 2*(x2 + y2)
    ], dim=1).view(-1, 3, 3)

    return rot

def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion[0]
    """Convert quaternion to rotation matrix."""
    # unpack quaternion
    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    # compute rotation matrix
    rotation_matrix = torch.tensor(
        [
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)],
        ],
        device=quaternion.device,
    )
    return rotation_matrix

def point_transform(points, env):
    translation = env.env.scene.rigid_objects['object2'].data.body_pos_w[:, :3]
    quaternion = env.env.scene.rigid_objects['object2'].data.body_quat_w[:, :4]
    # rotation = quaternion_to_rotation_matrix_2(quaternion)
    # points = points.unsqueeze(0)
    # # Repeat along the new dimension to get [2, 2000, 3]
    # points = points.repeat(len(quaternion), 1, 1)
    # points = torch.bmm(points.float(), rotation.T) + translation

    points_list = []
    for i in range(len(quaternion)):
        rotation = quaternion_to_rotation_matrix(quaternion[i])
        translation_i = translation[i,0]
        points_i = torch.matmul(points.float(), rotation.T) + translation_i
        points_list.append(points_i)

    points = torch.cat(points_list, dim=0)
    return points

def transform_points(points, env, obj_name="object2"):
    """
    Apply rigid-body transforms (quaternion + translation) to a [B,3] batch of points.
    """
    points = points.float()
    translation = env.env.scene.rigid_objects[obj_name].data.body_pos_w[:, :3].squeeze()   # [B,3]
    quaternion = env.env.scene.rigid_objects[obj_name].data.body_quat_w[:, :4].squeeze()  # [B,4]

    rotations = quaternion_to_rotation_matrix_batch(quaternion)  # [B,3,3]

    # [B,3] -> [B,1,3] for broadcast, then multiply by rotation
    rotated = torch.matmul(points.unsqueeze(1), rotations.transpose(1,2)).squeeze(1)  # [B,3]
    transformed = rotated + translation
    return transformed

def point_transform_2(points, env):
    # Convert points to float
    points = points.float()
    # Extract translation and quaternion
    translation = env.env.scene.rigid_objects['object2'].data.body_pos_w[:, :3].squeeze()  # shape: (B,3)
    quaternion = env.env.scene.rigid_objects['object2'].data.body_quat_w[:, :4].squeeze()  # shape: (B,4)

    # Get batch rotation matrices
    rotations = quaternion_to_rotation_matrix_batch(quaternion)  # (B,3,3)

    # Expand points to apply each rotation/translation
    # points: (N,3) -> (B,N,3)
    if points.dim() == 2:
        points = points.unsqueeze(0).expand(quaternion.size(0), -1, -1).float()


    # Apply rotation: (B,N,3) x (B,3,3) = (B,N,3)
    # Note: we use rotation.transpose(1,2) to match dimensions: (B,N,3)*(B,3,3)
    rotated_points = torch.matmul(points, rotations.transpose(1, 2))

    # Apply translation: (B,N,3) + (B,1,3) = (B,N,3)
    rotated_translated_points = rotated_points + translation.unsqueeze(1)

    # If you want the same final shape as before (concatenated), flatten B and N
    # This will yield (B*N, 3)
    points = rotated_translated_points#.reshape(-1, 3)

    return points

def inverse_transform_3(points, env):
    # Extract translation and quaternion
    translation = env.env.scene.rigid_objects['object2'].data.body_pos_w[:, :3].squeeze()  # shape: (B,3)
    quaternion = env.env.scene.rigid_objects['object2'].data.body_quat_w[:, :4].squeeze()  # shape: (B,4)

    # Get batch rotation matrices
    rotations = quaternion_to_rotation_matrix_batch(quaternion)  # (B,3,3)

    # Expand points to apply each rotation/translation
    # points: (B,3) -> (B,1,3)
    if points.dim() == 2:
        points = points.unsqueeze(1)

    points = points - translation.unsqueeze(1)

    rotated_points = torch.matmul(points, rotations)

    points = rotated_points#.reshape(-1, 3)

    return points

def point_transform_3(points, env):
    # Extract translation and quaternion
    translation = env.env.scene.rigid_objects['object2'].data.body_pos_w[:, :3].squeeze()  # shape: (B,3)
    quaternion = env.env.scene.rigid_objects['object2'].data.body_quat_w[:, :4].squeeze()  # shape: (B,4)

    # Get batch rotation matrices
    rotations = quaternion_to_rotation_matrix_batch(quaternion)  # (B,3,3)

    # Expand points to apply each rotation/translation
    # points: (B,3) -> (B,1,3)
    if points.dim() == 2:
        points = points.unsqueeze(1)


    # Apply rotation: (B,N,3) x (B,3,3) = (B,N,3)
    # Note: we use rotation.transpose(1,2) to match dimensions: (B,N,3)*(B,3,3)
    rotated_points = torch.matmul(points, rotations.transpose(1, 2))

    # Apply translation: (B,N,3) + (B,1,3) = (B,N,3)
    rotated_translated_points = rotated_points + translation.unsqueeze(1)

    # If you want the same final shape as before (concatenated), flatten B and N
    # This will yield (B*N, 3)
    points = rotated_translated_points#.reshape(-1, 3)

    return points


def quaternion_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions p and q in (x, y, z, w) format.
    p, q: Tensors of shape (..., 4).
    Returns: A tensor of shape (..., 4) representing p*q.
    """
    px, py, pz, pw = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Compute the product
    rw = pw * qw - px * qx - py * qy - pz * qz
    rx = pw * qx + px * qw + py * qz - pz * qy
    ry = pw * qy - px * qz + py * qw + pz * qx
    rz = pw * qz + px * qy - py * qx + pz * qw

    # Return in (x, y, z, w) format
    return torch.stack((rx, ry, rz, rw), dim=-1)

def on_pose(grasp_pcl, env):
    target_q = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=env.unwrapped.device)
    pos_distances, rot_distances = point_transform_to_ee(grasp_pcl, env, target_q)
    total_distance = torch.cat((pos_distances.squeeze(), rot_distances), dim=1)

    z = torch.zeros(len(rot_distances), 6, device=env.unwrapped.device, dtype=torch.float32)
    rowwise_comparison = torch.all(torch.isclose(total_distance, z, rtol=0.03, atol=0.03), dim=1)
    return rowwise_comparison

def tip_x_close(tips_pcl, env):
    translation = env.env.scene.rigid_objects['object'].data.body_pos_w[:, :, :3].squeeze()[:,0]  # shape: (B,3)
    ones = torch.ones(len(tips_pcl), 1, device=env.unwrapped.device, dtype=torch.float32).squeeze() * 0.1
    # return torch.isclose(tips_pcl.squeeze()[:,0], translation + ones, rtol=1e-2, atol=1e-2)
    return torch.greater(tips_pcl.squeeze()[:,0], translation + ones)

def tip_y_close(tips_pcl, env):
    translation = env.env.scene.rigid_objects['object'].data.body_pos_w[:, :, :3].squeeze()[:,1]  # shape: (B,3)
    ones = torch.ones(len(tips_pcl), 1, device=env.unwrapped.device, dtype=torch.float32).squeeze() * 0.01
    #return torch.all(torch.isclose(tips_pcl.squeeze(), translation - ones, rtol=0.1, atol=0.1))#[:,1], translation - ones, rtol=1e-2, atol=1e-2)
    return torch.greater(tips_pcl.squeeze()[:,1], translation - ones)


def point_transform_to_ee(points, env, target_q):

    ee_translation = env.env.scene.sensors['ee_frame'].data.target_pos_w[:, :3]  # shape: (B,1,3)
    ee_quaternion = env.env.scene.sensors['ee_frame'].data.target_quat_w[:, :4].squeeze()  # shape: (B,4)

    
    target_quaternion = target_q.unsqueeze(0).expand(ee_translation.size(0), -1, -1).squeeze()  # shape: (B,4)
    target_translation = points  # shape: (B,1,3)

    position_diff = target_translation - ee_translation
    # Get batch rotation matrices
    # ee_rotations = quaternion_to_rotation_matrix_batch(ee_quaternion)  # (B,3,3)
    # target_rotations = quaternion_to_rotation_matrix_batch(target_quaternion)  # (B,3,3)
    # rotation_diff = torch.matmul(target_rotations, ee_rotations.transpose(1, 2))

    # quaternion_diff = rotation_matrices_to_quaternions_batch(rotation_diff)
    quaternion_diff = quaternion_difference_batch(target_quaternion, ee_quaternion)
    # Lets check if the quaternion difference is the same as the rotation difference
    # print(torch.allclose(quaternion_diff, quaternion_diff2, atol=1e-6))
    rotation_diff = quaternion_to_rotation_vector_batch(quaternion_diff)

    return position_diff, rotation_diff
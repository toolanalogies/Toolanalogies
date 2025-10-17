import open3d as o3d
import numpy as np
import copy
import trimesh

def get_z_origin_mean(pcd):
    """Get mean z-value of the point cloud."""
    points = np.asarray(pcd.points)
    return points[:, 2].mean()

def get_z_origin_histogram(pcd, bin_size=0.005):
    """Estimate z-origin using histogram mode."""
    points = np.asarray(pcd.points)
    z_vals = points[:, 2]
    
    z_min, z_max = z_vals.min(), z_vals.max()
    bins = np.arange(z_min, z_max + bin_size, bin_size)
    hist, bin_edges = np.histogram(z_vals, bins=bins)
    
    max_bin_idx = np.argmax(hist)
    z_origin = 0.5 * (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1])
    return z_origin

def flip_point_cloud_about_z(pcd, z_origin):
    """Flip the point cloud about the plane z = z_origin."""
    pcd_flipped = copy.deepcopy(pcd)
    points = np.asarray(pcd_flipped.points)
    
    # Shift, reflect, shift back
    points[:, 2] -= z_origin
    points[:, 2] *= -1
    points[:, 2] += z_origin
    
    pcd_flipped.points = o3d.utility.Vector3dVector(points)
    # Flip normals as well
    return pcd_flipped

def load_point_cloud(path):
    """Load a point cloud from file."""
    pcd = o3d.io.read_point_cloud(path)
    return pcd


def merge_point_clouds(pcd1, pcd2):
    """Merge two point clouds."""
    merged = pcd1 + pcd2
    return merged

def estimate_normals(pcd, radius=0.01, max_nn=30):
    """Estimate normals for a point cloud."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_consistent_tangent_plane(30)
    return pcd

def alpha_shape_reconstruction(pcd, alpha=0.02):
    print("Running Alpha Shape reconstruction...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    return mesh



def run_pipeline(input_path):
    # Load
    pcd = load_point_cloud(input_path)

    # Flip
    z_origin = get_z_origin_mean(pcd)
    flipped_pcd = flip_point_cloud_about_z(pcd, z_origin)
    
    # Merge
    merged_pcd = merge_point_clouds(pcd, flipped_pcd)
    surface_mesh = alpha_shape_reconstruction(merged_pcd, alpha=0.03)
    return surface_mesh

if __name__ == "__main__":

	# save the o3d mesh
	input_mesh_path = "input.ply"
	output_mesh_path = "output.obj"
	surface_mesh = run_pipeline(input_mesh_path)
	o3d.io.write_triangle_mesh(output_mesh_path, surface_mesh)


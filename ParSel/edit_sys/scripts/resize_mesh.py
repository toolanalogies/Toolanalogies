
import argparse
import os
import sys
import open3d as o3d

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Load all .obj files from a specified directory using Open3D."
    )
    parser.add_argument(
        "path", 
        type=str, 
        help="Path to the directory containing .obj files"
    )
    args = parser.parse_args()

    # Extract the directory path
    directory_path = args.path

    # Check if path exists and is a directory
    if not os.path.exists(directory_path):
        print(f"Error: The path '{directory_path}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a directory.")
        sys.exit(1)

    # Get the list of .obj files in the directory
    obj_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".obj")]
    if not obj_files:
        print(f"No .obj files found in '{directory_path}'.")
        sys.exit(0)

    # Loop through each .obj file, load it, and print some basic info
    for obj_file in obj_files:
        obj_path = os.path.join(directory_path, obj_file)

        # Load the .obj file as an Open3D TriangleMesh
        print(f"Loading: {obj_path}")
        mesh = o3d.io.read_triangle_mesh(obj_path)
        if not mesh.is_empty():
            # Print out information about the mesh, e.g., bounding box
            aabb = mesh.get_axis_aligned_bounding_box()
            print(f"  - Bounding box: {aabb}")
            print(f"  - # of vertices: {len(mesh.vertices)}")
            print(f"  - # of triangles: {len(mesh.triangles)}\n")
            mesh.scale(0.01, center = (0,0,0))
            o3d.io.write_triangle_mesh(obj_path, mesh, write_ascii=False, compressed=False) 
        else:
            print("  - Failed to load or empty mesh.\n")

if __name__ == "__main__":
    main()
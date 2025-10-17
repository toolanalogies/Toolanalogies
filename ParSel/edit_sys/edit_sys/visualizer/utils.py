import open3d as o3d
import numpy as np
import copy
import torch as th
import sympy as sp
from .constants import COLORS
from edit_sys.data_loader.partnet import obb_to_mesh
from edit_sys.shape_system.shape_atoms import PART_INACTIVE, PART_ACTIVE
from edit_sys.shape_system.shape_atoms import Part

INDEX_BUMP = 100
INDEX_MULT = 10

def create_geom_map(parts_dict, relations_dict, intersection_dict, add_color=True):
    # create a name to object map.
    # additionally create a name to state map. 
    # Iff the state changes, remove or add the object. 
    name_to_geom = {}
    label_dict = {}
    for ind, part in parts_dict.items():
        name = f"{ind}_mesh"
        if add_color:
            for x in part['mesh'].meshes:
                x.mesh.paint_uniform_color(COLORS[ind])
        name_to_geom[name] = part['mesh']


        name = f"{ind}_obb"
        obb_mesh = obb_to_mesh(part['obb'])
        obb_mesh.compute_vertex_normals()
        # get the normals
        normals = np.array(obb_mesh.triangle_normals)
        # flip
        normals = -normals
        obb_mesh.triangle_normals = o3d.utility.Vector3dVector(normals)
        obb_mesh.paint_uniform_color(COLORS[ind])
        name_to_geom[name] = obb_mesh
        
        label_dict[f"{ind}_3d_label"] = (part['label'], part['obb'].center, None)

    for ind, relation in relations_dict.items():
        if relation['type'] == "R":
            axis, origin = relation['params']
            # make an arrow from origin to origin + axis
            relation_geom = create_arrow(origin, origin + axis)
        elif relation['type'] == "T":
            delta = relation['params'][0]
            part_ind = relation['indices'][0]
            part = parts_dict[part_ind]
            center = part['obb'].center
            relation_geom = create_arrow(center, center + delta)
        elif relation['type'] == "REF":
            # create plane
            plane_axis, plane_origin = relation['params']
            relation_geom = create_plane(plane_axis, plane_origin)
        name = f"{ind}_R"
        name_to_geom[name] = relation_geom
        label_dict[name + "_3d_label"] = (relation['name'], relation_geom.get_center(), None)

    for ind, intersection in intersection_dict.items():
        name = f"{ind}_I"
        point_cloud = o3d.geometry.PointCloud()
        points = intersection['points']
        point_cloud.points = o3d.utility.Vector3dVector(points)
        color = np.array([1, 0, 0])
        color_set = np.tile(color, (points.shape[0], 1))
        point_cloud.colors = o3d.utility.Vector3dVector(color_set)
        name_to_geom[name] = point_cloud
        # unfortunately 3D label has to be toggled separately
        label_dict[name + "_3d_label"] = (intersection['name'], point_cloud.get_center(), None)
    return name_to_geom, label_dict

def create_switch_board(name_to_geom, label_dict):
    switch_board = {}
    for name, geom in name_to_geom.items():
        switch_board[name] = False
    for name, (label, center, obj) in label_dict.items():
        switch_board[name] = False
    return switch_board

def update_switch_board(name_to_geom, label_dict, part_dict, shape):

    switch_board = {}
    for name, geom in name_to_geom.items():
        ind = int(name.split("_")[0])
        item_type = name.split("_")[-1]
        if item_type in ['mesh']:
            part_ind = ind
            part = [x for x in shape.partset if x.part_index == part_ind ][0]
            state = part.state[0]
            if state == PART_INACTIVE:
                switch_board[name] = False
            elif state == PART_ACTIVE:
                switch_board[name] = True
        else:
            switch_board[name] = False
    for name, (label, center, obj) in label_dict.items():
        ind = int(name.split("_")[0])
        item_type = name.split("_")[1]
        if item_type == "3d":
            part = [x for x in shape.partset if x.part_index == part_ind ][0]
            state = part.state[0]
            if state == PART_INACTIVE:
                switch_board[name] = False
            elif state == PART_ACTIVE:
                switch_board[name] = True
        else:
            switch_board[name] = False
    return switch_board

def create_arrow(start, end):
    # keep 75% as cylinder and 25% as cone.
    length = np.linalg.norm(end - start)
    cylinder_length = 0.75 * length
    cone_length = 0.25 * length
    cylinder_radius = 0.02
    cone_radius = 0.04
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius, cone_radius, cylinder_length, cone_length)
    arrow.compute_vertex_normals()
    axis = end - start
    axis = axis / np.linalg.norm(axis)
    z_axis = np.array([0, 0, 1])
    angle = np.arccos(np.dot(z_axis, axis))
    axis = np.cross(z_axis, axis)
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(start)
    
    return arrow

def create_plane(normal, origin, size=1.0):
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=0.01, depth=size)

    # Align the mesh with the normal vector
    # This is done by finding a rotation matrix that aligns the z-axis with the normal vector
    z_axis = np.array([0, 1, 0])
    axis = np.cross(z_axis, normal)
    angle = np.arccos(np.dot(z_axis, normal))
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    centering = np.array([size/2, 0.005, size/2])
    mesh.translate(-centering)
    
    mesh.rotate(R, center=(0, 0, 0))
    mesh.translate(origin)

    # Translate the mesh so that the given point lies on the plane
    mesh.compute_vertex_normals()
    return mesh


def update_model(shape, parts_dict, symbol_to_parts, 
                 symbol_value_dict, cube_coord_dict, name_to_geom, 
                 device="cuda"):
    multiplicity = 0
    for symbol, value in symbol_value_dict.items():
        relevant_parts = symbol_to_parts[symbol]
        for part in relevant_parts:
            if len(part.partset) > 0:
                # see which kind of execution is required.
                core_relation = part.core_relation
                output_bboxes = core_relation.execute_relation(symbol_value_dict)
                n_boxes = len(output_bboxes)
                if n_boxes > 1:
                    # The part count might be different.
                    # Re do this part.
                    multiplicity += 1
                    print("Multiple boxes found. Re-doing the part.")
                    parent_part = part
                    to_remove_indices = [x.part_index for x in parent_part.sub_parts]
                    # just remove all children
                    parent_part.sub_parts = {}
                    child = part.core_relation.primitives[0].part

                    original_index = child.part_index
                    original_part_dict = parts_dict[original_index]

                    cur_coords = cube_coord_dict[original_index]

                    new_index = len(shape.partset) + INDEX_BUMP * (multiplicity + 1)
                    new_children = []
                    output_bboxes = np.stack([np.asarray(x) for x in output_bboxes], axis=0).astype(np.float32)
                    output_bboxes = th.from_numpy(output_bboxes).to(device)
                    for index in range(n_boxes):
                        real_th = output_bboxes[index]

                        vertices = cur_coords[..., None] * real_th[None, ...]
                        vertices = vertices.sum(dim=1)

                        label = f"{child.original_label}_{index}"
                        cur_index = new_index + index
                        new_child = Part(label, child.primitive, part_index=cur_index,
                                        has_children=False, original_label=child.original_label,
                                        mode=child.mode)
                        new_children.append(new_child)
                        # replicate part dict
                        original_tm = original_part_dict[f"mesh"]
                        new_tm = o3d.visualization.rendering.TriangleMeshModel()
                        new_tm.materials = [x for x in original_tm.materials]
                        new_meshes = []
                        vertice_count = 0
                        for mesh_info in original_tm.meshes:
                            cur_mesh = copy.deepcopy(mesh_info.mesh)
                            material_idx = mesh_info.material_idx
                            new_mesh_info = o3d.visualization.rendering.TriangleMeshModel.MeshInfo(cur_mesh, "", material_idx)
                            cur_n_vertices = len(mesh_info.mesh.vertices)
                            cur_vertices = vertices[vertice_count: vertice_count + cur_n_vertices]
                            cur_vertices = cur_vertices.cpu().numpy().astype(np.float64)
                            creation = o3d.utility.Vector3dVector(cur_vertices)
                            new_mesh_info.mesh.vertices = creation
                            new_meshes.append(new_mesh_info)
                        new_tm.meshes = new_meshes
                        new_part_dict = dict(mesh=new_tm, obb=original_part_dict['obb'], label=child.original_label,
                                             objs=original_part_dict['objs'], )
                        parts_dict[cur_index] = new_part_dict
                        name_to_geom[f"{cur_index}_mesh"] = new_tm
                        # Other updates
                        cube_coord_dict[cur_index] = cube_coord_dict[original_index]
                    new_children = set(new_children)
                    parent_part.sub_parts = new_children
                    parent_part.state[0] = PART_INACTIVE

    return shape, parts_dict


def _update_part_mesh(part_id, part_eq, cube_coord_dict, 
                      symbol_value_dict, name_to_geom, device):
    mesh_name = f"{part_id}_mesh"
    cur_cube_coords = cube_coord_dict[part_id]
    realization = part_eq.subs(symbol_value_dict)
    real_np = np.asarray(realization).astype(np.float32)
    real_th = th.from_numpy(real_np).to(device)
    vertices = cur_cube_coords[..., None] * real_th[None, ...]
    vertices = vertices.sum(dim=1).cpu().numpy().astype(np.float64)
    corresponding_mesh = name_to_geom[mesh_name]
    
    vertice_count = 0
    for mesh_info in corresponding_mesh.meshes:
        cur_n_vertices = len(mesh_info.mesh.vertices)
        cur_vertices = vertices[vertice_count: vertice_count + cur_n_vertices]
        creation = o3d.utility.Vector3dVector(cur_vertices)
        mesh_info.mesh.vertices = creation
        vertice_count += cur_n_vertices


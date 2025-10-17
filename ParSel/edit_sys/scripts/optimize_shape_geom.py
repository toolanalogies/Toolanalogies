import json
import os
from pathlib import Path
import sys
import open3d as o3d
import numpy as np
import torch as th
import networkx as nx
import sympy as sp
import copy
import _pickle as cPickle

from edit_sys.data_loader.partnet_shape import get_obj
from scripts.shapetalk_data import load_valid_shapetalk_utterances
from edit_sys.shape_system.constants import PART_INACTIVE
from edit_sys.data_loader.partnet_shape import get_obj, local_to_cube, global_to_local
from edit_sys.visualizer.utils import create_geom_map
from pytorch3d.loss import chamfer_distance
from edit_sys.shape_system.proposer_utils import get_vert_face_map, Deformer
from edit_sys.shape_system.relations import PrimitiveRelation, FeatureRelation
from edit_sys.shape_system.shape_atoms import Part
N_POINTS = 2048
DEVICE = th.device("cuda:0")
LR = 0.001
N_ITERS = 1000

def rejection_creteria(shape, part_ind, avoid_ground=True, avoid_inactive=True):
    reject = False
    corresonding_part = [x for x in shape.partset if x.part_index == part_ind][0]
    if avoid_inactive and corresonding_part.state[0] == PART_INACTIVE:
        reject = True
    part_name = corresonding_part.label
    if avoid_ground and part_name == "ground":
        reject = True
    return reject

def get_pc_dict(part_dicts, shape, apply_cube=True, avoid_ground=True, avoid_inactive=False):
    mesh_dict = {}
    for part_ind, part_dict in part_dicts.items():
        reject = rejection_creteria(shape, part_ind, avoid_ground, avoid_inactive)
        if reject:
            continue
        mesh = o3d.geometry.TriangleMesh()
        for mini_mesh in part_dict['mesh'].meshes:
            mesh += mini_mesh.mesh
        
        area = mesh.get_surface_area()
        mesh_dict[part_ind] = (mesh, area)

    total_area = sum([area for mesh, area in mesh_dict.values()])

    pc_dict = {}
    for part_ind, (mesh, area) in mesh_dict.items():
        
        reject = rejection_creteria(shape, part_ind, avoid_ground, avoid_inactive)
        if reject:
            continue
        area_fraction = area / total_area
        total_points = int(N_POINTS * area_fraction)
        total_points = max(total_points, 1)
        pcs = mesh.sample_points_poisson_disk(total_points)
        points = np.asarray(pcs.points)
        if apply_cube:
            obb = part_dicts[part_ind]['obb']
            local_coords = global_to_local(points, obb) 
            cube_coords = local_to_cube(local_coords)
            points = th.from_numpy(cube_coords).float().to(DEVICE)
        else:
            points = th.from_numpy(points).float().to(DEVICE)
        pc_dict[part_ind] = points
    return pc_dict

def bbox_to_pc(bboxes_holder, source_pc_dict, shape, avoid_ground=True):
    out_cloud = []
    for (part_index, static_th) in bboxes_holder:

        reject = rejection_creteria(shape, part_index, avoid_ground)
        if reject:
            continue
        part = source_pc_dict[part_index]
        part = th.matmul(part, static_th)
        out_cloud.append(part)
    out_cloud = th.cat(out_cloud, dim=0)
    out_cloud = out_cloud.unsqueeze(0)
    return out_cloud

def get_base_opt_tuple(shape):

    bboxes_holder = []
    for part in shape.partset:
        if len(part.partset) > 0:
            continue
        else:
            static_expr = part.primitive.static_expression()
            static_np = np.array(static_expr).astype(np.float32)
            static_th = th.tensor(static_np, device=DEVICE, requires_grad=True)
            bboxes_holder.append((part.part_index, static_th))

    optimizable_params = [static_expr for _, static_expr in bboxes_holder]
    return optimizable_params, bboxes_holder

def save_deformed_meshes(processed_data, symbolic, bbox_holder, folder_name, symbol_to_parts, value):

    shape = symbolic[0]
    part_dict = processed_data[0]

    name_to_geom, _ = create_geom_map(*processed_data)

    # only if any edits are made to the counts
    cube_coord_dict = {}
    for part in shape.partset:
        part_index = part.part_index
        corresponding_mesh = name_to_geom[f"{part_index}_mesh"]
        obb = part_dict[part_index]['obb']
        point_set = [np.asarray(x.mesh.vertices) for x in corresponding_mesh.meshes]
        # points = np.asarray(corresponding_mesh.meshes[0].mesh.vertices)
        points = np.concatenate(point_set, axis=0)
        local_coords = global_to_local(points, obb) 
        cube_coords = local_to_cube(local_coords)
        # cube_coords = np.contiguousarray(cube_coords)
        cube_coords = th.from_numpy(cube_coords).float().to(DEVICE)
        cube_coord_dict[part_index] = cube_coords

    # Now for all parts, get mesh with the latest bboxes
    for part_id, real_th in bbox_holder:
        mesh_name = f"{part_id}_mesh"
        cur_cube_coords = cube_coord_dict[part_id].clone()
        real_th = real_th.detach()
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
    
    # TODO: ADD a UV map updated here.
    # Now in the save dir, we need to save the meshes
    obj_save_dir = os.path.join(folder_name, "objs")
    Path(obj_save_dir).mkdir(parents=True, exist_ok=True)
    save_dicts = []
    for part in shape.partset:
        if part.state[0] == PART_INACTIVE:
            continue
        part_index = part.part_index
        corresponding_mesh = name_to_geom[f"{part_index}_mesh"]
        # save TriangleModelFile
        obj_file = os.path.join(obj_save_dir, f"{part_index}.obj")
        mesh = o3d.geometry.TriangleMesh()
        for cur_mesh in corresponding_mesh.meshes:
            mesh += cur_mesh.mesh
        o3d.io.write_triangle_mesh(obj_file, mesh)
        obj_names = [f"{part_index}.obj"]
        cur_part_dict = {
            'name': part.original_label,
            'objs': obj_names,
            'children': [],
        }
        save_dicts.append(cur_part_dict)

    parent_node = {'objs': [], 'name': shape.label, 
                   'children': save_dicts, 'hier': 0}
    
    data_file = os.path.join(folder_name, "result_after_merging.json")
    with open(data_file, "w") as f:
        json.dump([parent_node], f)

def get_relation_signatures(post_shape):
    post_relation_signatures = []
    for relation in post_shape.all_relations():
        if isinstance(relation, PrimitiveRelation):
            parts = [x.part.label for x in relation.primitives]
        elif isinstance(relation , FeatureRelation):
            parts = [x.primitive.part.label for x in relation.features]
        parts = sorted(parts)
        parts_str = "_".join(parts)
        relation_signature = f"{relation.__class__.__name__}_{parts_str}"
        post_relation_signatures.append(relation_signature)
    post_relation_signatures = set(post_relation_signatures)
    return post_relation_signatures

def get_cd_and_distortion(match_parts_source, match_parts_target, source_shape, source_pc_dict, original_source_pc_dict, target_shape, target_pc_dict, bbox_holder):
    source_cd = []
    source_dist = []
    all_target_match_source_pcs = []
    for part in source_shape.partset:
        if part.state[0] == PART_INACTIVE:
            continue
        label = part.label
        source_pc = source_pc_dict[part.part_index]
        static_th = [x[1] for x in bbox_holder if x[0] == part.part_index][0]
        source_pc = th.matmul(source_pc, static_th)
        if label in match_parts_source:
            all_target_match_source_pcs.append(source_pc)
        else:
            source_pc = source_pc - source_pc.mean(dim=0)
            source_pc = source_pc.unsqueeze(0)
            original_pc = original_source_pc_dict[part.part_index]
            # recenter
            original_pc = original_pc - original_pc.mean(dim=0)
            original_pc = original_pc.unsqueeze(0)
            loss, _ = chamfer_distance(source_pc, original_pc)
            source_cd.append(loss.item())
            # Next also measure bbox arap distortion.
            static = part.primitive.static_expression()
            static_np = np.asarray(static).astype(np.float32)
            v_map, F = get_vert_face_map(part)
            seq_points = static_np[v_map]
            dynamic = [x[1] for x in bbox_holder if x[0] == part.part_index][0]
            dynamic_np = dynamic.detach().cpu().numpy()
            dynamic_points = dynamic_np[v_map]
            deform = Deformer(seq_points, dynamic_points, F)
            source_dist.append(deform.energy)
    
    all_target_pcs = []
    for part in target_shape.partset:
        if part.state[0] == PART_INACTIVE:
            continue
        label = part.label
        target_pc = target_pc_dict[part.part_index]
        if label in match_parts_target:
            all_target_pcs.append(target_pc)

    all_target_pcs = th.cat(all_target_pcs, dim=0)
    all_target_pcs = all_target_pcs.unsqueeze(0)
    all_target_match_source_pcs = th.cat(all_target_match_source_pcs, dim=0)
    all_target_match_source_pcs = all_target_match_source_pcs.unsqueeze(0)
    target_part_cd, _ = chamfer_distance(all_target_pcs, all_target_match_source_pcs)
    target_part_cd = target_part_cd.item() * 500
    source_part_cd = np.mean(source_cd) * 500
    source_distortion = np.mean(source_dist)

    return target_part_cd, source_part_cd, source_distortion

def get_connectedness(post_shape):
    graph = nx.Graph()
    edges = []
    for relation in post_shape.all_relations():
        if isinstance(relation, FeatureRelation):
            # only leaf parts
            part_indices = relation.uid
            part_1 = [x for x in post_shape.partset if x.part_index == part_indices[0]][0]
            part_2 = [x for x in post_shape.partset if x.part_index == part_indices[1]][0]
            if part_1.state[2] != 0 or part_2.state[2] != 0:
                continue
            edges.append(relation.uid)

    graph.add_edges_from(edges)
    connected_components = list(nx.connected_components(graph))
    if len(connected_components) == 1:
        connected = 1
    else:
        connected = 0
    return connected

def update_statistics(timing_info_file, info_row):
    if not os.path.exists(timing_info_file):
        with open(timing_info_file, 'w') as fd:
            fd.write("index, target_cd, source_cd, source_distortion, final_shape_cd, connected, relation_retained \n")
    # check if info_row already exists.
    prexisting_info = []
    with open(timing_info_file, 'r') as fd:
        prexisting_info = fd.readlines()

    prexisting_info = [[y.strip() for y in x.split(",")] for x in prexisting_info[1:]]
    prexisting_info = {x[0]:x[1:] for x in prexisting_info}
    prexisting_info[info_row[0]] = info_row[1:]
    with open(timing_info_file, 'w') as fd:
        fd.write("index, target_cd, source_cd, source_distortion, final_shape_cd, connected, relation_retained \n")
        for cur_info in prexisting_info:
            cur_info_row = [cur_info] + prexisting_info[cur_info]
            cur_info_row = [str(x) for x in cur_info_row]
            cur_info_row = ",".join(cur_info_row) + "\n"
            fd.write(cur_info_row)

def evaluate(index, target_category, redo_search, save_dir, 
         match_parts_source, match_parts_target, source_shape, source_pc_dict, original_source_pc_dict, 
         target_shape, target_pc_dict, bbox_holder, final_shape_cd, name, timing_info_file):
    target_part_cd, source_part_cd, source_distortion = get_cd_and_distortion(match_parts_source, match_parts_target, source_shape, source_pc_dict, original_source_pc_dict, target_shape, target_pc_dict, bbox_holder)
    
    temp_data_dir = os.path.join(save_dir, "results")
    Path(temp_data_dir).mkdir(parents=True, exist_ok=True)
    selected_obj = os.path.join(save_dir, "results", f"{name}.pkl") 
    post_data, post_symbolic = get_obj(selected_obj, redo_search=redo_search, data_dir=save_dir, mode="new",
                                            add_ground=False)
    post_shape = post_symbolic[0]
    post_signatures = get_relation_signatures(post_shape)
    original_signatures = get_relation_signatures(source_shape)
    retained_relations = post_signatures.intersection(original_signatures)
    all_relations = post_signatures.union(original_signatures)
    relation_retained = len(retained_relations) / len(original_signatures)
    # Metric for disjointness.
    connected = get_connectedness(post_shape)
    # Save to the statistics csv as we do
    info_row = [f"{target_category}_{index}", target_part_cd, source_part_cd, 
                source_distortion, final_shape_cd, connected, 
                relation_retained]
    update_statistics(timing_info_file, info_row)
    print("Target CD", target_part_cd)
    print("Source CD", source_part_cd)
    print("Source Distortion", source_distortion)
    print("Final shape CD", final_shape_cd)
    print("Connected", connected)
    print("Retained fraction", relation_retained)

def get_input_info(target_category, index, general_dir):

    file_name = os.path.join(general_dir, "annotated_info.pkl")
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            all_info = cPickle.load(f)
    else:
        all_info = {}
    key = (target_category, index)
    if key in all_info:
        edit_request, match_parts_source, match_parts_target = all_info[key]
    else:
        edit_request = "I want to make the bookshelf taller by extending its top, while keeping its bottom panel as it is."
        match_parts_source = ["top_panel", "back_panel"]
        match_parts_target = ["top_panel", "back_panel"]
        all_info[key] = (edit_request, match_parts_source, match_parts_target)
        with open(file_name, "wb") as f:
            cPickle.dump(all_info, f)

    return edit_request, match_parts_source, match_parts_target
if __name__ == "__main__":
    index = 11
    target_category = "Chair"
    MODE = "V2"
    SAVE_ALL = True
    data_dir = "/media/aditya/OS/data/partnet/data_v0/"
    general_dir = f"/media/aditya/OS/data/partnet/optimization_results/"
    save_dir = os.path.join(general_dir, f"geom_{MODE}/")
    output_dir = "/media/aditya/OS/data/edit_vpi/outputs/optimization/"
    
    if target_category == "Chair":
        only_hard_context = False
    else:
        only_hard_context = True
    dataset = load_valid_shapetalk_utterances(target_category, only_hard_context=only_hard_context)

    item = dataset[index]
    source_anno_id = item['source_anno_id']
    target_anno_id = item['target_anno_id']

    # store the converted form, and the set of parts to match.
    edit_request, match_parts_source, match_parts_target = get_input_info(target_category, index, general_dir)

    # Load the shapes
    selected_obj = os.path.join(data_dir, f"{source_anno_id}", f"{source_anno_id}.pkl")
    source_data, source_symbolic = get_obj(selected_obj, redo_search=False, data_dir=data_dir, mode="new",
                                            add_ground=False)
    to_match_target = [x.part_index for x in source_symbolic[0].partset if x.label in match_parts_source]
    to_match_source = [x.part_index for x in source_symbolic[0].partset if x.part_index not in to_match_target]

    part_dicts = source_data[0]
    source_shape = source_symbolic[0]
    source_pc_dict = get_pc_dict(part_dicts, source_shape)
    original_source_pc_dict = get_pc_dict(part_dicts, source_shape, apply_cube=False)
    # Do the same for target
    selected_obj = os.path.join(data_dir, f"{target_anno_id}", f"{target_anno_id}.pkl")
    target_data, target_symbolic = get_obj(selected_obj, redo_search=False, data_dir=data_dir, mode="new",
                                            add_ground=False)

    part_dicts = target_data[0]
    target_shape = target_symbolic[0]
    target_pc_dict = get_pc_dict(part_dicts, target_shape, apply_cube=False)
    # partwise point cloud
    all_target_points = th.cat([pc for pc in target_pc_dict.values()], dim=0)
    all_target_points = all_target_points.unsqueeze(0)


    # Base optimization mode:
    opt_param, bbox_holder = get_base_opt_tuple(source_shape)
    
    # Else based on edit
    # OPT
    optim = th.optim.Adam(opt_param, lr=LR)

    if MODE == "V1":
        for i in range(N_ITERS):
            pred_points = bbox_to_pc(bbox_holder, source_pc_dict, source_shape)
            loss, _ = chamfer_distance(pred_points, all_target_points)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if i % 100 == 0:
                print(loss.item())
    elif MODE == "V2":
        # Gather pcs in separate binx.
        to_match_in_target = [x.part_index for x in target_shape.partset if x.label in match_parts_target]
        target_pc = th.cat([pc for x, pc in target_pc_dict.items() if x in to_match_in_target], dim=0)
        target_pc = target_pc.unsqueeze(0)
        source_pc = th.cat([pc for x, pc in original_source_pc_dict.items() if x in to_match_source], dim=0)
        source_pc = source_pc.unsqueeze(0)
        # OPT
        target_bbox_holders = [(x, y) for x, y in bbox_holder if x in to_match_target]
        source_bbox_holders = [(x, y) for x, y in bbox_holder if x in to_match_source]
        total_points = target_pc.shape[1] + source_pc.shape[1]
        weight_target = target_pc.shape[1] / total_points
        weight_source = source_pc.shape[1] / total_points
        for i in range(N_ITERS):

            pred_points = bbox_to_pc(target_bbox_holders, source_pc_dict, source_shape)
            loss_target, _ = chamfer_distance(pred_points, target_pc)
            pred_source = bbox_to_pc(source_bbox_holders, source_pc_dict, source_shape)
            loss_source, _ = chamfer_distance(pred_source, source_pc)
            loss = loss_target * weight_target + loss_source * weight_source
            loss.backward()
            optim.step()
            optim.zero_grad()
            if i %100 == 0:
                print(loss.item())

    final_shape_cd = loss.item() * 500

    # SAVE
    name = f"{source_anno_id}_{target_anno_id}"
    folder_name = os.path.join(save_dir, name)
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(folder_name, "result_after_merging.json")
    if SAVE_ALL or not os.path.exists(output_file):
        save_deformed_meshes(source_data, source_symbolic, bbox_holder, folder_name)

    timing_dir = os.path.join(output_dir, "statistics")
    Path(timing_dir).mkdir(parents=True, exist_ok=True)
    timing_info_file = os.path.join(timing_dir, f"geom_{MODE}_baseline.csv")

    if SAVE_ALL:
        redo_search = True
    else:
        redo_search = False
    # Measure and print statistics.
    evaluate(index, target_category, redo_search, save_dir, 
             match_parts_source, match_parts_target, source_shape, source_pc_dict, original_source_pc_dict, 
             target_shape, target_pc_dict, bbox_holder, final_shape_cd, 
             name, timing_info_file)

            


# We will use differentiable optimization of the bbox parameters. 

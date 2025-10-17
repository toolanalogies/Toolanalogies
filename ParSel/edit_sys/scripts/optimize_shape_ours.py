import copy
import json
import os
from pathlib import Path
import sys
import open3d as o3d
import numpy as np
import torch as th
import networkx as nx

import time
import _pickle as cPickle
import edit_sys.llm as prompters
import edit_sys.shape_system.parallel_prop as parallel_prop
from edit_sys.llm.common import KEY, MODEL, TEMPERATURE, SEED, MODE
from scripts.local_config import METHOD_MARKER, HEURISTIC, EDIT_TYPE
from edit_sys.weave.new_algo import algo_v4
from edit_sys.shape_system.edits import load_edits

from edit_sys.data_loader.partnet_shape import get_obj
from scripts.shapetalk_data import load_valid_shapetalk_utterances
from scripts.optimize_shape_geom import (get_pc_dict, chamfer_distance, save_deformed_meshes,
                                         bbox_to_pc, get_input_info, evaluate)
from scripts.optimize_shape_geom import (N_ITERS, LR, DEVICE, PART_INACTIVE)
# Now to do for programmatic.
from edit_sys.shape_system.shape_atoms import Part

LR = 0.05
INDEX_BUMP = 100
sys.path.insert(0, "../external/sympytorch")
import sympytorch, sympy as sp

th.set_default_device('cuda')

def get_base_opt_tuple(shape):
    x_init = th.tensor(0.0, device=DEVICE, requires_grad=True)
    opt_param = [x_init]
    symbol = sp.Symbol("X")
    symbol_to_parts = {symbol: []}
    inner_list = []
    for part in shape.partset:
        part_eq = part.primitive.dynamic_expression()
        if symbol in part_eq.free_symbols:
            inner_list.append(part)
    for relation in shape.all_relations():
        if hasattr(relation, 'dynamic_expression'):
            relation_eq = relation.dynamic_expression()
            for eq in relation_eq:
                if symbol in eq.free_symbols:
                    inner_list.append(
                        relation.parent_part)
                break
    symbol_to_parts[symbol] = list(set(inner_list))
    return opt_param, symbol_to_parts

def shape_to_bboxes(opt_param, shape, symbol_to_parts):

    symbol = sp.Symbol("X")
    relevant_parts = symbol_to_parts[symbol]
    symbol_value_dict = {symbol: opt_param.detach().item()}
    bbox_holder = []
    for part in relevant_parts:
        if len(part.partset) > 0:
            # see which kind of execution is required.
            core_relation = part.core_relation
            output_bboxes = core_relation.execute_relation(symbol_value_dict)
            n_boxes = len(output_bboxes)
            if n_boxes == 1:
                # this means take the parent mesh, and apply the edit
                part_index = part.part_index
                part_eq = part.primitive.dynamic_expression()
                if not hasattr(part, 'sympytorch_expr'):
                    part.sympytorch_expr = sympytorch.SymPyModule(expressions=part_eq)
                bbox = _update_part_mesh(part, opt_param)
                bbox_holder.append((part.part_index, bbox))
            else:
                part_id = part.part_index
                primitive = part.core_relation.primitives[0]
                true_id = primitive.part.part_index
                for index in range(n_boxes):
                    # Temp Hack
                    if not isinstance(output_bboxes[index], th.Tensor):
                        static_expr = output_bboxes[index]
                        static_np = np.array(static_expr).astype(np.float32)
                        static_th = th.tensor(static_np, device=DEVICE, requires_grad=False)
                        bbox_holder.append((true_id, static_th))
                    else:
                        bbox_holder.append((true_id, output_bboxes[index]))
        else:
            part_eq = part.primitive.dynamic_expression()
            if not hasattr(part, 'sympytorch_expr'):
                part.sympytorch_expr = sympytorch.SymPyModule(expressions=part_eq)
            bbox = _update_part_mesh(part, opt_param)
            bbox_holder.append((part.part_index, bbox))
    # for the reminder use the orginal
    for part in shape.partset:
        if part.state[0] == PART_INACTIVE:
            continue
        else:
            if not part in relevant_parts:
                if not hasattr(part, 'static_expression_tensor'):
                    static_expr = part.primitive.static_expression()
                    static_np = np.array(static_expr).astype(np.float32)
                    static_th = th.tensor(static_np, device=DEVICE, requires_grad=True)
                    part.static_expression_tensor = static_th
                static_th = part.static_expression_tensor
                bbox_holder.append((part.part_index, static_th))
    return bbox_holder

def _update_part_mesh(part, opt_param):
    mod = part.sympytorch_expr
    part_shape = part.primitive.static_expression().shape
    real_th = mod(X=opt_param).reshape(part_shape)
    # vertices = cur_cube_coords[..., None] * real_th[None, ...]
    # vertices = vertices.sum(dim=1)
    return real_th

def update_model(shape, bbox_holder, parts_dict, source_pc_dict, original_source_pc_dict, symbol_to_parts, value):
    symbol = sp.Symbol('X')
    relevant_parts = symbol_to_parts[symbol]
    symbol_value_dict = {symbol: value.detach().item()}
    mod_bbox_holder = None
    for part in relevant_parts:
        if len(part.partset) > 0:
            # see which kind of execution is required.
            core_relation = part.core_relation
            output_bboxes = core_relation.execute_relation(symbol_value_dict)
            n_boxes = len(output_bboxes)
            if n_boxes > 1:
                # The part count might be different.
                # Re do this part.
                parent_part = part
                to_remove_indices = [x.part_index for x in parent_part.sub_parts]
                # just remove all children
                parent_part.sub_parts = {}
                child = part.core_relation.primitives[0].part

                original_index = child.part_index
                original_part_dict = parts_dict[original_index]

                new_index = len(shape.partset) + INDEX_BUMP
                new_children = []
                mod_bbox_holder = [x for x in bbox_holder if x[0] != original_index]
                for index in range(n_boxes):
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
                    for mesh_info in original_tm.meshes:
                        cur_mesh = copy.deepcopy(mesh_info.mesh)
                        material_idx = mesh_info.material_idx
                        new_mesh_info = o3d.visualization.rendering.TriangleMeshModel.MeshInfo(cur_mesh, "", material_idx)
                        new_meshes.append(new_mesh_info)
                    new_tm.meshes = new_meshes
                    new_part_dict = dict(mesh=new_tm, obb=original_part_dict['obb'], label=child.original_label)
                    parts_dict[cur_index] = new_part_dict

                    static_expr = output_bboxes[index]
                    static_np = np.array(static_expr).astype(np.float32)
                    static_th = th.tensor(static_np, device=DEVICE, requires_grad=True)
                    # Other updates
                    mod_bbox_holder.append((cur_index, static_th))
                    source_pc_dict[cur_index] = source_pc_dict[original_index]
                    original_source_pc_dict[cur_index] = original_source_pc_dict[original_index]
                new_children = set(new_children)
                parent_part.sub_parts = new_children
                # for remove_index in to_remove_indices:
                #     parts_dict.pop(remove_index)
    if mod_bbox_holder is None:
        mod_bbox_holder = bbox_holder
    return shape, mod_bbox_holder, parts_dict, source_pc_dict, original_source_pc_dict

if __name__ == "__main__":
    target_category = "Bookshelf"
    index = 10
    SAVE_ALL = True
    OPTIM_MODE = "V2"
    data_dir = "/media/aditya/OS/data/partnet/data_v0/"
    general_dir = f"/media/aditya/OS/data/partnet/optimization_results/"
    save_dir = f"/media/aditya/OS/data/partnet/optimization_results/program_{OPTIM_MODE}/"
    output_dir = "/media/aditya/OS/data/edit_vpi/outputs/optimization"
    
    if target_category == "Chair":
        only_hard_context = False
    else:
        only_hard_context = False
    dataset = load_valid_shapetalk_utterances(target_category, only_hard_context=only_hard_context)


    item = dataset[index]
    source_anno_id = item['source_anno_id']
    target_anno_id = item['target_anno_id']

    # store the converted form, and the set of parts to match.
    edit_request, match_parts_source, match_parts_target = get_input_info(target_category, index, general_dir)

    selected_obj = os.path.join(data_dir, f"{target_anno_id}", f"{target_anno_id}.pkl")
    target_data, target_symbolic = get_obj(selected_obj, redo_search=False, data_dir=data_dir, mode="new",
                                            add_ground=False)

    part_dicts = target_data[0]
    target_shape = target_symbolic[0]
    target_pc_dict = get_pc_dict(part_dicts, target_shape, apply_cube=False)
    # partwise point cloud
    all_target_points = th.cat([pc for pc in target_pc_dict.values()], dim=0)
    all_target_points = all_target_points.unsqueeze(0)
    # hack
    # all_target_points = all_target_points * 1.5
    # all_target_points = all_target_points[:, :10, :]


    # Load / Generate the program:
    program_dir = os.path.join(output_dir, f"program_{OPTIM_MODE}")
    Path(program_dir).mkdir(parents=True, exist_ok=True)
    program_file = os.path.join(output_dir, f"program_{OPTIM_MODE}", f"{source_anno_id}_{target_anno_id}.pkl")

    selected_obj = os.path.join(data_dir, f"{source_anno_id}", f"{source_anno_id}.pkl")
    if os.path.exists(program_file):

        # Load the shapes
        source_data, source_symbolic = get_obj(selected_obj, redo_search=False, data_dir=data_dir, mode="new",
                                                add_ground=True)

        part_dicts = source_data[0]
        source_shape = source_symbolic[0]

        edit_gens = cPickle.load(open(program_file, "rb"))
        all_edits = load_edits(source_shape, edit_gens)
        all_edits = all_edits[0]
        ground_part = source_shape.get('ground')
        source_shape.sub_parts.remove(ground_part)
        # set active in active.
    else:

        source_data, source_symbolic = get_obj(selected_obj, redo_search=False, data_dir=data_dir, mode="new",
                                                add_ground=True)
        shape = source_symbolic[0]
        prompter_class = getattr(prompters, METHOD_MARKER)
        prompter = prompter_class(MODE, KEY, MODEL, TEMPERATURE, SEED)
        edit_proposer = parallel_prop.ParallelEditProposer(HEURISTIC)
        start_time = time.time()
        all_edits, log_info, any_breaking, _ = algo_v4(shape, edit_request, prompter, edit_proposer, EDIT_TYPE)

        save_format = [x.save_format() for x in all_edits]

        cPickle.dump(save_format, open(program_file, "wb"))
        ground_part = shape.get('ground')
        shape.sub_parts.remove(ground_part)
        source_shape = shape
    
    source_pc_dict = get_pc_dict(part_dicts, source_shape, avoid_inactive=False)
    original_source_pc_dict = get_pc_dict(part_dicts, source_shape, apply_cube=False, avoid_inactive=False)
    # Do the same for target
    # Base optimization mode:
    for edit in all_edits:
        edit.propagate()
        
    opt_param, symbol_to_parts = get_base_opt_tuple(source_shape)
    
    # Else based on edit
    # OPT
    optim = th.optim.Adam(opt_param, lr=LR)

    if OPTIM_MODE == "V1":
        for i in range(N_ITERS):
            bbox_holder = shape_to_bboxes(opt_param[0], source_shape, symbol_to_parts)
            pred_points = bbox_to_pc(bbox_holder, source_pc_dict, source_shape)
            loss, _ = chamfer_distance(pred_points, all_target_points)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if i %100 == 0:
                print(loss.item())
    elif OPTIM_MODE == "V2":
        # Create the smaller pc
        to_match_in_target = [x.part_index for x in target_shape.partset if x.label in match_parts_target]
        target_pc = th.cat([pc for x, pc in target_pc_dict.items() if x in to_match_in_target], dim=0)
        target_pc = target_pc.unsqueeze(0)
        to_match_target = [x.part_index for x in source_symbolic[0].partset if x.label in match_parts_source]
        
        for i in range(N_ITERS):
            bbox_holder = shape_to_bboxes(opt_param[0], source_shape, symbol_to_parts)
            temp_bbox_holder = [x for x in bbox_holder if x[0] in to_match_target]
            pred_points = bbox_to_pc(temp_bbox_holder, source_pc_dict, source_shape)
            loss, _ = chamfer_distance(pred_points, target_pc)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if i %100 == 0:
                print(loss.item())
        
        bbox_holder = shape_to_bboxes(opt_param[0], source_shape, symbol_to_parts)
        pred_points = bbox_to_pc(bbox_holder, source_pc_dict, source_shape)
        loss, _ = chamfer_distance(pred_points, all_target_points)
    final_shape_cd = loss.item() * 500

    # SAVE
    name = f"{source_anno_id}_{target_anno_id}"
    folder_name = os.path.join(save_dir, name)
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(folder_name, "result_after_merging.json")
    # Update Shape?

    # Would be nice to update the shape here itself.
    # Basically update part dict, as well as the mesh
    parts_dict = source_data[0]
    source_shape = source_symbolic[0]
    output = update_model(source_shape, bbox_holder, parts_dict, source_pc_dict, original_source_pc_dict, symbol_to_parts, opt_param[0])
    source_shape, bbox_holder, parts_dict, source_pc_dict, original_source_pc_dict = output
    source_data = (parts_dict, source_data[1], source_data[2], source_data[3])
    source_symbolic = (source_shape, source_symbolic[1], source_symbolic[2]) 

    if SAVE_ALL or not os.path.exists(output_file):
        save_deformed_meshes(source_data, source_symbolic, bbox_holder, 
                                                folder_name, symbol_to_parts, opt_param[0])


    # Measure and print statistics.

    timing_dir = os.path.join(output_dir, "statistics")
    Path(timing_dir).mkdir(parents=True, exist_ok=True)
    timing_info_file = os.path.join(timing_dir, f"program_{OPTIM_MODE}.csv")

    if SAVE_ALL:
        redo_search = True
    else:
        redo_search = False
    # Measure and print statistics.
    evaluate(index, target_category, redo_search, save_dir, 
             match_parts_source, match_parts_target, 
             source_shape, source_pc_dict, original_source_pc_dict, 
             target_shape, target_pc_dict, bbox_holder,
             final_shape_cd, name, timing_info_file)

            
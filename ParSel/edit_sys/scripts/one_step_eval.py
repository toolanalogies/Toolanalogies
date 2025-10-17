from collections import defaultdict
import os
import _pickle as cPickle
import argparse
import openai
from pathlib import Path
import time

import edit_sys.llm.base_prompter as prompter_lib
from edit_sys.state import State
from edit_sys.llm.common import KEY, MODEL, TEMPERATURE, SEED, prompt_cost, completion_cost
from edit_sys.data_loader.partnet_shape import get_obj
import edit_sys.shape_system.proposal_mechanism as proposal_mechanism
from edit_sys.weave.simple_algo import new_algorithm, algo_v2
from edit_sys.shape_system.edits import *
import csv
from scripts.local_config import (DATA_DIR, DEFAULT_OUTPUT_DIR, DATASET_INDEX, METHOD_MARKER, METADATA_FILE)
SKIP_ONE_STEP = False

SUBSITUTE_VALUE = 0.5

def main(dataset_index, output_dir):
    raise NotImplementedError("This script is not yet implemented")
    # assert method_marker is not prompter_lib.HumanPrompter
    # Add support
    all_data = cPickle.load(open(METADATA_FILE, "rb"))
    item = all_data[dataset_index]
    shape_id = item['shape_id']
    edit_request = item['edit_request']
    selected_obj = os.path.join(DATA_DIR, f"{shape_id}", f"{shape_id}.pkl")
    processed_data, symbolic_data = get_obj(selected_obj, redo_search=False, data_dir=DATA_DIR, mode="new",
                                            add_ground=True)
    shape = symbolic_data[0]


    # Load the prompter
    # This will be changed for different baselines
    gt_method_marker = "HumanPrompter"
    save_dir = os.path.join(output_dir, "logs", gt_method_marker, selected_class)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(save_dir, f"logs_{shape_id}_{prompt_id}.pkl")
    save_dir = os.path.join(output_dir, "programs", gt_method_marker, selected_class)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    program_file = os.path.join(save_dir, f"programs_{shape_id}_{prompt_id}.pkl")
    prompter = prompter_lib.OneStepEvaluator(KEY, MODEL, TEMPERATURE, SEED, log_file=log_file, program_file=program_file)
    # prompter = HumanPrompter(KEY, MODEL, TEMPERATURE, SEED)
    method_marker = prompter.method_marker

    # gt_program, pred_program = get_programs(shape_id, prompt_id, selected_class, output_dir, method_marker, prompter)
    
    # Run the algorithm
    if not SKIP_ONE_STEP:
        prompt_dir = os.path.join(output_dir, "prompts", gt_method_marker, f"{selected_class}_prompts_{shape_id}_{prompt_id}")
        start_time = time.time()
        evaluation_list = prompter.evaluate_one_step(shape, edit_request, prompt_dir)
        end_time = time.time()

        evaluation_metrics = prompter.evaluate_instance_level(evaluation_list)
    else:
        evaluation_metrics = {}
        evaluation_list = []
    evaluation_metrics['step'] = "one_step"
    # Save information.
    evaluation_list.append(evaluation_metrics)
    # Also add the program comparison result:
    # Now calcuate precision, recall, and F1, IoU, # partial match
    
    # eval_dict = get_program_level_metrics(shape, gt_program, pred_program)
    # evaluation_list.append(eval_dict)
    method_marker = f"{method_marker}_edit_change"
    save_dir = os.path.join(output_dir, "eval", method_marker, selected_class)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_file = os.path.join(save_dir, f"eval_{shape_id}_{prompt_id}.pkl")
    cPickle.dump(evaluation_list, open(save_file, "wb"))

def get_programs(shape_id, prompt_id, selected_class, output_dir, method_marker, prompter):
    valid_types = (PartTranslate, RestrictedTranslate,
                   PartRotate, RestrictedRotate,
                   PartScale1D, RestrictedScale1D,
                   PartScale2D, RestrictedScale2D,
                   PartScale3D, RestrictedScale3D,
                   PartShear, ChangeCount, ChangeDelta)
    
    gt_program = [x for x in prompter.gt_program if issubclass(x[0].edit_class, valid_types)]
    gt_program = [x for x in gt_program if not evaluate_equals_zero(x[0].amount, mode=3, value=4)]
    # pred program
    save_dir = os.path.join(output_dir, "programs", method_marker, selected_class)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    pred_program_file = os.path.join(save_dir, f"programs_{shape_id}_{prompt_id}.pkl")
    pred_program = cPickle.load(open(pred_program_file, "rb"))

    pred_program = [x for x in pred_program if issubclass(x[0].edit_class, valid_types)]
    pred_program = [x for x in pred_program if not evaluate_equals_zero(x[0].amount, mode=3, value=4)]
    # also only programs with meaningful amount of edit.

    return gt_program, pred_program

def get_program_level_metrics(shape, gt_program, pred_program):
    partial_match_dict = {}
    full_match_list = []
    for pred_ind, pred_edit in enumerate(pred_program):
        partial_match_dict[pred_ind] = False
        full_match = False
        for gt_edit in gt_program:
            number_match = (gt_edit[2] == pred_edit[2])
            type_match = (gt_edit[0].edit_class == pred_edit[0].edit_class)
            if number_match and type_match:
                partial_match = True
                full_match = True
                for param_name, gt_param in gt_edit[0].param_dict.items():
                    param_value = gt_edit[0].param_dict[param_name]
                    if isinstance(gt_param, str):
                        if param_value != gt_param:
                            full_match = False
                            break
                    else:
                        delta = param_value - gt_param
                        delta = delta.norm()
                        if not evaluate_equals_zero(delta, mode=3, value=4):
                            full_match = False
                            break
                if partial_match:
                    partial_match_dict[pred_ind] = True
            if full_match:
                break
        full_match_list.append(full_match)
    
    # Now the inverse for gt_program:
    true_positives = sum(full_match_list)
    false_positives = len(full_match_list) - true_positives
    
    p_tp = sum(partial_match_dict.values())
    p_fp = len(partial_match_dict) - p_tp
    
    partial_match_dict = {}
    full_match_list = []
    for gt_ind, gt_edit in enumerate(gt_program):
        partial_match_dict[pred_ind] = False
        full_match = False
        for pred_edit in pred_program:
            number_match = (gt_edit[2] == pred_edit[2])
            type_match = (gt_edit[0].edit_class == pred_edit[0].edit_class)
            if number_match and type_match:
                partial_match = True
                full_match = True
                for param_name, gt_param in gt_edit[0].param_dict.items():
                    param_value = pred_edit[0].param_dict[param_name]
                    if isinstance(gt_param, str):
                        if param_value != gt_param:
                            full_match = False
                            break
                    else:
                        delta = param_value - gt_param
                        delta = delta.norm()
                        if not evaluate_equals_zero(delta, mode=3, value=4):
                            full_match = False
                            break
                if partial_match:
                    partial_match_dict[pred_ind] = True
            if full_match:
                break
        full_match_list.append(full_match)
    
    # no true negatives, but there are false negatives
    correct_gt = sum(full_match_list)
    false_negatives = len(full_match_list) - correct_gt
    
    p_corr = sum(partial_match_dict.values())
    p_fn = len(partial_match_dict) - p_corr
    
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = correct_gt / (correct_gt + false_negatives + 1e-10)
    f1 = true_positives / (true_positives + 0.5 * (false_positives + false_negatives) + 1e-10)
    iou = true_positives / (true_positives + false_positives + false_negatives + 1e-10)
    partial_precision = p_tp / (p_tp + p_fp + 1e-10)
    partial_recall = p_corr / (p_corr + p_fn + 1e-10)
    partial_f1 = p_tp / (p_tp + 0.5 * (p_fp + p_fn) + 1e-10)
    partial_iou = p_tp / (p_tp + p_fp + p_fn + 1e-10)
    
    eval_dict = {
        "step": "gt_compare",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "partial_precision": partial_precision,
        "partial_recall": partial_recall,
        "partial_f1": partial_f1,
        "partial_iou": partial_iou,
        "pred_len": len(pred_program),
        "gt_len": len(gt_program),
    }
    # also get the l2 for all the edited cuboids
    # How? Get the L2 on the dynamic expr for all cuboids.
    # Apply to shape collect per label dynamic expr.
    edits = []
    for edit_content in pred_program:
        edit_gen, operand_type, index = edit_content
        if operand_type == "part":
            operand = [x for x in shape.partset if x.part_index == index][0]
        else:
            operand = [x for x in shape.all_relations() if x.relation_index == index][0]
        edit = edit_gen.employ(operand)
        edits.append(edit)
        edit.propagate()
    
    pred_part_to_expr = {}
    for part in shape.partset:
        part_expr = part.primitive.dynamic_expression()
        part_np = part_expr.subs({MAIN_VAR: SUBSITUTE_VALUE})
        part_np = np.asarray(part_np).astype(np.float32)
        pred_part_to_expr[part.full_label] = part_np
        part.primitive.edit_sequence = []
    
    shape.clean_up_motion()
    edits = []
    for edit_content in gt_program:
        edit_gen, operand_type, index = edit_content
        if operand_type == "part":
            operand = [x for x in shape.partset if x.part_index == index][0]
        else:
            operand = [x for x in shape.all_relations() if x.relation_index == index][0]
        edit = edit_gen.employ(operand)
        edits.append(edit)
        edit.propagate()

    gt_part_to_expr = {}
    
    for part in shape.partset:
        part_expr = part.primitive.dynamic_expression()
        part_np = part_expr.subs({MAIN_VAR: SUBSITUTE_VALUE})
        part_np = np.asarray(part_np).astype(np.float32)
        gt_part_to_expr[part.full_label] = part_np
        part.primitive.edit_sequence = []
    
    l2 = 0
    for part_label, pred_expr in pred_part_to_expr.items():
        gt_expr = gt_part_to_expr[part_label]
        l2 += np.linalg.norm(pred_expr - gt_expr)
    eval_dict["l2"] = l2
    for key, value in eval_dict.items():
        print(f"{key}: {value}")

    
    return eval_dict

    
if __name__ == "__main__":
    # Argument parser for setting PROMPT_ID
    parser = argparse.ArgumentParser()
    # parser.add_argument("--shape_ind", help="Shape Index", type=int, default=SHAPE_INDEX)
    # parser.add_argument("--prompt_ind", help="Prompt Index", type=int, default=PROMPT_ID)
    # parser.add_argument("--tuple_ind", help="Tuple Index", type=int, default=TUPLE_INDEX)
    # parser.add_argument("--obj_class", help="selected class", type=str, default=SHAPE_CATEGORY)
    parser.add_argument("--DATASET_INDEX", help="dataset index", type=int, default=DATASET_INDEX)
    parser.add_argument("--output_dir", help="output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method_marker", help="forsaving", type=str, default=METHOD_MARKER)
    args = parser.parse_args()
    
    selected_class, tuple_ind = get_simple_dataset_item(args.DATASET_INDEX)
    shape_ind, prompt_ind = get_shape_prompt_tuple(tuple_ind, selected_class)
    main(shape_ind, prompt_ind, selected_class, args.output_dir)

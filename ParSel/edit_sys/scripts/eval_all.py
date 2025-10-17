from collections import defaultdict
import os
import _pickle as cPickle
import argparse
import openai
from pathlib import Path
import time

from edit_sys.llm.base_prompter import OneStepEvaluator, HumanPrompter
from edit_sys.state import State
from edit_sys.llm.common import KEY, MODEL, TEMPERATURE, SEED, prompt_cost, completion_cost
from edit_sys.data_loader.partnet_shape import get_obj
import edit_sys.shape_system.proposal_mechanism as proposal_mechanism
from edit_sys.shape_system.visualization import generate_gif
from edit_sys.edit_modes.editing_algo import new_algorithm, algo_v2
from edit_sys.shape_system.edits import *
import csv
from scripts.local_config import (DATA_DIR, METADATA_FILE, DEFAULT_OUTPUT_DIR, 
                                  SHAPE_INDEX, PROMPT_ID, SHAPE_CATEGORY, TUPLE_INDEX,
                                  DATASET_INDEX, METHOD_MARKER,
                                  SAVE_WITH_RESPONSES,)
from scripts.solve_edit import get_all_valid_files, get_all_edit_requests, get_simple_dataset_item, get_shape_prompt_tuple

def main(output_dir):
    gt_method_marker = "HumanPrompter"
    ablation_markers = [
        "LLMPrompterV1",
        "LLMPrompterV1_n_calls_5",
        "LLMPrompterV1_no_cot",
        "LLMPrompterV1_nocot_v2",
        "LLMPrompterV1_edit_change",
        # "PureGeometryPrompter",
        # "UpfrontLLMPrompter"  + "_n_calls_5",
        # "NoRelationSelectLLMPrompter",
        # "NoEditSelectLLMPrompter",
        # "APIOnlyLLMPrompter",
        # "HumanPrompter"
    ]
    # for method_marker in ablation_markers:
    #     for dataset_index in range(0, 31, 3):
    #         selected_class, tuple_ind = get_simple_dataset_item(dataset_index)
    #         shape_ind, prompt_ind = get_shape_prompt_tuple(tuple_ind, selected_class)
    #         save_dir = os.path.join(output_dir, "programs", method_marker, selected_class)
    #         Path(save_dir).mkdir(parents=True, exist_ok=True)
    #         save_name = os.path.join(save_dir, f"programs_{shape_ind}_{prompt_ind}.pkl")
    #         if os.path.exists(save_name):
    #             continue
    #         else:
    #             print("Not found", method_marker, dataset_index)
    print("done")
    dataset = range(0, 31)

    # gather the list in a default dict.
    method_to_scores = defaultdict(list)
    for method_marker in ablation_markers:
        print("here", method_marker)
        metrics = defaultdict(list)
        for dataset_index in dataset:
            selected_class, tuple_ind = get_simple_dataset_item(dataset_index)
            shape_ind, prompt_ind = get_shape_prompt_tuple(tuple_ind, selected_class)
            save_dir = os.path.join(output_dir, "eval", method_marker, selected_class)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_name = os.path.join(save_dir, f"eval_{shape_ind}_{prompt_ind}.pkl")
            if os.path.exists(save_name):
                eval_info = cPickle.load(open(save_name, "rb"))
                complete_correct = True
                sym_correct = True
                select_edit_correct = True
                contact_correct = True
                for info_dict in eval_info:
                    step = info_dict['step']
                    if step == "sym_relation":
                        key = "sym_relation_per"
                        metrics[key].append(float(info_dict['metric']))
                        if not info_dict['metric']:
                            complete_correct = False
                            sym_correct = False
                    elif step == "select_edit_option":
                        key = "select_edit_per"
                        metrics[key].append(float(info_dict['metric']))
                        if not info_dict['metric']:
                            complete_correct = False
                            select_edit_correct = False
                    elif step == "contact_relation":
                        key = "contact_relation_per"
                        metrics[key].append(float(info_dict['metric']))
                        if not info_dict['metric']:
                            complete_correct = False
                            contact_correct = False
                    metrics['complete_correct'].append(float(complete_correct))
                    metrics['sym_correct'].append(float(sym_correct))
                    metrics['select_edit_correct'].append(float(select_edit_correct))
                    metrics['contact_correct'].append(float(contact_correct))

                # one_step_info = [x for x in eval_info if x['step'] == "one_step"][0]
                # for key, value in one_step_info.items():
                #     metrics[key].append(value)
                # eval_final = [x for x in eval_info if x['step'] == "gt_compare"][0]
                # for key, value in eval_final.items():
                #     metrics[key].append(value)
            else:
                print("Not found", method_marker, dataset_index)
        print("here", method_marker)
        for key, value in metrics.items():
            if isinstance(value[0], float):
                values = np.array(value)
                avg = np.mean(values)
                median = np.median(values)
                method_to_scores[method_marker].append((key, avg, median))

    keys = ['complete_correct', 
            'sym_relation_per',
            'contact_relation_per',
            'selected_edit_per',
    ]
    indices = [
        1, 0, 5, 6
    ]

    for key, value in method_to_scores.items():
        print("-------------------")
        header = ["Method"]
        for index in indices:
        # for value_pair in value:
            value_pair = value[index]
            header.append(value_pair[0])
        break
    header_str = "|".join(header)
    brack = "|".join(["---" for x in header])
    print(header_str)
    print(brack)

    for key, value in method_to_scores.items():
        avg_values = []
        for index in indices:
        # for value_pair in value:
            value_pair = value[index]
            avg_values.append(f"{value_pair[1]:.2f}")
        print(f"{key}|{'|'.join(avg_values)}")

    # print("-------------------")
    # print(header_str)
    # print(brack)
    # for key, value in method_to_scores.items():
    #     # value = method_to_scores[key]
    #     median_values = []
    #     for value_pair in value:
    #         median_values.append(f"{value_pair[2]:.2f}")
    #     print(f"{key}|{'|'.join(median_values)}")

if __name__ == "__main__":
    # Argument parser for setting PROMPT_ID
    parser = argparse.ArgumentParser()
    # parser.add_argument("--shape_ind", help="Shape Index", type=int, default=SHAPE_INDEX)
    # parser.add_argument("--prompt_ind", help="Prompt Index", type=int, default=PROMPT_ID)
    # parser.add_argument("--tuple_ind", help="Tuple Index", type=int, default=TUPLE_INDEX)
    # parser.add_argument("--obj_class", help="selected class", type=str, default=SHAPE_CATEGORY)
    parser.add_argument("--DATASET_INDEX", help="dataset index", type=int, default=DATASET_INDEX)
    parser.add_argument("--output_dir", help="output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    main(args.output_dir)
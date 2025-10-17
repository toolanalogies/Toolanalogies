from collections import defaultdict
import os
import _pickle as cPickle
import argparse
from pathlib import Path
import time

import edit_sys.llm as prompters #import HumanPrompter, LLMPrompterV1, PureGeometryPrompter
from edit_sys.state import State
from edit_sys.llm.common import KEY, MODEL, TEMPERATURE, SEED, prompt_cost, completion_cost, MODE
from edit_sys.data_loader.partnet_shape import get_obj
import edit_sys.shape_system.proposal_mechanism as proposal_mechanism
import edit_sys.shape_system.parallel_prop as parallel_prop
from edit_sys.weave.simple_algo import algo_v2, algo_v3
from edit_sys.weave.new_algo import algo_v4
from edit_sys.visualizer.shape_renderer import ShapeRenderer
import csv
import edit_sys
from scripts.local_config import (DATA_DIR, METADATA_FILE, DEFAULT_OUTPUT_DIR,
                                  DATASET_INDEX,METHOD_MARKER, REPETITIONS, TEMP_INDEX, EDIT_FEATURE, EDIT_PROMPT, TASK)

def cleanup_list_recursive(all_edits):
    all_items = []
    for item in all_edits:
        if isinstance(item, list):
            all_items.extend(cleanup_list_recursive(item))
        else:
            all_items.append(item)
    return all_items

def save_prompts(save_dir, log_info):
    # clear the directory
    for file in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
    step = 0
    for info in log_info:
        if "prompt" in info.keys():
            print("has prompt")
            step_type = info['step']
            file_name = os.path.join(save_dir, f"{step_type}_{step}.md")
            print(file_name)
            with open(file_name, "w") as f:
                f.write(info['prompt'])
            step += 1
        if 'prompt_ncot' in info.keys():
            print("has prompt_ncot")
            step_type = info['step']
            file_name = os.path.join(save_dir, f"{step_type}_{step}_nocot.md")
            print(file_name)
            with open(file_name, "w") as f:
                f.write(info['prompt'])

def save_prompts_and_responses(save_dir, log_info):
    for file in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
    step = 0
    for info in log_info:
        if "prompt" in info.keys():
            print("has prompt")
            step_type = info['step']
            file_name = os.path.join(save_dir, f"{step_type}_{step}.md")
            print(file_name)
            cur_info = [info['prompt'], info['response']]
            cur_info = ("\n\n" + "--" * 10 + "RESPONSE" + "--" * 10 +"\n\n" ).join(cur_info)
            with open(file_name, "w") as f:
                f.write(cur_info)
            step += 1
    

def main(dataset_index, temp_index, output_dir, method_marker, load_gt, repetitions, edit_feature=None, edit_prompt=None, task=None):

    # Load shape
    all_data = cPickle.load(open(METADATA_FILE, "rb"))
    item = all_data[dataset_index]
    # shape_id = str(int(item['shape_id']) + 11)
    # shape_id = TEMP_INDEX# item['shape_id']

    if temp_index is not None:
        shape_id = temp_index# item['shape_id']
    else:
        shape_id = item['shape_id']
    edit_request = item['edit_request']
    # req=input("Enter request to LLM")
    # edit_request="Make the handle of the stick longer in the left-right direction"
    edit_request= edit_prompt
    selected_obj = os.path.join(DATA_DIR, f"{shape_id}", f"{shape_id}.pkl")
    processed_data, symbolic_data = get_obj(selected_obj, redo_search=False, data_dir=DATA_DIR, mode="new",
                                            add_ground=False)
    
    prompter_class = getattr(prompters, method_marker)
    if issubclass(prompter_class, prompters.APIOnlyVisionPrompter):
        img_dir = os.path.join(output_dir, "shape_renders")
        Path(img_dir).mkdir(parents=True, exist_ok=True)
        img_name = os.path.join(img_dir, f"{dataset_index}.png")
        image_gen = ShapeRenderer(processed_data, symbolic_data, img_name)
        # MODEL = "gpt-4-vision-preview"

    shape = symbolic_data[0]
    if 'category' in item.keys():
        shape.label = item['category']
    print(edit_request)

    # internal_save_path = os.path.join(output_dir, "ours_pred")
    internal_save_path = os.path.join(output_dir, "qualitative")
    # internal_save_path = os.path.join(output_dir, "ours_pred_no_vote")
    # internal_save_path = os.path.join(output_dir, "ours_pred_no_cot")
    # internal_save_path = os.path.join(output_dir, "ours_pred_no_examples")
    # internal_save_path = os.path.join(output_dir, "ours_pred_nothing")
    Path(internal_save_path).mkdir(parents=True, exist_ok=True)
    internal_save_path = os.path.join(internal_save_path, f"{dataset_index}.pkl")
    prompter = prompter_class(MODE, KEY, MODEL, TEMPERATURE, SEED, internal_save_path=internal_save_path)
    prompter.repetitions = repetitions
    # saved_program = []
    if load_gt:
        gt_method_marker = "GT"
        save_dir = os.path.join(output_dir, "programs", gt_method_marker)
        program_file = os.path.join(save_dir, f"programs_{dataset_index}.pkl")
        gt_program = cPickle.load(open(program_file, "rb"))
        prompter.edit_program = gt_program
        prompter.dataset_index = dataset_index
        prompter.output_dir = output_dir
    if issubclass(prompter_class, prompters.APIOnlyVisionPrompter):
        start_time = time.time()
        # Generate image here.
        image = image_gen.generate_image()
        try:
            all_edits, log_info = prompter.api_call_single_pass_program(shape, edit_request, en_image=image)
            # resolve list nesting:
            all_edits = cleanup_list_recursive(all_edits)
        except:
            all_edits = []
            log_info = [{"step": "FAILURE"}]
        end_time = time.time()
    elif issubclass(prompter_class, prompters.APIOnlyLLMPrompter):
        # Run the algorithm
        start_time = time.time()
        try:
            all_edits, log_info = prompter.api_call_single_pass_program(shape, edit_request)
            # resolve list nesting:
            all_edits = cleanup_list_recursive(all_edits)
            step = 0
        except:
            all_edits = []
            log_info = [{"step": "FAILURE"}]
        end_time = time.time()
    
    else:
        edit_proposer = parallel_prop.ParallelEditProposer()

        # Run the algorithm
        start_time = time.time()
        # all_edits, log_info = algo_v2(shape, edit_request, prompter, edit_proposer)

        all_edits, log_info, any_breaking, _ = algo_v4(shape, edit_request, prompter, edit_proposer)
        end_time = time.time()

    # Save information.
    method_marker = f"{method_marker}_{repetitions}"

    cur_info = {
        "step": "start",
        'dataset_index': dataset_index,
        "prompter_type": method_marker,
        "edit_request": edit_request,
    }
    log_info.insert(0, cur_info)

    # Compute total tokens:
    prompt_tokens = State.n_prompt_tokens
    completion_tokens = State.n_completion_tokens
    cost = prompt_tokens * prompt_cost[MODEL] + completion_tokens * completion_cost[MODEL]
    total_time = end_time - start_time 
    cur_info = {
        "step": "end",
        "total_cost": cost/ 1_000.0,
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "total_calls": State.n_api_calls,
        "total_api_time": State.api_time,
        "solver_time": total_time - State.api_time,
        "total_time": total_time
    }
    for x in cur_info:
        print(f"{x}: {cur_info[x]}")
    for edit in all_edits:
        print(edit)
    # save the information
    timing_dir = os.path.join(output_dir, "statistics")
    Path(timing_dir).mkdir(parents=True, exist_ok=True)
    timing_info_file = os.path.join(timing_dir, f"{method_marker}.csv")
    
    info_row = [f"{dataset_index}", State.api_time, total_time - State.api_time, State.n_api_calls, cost/1000, total_time]
    if not os.path.exists(timing_info_file):
        with open(timing_info_file, 'w') as fd:
            fd.write("shape_id, api_time, solving_time, cost, total_time, n_api_calls \n")
    # check if info_row already exists.
    prexisting_info = []
    with open(timing_info_file, 'r') as fd:
        prexisting_info = fd.readlines()

    prexisting_info = [[y.strip() for y in x.split(",")] for x in prexisting_info[1:]]
    prexisting_info = {x[0]:x[1:] for x in prexisting_info}
    prexisting_info[info_row[0]] = info_row[1:]
    with open(timing_info_file, 'w') as fd:
        fd.write("shape_id, api_time, solving_time, cost, total_time, n_api_calls \n")
        for cur_info in prexisting_info:
            cur_info_row = [cur_info] + prexisting_info[cur_info]
            cur_info_row = [str(x) for x in cur_info_row]
            cur_info_row = ",".join(cur_info_row) + "\n"
            fd.write(cur_info_row)

    log_info.append(cur_info)

    # Function to save all the prompts as steps.
    save_dir = os.path.join(output_dir, "logs", method_marker, task)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_name = os.path.join(save_dir, edit_feature + f"logs_{dataset_index}.pkl")
    print(f"Saving the information at {save_name}")
    cPickle.dump(log_info, open(save_name, "wb"))

    save_format = [x.save_format() for x in all_edits]
    save_dir = os.path.join(output_dir, "programs", method_marker, task)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_name = os.path.join(save_dir, edit_feature + f"_programs_{dataset_index}.pkl")
    cPickle.dump(save_format, open(save_name, "wb"))

if __name__ == "__main__":
    # Argument parser for setting PROMPT_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_index", help="dataset index", type=int, default=DATASET_INDEX)
    parser.add_argument("--temp_index", help="temp_index", type=int, default=TEMP_INDEX)
    parser.add_argument("--output_dir", help="output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method_marker", help="method_marker", type=str, default=METHOD_MARKER)
    parser.add_argument("--repetitions", help="mode", type=int, default=REPETITIONS)
    parser.add_argument("--edit_feature", help="edit_feature", type=str, default=EDIT_FEATURE)
    parser.add_argument("--edit_prompt", help="edit_prompt", type=str, default=EDIT_PROMPT)
    parser.add_argument("--task", help="task", type=str, default=TASK)
    # add flat for loading gt
    parser.add_argument("--load_gt", help="load_gt", action="store_true")
    args = parser.parse_args()
    main(args.dataset_index, args.temp_index, args.output_dir, args.method_marker, args.load_gt, args.repetitions, args.edit_feature, args.edit_prompt, args.task)


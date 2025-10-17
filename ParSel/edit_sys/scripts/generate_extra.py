from collections import defaultdict
import os
import _pickle as cPickle
import argparse
from pathlib import Path
import time

import edit_sys.shape_system.parallel_prop as parallel_prop
from edit_sys.weave.new_algo import algo_v4
import edit_sys.llm as prompters #import HumanPrompter, LLMPrompterV1, PureGeometryPrompter
from edit_sys.state import State
from edit_sys.llm.common import KEY, MODEL, TEMPERATURE, SEED, prompt_cost, completion_cost, MODE
from edit_sys.data_loader.partnet_shape import get_obj
from edit_sys.visualizer.shape_renderer import ShapeRenderer
from scripts.local_config import (DATA_DIR, METADATA_FILE, DEFAULT_OUTPUT_DIR,TEMP_INDEX,
                                  DATASET_INDEX,METHOD_MARKER, EXTRA_METHOD_MARKER, REPETITIONS)


def main(dataset_index, output_dir, method_marker, repetitions, extra_method_marker):

    # Load shape
    all_data = cPickle.load(open(METADATA_FILE, "rb"))
    item = all_data[dataset_index]
    # shape_id = str(int(item['shape_id']) + 11)

    if TEMP_INDEX is not None:
        shape_id = TEMP_INDEX# item['shape_id']
    else:
        shape_id = item['shape_id']

    edit_request = item['edit_request'] 
    selected_obj = os.path.join(DATA_DIR, f"{shape_id}", f"{shape_id}.pkl")
    processed_data, symbolic_data = get_obj(selected_obj, redo_search=False, data_dir=DATA_DIR, mode="new",
                                            add_ground=True)
    
    prompter_class = getattr(prompters, extra_method_marker)
    valid_prompter_classes = [prompters.EditExtensionPrompter, 
                              prompters.ProceduralPrompter, 
                              prompters.MotionPrompter]
    assert issubclass(prompter_class, tuple(valid_prompter_classes)), f"Invalid prompter class: {prompter_class}"
    
    img_dir = os.path.join(output_dir, "shape_renders")
    Path(img_dir).mkdir(parents=True, exist_ok=True)
    img_name = os.path.join(img_dir, f"{dataset_index}.png")
    image_gen = ShapeRenderer(processed_data, symbolic_data, img_name)

    shape = symbolic_data[0]
    if 'category' in item.keys():
        shape.label = item['category']
    print(edit_request)
    prompter = prompter_class(MODE, KEY, MODEL, TEMPERATURE, SEED)
    # saved_program = []
    if issubclass(prompter_class, prompters.APIOnlyVisionPrompter):
        start_time = time.time()
        # Generate image here.
        image = image_gen.generate_image()
        output, log_info = prompter.api_call_get_output(shape, edit_request, en_image=image)
            # resolve list nesting:
        end_time = time.time()
    else:
        raise NotImplementedError("Only APIOnlyVisionPrompter is supported for now.")
    # Save information.

    for out in output:
        print(out)
    cur_info = {
        "step": "start",
        'dataset_index': dataset_index,
        "prompter_type": extra_method_marker,
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

    log_info.append(cur_info)
    dir_maker = f"{extra_method_marker}_{repetitions}"
    save_dir = os.path.join(output_dir, "logs", dir_maker)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_name = os.path.join(save_dir, f"logs_{dataset_index}.pkl")
    print(f"Saving the information at {save_name}")
    cPickle.dump(log_info, open(save_name, "wb"))
    # save the information
    save_dir = os.path.join(output_dir, "extras", dir_maker)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_name = os.path.join(save_dir, f"extra_{dataset_index}.pkl")
    cPickle.dump(output, open(save_name, "wb"))

    # Now lets do the next step here itself:
    State.reset()

    prompter_class = getattr(prompters, method_marker)
    prompter = prompter_class(MODE, KEY, MODEL, TEMPERATURE, SEED)
    prompter.repetitions = repetitions
    edit_proposer = parallel_prop.ParallelEditProposer()

    all_programs = {}
    program_signature = {}
    for ind, edit_request in enumerate(output):
        processed_data, symbolic_data = get_obj(selected_obj, redo_search=False, data_dir=DATA_DIR, mode="new",
                                                add_ground=True)
        shape = symbolic_data[0]
        # Run the algorithm
        shape = symbolic_data[0]
        if 'category' in item.keys():
            shape.label = item['category']

        start_time = time.time()
        all_edits, log_info, any_breaking, state_info = algo_v4(shape, edit_request, prompter, edit_proposer)
        end_time = time.time()
        
        save_format = [x.save_format() for x in all_edits]
        all_programs[ind] = save_format
        # signature = ''
        # for relation in shape.all_relations():
        #     key = f"r_{relation.relation_index}"
        #     signature += f"{state_info[key]}"
        # for part in shape.partset:
        #     key = f"p_{part.part_index}"
        #     signature += f"{state_info[key]}"
        # program_signature[ind] = signature
    # map the signatures to indices:

    # signature_to_index = defaultdict(list)
    # for ind, signature in program_signature.items():
    #     signature_to_index[signature].append(ind)
    # # get max
    # max_len = 0
    # max_signature = None
    # for signature, indices in signature_to_index.items():
    #     if len(indices) > max_len:
    #         max_len = len(indices)
    #         max_signature = signature
    # # Now select a subset based on those with matching signatures.
    # selected_indices = signature_to_index[max_signature]
    # all_programs = {ind: all_programs[ind] for ind in selected_indices}
    # Now select a subset based on those with matching signatures.


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

    log_info.append(cur_info)
    save_dir = os.path.join(output_dir, "logs", dir_maker)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_name = os.path.join(save_dir, f"logs_{dataset_index}.pkl")
    print(f"Saving the information at {save_name}")
    cPickle.dump(log_info, open(save_name, "wb"))
    # save the information
    save_dir = os.path.join(output_dir, "programs", dir_maker)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_name = os.path.join(save_dir, f"programs_{dataset_index}.pkl")
    cPickle.dump(all_programs, open(save_name, "wb"))

if __name__ == "__main__":
    # Argument parser for setting PROMPT_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_index", help="dataset index", type=int, default=DATASET_INDEX)
    parser.add_argument("--output_dir", help="output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method_marker", help="method_marker", type=str, default=METHOD_MARKER)
    parser.add_argument("--extra_method_marker", help="method_marker_2", type=str, default=EXTRA_METHOD_MARKER)
    parser.add_argument("--repetitions", help="mode", type=int, default=REPETITIONS)
    args = parser.parse_args()
    main(args.dataset_index, args.output_dir, args.method_marker, args.repetitions, args.extra_method_marker)


import argparse
import os
from edit_sys.visualizer.base import EditSysApp
from scripts.local_config import (DEFAULT_OUTPUT_DIR, DATA_DIR, METHOD_MARKER, DATASET_INDEX, TEMP_INDEX, METADATA_FILE, REPETITIONS, CLASSIFIER_MODE, TARGET_MESH_PATH, EDIT_FEATURE, TASK, RUN_VIS_MODE, SS)
import open3d.visualization.gui as gui
import _pickle as cPickle


if __name__ == "__main__":
    # Input is a data.
    # Should have the obj
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset_index", help="dataset index", type=int, default=DATASET_INDEX)
    parser.add_argument("--method_marker", help="forsaving", type=str, default=METHOD_MARKER)
    parser.add_argument("--temp_index", help="temp_index", type=int, default=TEMP_INDEX)
    parser.add_argument("--classifier_mode", help="classifier_mode", type=bool, default=CLASSIFIER_MODE)
    parser.add_argument("--target_mesh_path", help="target_mesh_path", type=str, default=TARGET_MESH_PATH)
    parser.add_argument("--edit_feature", help="edit_feature", type=str, default=EDIT_FEATURE)
    parser.add_argument("--task", help="task", type=str, default=TASK)
    parser.add_argument("--run_vis_mode", help="run_vis_mode", type=str, default=RUN_VIS_MODE)
    parser.add_argument("--ss", help="scale factor for the slider in the GUI", type=float, default=SS)

    args = parser.parse_args()

    all_data = cPickle.load(open(METADATA_FILE, "rb"))
    item = all_data[args.dataset_index]
    # shape_id = str(int(item['shape_id']) + 11)
    # Temp: Just see the size of program files
    temp_index = args.temp_index
    if temp_index is not None:
        shape_id = temp_index# item['shape_id']
    else:
        shape_id = item['shape_id']
    edit_request = item['edit_request']
    selected_obj = os.path.join(DATA_DIR, f"{shape_id}", f"{shape_id}.pkl")
    # edit_request_list = get_all_edit_requests(selected_class)
    # edit_request = edit_request_list[program_index]
    method_marker = args.method_marker
    # DATA_DIR = "/media/aditya/OS/data/compat/processed/"
    # method_marker=f"{method_marker}_{REPETITIONS}_v2"
    method_marker=f"{method_marker}_{REPETITIONS}"

    # method_marker=f"{method_marker}_{REPETITIONS}"
    # method_marker=f"{method_marker}_{REPETITIONS}_no_rel_types"
    # method_marker=f"{method_marker}_{REPETITIONS}_final"
    # empty_ones = []
    # for i in range(50):
    #     item = all_data[i]
    #     program_dir = os.path.join(args.output_dir, "programs", method_marker)
    #     program_file = os.path.join(program_dir, f"programs_{i}.pkl")
    #     edit_gens = cPickle.load(open(program_file, "rb"))
    #     print(i, len(edit_gens))
    #     if len(edit_gens) == 0:
    #         empty_ones.append(i)
    # method_marker = "GT"
    # print(empty_ones)

    gui.Application.instance.initialize()
    app = EditSysApp(args.dataset_index, 
                     shape_id,
                     selected_obj, 
                     method_marker=method_marker,
                     data_dir=DATA_DIR, 
                     redo_search=False,
                     output_dir=DEFAULT_OUTPUT_DIR,
                     classifier_mode=args.classifier_mode,
                     target_mesh_path=args.target_mesh_path,
                     edit_feature=args.edit_feature,
                     task=args.task,
                     run_vis_mode=args.run_vis_mode,
                     ss= args.ss
                     )
    print(edit_request)
    print("shape index", selected_obj)
    print(args.dataset_index)
    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()

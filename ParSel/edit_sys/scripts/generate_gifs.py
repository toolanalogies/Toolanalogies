import argparse
import os
import numpy as np
from pathlib import Path
import sys
import _pickle as cPickle

import open3d.visualization.gui as gui
from edit_sys.visualizer.gif_maker_v2 import GifCreatorV2

from scripts.local_config import (DATA_DIR, DEFAULT_OUTPUT_DIR, METHOD_MARKER, REPETITIONS,
                                  METADATA_FILE, DATASET_INDEX)

if __name__ == "__main__":
    # Input is a data.
    # Should have the obj
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset_index", help="dataset index", type=int, default=DATASET_INDEX)
    parser.add_argument("--method_marker", help="forsaving", type=str, default=METHOD_MARKER)
    parser.add_argument("--temp", help="forsaving", type=str, default="v1")
    args = parser.parse_args()

    all_data = cPickle.load(open(METADATA_FILE, "rb"))
    item = all_data[args.dataset_index]
    # shape_id = str(int(item['shape_id']) + 11)
    shape_id = item['shape_id']
    edit_request = item['edit_request']
    selected_obj = os.path.join(DATA_DIR, f"{shape_id}", f"{shape_id}.pkl")
    # edit_request_list = get_all_edit_requests(selected_class)
    # edit_request = edit_request_list[program_index]
    method_marker = args.method_marker
    if args.temp == "v1":
        method_marker = f"{method_marker}_{REPETITIONS}_final"
    elif args.temp == "v2":
        method_marker = f"{method_marker}_{REPETITIONS}"
    elif args.temp == "v3":
        method_marker = f"{method_marker}_{REPETITIONS}_no_rel_types"
    

    app = GifCreatorV2(edit_request, 
                     args.dataset_index, 
                     shape_id,
                     selected_obj, 
                     method_marker=method_marker,
                     data_dir=DATA_DIR, 
                     redo_search=False,
                     output_dir=DEFAULT_OUTPUT_DIR)
    print(edit_request)
    # close the application

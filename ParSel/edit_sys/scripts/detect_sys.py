""" DEPRECIATED """

import numpy as np
np.set_printoptions(precision=3, suppress=True)

from edit_sys.data_loader.partnet import main_loader
from edit_sys.data_loader.io import enrich_data, get_save_format

import time
import argparse
import os
import _pickle as cPickle
from pathlib import Path

metadata_file = "/media/aditya/OS/data/partnet/metadata/class_wise_dict.pkl"
N_COUNTS = 50
from edit_sys.data_loader.constants import DATA_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=int, default=3116)
    parser.add_argument("--class_id", type=int, default=0)
    # parser.add_argument("--model_id", type=int, default=22235)
    parser.add_argument("--data_dir", type=str, default="/media/aditya/OS/data/partnet/")
    parser.add_argument("--save_dir", type=str, default="/media/aditya/OS/data/metadata/processed_v5/")
    args = parser.parse_args()
    # create path using Patlib
    Path.mkdir(Path(args.save_dir), parents=True, exist_ok=True)

    # parts, label_to_sel_syms, intersections_indices, sym_anno, contact_anno = load_model(str(args.model_id), args.data_dir)
    # for line in sym_anno:
    #     print(line)
    # for line in contact_anno:
    #     print(line)
    # hierarchy - Manual?
    failure_count = 0
    class_list = ["Chair", "Lamp", "Vase", "StorageFurniture", "Table", "Bed",]
    # class_list = ["Table", "Lamp", "Vase"]
    all_objs = []
    selected_class_list = [class_list[args.class_id]]

    
    for selected_class in selected_class_list:
        obj_list = cPickle.load(open(metadata_file, "rb"))[selected_class]
        selected_objs = np.random.choice(obj_list, N_COUNTS * 2, replace=False)
        selected_objs = [(x, selected_class) for x in selected_objs]
        save_dir = os.path.join(args.save_dir, selected_class)
        Path.mkdir(Path(save_dir), parents=True, exist_ok=True)
        all_objs.extend(selected_objs)
    
    success_count = 0
    # shuffle them
    # np.random.shuffle(all_objs)
    for cur_obj, selected_class_list in all_objs:    
        model_id = str(cur_obj['anno_id'])

        try:
            save_file = os.path.join(save_dir, f"{model_id}.pkl") 
            model, clean_relations, intersections = main_loader(model_id, DATA_DIR)
            model = get_save_format(model)
            items = (model, clean_relations, intersections)
            cPickle.dump(items, open(save_file, "wb"))
            print("SAVED MODEL!")
            success_count += 1

            # parts, label_to_sel_syms, intersections_indices = main_loader(model_id, args.data_dir)
            # parts = get_save_format(parts)
            # items = (parts, label_to_sel_syms, intersections_indices)
            # save_dir = os.path.join(args.save_dir, selected_class)
        except:
            print(f"Failed for {model_id}")
            failure_count += 1
        if success_count == N_COUNTS * len(class_list):
            break
        print(f"Failed for {failure_count} models")
import os
from edit_sys.data_loader.partnet_shape import get_obj
from scripts.local_config import (DATA_DIR, TUPLE_DIR)
from scripts.solve_edit import get_all_valid_files, get_simple_dataset_item, get_shape_prompt_tuple


if __name__ == "__main__":

    dataset_file = os.path.join(TUPLE_DIR, "simple_dataset.txt")
    with open(dataset_file, "r") as f:
        dataset = f.readlines()
    dataset = [x.strip() for x in dataset][20: 52]
    for DATASET_INDEX, line in enumerate(dataset):
        print(f"{DATASET_INDEX}: {line}")
        selected_class, tuple_ind = get_simple_dataset_item(
            DATASET_INDEX)
        shape_ind, prompt_ind = get_shape_prompt_tuple(
            tuple_ind, selected_class)

        file_list = get_all_valid_files(selected_class=selected_class)
        selected_obj = file_list[shape_ind]
        processed_data, symbolic_data = get_obj(
            selected_obj, redo_search=True, data_dir=DATA_DIR, mode="new")

import os
import time
from pathlib import Path
from edit_sys.data_loader.partnet_shape import get_obj
from edit_sys.data_loader.compat_to_partnet import StylizedShapeLoader
from scripts.local_config import DATA_DIR
import _pickle as cPickle
COMPAT_ZIP_PATH = "/media/aditya/OS/data/compat/3DCoMPaT_ZIP.zip"
COMPAT_META_DIR = "/home/aditya/projects/llm_vpi/3DCoMPaT-v2/metadata"
VALID_CLASSES = [
    # "airplane",    # 0 26
    # "bag",         # 27 129
    "basket",        # 130 222
    # "bbq_grill",   # 223 237
    "bed",           # 238 656
    "bench",         # 657 1044
    "bicycle",       # 1045 1072
    # "bird_house",  # 1073 1193
    # "boat",        # 1194 1241 
    "cabinet",       # 1242 1590
    "candle_holder", # 1591 1872
    # "car",         # 1873 1902
    "chair",         # 1903 2848
    "clock",         # 2849 3043
    "coat_rack",     # 3044 3149
    # "curtain",     # 3150 3228
    # "dishwasher",  # 3229 3257
    "dresser",       # 3258 3506
    "fan",           # 3507 3549
    "faucet",        # 3550 3682
    "garbage_bin",   # 3683 3784
    "gazebo",        # 3785 3840
    "jug",           # 3841 4045
    "ladder",        # 4046 4077
    "lamp",          # 4078 4508
    "love_seat",     # 4509 4647
    "ottoman",       # 4648 4932
    # "parasol",     # 4933 4990
    "planter",       # 4991 5439
    "shelf",         # 5440 5564
    # "shower",      # 5565 5657
    "sinks",         # 5658 5727
    "skateboard",    # 5728 5760
    "sofa",          # 5761 6039
    "sports_table",  # 6040 6078
    "stool",         # 6079 6404
    "sun_lounger",   # 6405 6442
    "table",         # 6443 7420
    "toilet",        # 7421 7532
    "tray",          # 7533 7661
    "trolley",       # 7662 7836
    "vase"           # 7837 8075
]



def get_config(zip_path, meta_dir, split, semantic_level):
    config = dict(
        zip_path=zip_path,
        meta_dir=meta_dir,
        split=split,
        semantic_level=semantic_level,
        n_compositions=1,
        get_mats=False,
        n_points=None,
        load_mesh=True,
        shape_only=False,
        get_normals=False,
        seed=None,
        shuffle=False,
    )
    return config
if __name__ == "__main__":
    config = get_config(
        zip_path=COMPAT_ZIP_PATH,
        meta_dir=COMPAT_META_DIR,
        split="train",
        semantic_level="fine"
    )
    data_loader = StylizedShapeLoader(**config)
    # for index in range(10):
    index_list_2 = list(range(0, 8000, 347))
    index_list = [
        # 6400, 300, 390,
        # 650, 725,735,745,
        # 765,775,825, 855,
        # 875, 925, 945,
        # 1022, 1070, 2025,2069,
        # 2075,2095,2120,2133,
        # 2140, 2147, 2154,2160,
        # 2164,2168, 2250,2278,
        # 2350,2700,3050,3521,
        # 3810,4050,
        4560,4520,
        5560,5460,6290,6390,
        6590,6595,6632, 6745,6755,
        6855,
    ]
    index_list += index_list_2
    time_list = []
    # pkl_file = "/media/aditya/OS/data/compat/metadata/compat_edits.pkl"
    # rows = cPickle.load(open(pkl_file, "rb"))
    # index_list = [int(row["shape_id"]) for row in rows]
    index_list = [2553, 2563, 1252, 1271]
    for index in index_list:
        print("here at index", index)
        shape_id, style_id, shape_label, obj = data_loader.__getitem__(index) 
        save_dir = os.path.join(DATA_DIR, f"{index}")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        data_loader.save_to_our_form(index, obj, shape_label, save_dir, allow_split=False)
        print(save_dir)
        # Also use the get_obj function
        selected_obj_file = os.path.join(DATA_DIR, f"{index}", f"{index}.pkl")
        path = os.path.join(DATA_DIR, f"{index}")
        Path(path).mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        _,_ = get_obj(selected_obj_file, redo_search=True,
                    data_dir=DATA_DIR, mode="new", add_ground=False)
        end_time = time.time()
        time_list.append(end_time - start_time)
    print("Time taken for each shape", time_list)
    # save the time list
    time_list_file = os.path.join("/media/aditya/OS/data/compat/metadata/", "time_list.pkl")
    cPickle.dump(time_list, open(time_list_file, "wb"))
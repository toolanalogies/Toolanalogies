import os
import argparse
from scripts.local_config import (DATA_DIR, METADATA_FILE, DEFAULT_OUTPUT_DIR, IMAGE_DIR,
                                  DATASET_INDEX,METHOD_MARKER, REPETITIONS, TEMP_INDEX)
def main():
    id=input("Enter Dataset ID")
    part=input("Enter part")
    human_prompt =input("Enter human prompt")
    

    os.system("conda activate parsel")
    os.system("python scripts/solve_edit.py --dataset_index", id)
    os.system("python scripts/run_vis.py")

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="ID", type=int, default=DATASET_INDEX)
    parser.add_argument("--part", help = "part", type=str, default = "")
    parser.add_argument("--image_dir", help = "image directory", type = str, default = IMAGE_DIR)
    parser.add_argument("--dataset_index", help="dataset index", type=int, default=DATASET_INDEX)
    parser.add_argument("--temp_index", help="temp_index", type=int, default=TEMP_INDEX)
    parser.add_argument("--output_dir", help="output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method_marker", help="method_marker", type=str, default=METHOD_MARKER)
    parser.add_argument("--repetitions", help="mode", type=int, default=REPETITIONS)
    # add flat for loading gt
    parser.add_argument("--load_gt", help="load_gt", action="store_true")
    args = parser.parse_args()
    main(args.dataset_index, args.temp_index, args.output_dir, args.method_marker, args.load_gt, args.repetitions)
    main()
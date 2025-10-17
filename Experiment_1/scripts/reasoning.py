import json
import os
import subprocess

def main():

    # Example data you want to write
    local_config = {
        "shape_features": [
            "blade_length",
            "shaft_length",
            "blade_shaft_angle",
            "blade_thickness",
            "shaft_diameter",
            "blade_width",
        ],
        "data_paths": {
            "blade_length": "path_to_tool_dataset/blade_length",
            "shaft_length": "path_to_tool_dataset/shaft_length",
            "blade_shaft_angle": "path_to_tool_dataset/blade_shaft_angle",
            "blade_thickness": "path_to_tool_dataset/blade_thickness",
            "shaft_diameter": "path_to_tool_dataset/shaft_diameter",
            "blade_width": "path_to_tool_dataset/blade_width",
        },
    }

    for feature in local_config["shape_features"]:
        data_path = local_config["data_paths"][feature]
        edit_config = {
            "data_path": data_path,
            "feature": feature
        }
        edit_config_path = "path to edit_config.json"
        write_json_to_path(edit_config, edit_config_path)

        log_path = "log_path"+feature+".txt"

        command = [
        "python",          # or "python3"
        "run_task.py",
        "--num_envs",
        "10",
        "--task",
        "Isaac-Lift-and-Place-Franka-IK-Rel-v1",
        "--shape_data_path",
        "{0}".format(data_path),
        #"--headless",
        ]
        with open(log_path, "w") as logfile:
            result = subprocess.run(command, stdout=logfile, text=True, check=True)

        print(result.stdout)
        print(result.stderr)
        print("MESSI")


if __name__ == "__main__":
    # run the main function
    main()
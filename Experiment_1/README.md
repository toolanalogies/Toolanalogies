Requirements:
 Tested with Isaac 4.2 and IsaacLab v1.2.0-46-gbe526037
 Other required packages can be seen in environment.yml
 To create usd files for Isaac from obj files 
 python dataset_obj_usd_converter.py --input 'dataset_folder_path'

 To use the Isaac env the following paths should be set:
 In lift_env_cfg.py under sim_scene 'path to work_table.usd in the dataset' 2 times
 In joint_pos_env_cfg.py under sim_scene 'path to hockey_ball.usd in the dataset'
 In joint_pos_env_cfg.py under sim_scene and in reasoning.py 'path to edit_config.json' should be edited and be the same.
 In reasoning.py "path_to_log" if you want to see terminal logs.

 In addition the scene inside the sim_scene folder should be added to manipulation scenes inside Isaaclab.

 run reasoning.py inside scripts to observe simulation runs for counterfactual reasoning
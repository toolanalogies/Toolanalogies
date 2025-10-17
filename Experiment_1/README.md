# Counterfactual Tool Reasoning in IsaacLab

This repository provides scripts and configuration files for running counterfactual reasoning experiments in NVIDIA Isaac Sim and IsaacLab.

## Requirements
- **Isaac Sim:** 4.2  
- **IsaacLab:** v1.2.0-46-gbe526037  
- Additional dependencies are listed in `environment.yml`.

## Dataset Conversion
To generate `.usd` files for Isaac from `.obj` datasets:
```bash
python dataset_obj_usd_converter.py --input 'dataset_folder_path'
```

## Environment Configuration

Before running experiments, update the following paths:

- **`lift_env_cfg.py`** → under `sim_scene`, set the path to `work_table.usd` (appears twice).  
- **`joint_pos_env_cfg.py`** → under `sim_scene`, set the path to `hockey_ball.usd`.  
- **`joint_pos_env_cfg.py`** and **`reasoning.py`** → ensure both reference the same `edit_config.json`.  
- **`reasoning.py`** → optionally update `"path_to_log"` to enable terminal logging.

Make sure the scene defined in `sim_scene` is added to **Manipulation Scenes** inside IsaacLab.

---

## Running Simulations

To observe simulation runs for counterfactual reasoning, execute:
```bash
python scripts/reasoning.py
```

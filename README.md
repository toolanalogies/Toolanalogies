# Creative Robot Tool Use by Counterfactual Reasoning #
This repository contains the implementation of ToolAnalogy, a causal reasoning framework for creative robot tool use. The system enables a robot to identify and repurpose novel objects as functional substitutes for unavailable tools by reasoning over causal, task-relevant features discovered via counterfactual experiments in simulation.
## Gridenvironment ##
This folder contains the GridEnvironment (Appendix A1), which introduces the core concept in an idealized setting where object properties are fully known and can be modified within the environment to generate counterfactual experiments. Refer to the readme inside its folder for installation.
## Part Segmentation ##
To enable features and edits that rely on object subcomponents, we perform part-level segmentation using SAMPart3D, which identifies and separates meaningful generic parts (e.g., handle, head) of 3D objects. This allows part level object edits. Part names are found after the segmentation via VLM. SAMPart3D can be installed from https://github.com/Pointcept/SAMPart3D.
## Object Editing ##
For object editing, we use ParSEL (Ganeshan et al., 2024), which edits 3D objects based on semantic shape features through program synthesis. Given object parts and an edit request, it generates shape variations corresponding to the specified feature (e.g., tip angle, length). Physical properties such as mass are edited directly within the simulator.
Installation of Parsel and its examples can be found in Parsel folder. The codebase is obtained from the authors of Ganeshan et al., 2024, and edited for our purposes.
## Experiment 1 ##
This folder contains the files for the first experiment (Table-top Pulling) where the robot could pull a hockey puck with a hockey stick. In the folder, you can find the simulator scenes, the object dataset consists of edited shape features, counterfactual reasoning scripts, task and tool images.

## Human survey - copy ##
Here, you can find a copy of the questionnaire filled by our participants. 
https://forms.gle/tBCrJiK8r6pUgQzB8

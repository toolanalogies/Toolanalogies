# Code for ParSEL.

Note - temporary pre-release version. 

# Instructions

You will need to configure your python environment with the following packages:

```python
open3d==0.17.0+f3960d9
torch==2.1.0 # might be compatible with other versions
sympy
https://github.com/mavillan/py-hausdorff
openai
```
Other requirements can be seen in environment.yml

1. Unzip the dataset file, and paste the locations in `scripts/local_config.py`.

2. Store your openai config in `.credentials/open_ai.key`. Refer to `edit_sys/llm/common.py` to see how it is processed.

3. Edit 

4. Run 'python scripts/solve_edit.py to run to produce the edit program with a given prompt in `local_config.py`.

5. Run `python scripts/run_vis.py` to run the visualizer. RUN_VIS_MODE in `local_config.py` should be GUI to use the scale to visualize the edit. Save mode creates 10 different instances of edits to create the dataset for that feature and the prompt.
import os

# Datamode
# DATA_MODE = "PARTNET"
DATA_MODE = "GT_EVAL"
#DATA_MODE = "COMPAT"
PROJECT_DIR = "/path_to/ParSel/media/reviewer/OS"
#TODO verify PARTNET and COMPAT need / don't need images?
if DATA_MODE == "GT_EVAL":
    BASE_DIR = os.path.join(PROJECT_DIR, "data/edit_vpi/dataset_1/")
    DATA_DIR = os.path.join(BASE_DIR, "shapes/")
    IMAGE_DIR = os.path.join(PROJECT_DIR, "data/images/")
    METADATA_FILE = os.path.join(BASE_DIR, "metadata/dataset_mod.pkl")
    DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "data/edit_vpi/outputs")


METHOD_MARKER = "UpfrontParallelLLMPrompterV3"  #LLM asks the questions rather than human

EXTRA_METHOD_MARKER = "EditExtensionPrompter"
EXTRA_METHOD_MARKER = "ProceduralPrompter"

REPETITIONS = 1
DATASET_INDEX = 2

TEMP_INDEX = 50005
# 70001 = pull, 70002 = scoop, 50005 = reach
=
VOTE_NUM = 5
RUN_VIS_MODE = "save"  # GUI or save
SAVE_COPY_NUM = 10  # How many copies to save when saving
CLASSIFIER_MODE = False
TARGET_MESH_PATH = "PATH/TO/TARGET/MESH"
EDIT_FEATURE = "length"
EDIT_PROMPT = "Scale the hockeystick_shaft along its principal_axis 0"
TASK = "pull"
SS = 1.0  # scale factor for the slider in the GUI


import itertools
import sympy as sp

MOVE_LIMIT = 0.35
TRIALS = 50
QUANT_COMPARE_THRESHOLD_MODE_1 = 0.025
QUANT_COMPARE_THRESHOLD_MODE_2 = 0.075
UPPER_BOUND = 5
NUMERIC_EVAL = True

COMPARISON_THRESHOLD = 0.01
DOT_THRESHOLD = 0.95
TEST_VALUE = 0.3

RELATION_ACTIVE = 0
RELATION_INACTIVE = 1
RELATION_CONSUMED = 2

RELATION_STABLE = 3
RELATION_BROKEN = 4

RELATION_UNCHECKED = 5
RELATION_RETAINED = 6
RELATION_REJECTED = 7


NOTHING_TO_DO = 0
RESOLVE_RELATION = 1
UPDATE_RELATION = 2

PART_ACTIVE = 0
PART_INACTIVE = 1

PART_UNEDITED = 2
PART_EDITED = 3

# Text to Code
FACE_CENTER_INDICES = {
    "down": 0,
    "up": 1,
    "left": 4,
    "right": 5,
    "back": 2,
    "front": 3,
}
CORNER_INDICES = {}

set_of_sets = [
    set(["down", "left", "back"]),
    set(["down", "right", "back"]),
    set(["up", "right", "back"]),
    set(["up", "left", "back"]),
    set(["down", "left", "front"]),
    set(["down", "right", "front"]),
    set(["up", "right", "front"]),
    set(["up", "left", "front"]),
]
value_list = list(range(8))
for ind, cur_set in enumerate(set_of_sets):
    for perm in itertools.permutations(cur_set):
        key = "_".join(perm)
        CORNER_INDICES[key] = ind

EDGE_CENTER_INDICES = {}
set_of_sets_2 = [
    set(["down", "back"]),
    set(["right", "back"]),
    set(["up", "back"]),
    set(["left", "back"]),
    set(["down", "front"]),
    set(["right", "front"]),
    set(["up", "front"]),
    set(["left", "front"]),
    set(["left", "down"]),
    set(["right", "down"]),
    set(["right", "up"]),
    set(["left", "up"]),
]
value_list = list(range(8))
for ind, cur_set in enumerate(set_of_sets_2):
    for perm in itertools.permutations(cur_set):
        key = "_".join(perm)
        EDGE_CENTER_INDICES[key] = ind

# Code to Text
ANNOTATE_DP_THRESHOLD = 0.9
ANNOTATE_DIST_THRESHOLD = 0.25
DIR_MAT = sp.Matrix(
    [[1, 0, 0], 
     [0, 1, 0], 
     [0, 0, 1], 
     [-1, 0, 0], 
     [0, -1, 0], 
     [0, 0, -1]]
)
N_DIR = DIR_MAT.shape[0]
DIR_NAMES = ["right", 
              "up", 
              "front", 
              "left", 
              "down", 
              "back",
              "primary-up",
              "primary-down",
              "secondary-up",
              "secondary-down"
              ]
DIR_CODE = [
    "RightUnitVector()",
    "UpUnitVector()",
    "FrontUnitVector()",
    "LeftUnitVector()",
    "DownUnitVector()",
    "BackUnitVector()",
    "PrimaryUpUnitVector()",
    "PrimaryDownUnitVector()",
    "SecondaryUpUnitVector()",
    "SecondaryDownUnitVector()"
]


LOCAL_COORD_MAT = sp.Matrix(
    [   
        [0, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 1],
        [1, -1, 0],
        [0, -1, -1],
        [-1, -1, 0],

        [0, 1, 1],
        [1, 1, 0],
        [0, 1, -1],
        [-1, 1, 0],
        
        [-1, 0, 1],
        [1, 0, 1],
        [1, 0, -1],
        [-1, 0, -1],

        [-1, -1, 1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, 1],
        [1, 1, 1],
        [1, 1, -1],
        [-1, 1, -1],
    ]
)
N_LOCAL_COORDS = LOCAL_COORD_MAT.shape[0]


COORD_LABELS = [
    "origin",
    "down face center",
    "up face center",
    "front face center",
    "back face center",
    "left face center",
    "right face center",
    "front down edge center",
    "right down edge center",
    "back down edge center",
    "left down edge center",

    "front up edge center",
    "right up edge center",
    "back up edge center",
    "left up edge center",

    "left front edge center",
    "right front edge center",
    "right back edge center",
    "left back edge center",

    "left front down corner",
    "right front down corner",
    "right back down corner",
    "left back down corner",
    "left front up corner",
    "right front up corner",
    "right back up corner",
    "left back up corner",
]

POINT_NAMES = [
    "center",
    "face_center(down)",
    "face_center(up)",
    "face_center(front)",
    "face_center(back)",
    "face_center(left)",
    "face_center(right)",
    "edge_center(front-down)",
    "edge_center(right-down)",
    "edge_center(back-down)",
    "edge_center(left-down)",
    "edge_center(front-up)",
    "edge_center(right-up)",
    "edge_center(back-up)",
    "edge_center(left-up)",
    "edge_center(front-left)",
    "edge_center(front-right)",
    "edge_center(back-right)",
    "edge_center(back-left)",
    "corner(left-front-down)",
    "corner(right-front-down)",
    "corner(right-back-down)",
    "corner(left-back-down)",
    "corner(left-front-up)",
    "corner(right-front-up)",
    "corner(right-back-up)",
    "corner(left-back-up)",
]

PROMPT_SIGS = [
    "$part.center()",
    "$part.face_center('down')",
    "$part.face_center('up')",
    "$part.face_center('front')",
    "$part.face_center('back')",
    "$part.face_center('left')",
    "$part.face_center('right')",
    "$part.edge_center('front', 'down')",
    "$part.edge_center('right', 'down')",
    "$part.edge_center('back', 'down')",
    "$part.edge_center('left, 'down')",
    "$part.edge_center('front', 'up')",
    "$part.edge_center('right', 'up')",
    "$part.edge_center('back', 'up')",
    "$part.edge_center('left', 'up')",
    "$part.edge_center('front', 'left')",
    "$part.edge_center('front', 'right')",
    "$part.edge_center('back', 'right')",
    "$part.edge_center('back', 'left')",
    "$part.corner('left', 'front', 'down')",
    "$part.corner('right', 'front', 'down')",
    "$part.corner('right', 'back', 'down')",
    "$part.corner('left', 'back', 'down')",
    "$part.corner('left', 'front', 'up')",
    "$part.corner('right', 'front', 'up')",
    "$part.corner('right', 'back', 'up')",
    "$part.corner('left', 'back', 'up')",
]

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import gym
from gym import spaces
from PIL import Image as PILim

grid_height = 3
grid_width = 5
objects = {'box', 'wall', 'red_ball', 'green_ball', 'agent'}
tile = {'normal', 'lava'}
actions = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
features = [0, 0, 0, 0, 0, 0, 0, 0, 0]
textile_features_key = ['id', 'x', 'y', 'color', 'termination']
object_features_key = ['id', 'x', 'y', 'controllable', 'rigid', 'movable', 'color', 'shape', 'termination']

initial_state={'dimensions': {'height': 3, 'width': 5},
               'textiles': {'lava0': {'id': '!', 'x': 4, 'y': 2, 'color': 'orange', 'termination': True}},
               'objects': {'wall0': {'id': 'W','x': 1, 'y': 1, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False},
                           'wall1': {'id': 'W','x': 4, 'y': 0, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False},
                           'wall2': {'id': 'W','x': 4, 'y': 1, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False},
                           'wall3': {'id': 'W','x': 4, 'y': 2, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False},  
                           'red_ball': {'id': 'R','x': 3, 'y': 1, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'red', 'shape': 'round', 'termination': True},
                           'red_ball2': {'id': 'R','x': 2, 'y': 0, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'red', 'shape': 'round', 'termination': True},  
                           'green_ball': {'id': 'G','x': 3, 'y': 2, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'green', 'shape': 'round', 'termination': True},
                           'agent': {'id': 'A','x': 0, 'y': 0, 'controllable': True, 'rigid': True, 'movable': True, 'color': 'blue', 'shape': 'triangle', 'termination': False},
                           'box': {'id': 'B','x': 1, 'y': 0, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'brown', 'shape': 'square', 'termination': False}}}

# %%
grid_height = 6
grid_width = 6
objects = {'box', 'wall', 'red_ball', 'green_ball', 'agent'}
tile = {'normal', 'lava'}
actions = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
features = [0, 0, 0, 0, 0, 0, 0, 0, 0]
features_key = ['id', 'x', 'y', 'controllable', 'rigid', 'movable', 'color', 'shape', 'termination']


# %%
initial_state={'dimensions': {'height': 3, 'width': 5},
               'textiles': {'lava0': {'id': '!', 'x': 4, 'y': 2, 'color': 'orange', 'termination': True}},
               'objects': {'wall0': {'id': 'W','x': 1, 'y': 1, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False},
                           'wall1': {'id': 'W','x': 4, 'y': 0, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False},
                           'wall2': {'id': 'W','x': 4, 'y': 1, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False},
                           'wall3': {'id': 'W','x': 4, 'y': 2, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False},  
                           'red_ball': {'id': 'R','x': 3, 'y': 1, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'red', 'shape': 'round', 'termination': True},
                           'red_ball2': {'id': 'R','x': 2, 'y': 0, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'red', 'shape': 'round', 'termination': True},  
                           'green_ball': {'id': 'G','x': 3, 'y': 2, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'green', 'shape': 'round', 'termination': True},
                           'agent': {'id': 'A','x': 0, 'y': 0, 'controllable': True, 'rigid': True, 'movable': True, 'color': 'blue', 'shape': 'triangle', 'termination': False},
                           'box': {'id': 'B','x': 1, 'y': 0, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'brown', 'shape': 'square', 'termination': False}}}

# %%
# for name in initial_state['objects']:
#     print(name)

# for obj in initial_state['objects'].values():
#     print(obj)

# for name, obj in initial_state['objects'].items():
#     print(name, obj)

# %%
def renderer(state):
    grid = np.full((2, state['dimensions']['height'], state['dimensions']['width']), '.', dtype=object)

    for value in state['textiles'].values():
        grid[0,value['y'],value['x']] = value['id']

    for value in state['objects'].values():
        grid[1,value['y'],value['x']] = value['id']
    return grid

# %%
def fancy_renderer(state):
    grid = renderer(state)
    return grid[1] + grid[0]

# %%
# print(fancy_renderer(initial_state))

# %%
def moving_boxes(state, action, cur_obj, cur_name):
    
    next_state = deepcopy(state)
    next_state['objects'][cur_name]['x'] += actions[action][0]
    next_state['objects'][cur_name]['y'] += actions[action][1]
    grid = renderer(state)
    tile_char = grid[0,next_state['objects'][cur_name]['y'],next_state['objects'][cur_name]['x']]
    obj_char = grid[1,next_state['objects'][cur_name]['y'],next_state['objects'][cur_name]['x']]    
    #state['objects'][cur_name]['x'] += actions[action][0]
    #state['objects'][cur_name]['y'] += actions[action][1]

    if obj_char == '.':
        return next_state, True
    for name, obj in state['objects'].items():
        if obj['id'] == obj_char:
            next_obj = obj
            next_name = name

    if not next_obj['movable']:
        return state, False
    else:
        temp_state, movable_bool = moving_boxes(state, action, next_obj, next_name)


    if movable_bool:
        temp_state['objects'][cur_name]['x'] += actions[action][0]
        temp_state['objects'][cur_name]['y'] += actions[action][1]

        return temp_state, True
    else:
        return state, False

# %%
def step(state, action):
    next_state = deepcopy(state)
    next_state['objects']['agent']['x'] += actions[action][0]
    next_state['objects']['agent']['y'] += actions[action][1]
    grid = renderer(state)
    done = False
    tile_char = grid[0,next_state['objects']['agent']['y'],next_state['objects']['agent']['x']]
    obj_char = grid[1,next_state['objects']['agent']['y'],next_state['objects']['agent']['x']]
    reward = 0
    for tile in state['textiles'].values():
        if tile['id'] == tile_char:
            done = tile['termination']
            if tile['color'] == 'orange':
                reward = -1
            if done:
                return next_state, reward, True

    if obj_char == '.':
        return next_state, reward, False

    for name, obj in state['objects'].items():
        if obj['id'] == obj_char:
            cur_obj = obj
            cur_name = name


    if cur_obj['termination']:
        if cur_obj['color'] == 'green':
            reward = 1
        elif cur_obj['color'] == 'red':
            reward = -1
        return next_state, reward, True
    if not cur_obj['movable']:
        return state, 0, False
    else:
        temp_state, movable_bool = moving_boxes(state, action, cur_obj, cur_name)
        if movable_bool:
            temp_state['objects']['agent']['x'] += actions[action][0]
            temp_state['objects']['agent']['y'] += actions[action][1]
            return temp_state, 0, False
        else:
            return state, 0, False

if __name__ == "__main__":
   

    # %%
    test_arr = np.full((2,2), '.', dtype=object)
    test_arr[0][0] += "W" 
    print(test_arr)

    # %%
    print(fancy_renderer(initial_state))
    next_state, reward, done = step(initial_state, 'right')
    print(fancy_renderer(next_state))

    next_state, reward, done = step(next_state, 'right')
    print(fancy_renderer(next_state))

    next_state, reward, done = step(next_state, 'right')
    print(fancy_renderer(next_state))

    next_state, reward, done = step(next_state, 'down')
    print(fancy_renderer(next_state))
    print(done,reward)

    next_state, reward, done = step(next_state, 'left')
    print(fancy_renderer(next_state))

    print(done,reward)
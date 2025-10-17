import numpy as np
import pygame
import Conceptgrid
from Conceptgrid.envs.Concept_classes2 import GridEnv2
from Conceptgrid.envs.grid2pddl import PDDLWriter
import subprocess
import copy
import os

import sys
from PIL import Image

all_objects = {'objects':{'wall': {'id': 'W','x': 2, 'y': 2, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'goal': False},
                        'green_ball': {'id': 'G','x': 4, 'y': 3, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'green', 'shape': 'round', 'goal': True},
                        'agent': {'id': 'A','x': 1, 'y': 1, 'controllable': True, 'rigid': True, 'movable': True, 'color': 'blue', 'shape': 'triangle', 'goal': False},
                        'box': {'id': 'B','x': 2, 'y': 4, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'brown', 'shape': 'square', 'goal': False},
                        'red_ball': {'id': 'R','x': 4, 'y': 4, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'red', 'shape': 'round', 'goal': True}}}

dict_state={'dimensions': {'height': 10, 'width': 10},
            'textiles': {'lava': {'id': '!', 'x': 4, 'y': 3, 'color': 'orange', 'textile_termination': 'death'}, 'goal': {'id': '?', 'x': 6, 'y': 6, 'color': 'white', 'textile_termination': 'goal'}},
            'objects': {'wall': {'id': 'W','x': 2, 'y': 2, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'goal': False},
                        'green_ball': {'id': 'G','x': 4, 'y': 3, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'green', 'shape': 'round', 'goal': True},
                        'agent': {'id': 'A','x': 1, 'y': 1, 'controllable': True, 'rigid': True, 'movable': True, 'color': 'blue', 'shape': 'triangle', 'goal': False},
                        'box': {'id': 'B','x': 2, 'y': 4, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'brown', 'shape': 'square', 'goal': False},
                        }}

deficient_state={'dimensions': {'height': 10, 'width': 10},
            'textiles': {'lava': {'id': '!', 'x': 4, 'y': 3, 'color': 'orange', 'textile_termination': 'death'}, 'goal': {'id': '?', 'x': 6, 'y': 6, 'color': 'white', 'textile_termination': 'goal'}},
            'objects': {'wall': {'id': 'W','x': 2, 'y': 2, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'goal': False},
                        'green_ball': {'id': 'G','x': 4, 'y': 3, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'green', 'shape': 'round', 'goal': True},
                        'agent': {'id': 'A','x': 1, 'y': 1, 'controllable': True, 'rigid': True, 'movable': True, 'color': 'blue', 'shape': 'triangle', 'goal': False},
                        }}

saved_dic = copy.deepcopy(dict_state)

object_features_key = ['id', 'x', 'y', 'controllable', 'rigid', 'movable', 'color', 'shape', 'goal']
object_features_key_no_pose = ['id', 'controllable', 'rigid', 'movable', 'color', 'shape', 'goal']
object_features_key_ranges = {
    'id': ['W', 'G', 'A', 'B', 'R'], 
    'controllable': [True, False], 'rigid': [True, False], 'movable': [True, False], 'goal': [True, False],
    'color': ['black', 'red', 'blue', 'green', 'brown'], 'shape': ['square', 'round', 'triangle']}
        

env = GridEnv2(dict_state = dict_state, state_mode = 'obj', randomize = False, tool_usage = True, change_object_size= False)

pddl_write = PDDLWriter()
pddl_domain, pddl_problem = pddl_write.grid2pddl(env)
with open('path to domain.pddl', "w") as domain_file:
        domain_file.write(pddl_domain)

with open('path to problem.pddl', "w") as problem_file:
        problem_file.write(pddl_problem)

subprocess.call("python path to fast-downward.py \
                    path to domain.pddl path to problem.pddl --search 'astar(blind())'", shell=True)

action_list = []

with open('path to sas_plan', 'r') as file:
    for line in file:
        line = line.strip()

        # Split the line into words
        words = line.split()

        # Now 'words' is a list of all words in this particular line
        print(words)
        if words[0] == '(move_agent':
            action_list.append(words[2])
print(action_list)

BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
WHITE = (200, 200, 200)
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800

window = (606,606)
size = (101, 101)
fps = 5
wait = False

env = GridEnv2(dict_state = dict_state, state_mode = 'obj', randomize = False, tool_usage = True, change_object_size= False)
pygame.init()
screen = pygame.display.set_mode(window)
clock = pygame.time.Clock()
running = True

done = False
duration = 0

reward_list = []

obs = env.reset()
st = 0
while running:
   
    # Rendering.
    image = env.image_renderer()
    if size != window:
        image = Image.fromarray(image)
        image = image.resize(window, resample=Image.NEAREST)
        image = np.array(image)
    surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(fps)
    pygame.time.delay(100)
    
   
    obs, reward, done, info = env.step(env.act(action_list[st]))
    reward_list.append(reward)
    if done:
            running = False       
    duration += 1
    if duration >200:
        running = False
    st += 1

reward_sum = sum(reward_list)


relevant_attributes = set()

for i_name, v_range in object_features_key_ranges.items():
    dict_state = copy.deepcopy(saved_dic)
    for val in v_range:
        dict_state['objects']['box'][i_name] = val

        env = GridEnv2(dict_state = dict_state, state_mode = 'obj', randomize = False, tool_usage = True, change_object_size= False)
        running = True
        done = False
        duration = 0
        temp_reward_list = []

        obs = env.reset()
        st = 0
        while running:
            obs, reward, done, info = env.step(env.act(action_list[st]))
            temp_reward_list.append(reward)
            if done:
                    running = False       
            duration += 1
            if duration > len(action_list) - 1:
                running = False
            st += 1
        temp_reward_sum = sum(temp_reward_list)
        if (temp_reward_sum != reward_sum):
            relevant_attributes.add((i_name, saved_dic['objects']['box'][i_name]))
print(relevant_attributes)

fitting_objects = []

for i_name, v_range in all_objects['objects'].items():
    for n, val in relevant_attributes:
        if (v_range[n] == val):
            fitting_objects.append(v_range)

print(fitting_objects)

for val in fitting_objects:
    new_state = copy.deepcopy(deficient_state)
    new_state['objects']['replacement'] = val
    new_state['objects']['replacement']['x'] = 2
    new_state['objects']['replacement']['y'] = 4
    print(new_state)
    env = GridEnv2(dict_state = new_state, state_mode = 'obj', randomize = False, tool_usage = True, change_object_size= False)
    pygame.init()
    screen = pygame.display.set_mode(window)
    clock = pygame.time.Clock()
    running = True
    done = False
    duration = 0

    obs = env.reset()
    st = 0
    while running:
    
        # Rendering.
        image = env.image_renderer()
        if size != window:
            image = Image.fromarray(image)
            image = image.resize(window, resample=Image.NEAREST)
            image = np.array(image)
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(fps)
        pygame.time.delay(100)
        obs, reward, done, info = env.step(env.act(action_list[st]))
        if done:
                running = False       
        duration += 1
        if duration >200:
            running = False
        st += 1



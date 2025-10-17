import numpy as np
import pygame
import Conceptgrid
from Conceptgrid.envs.Concept_classes2 import GridEnv2

import matplotlib.pyplot as plt
import os


import sys
from PIL import Image

action_list = []

with open('path to sas plan', 'r') as file:
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

env = GridEnv2(state_mode = 'obj', randomize = False, tool_usage = True, change_object_size= False)

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

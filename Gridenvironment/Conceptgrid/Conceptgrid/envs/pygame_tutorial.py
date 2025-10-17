import pygame 
import sys
import numpy as np
from Concept_classes2 import GridEnv2
from PIL import Image
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
WHITE = (200, 200, 200)
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800



def pygame_render(initial_state):
    global SCREEN, CLOCK
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(WHITE)

    while True:
        drawGrid()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                im=pygame.surface()
                print(np.shape(im))
                im_arr=pygame.image.tobytes()
                print(np.shape(im_arr))
                pygame.quit()
                sys.exit()

        pygame.display.update()


def drawGrid():
    blockSize = 80 #Set the size of the grid block
    for x in range(0, WINDOW_WIDTH, blockSize):
        for y in range(0, WINDOW_HEIGHT, blockSize):
            rect = pygame.Rect(x, y, blockSize, blockSize)
            pygame.draw.rect(SCREEN, BLACK, rect, 1)
    snake = [(0,0), (0,1), (1,1), (1,2), (1,3)]

    # head

    x, y = snake[0]
    rect = pygame.Rect(x*blockSize, y*blockSize, blockSize, blockSize)
    pygame.draw.circle(SCREEN, GREEN, ((x+0.5)*blockSize, (y+0.5)*blockSize), blockSize/2)

    # tail

    for x, y in snake[1:]:
        rect = pygame.Rect(x*blockSize, y*blockSize, blockSize, blockSize)
        pygame.draw.circle(SCREEN, GREEN, ((x+0.5)*blockSize, (y+0.5)*blockSize), blockSize/2)

if __name__ == "__main__":
    new_Grid = GridEnv2(state_mode = 'obj', randomize = False, tool_usage = True, change_object_size= False)

    window = (606,606)
    size = (101, 101)
    fps = 5
    wait = False

    keymap = {
    pygame.K_a: 'left',
    pygame.K_d: 'right',
    pygame.K_w: 'up',
    pygame.K_s: 'down'}


    # Inspired from crafter

    pygame.init()
    screen = pygame.display.set_mode(window)
    clock = pygame.time.Clock()
    running = True

    done = False
    duration = 0
    while running:

        # Rendering.
        image = new_Grid.image_renderer()
        if size != window:
            image = Image.fromarray(image)
            image = image.resize(window, resample=Image.NEAREST)
            image = np.array(image)
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(fps)

        action = None
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in keymap.keys():
                action = keymap[event.key]
        if action is None:
            pressed = pygame.key.get_pressed()
            for key, action in keymap.items():
                if pressed[key]:
                    break
            else:
                if wait:
                    continue
                else:
                    action = 'noop'
        if action != 'noop':
            _, reward, done, _ = new_Grid.step(new_Grid.act(action))#_, reward, done, _, _ = new_Grid.step(new_Grid.act(action))
            print(reward)
        if done:
            obs = new_Grid.reset()
            done = False
            #running = False       
        duration += 1
    pygame.quit()
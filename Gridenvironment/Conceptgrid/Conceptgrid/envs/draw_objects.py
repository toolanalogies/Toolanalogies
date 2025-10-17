import numpy as np
from PIL import Image as im
c_green = np.array([0,255,0])
c_white = np.array([255,255,255])
c_red = np.array([255,0,0])
c_blue = np.array([0,0,255])

def green_ball(block):
    block[0,4,:] = c_green
    block[1,2:7,:] = c_green
    block[2,1:8,:] = c_green
    block[3,1:8,:] = c_green
    block[4,:,:] = c_green
    block[5:,:,:] = np.flip(block[:4,:,:],0)
    return block

def red_ball(block):
    block[0,4,:] = c_red
    block[1,2:7,:] = c_red
    block[2,1:8,:] = c_red
    block[3,1:8,:] = c_red
    block[4,:,:] = c_red
    block[5:,:,:] = np.flip(block[:4,:,:],0)
    return block

def box(block):
    block[1:-1, 1:-1 , :] = np.array([150, 75, 0])
    return block

def agent(block):
    block[0,[0, 1, 7, 8],:] = c_blue
    block[1,[0, 1, 2, 6, 7, 8],:] = c_blue
    block[2,[1, 2, 3, 5, 6, 7],:] = c_blue
    block[3,2:7,:] = c_blue
    block[4,3:6,:] = c_blue
    block[5:,:,:] = np.flip(block[:4,:,:],0)
    return block

def wall(block):
    block[:, : , :] = 127
    return block

def lava(block):
    block = np.load('/path_to/Conceptgrid/Conceptgrid/envs/lava_good.npy')
    return block

def goal(block):
    block[: ,: , :] = c_white
    return block

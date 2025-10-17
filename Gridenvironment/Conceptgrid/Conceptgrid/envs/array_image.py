# Python program to convert
# numpy array to image
  
# import required libraries
import numpy as np
from PIL import Image as im
  
# define a main function
def image_grid():
  
    # create a numpy array from scratch
    # using arange function.
    # 1024x720 = 737280 is the amount 
    # of pixels.
    # np.uint8 is a data type containing
    # numbers ranging from 0 to 255 
    # and no non-negative integers
    grid_im = np.zeros((101,101,3), np.uint8)
    for i in range(11):
        grid_im[i*10, :, :] = 255
        grid_im[:, i*10, :] = 255
    # creating image object of
    # above array
    np.save('grid', grid_im)
    grid = im.fromarray(grid_im)
    
    # saving the final output 
    # as a PNG file
    grid.save('grid.png')
    
    box_im = np.zeros((9,9,3), np.uint8)
    box_im[:, : , 0] = 150
    box_im[:, : , 1] = 75
    np.save('box', box_im)
    box = im.fromarray(box_im)
    box.save('box.png')

    agent_im = np.zeros((9,9,3), np.uint8)
    agent_im[0,[0, 1, 7, 8],2] = 255
    agent_im[1,[0, 1, 2, 6, 7, 8],2] = 255
    agent_im[2,[1, 2, 3, 5, 6, 7],2] = 255
    agent_im[3,2:7,2] = 255
    agent_im[4,3:6,2] = 255
    agent_im[5:,:,:] = np.flip(agent_im[:4,:,:],0)
    np.save('agent', agent_im)
    agent = im.fromarray(agent_im)
    agent.save('agent.png')



    lava_im = np.zeros((9,9,3), np.uint8)
    lava_list = [np.array([255,0,0]), np.array([255,255,0]), np.array([255,165,0])]
    for i in range(9):
        for j in range(9):
            lava_im[i,j] = lava_list[np.random.randint(0,3)] #lava_list[np.mod(i+j,3)]
    np.save('lava', lava_im)
    lava = im.fromarray(lava_im)
    lava.save('lava.png')

    red_ball_im = np.zeros((9,9,3), np.uint8)
    red_ball_im[0,4,0] = 255
    red_ball_im[1,2:7,0] = 255
    red_ball_im[2,1:8,0] = 255
    red_ball_im[3,1:8,0] = 255
    red_ball_im[4,:,0] = 255
    red_ball_im[5:,:,:] = np.flip(red_ball_im[:4,:,:],0)
    np.save('red_ball', red_ball_im)
    red_ball = im.fromarray(red_ball_im)
    red_ball.save('red_ball.png')

    green_ball_im = np.zeros((9,9,3), np.uint8)
    green_ball_im[0,4,1] = 255
    green_ball_im[1,2:7,1] = 255
    green_ball_im[2,1:8,1] = 255
    green_ball_im[3,1:8,1] = 255
    green_ball_im[4,:,1] = 255
    green_ball_im[5:,:,:] = np.flip(green_ball_im[:4,:,:],0)
    np.save('green_ball', green_ball_im)
    green_ball = im.fromarray(green_ball_im)
    green_ball.save('green_ball.png')

    wall_im = np.zeros((9,9,3), np.uint8)
    wall_im[:, : , :] = 127
    np.save('wall', wall_im)
    wall = im.fromarray(wall_im)
    wall.save('wall.png')


  
# driver code
if __name__ == "__main__":
    
  # function call
  image_grid()
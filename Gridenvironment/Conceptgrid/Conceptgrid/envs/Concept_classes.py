# %%
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import gym
from gym import spaces
from PIL import Image as PILim
from Conceptgrid.envs import draw_objects as draw
from Conceptgrid.wrappers import StateModeWrapper
#import draw_objects as draw


class GridObj():
    def __init__(self, id, x, y, controllable, rigid, movable, color, shape, termination):
        self.id = id
        self.x = x
        self.y = y
        self.controllable = controllable
        self.rigid = rigid
        self.movable = movable
        self.color = color
        self.shape = shape
        self.termination = termination

class Textiles():
    def __init__(self, id, x, y, color, termination):
        self.id = id
        self.x = x
        self.y = y
        self.color = color
        self.termination = termination


class GridState():
    textile_features_key = ['id', 'x', 'y', 'color', 'termination']
    object_features_key = ['id', 'x', 'y', 'controllable', 'rigid', 'movable', 'color', 'shape', 'termination']
    def __init__(self, state_dict):
        self.height = state_dict['dimensions']['height']
        self.width = state_dict['dimensions']['width']
        self.textiles = {}
        self.objects = {}
        for textile_name, textile_value in state_dict['textiles'].items():
            params = [textile_value[key] for key in self.textile_features_key]
            self.textiles[textile_name] = Textiles(*params)
        for object_name, object_value in state_dict['objects'].items():
            params = [object_value[key] for key in self.object_features_key]
            self.objects[object_name] = GridObj(*params)
          

class SubObj(GridObj):
    def __init__(self, id, x, y, controllable, rigid, movable, color, shape, termination):
        super(SubObj, self).__init__(id, x, y, controllable, rigid, movable, color, shape, termination)



class GridEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    actions = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
    dict_state={'dimensions': {'height': 6, 'width': 6},
               'textiles': {'lava0': {'id': '!', 'x': 3, 'y': 3, 'color': 'orange', 'termination': True}, 'lava1': {'id': '!', 'x': 3, 'y': 4, 'color': 'orange', 'termination': True}},
               'objects': {'wall0': {'id': 'W','x': 2, 'y': 2, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False},
                           #'wall1': {'id': 'W','x': 4, 'y': 0, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False},
                           #'wall2': {'id': 'W','x': 4, 'y': 1, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False},
                           #'wall3': {'id': 'W','x': 4, 'y': 2, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False},  
                           'red_ball': {'id': 'R','x': 4, 'y': 1, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'red', 'shape': 'round', 'termination': True},
                           #'red_ball2': {'id': 'R','x': 2, 'y': 0, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'red', 'shape': 'round', 'termination': True},  
                           'green_ball': {'id': 'G','x': 3, 'y': 4, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'green', 'shape': 'round', 'termination': True},
                           'agent': {'id': 'A','x': 1, 'y': 1, 'controllable': True, 'rigid': True, 'movable': True, 'color': 'blue', 'shape': 'triangle', 'termination': False},
                           'box': {'id': 'B','x': 2, 'y': 4, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'brown', 'shape': 'square', 'termination': False}}}

    feature_map = {
        'textile_id' : ['!'],
        'textile_color' : ['orange'],
        'object_id' : ['A','B','G','R','W'],
        'object_color' : ['red', 'blue', 'green', 'brown', 'black'],
        'object_shape' : ['square', 'triangle', 'round']
    }

    # box_im = np.load('box.npy')
    # agent_im = np.load('agent.npy')
    # lava_im = np.load('lava_good.npy')
    # red_ball_im = np.load('red_ball.npy')
    # green_ball_im = np.load('green_ball.npy')
    # wall_im = np.load('wall.npy')

    image_map = {
        '!' : draw.lava,
        'W' : draw.wall,
        'R' : draw.red_ball,
        'G' : draw.green_ball,
        'A' : draw.agent,
        'B' : draw.box
    }

    # dictionary = {'!': 1, 'orange': 1, 
    #             'A': np.array([1, 0, 0, 0, 0]), 'B': np.array([0, 1, 0, 0, 0]), 'G': np.array([0, 0, 1, 0, 0]), 'R': np.array([0, 0, 0, 1, 0]), 'W': np.array([0, 0, 0, 0, 1]),
    #             'red': np.array([1, 0, 0, 0, 0]), 'blue': np.array([0, 1, 0, 0, 0]), 'green': np.array([0, 0, 1, 0, 0]), 'brown': np.array([0, 0, 0, 1, 0]), 'black': np.array([0, 0, 0, 0, 1]),
    #             'square': np.array([1, 0, 0]), 'triangle': np.array([0, 1, 0]), 'round': np.array([0, 0, 1])}

    action_set = list(actions.values())

    def __init__(self, max_height = 10, max_width = 10, min_height = 3, min_width = 3, max_eps_len = 200, state_mode = 'tabular'):
        super(GridEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.state = None
        self.state_mode = state_mode
        self.max_height = max_height
        self.min_height = min_height
        self.max_width = max_width
        self.min_width = min_width
        self.max_eps_len = max_eps_len
        self.action_space = spaces.Discrete(len(self.actions))
        self.grid_start = (0,0)
        self.timestep = 0
        self.gamma = 0.9999
        # 2 types of state representation rgb or object feature representation as a 3D array.
        if state_mode == 'rgb':
            self.observation_space = spaces.Box(low=0, high=255, shape=
                        (max_height * 10 + 1, max_width * 10 + 1, 3), dtype=np.uint8)
        elif state_mode == 'tabular':
            self.observation_space = spaces.Discrete(32)
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=
                        (max_height, max_width, 20), dtype=np.uint8)
        self.initial_state = self.add_walls_around()
        self.reset()

    def add_walls_around(self):
        counter = 0
        for i in range(self.dict_state['dimensions']['height']):
            counter = counter + 1
            self.dict_state['objects']['wall'+str(counter)] = {'id': 'W','x': 0, 'y': i, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False}
        for i in range(self.dict_state['dimensions']['height']):
            counter = counter + 1
            self.dict_state['objects']['wall'+str(counter)] = {'id': 'W','x': self.dict_state['dimensions']['width']-1, 'y': i, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False}
        
        for i in range(1, self.dict_state['dimensions']['width']-1):
            counter = counter + 1
            self.dict_state['objects']['wall'+str(counter)] = {'id': 'W','x': i, 'y': 0, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False}
        
        for i in range(1, self.dict_state['dimensions']['width']-1):
            counter = counter + 1
            self.dict_state['objects']['wall'+str(counter)] = {'id': 'W','x': i, 'y': self.dict_state['dimensions']['height']-1, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False}



        return self.dict_state

    def reset(self):
        self.done = False
        self.timestep = 0
        self.state = GridState(self.initial_state)
        self.grid_start = (0,0) # not used at the moment
        obs = self.observation_function()

        return obs#, {}   #TODO Info is deleted.

    def get_state(self):
        return self.state

    def encode(self,state):
        agent_x = state.objects['agent'].x
        agent_y = state.objects['agent'].y
        box_x = state.objects['box'].x
        agent_num = 4*(agent_x-1) + (agent_y-1)
        agent_num += (box_x-2)*16
        return agent_num


    def decode(self,num):
        box_x = (num // 16) + 2
        agent_num = num % 16
        agent_x = (agent_num // 4) + 1
        agent_y = (agent_num % 4) + 1
        return box_x, agent_x, agent_y
        
    def observation_function(self):
        #start_x, start_y = self.grid_start   # not used at the moment
        if self.state_mode == 'obj':
            observation = np.zeros((10, 10, 20))
            dictionary = {}
            for v in self.feature_map.values():
                for i,x in enumerate(v):
                    vec = np.zeros(len(v))
                    vec[i] = 1
                    dictionary[x]= vec
            for obj in self.state.objects.values():
                observation[obj.y, obj.x, 0:5] = dictionary[obj.id]
                observation[obj.y, obj.x, 5] = int(obj.controllable)
                observation[obj.y, obj.x, 6] = int(obj.rigid)
                observation[obj.y, obj.x, 7] = int(obj.movable)
                observation[obj.y, obj.x, 8:13] = dictionary[obj.color]
                observation[obj.y, obj.x, 13:16] = dictionary[obj.shape]
                observation[obj.y, obj.x, 16] = int(obj.termination)
            for txl in self.state.textiles.values():
                observation[txl.y, txl.x, 17] = dictionary[txl.id]
                observation[txl.y, txl.x, 18] = dictionary[txl.color]
                observation[txl.y, txl.x, 19] = int(txl.termination)
        elif self.state_mode == 'tabular':
            return self.encode(self.state)
        else:
            return self.image_renderer()

        return observation

        
            
    def moving_boxes(self, state, action, cur_name):
        
        next_state = deepcopy(state)
        next_state.objects[cur_name].x += self.action_set[action][0]
        next_state.objects[cur_name].y += self.action_set[action][1]
        grid = self.grid_renderer()
        #tile_char = grid[0,next_state.objects[cur_name].y,next_state.objects[cur_name].x]
        obj_char = grid[1,next_state.objects[cur_name].y,next_state.objects[cur_name].x]    
        #state['objects'][cur_name]['x'] += actions[action][0]
        #state['objects'][cur_name]['y'] += actions[action][1]

        if obj_char == '.':
            return next_state, True
        for name, obj in state.objects.items():
            if obj.id == obj_char:
                next_obj = obj
                next_name = name

        if not next_obj.movable:
            return state, False
        else:
            temp_state, movable_bool = self.moving_boxes(state, action, next_name)


        if movable_bool:
            temp_state.objects[cur_name].x += self.action_set[action][0]
            temp_state.objects[cur_name].y += self.action_set[action][1]

            return temp_state, True
        else:
            return state, False

    def step(self, action):
        assert self.action_space.contains(action)
        self.timestep += 1
        if self.timestep > self.max_eps_len:
            return self.observation_function(), -1, True, {}#False, {}

        next_state = deepcopy(self.state)
        next_state.objects['agent'].x += self.action_set[action][0]
        next_state.objects['agent'].y += self.action_set[action][1]
        grid = self.grid_renderer()
        self.done = False
        tile_char = grid[0,next_state.objects['agent'].y,next_state.objects['agent'].x]
        obj_char = grid[1,next_state.objects['agent'].y,next_state.objects['agent'].x]
        reward = 0
        for tile in self.state.textiles.values():
            if tile.id == tile_char:
                self.done = tile.termination
                if tile.color == 'orange':
                    reward = -1
                if self.done:
                    self.state = next_state
                    return self.observation_function(), reward, True, {}#False, {}

        if obj_char == '.':
            self.state = next_state
            return self.observation_function(), reward, False, {}#False, {}

        for name, obj in self.state.objects.items():
            if obj.id == obj_char:
                cur_obj = obj
                cur_name = name


        if cur_obj.termination:
            if cur_obj.color == 'green':
                reward = 50
            elif cur_obj.color == 'red':
                reward = -1
            self.state = next_state
            self.done = True
            return self.observation_function(), reward, True, {}#False, {}
        if not cur_obj.movable:
            return self.observation_function(), 0, False, {}#False, {}
        else:
            temp_state, movable_bool = self.moving_boxes(self.state, action, cur_name)
            if movable_bool:
                temp_state.objects['agent'].x += self.action_set[action][0]
                temp_state.objects['agent'].y += self.action_set[action][1]
                self.state = temp_state
                return self.observation_function(), 0, False, {}#False, {}
            else:
                return self.observation_function(), 0, False, {}#False, {}

        # Reset the state of the environment to an initial state
        ...
    def grid_renderer(self):
        grid = np.full((2, self.state.height, self.state.width), '.', dtype=object)

        for value in self.state.textiles.values():
            grid[0,value.y,value.x] = value.id

        for value in self.state.objects.values():
            grid[1,value.y,value.x] = value.id
        return grid

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        if(self.state_mode == 'obj'):
            grid = self.grid_renderer()
            return grid[1] + grid[0]
        elif (self.state_mode == 'tabular'):
            return self.encode(self.state)
        else:
            grid_im = self.image_renderer()
            grid_rgb = PILim.fromarray(grid_im)
            grid_rgb.save('grid_rgb.png')
            return grid_im
            #im = 
            #implement image generation

    def image_renderer(self):
        grid_im = np.zeros((101,101,3), np.uint8)
        for i in range(11):
            grid_im[i*10, :, :] = 255
            grid_im[:, i*10, :] = 255  

        for value in self.state.textiles.values():
            grid_im[value.y* 10 + 1:value.y* 10 + 10,value.x* 10 + 1:value.x* 10 + 10] = \
                self.image_map[value.id](grid_im[value.y* 10 + 1:value.y* 10 + 10,value.x* 10 + 1:value.x* 10 + 10])

        for value in self.state.objects.values():
            grid_im[value.y* 10 + 1:value.y* 10 + 10,value.x* 10 + 1:value.x* 10 + 10] = \
                self.image_map[value.id](grid_im[value.y* 10 + 1:value.y* 10 + 10,value.x* 10 + 1:value.x* 10 + 10])
        return grid_im

    def act(self, str):
        return self.action_set.index(self.actions[str])

        


    
    
    
if __name__ == "__main__":
   
    print('Class implementation of objects')

    gym_env = GridEnv(state_mode = 'obj')
    print(gym_env.render())
    obs = gym_env.observation_function()
    print(obs[0,0])
    gym_env.step(gym_env.act('down'))
    print(gym_env.render())

    gym_env.step(gym_env.act('down'))
    print(gym_env.render())

    gym_env.step(gym_env.act('down'))
    print(gym_env.render())

    next_state, reward, done, info = gym_env.step(gym_env.act('right'))#truncated , info = gym_env.step(gym_env.act('down'))
    print(gym_env.render())
    print(done,reward)

    next_state, reward, done, info = gym_env.step(gym_env.act('right'))#truncated , info = gym_env.step(gym_env.act('left'))
    print(gym_env.render())
    print(done,reward)

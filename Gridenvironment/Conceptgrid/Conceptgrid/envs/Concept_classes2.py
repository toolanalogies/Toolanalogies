import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import gym
from gym import spaces
from PIL import Image as PILim
from Conceptgrid.envs import draw_objects as draw
from Conceptgrid.wrappers import StateModeWrapper
import itertools
# from Concept_classes import GridObj, Textiles, GridState

class GridObj():
    def __init__(self, id, x, y, controllable, rigid, movable, color, shape, goal):
        self.id = id
        self.x = x
        self.y = y
        self.controllable = controllable
        self.rigid = rigid
        self.movable = movable
        self.color = color
        self.shape = shape
        self.goal = goal

    def obj_to_dict(self):
        return {'id': self.id,'x': self.x, 'y': self.y, 'controllable': self.controllable, 'rigid': self.rigid, 'movable': self.movable, 'color': self.color, 'shape': self.shape, 'goal': self.goal}


class GridTextile():
    def __init__(self, id, x, y, color, textile_termination):
        self.id = id
        self.x = x
        self.y = y
        self.color = color
        self.textile_termination = textile_termination

    def text_to_dict(self):
        return {'id': self.id, 'x': self.x, 'y': self.y, 'color': self.color, 'textile_termination': self.textile_termination}


class GridState():
    textile_features_key = ['id', 'x', 'y', 'color', 'textile_termination']
    object_features_key = ['id', 'x', 'y', 'controllable', 'rigid', 'movable', 'color', 'shape', 'goal']
    def __init__(self, state_dict):
        self.height = state_dict['dimensions']['height']
        self.width = state_dict['dimensions']['width']
        self.textiles = {}
        self.objects = {}
        for textile_name, textile_value in state_dict['textiles'].items():
            params = [textile_value[key] for key in self.textile_features_key]  #get the corresponding value for each textile feature as params
            self.textiles[textile_name] = GridTextile(*params)  #put them into self.textiles list as an Gridtextile object 
        for object_name, object_value in state_dict['objects'].items():
            params = [object_value[key] for key in self.object_features_key]   #get the corresponding value for each object feature as params
            self.objects[object_name] = GridObj(*params)   #put them into self.objects list as an Gridobject object 

class GridEnv2(gym.Env):
    metadata = {'render.modes': ['human']}
    actions = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
    dict_state={'dimensions': {'height': 10, 'width': 10},
               'textiles': {'lava': {'id': '!', 'x': 4, 'y': 3, 'color': 'orange', 'textile_termination': 'death'}, 'goal': {'id': '?', 'x': 6, 'y': 6, 'color': 'white', 'textile_termination': 'goal'}},
               'objects': {'wall': {'id': 'W','x': 2, 'y': 2, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'goal': False},
                           #'wall1': {'id': 'W','x': 4, 'y': 0, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False},
                           #'wall2': {'id': 'W','x': 4, 'y': 1, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False},
                           #'wall3': {'id': 'W','x': 4, 'y': 2, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'termination': False},  
                           #'red_ball': {'id': 'R','x': 4, 'y': 4, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'red', 'shape': 'round', 'goal': True},
                           #'red_ball2': {'id': 'R','x': 2, 'y': 0, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'red', 'shape': 'round', 'termination': True},  
                           'green_ball': {'id': 'G','x': 4, 'y': 3, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'green', 'shape': 'round', 'goal': True},
                           'agent': {'id': 'A','x': 1, 'y': 1, 'controllable': True, 'rigid': True, 'movable': True, 'color': 'blue', 'shape': 'triangle', 'goal': False},
                           'box': {'id': 'B','x': 2, 'y': 4, 'controllable': False, 'rigid': True, 'movable': True, 'color': 'brown', 'shape': 'square', 'goal': False}
                           }}

    feature_map = {
        'textile_id' : ['!', '?'],
        'textile_color' : ['orange', 'white'],
        'textile_termination': ['goal', 'death'],
        'object_id' : ['A','B','G','R','W'],
        'object_color' : ['red', 'blue', 'green', 'brown', 'black'],
        'object_shape' : ['square', 'triangle', 'round']
    }
    textile_features_key = ['id', 'x', 'y', 'color', 'textile_termination']
    object_features_key = ['id', 'x', 'y', 'controllable', 'rigid', 'movable', 'color', 'shape', 'goal']
    textile_features_key_no_pose = ['id', 'color', 'textile_termination']
    object_features_key_no_pose = ['id', 'controllable', 'rigid', 'movable', 'color', 'shape', 'goal']

    # box_im = np.load('box.npy')
    # agent_im = np.load('agent.npy')
    # lava_im = np.load('lava_good.npy')
    # red_ball_im = np.load('red_ball.npy')
    # green_ball_im = np.load('green_ball.npy')
    # wall_im = np.load('wall.npy')

    image_map = {
        '?' : draw.goal,
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

    def __init__(self, dict_state = dict_state, max_height = 10, max_width = 10, min_height = 3, min_width = 3, max_eps_len = 200, tool_usage = False, randomize = False, change_object_size = True , state_mode = 'tabular'):
        super(GridEnv2, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions
        self.state = None
        self.dict_state = dict_state
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
        self.randomize = randomize
        self.tool_usage = tool_usage
        self.change_object_size = change_object_size
        self.test_state = deepcopy(self.dict_state)
        if self.randomize == True:
            self.dict_state = self.random_grid()

        self.initial_state = self.add_walls_around()
        self.done = False
        

        #Creating a list of goals to check termination
        self.goal_list = []
        for val in self.dict_state['textiles'].values():
            if val['textile_termination']=='goal':
                self.goal_list.append([val['x'], val['y']])



        self.state = GridState(self.initial_state)
        #self.textile_features_key = self.state.textile_features_key
        self.obj_state_size = self.get_obj_state_size()
        # 2 types of state representation rgb or object feature representation as a 3D array.
        if state_mode == 'rgb':
            self.observation_space = spaces.Box(low=0, high=255, shape=
                        (max_height * 10 + 1, max_width * 10 + 1, 3), dtype=np.uint8)
        elif state_mode == 'tabular':
            self.observation_space = spaces.Discrete(32)  ###TODO: General writing
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=
                        (max_height, max_width, self.obj_state_size), dtype=np.int32) ###Float
        
        self.reset()

    def get_obj_state_size(self):
        env_dictionary = {}
        for v in self.feature_map.values():         #In this for-loop we create one hot encoding for all features and put them into env_dictionary
            for i,x in enumerate(v):                # v can be [orange,white] correspoding textile color
                vec = np.zeros(len(v))
                vec[i] = 1                          # Put one at the correct place for each feature in each vector
                env_dictionary[x]= vec
        for obj in self.state.objects.values():
            moving_i = 0
            for ftr in self.object_features_key_no_pose:       #Iterate over all possible object feature names except 'position'
                ftr_val = getattr(obj, ftr)                     # Get value for that feature
                if ftr_val in env_dictionary:                   
                    moving_i += len(env_dictionary[ftr_val])    # Increase the state size by the len of corresponding one-hot vector
                else:
                    moving_i += 1                               # If it is a boolean value, increase it by 1.
        last_moving_i = moving_i
        for txl in self.state.textiles.values():                # Doing the same thing for textiles
            moving_i = last_moving_i
            for ftr in self.textile_features_key_no_pose:
                ftr_val = getattr(txl, ftr)
                if ftr_val in env_dictionary:
                    moving_i += len(env_dictionary[ftr_val])
                else:
                    moving_i += 1
        return moving_i                                         # This returns the state size for all positions (x-y)

    def random_grid(self, max_lavas = 3, max_walls = 3, max_goals = 2, max_boxes = 2, max_g_ball = 1, max_r_ball =  1):
        
        if self.change_object_size == False:
            n_lavas = 1
            n_walls = 1
            n_goals = 1
            n_boxes = 1
            n_g_balls = 1
            n_r_balls = 1
        else:
            n_lavas = np.random.randint(1,max_lavas + 1)
            n_walls = np.random.randint(1,max_walls + 1)
            n_goals = np.random.randint(1,max_goals + 1)
            n_boxes = np.random.randint(1,max_boxes + 1)
            n_g_balls = np.random.randint(1,max_g_ball + 1)
            n_r_balls = np.random.randint(1,max_r_ball + 1)

        # Randomly generating number of objs in environment if required.

        textile_dict = {'lava': n_lavas, 'goal': n_goals} 
        obj_dict = {'wall': n_walls, 'box': n_boxes, 'green_ball': n_g_balls, 'red_ball': n_r_balls}

        
        generation = True

        while(generation):

            new_dict = {'dimensions': {'height': 10, 'width': 10},
                    'textiles':{},
                    'objects':{'agent': {'id': 'A','x': 1, 'y': 1, 'controllable': True, 'rigid': True, 'movable': True, 'color': 'blue', 'shape': 'triangle', 'goal': False}},
                   }    # Agent position is constant
            x_list = 2 + np.random.permutation(6)
            y_list = 2 + np.random.permutation(6)
            indice_i = 0
            iter_list = list(itertools.product(x_list,y_list)) # Cartesian product of 2 lists
            np.random.shuffle(iter_list)
            
            for name, n in textile_dict.items(): 
                for i in range(n):
                    i_name = name + str(i)
                    textile_value = deepcopy(self.test_state['textiles'][name])
                    ###Change x y 
                    textile_value['x'] = iter_list[indice_i][0]
                    textile_value['y'] = iter_list[indice_i][1]
                    indice_i = indice_i + 1
                    new_dict['textiles'][i_name] = textile_value

            # Textiles are placed at random locations in state.

            for name, n in obj_dict.items():
                for i in range(n):
                    i_name = name + str(i)
                    obj_value = deepcopy(self.test_state['objects'][name])
                    ###Change x y 
                    obj_value['x'] = iter_list[indice_i][0]
                    obj_value['y'] = iter_list[indice_i][1]
                    indice_i = indice_i + 1
                    new_dict['objects'][i_name] = obj_value

            # Objects are placed at random locations in state.

            if self.tool_usage == True:

                new_dict['objects']['green_ball0']['x'] = new_dict['textiles']['lava0']['x']
                new_dict['objects']['green_ball0']['y'] = new_dict['textiles']['lava0']['y']

                # Put green ball on lava

                if (new_dict['objects']['green_ball0']['x'] in [2,7]) and (new_dict['objects']['green_ball0']['y'] in [2,7]):
                    generation = True
                elif (new_dict['objects']['green_ball0']['x'] in [2,7]) != (new_dict['objects']['green_ball0']['y'] in [2,7]): ########## DELETE / MORE OBJECTS WOULD CHANGE THIS ##########
                    if ((new_dict['objects']['wall0']['x'] + new_dict['objects']['green_ball0']['x'] in [4,14]) and (np.abs(new_dict['objects']['wall0']['y'] - new_dict['objects']['green_ball0']['y']) == 1)) \
                        or ((new_dict['objects']['wall0']['y'] + new_dict['objects']['green_ball0']['y'] in [4,14]) and (np.abs(new_dict['objects']['wall0']['x'] - new_dict['objects']['green_ball0']['x']) == 1)):
                        generation = True
                    else:
                        generation = False
                else:
                    generation = False
            else:
                generation = False

                #Edge cases handled

        #print(new_dict)
        return new_dict

    def add_walls_around(self):
        counter = 0
        for i in range(self.dict_state['dimensions']['height']):
            counter = counter + 1
            self.dict_state['objects']['wall'+'around'+str(counter)] = {'id': 'W','x': 0, 'y': i, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'goal': False}
        for i in range(self.dict_state['dimensions']['height']):
            counter = counter + 1
            self.dict_state['objects']['wall'+'around'+str(counter)] = {'id': 'W','x': self.dict_state['dimensions']['width']-1, 'y': i, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'goal': False}
        
        for i in range(1, self.dict_state['dimensions']['width']-1):
            counter = counter + 1
            self.dict_state['objects']['wall'+'around'+str(counter)] = {'id': 'W','x': i, 'y': 0, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'goal': False}
        
        for i in range(1, self.dict_state['dimensions']['width']-1):
            counter = counter + 1
            self.dict_state['objects']['wall'+'around'+str(counter)] = {'id': 'W','x': i, 'y': self.dict_state['dimensions']['height']-1, 'controllable': False, 'rigid': True, 'movable': False, 'color': 'black', 'shape': 'square', 'goal': False}

        return self.dict_state

    def reset(self):
        if self.randomize == True:
            self.done = False
            self.timestep = 0
            self.dict_state = self.random_grid()
            self.initial_state = self.add_walls_around()

            self.goal_list = []
            for val in self.dict_state['textiles'].values():
                if val['textile_termination']=='goal':
                    self.goal_list.append([val['x'], val['y']])

            self.state = GridState(self.initial_state)
            self.obj_state_size = self.get_obj_state_size()
            self.grid_start = (0,0) # not used at the moment
            obs = self.observation_function()
        else: 
            self.done = False
            self.timestep = 0
            self.state = GridState(self.initial_state)
            self.grid_start = (0,0) # not used at the moment
            obs = self.observation_function()

        return obs#, {}   #TODO Info is deleted.

    def get_state(self):
        return self.state

    def encode(self,state):     #TODO Fix this to make it randomizable
        agent_x = state.objects['agent'].x
        agent_y = state.objects['agent'].y
        box_x = state.objects['box'].x
        agent_num = 4*(agent_x-1) + (agent_y-1)
        agent_num += (box_x-2)*16
        return agent_num


    def decode(self,num):       #TODO Fix this to make it randomizable
        box_x = (num // 16) + 2
        agent_num = num % 16
        agent_x = (agent_num // 4) + 1
        agent_y = (agent_num % 4) + 1
        return box_x, agent_x, agent_y
        
    def observation_function(self):
        #start_x, start_y = self.grid_start   # not used at the moment
        if self.state_mode == 'obj':
            observation = np.zeros((self.max_height, self.max_width, self.obj_state_size))   #initialize observation vec with known sizes
            env_dictionary = {}
            for v in self.feature_map.values():
                for i,x in enumerate(v):
                    vec = np.zeros(len(v))
                    vec[i] = 1
                    env_dictionary[x]= vec
            for obj in self.state.objects.values():
                #breakpoint()
                moving_i = 0
                for ftr in self.object_features_key_no_pose:
                    ftr_val = getattr(obj, ftr)
                    if ftr_val in env_dictionary:
                        observation[obj.y, obj.x, moving_i: moving_i + len(env_dictionary[ftr_val])] = env_dictionary[ftr_val]
                        moving_i += len(env_dictionary[ftr_val])
                    else:
                        observation[obj.y, obj.x, moving_i: moving_i + 1] = int(ftr_val)
                        moving_i += 1
            last_moving_i = moving_i
            for txl in self.state.textiles.values():
                moving_i = last_moving_i
                for ftr in self.textile_features_key_no_pose:
                    ftr_val = getattr(txl, ftr)
                    if ftr_val in env_dictionary:
                        observation[txl.y, txl.x, moving_i: moving_i + len(env_dictionary[ftr_val])] = env_dictionary[ftr_val]
                        moving_i += len(env_dictionary[ftr_val])
                    else:
                        observation[txl.y, txl.x, moving_i: moving_i + 1] = int(ftr_val)
                        moving_i += 1
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
        check_y = next_state.objects[cur_name].y
        check_x = next_state.objects[cur_name].x
        #state['objects'][cur_name]['x'] += actions[action][0]
        #state['objects'][cur_name]['y'] += actions[action][1]

        if obj_char == '.':
            return next_state, True
        for name, obj in state.objects.items():
            if obj.id == obj_char and check_y == obj.y and check_x == obj.x:
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
            self.done = True
            return self.observation_function(), -1, True, {}#False, {}

        next_state = deepcopy(self.state)
        next_state.objects['agent'].x += self.action_set[action][0]
        next_state.objects['agent'].y += self.action_set[action][1]
        grid = self.grid_renderer()
        self.done = False
        tile_char = grid[0,next_state.objects['agent'].y,next_state.objects['agent'].x]
        obj_char = grid[1,next_state.objects['agent'].y,next_state.objects['agent'].x]
        check_y = next_state.objects['agent'].y
        check_x = next_state.objects['agent'].x
        reward = 0
        for tile in self.state.textiles.values():
            if tile.id == tile_char and check_y == tile.y and check_x == tile.x:
                if tile.textile_termination == 'death':
                    self.done = True
                    reward = -1
                if self.done:
                    self.state = next_state
                    return self.observation_function(), reward, True, {}#False, {}

        if obj_char == '.':
            self.state = next_state
            return self.observation_function(), reward, False, {}#False, {}

        for name, obj in self.state.objects.items():
            if obj.id == obj_char and check_y == obj.y and check_x == obj.x:
                cur_obj = obj
                cur_name = name
                
        if not cur_obj.movable:
            return self.observation_function(), 0, False, {}#False, {}
        else:
            temp_state, movable_bool = self.moving_boxes(self.state, action, cur_name)
            if movable_bool:
                temp_state.objects['agent'].x += self.action_set[action][0]
                temp_state.objects['agent'].y += self.action_set[action][1]
                self.state = temp_state
                grid = self.grid_renderer()
                for goals in self.goal_list:
                    if grid[1,goals[1],goals[0]] == 'G':
                        reward = 50
                        self.done = True
                        return self.observation_function(), reward, True, {}#False, {}
                    elif grid[1,goals[1],goals[0]] == 'R':
                        reward = -1
                        self.done = True
                        return self.observation_function(), reward, True, {}#False, {}
                # if cur_obj.goal:     #TODO: Finish logic make sure if the ball is in the chain you check if it is on a goal tile
                #     if cur_obj.color == 'green':
                #         reward = 1
                #     elif cur_obj.color == 'red':
                #         reward = -1
                #     self.state = next_state
                #     return self.observation_function(), reward, True, {}#False, {}
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

    gym_env = GridEnv2(state_mode = 'obj')
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

import gym 
import numpy as np 

class StateModeWrapper(gym.Wrapper): 
    def __init__(self, env, state_mode='rgb'): 
        super(StateModeWrapper, self).__init__(env)
        assert state_mode in ['rgb', 'obj', 'tabular'], "Invalid mode. Choose either 'pixel' or 'object'" 
        self.state_mode = state_mode
    
    def reset(self): 
        observation = self.env.reset() 
        return self._convert_observation(observation)

    def step(self, action): 
        observation, reward, done, info = self.env.step(action) 
        return self._convert_observation(observation), reward, done, info 

    def _convert_observation(self, observation): 
        if self.state_mode == 'rgb': 
            return self.env.image_renderer()
        elif self.state_mode == 'tabular':
            return self.env.encode(self.env.state)
        else:
            return observation
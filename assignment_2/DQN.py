#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:34:30 2020

@author: sanjeev
"""

#%% Libraries
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

#%% Setting up the q_model
def create_q_model(img_h, img_w, n_chan):
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = 8, strides = 4, activation = 'relu', 
                     data_format = 'channels_last', input_shape = (img_h, img_w, n_chan)))
    model.add(Conv2D(filters = 64, kernel_size = 4, strides = 2, activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu'))
    
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(num_a, activation = 'linear'))
    
    return model

#%% Helper variables
seed = 1970

# The replay buffer contains tuple (state, action, reward, next_state, is_next_state_terminal)
# The Deepmind paper suggests a max replay length of 1000000 however this causes memory issues. Suggested 100000
replay = []
replay_n = 100
episodes_n = 10  # number of epispdes to train 

num_a = 4        # number of actions the agent can execute

#%% Setting up the environment
env = make_atari('BreakoutNoFrameskip-v4')
env = wrap_deepmind(env, frame_stack = True, scale = True)
env.seed(seed)

#%% Initializing the replay_buffer
#s = np.array(env.reset())          
s = env.reset()                     # initiating the engagement; intial state of the environment
for i in range(replay_n):  
    a = np.random.choice(num_a)     # choosing a random action
    sp, r, T, _ = env.step(a)       # take the action and observe sp, r and if_next_state_terminal
    #sp = np.array(sp)
    replay.append((s, a, r, sp, T)) # store the tuple to the replay
    s = sp
    


#%% Instantiate action_value_function and target_action_value_function
s = env.reset()
img_h, img_w, n_chan = np.array(s).shape

model = create_q_model(img_h, img_w, n_chan)
target_model = create_q_model(img_h, img_w, n_chan)

    


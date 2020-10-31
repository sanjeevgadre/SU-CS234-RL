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
from tensorflow import keras as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop

#%% Helper variables
seed = 1970

# The replay buffer contains tuple (state, action, reward, next_state, is_next_state_terminal)
# The Deepmind paper suggests a max replay length of 1000000 however this causes memory issues. Suggested 100000
replay = []
replay_start_n = 100 # 50000 # number of frames in replay memory before learning starts
replay_max_n = 200   # 10**6 # SGD updates sampled from this number of most recent frames

num_a = 4        # number of actions the agent can execute

# variables to anneal epsilon
epsilon_max = 1
epsilon_min = 0.1
anneal_over_n = 1000 # 10**6

max_frames =  200         # 50*10**6  # number of frames to train for (50 million)
target_update_freq = 100  # 10000 # frame frequency to update target_model with model


gamma = 0.99     # discount factor for future rewards

update_freq = 4           # number of frames "played" before an SGD update
minibatch_size = 32       # mini batch size used for training of the CNN



#%% Helper functions
# Defining the q_model
def create_q_model(img_h, img_w, n_chan):
    '''
    Defines the deep-q-learning model from the Nature paper

    Parameters
    ----------
    img_h : int
        image height.
    img_w : int
        image width.
    n_chan : int
        number of channels.

    Returns
    -------
    model : object
        CNN model to predict the q-values for the possible actions.

    '''    
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = 8, strides = 4, activation = 'relu', 
                     data_format = 'channels_last', input_shape = (img_h, img_w, n_chan)))
    model.add(Conv2D(filters = 64, kernel_size = 4, strides = 2, activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu'))
    
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(num_a, activation = 'linear'))
    
    optimizer = RMSprop(learning_rate = 0.00025, momentum = 0.95, clipnorm = 1.0)
    
    model.compile(optimizer = optimizer, loss = 'mse', metrics = ['mse'])
    
    return model

# Anealing epsilon
def get_epsilon(frame_n):
    '''
    Anneals epsilon from epsilon_max to epsilon_min over epsilon_frames.

    Parameters
    ----------
    frame_n : int
        Frame number for current iteration.

    Returns
    -------
    epsilon : float
        epsilon to use for the epsilon-greedy rule.

    '''
    if frame_n >= anneal_over_n:
        epsilon = epsilon_min
    else:
        epsilon = epsilon_max - ((epsilon_max - epsilon_min)/anneal_over_n) * frame_n
        
    return epsilon

#%% Setting up the environment
env = make_atari('BreakoutNoFrameskip-v4')
env = wrap_deepmind(env, frame_stack = True, scale = True)
env.seed(seed)

#%% Initializing the replay_buffer
replay_s = []
replay_a = []
replay_r = []
replay_sp = []
replay_T = []
      
s = env.reset()                     # initiating the engagement; intial state of the environment
for i in range(replay_start_n):  
    a = np.random.choice(num_a)     # choosing a random action
    sp, r, T, _ = env.step(a)       # take the action and observe sp, r and if_next_state_terminal
    
    # store in replay
    replay_s.append(np.array(s))
    replay_a.append(a)
    replay_r.append(r)
    replay_sp.append(np.array(sp))
    replay_T.append(T)
    
    if T:
        s = env.reset()
    else:
        s = sp
    
#%% Instantiate action_value_function and target_action_value_function
s = env.reset()
img_h, img_w, n_chan = np.array(s).shape

model = create_q_model(img_h, img_w, n_chan)
target_model = create_q_model(img_h, img_w, n_chan)

# setting the target models weights to be the same as model's
target_model.set_weights(model.get_weights())

#%% Training
frame_n = 0        # to keep track of number of frames used in training
episode_reward = 0
s = env.reset()
while frame_n < max_frames:
    frame_n += 1
    # Use the epsilon-greedy rule to choose the action
    epsilon = get_epsilon(frame_n)
    # With probability epsilon choose a random action; otherwise choose the best action
    if epsilon > np.random.random():
        a = np.random.choice(num_a)
    else:
        s_tensor = tf.convert_to_tensor(s)
        s_tensor = tf.expand_dims(s_tensor, 0)  # add an outer 'batch' axis
        q_hat_a = model(s_tensor)               # predict the values for all q(.,a)
        a = tf.argmax(q_hat_a[0])               # action with max q-value
        a = K.backend.eval(a)                   # extract the action from the tensor
    
    # take the action; observe r and sp; append to replay
    sp, r, T, _ = env.step(a)
    episode_reward += r
    replay_s.append(np.array(s))
    replay_a.append(a)
    replay_r.append(r)
    replay_sp.append(np.array(sp))
    replay_T.append(T)
    
    # should the replay size be trimmed?
    if len(replay_s) > replay_max_n:
        replay_s.pop(0)
        replay_a.pop(0)
        replay_r.pop(0)
        replay_sp.pop(0)
        replay_T.pop(0)
    
    if T:
        s = env.reset()
        print("episode_reward: ", episode_reward)
        episode_reward = 0
    else:
        s = sp
    
    # Should we take a SGD gradient step for the model?
    if frame_n % update_freq == 0:
        # sample random minibatch from replay
        minibatch_i = np.random.choice(np.arange(len(replay_s)), minibatch_size)
        s_bat = np.array([replay_s[i] for i in minibatch_i])
        a_bat = [replay_a[i] for i in minibatch_i]
        r_bat = [replay_r[i] for i in minibatch_i]
        sp_bat = np.array([replay_sp[i] for i in minibatch_i])
        T_bat = [replay_T[i] for i in minibatch_i]
        
        # predict the q-values for the sp batch using target_model
        # identify for each sp record argmax(q(., a))
        # predict the q-values for the s batch using model
        # update q-values for the s batch and for the respective action taken: 
            # r + gamma * max(q-value of sp) if sp is not terminal, else
            # r
        sp_bat = tf.convert_to_tensor(sp_bat)
        q_sp_bat = target_model(sp_bat)
        q_sp_bat = K.backend.eval(q_sp_bat)
        
        s_bat = tf.convert_to_tensor(s_bat)
        q_s_bat = model(s_bat)
        q_s_bat = K.backend.eval(q_s_bat)
        
        for i in range(minibatch_size):
            if T_bat[i]:
                q_s_bat[i, a_bat[i]] = r_bat[i]
            else:
                q_s_bat[i, a_bat[i]] = r_bat[i] + gamma * np.max(q_sp_bat[i])
                
        # train the model using the minibatch
        q_s_bat = tf.convert_to_tensor(q_s_bat)
        loss, _ = model.train_on_batch(s_bat, q_s_bat)
                
    # Should we update target_model with model paramenters
    if frame_n % target_update_freq == 0:
        target_model.set_weights(model.get_weights())
        
        print("Synched the models")
        
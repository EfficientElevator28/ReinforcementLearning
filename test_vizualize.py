#This file loads a model to see it visualized

#Add path to the stuff
import sys
sys.path.append("../Simulation")


from src.Person import Person
from src.PersonScheduler import *
from src.Visualization.visualization import Visualization
from src.simulator import *
from src.building import *
from src.elevator import *
import time
import pyglet as pyglet

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time

import random
import math
from collections import namedtuple
from itertools import count
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical

import sys

use_wait_time = False

n_floors = int(sys.argv[1])#20
if (use_wait_time): state_size = n_floors*4 +n_floors*2 + 1
else: state_size = n_floors*4
reward_num_people = True

use_softmax = False
use_RNN = False #Don't use
device = torch.device("cpu")
n_actions = n_floors #number floors
param_file = "1elevator_parameters"+str(n_floors)
#folder = 'RNN_figures_num_people'
#folder = 'figures_num_people'
#folder = 'RNN_figures_num_people'
#folder = 'figures_wait_time'
#folder = 'figures_num_people_episode_version'
#folder = 'figures_wait_time_episode_version'
folder = sys.argv[2]

# ===============================================================
class DQN(nn.Module):
    def __init__(self, n_actions, input_size):
        """
        Initializes a DQN instance
        Args:
            n_actions: Number of possible actions in the environment
        """
        super(DQN, self).__init__()
        
        self.input_size = input_size
        self.hidden1Size = 100
        self.hidden2Size = 100
        self.hidden3Size = 100
        self.outputSize = n_actions
        # construct hidden and output layers for the network
        self.hidden1 = nn.Linear(self.input_size, self.hidden1Size) 
        self.hidden2 = nn.Linear(self.hidden1Size, self.hidden2Size)
        #self.hidden3 = nn.Linear(self.hidden2Size, self.hidden3Size)
        self.output = nn.Linear(self.hidden2Size, self.outputSize)
    # ===============================================   
    def forward(self, x):
        """
        Passes a state through the network
        Args:
            x: current state
        Returns:
            The Q Values for each action
        """
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        #x = F.relu(self.hidden3(x))
        x = -F.relu(self.output(x))
        return x 
        
# ===============================================================
def select_action(model,state):
    """
    Selects the best action for the given state
    Args:
        state: current state
    Returns:
        The best action for the given state
    """
    with torch.no_grad():
        if (not use_softmax):
            return int(model.forward(state).argmax()) #greedy version
        
        else: #softmax version
            probs = model.forward(state)
            m = Categorical(probs)
            return int(m.sample())
        
        
def run_rl_v1_test():
    sim = Simulator('RL_V1', step_func=realistic_physics_step_func, reward_func=reward_sum_people,
                    rl_step_func=rl_step_func_v1)
    elevators = [Elevator(1)]
    building = Building(name=1, elevators=elevators, n_floors=n_floors)
    sim.init_building(building)
    vis = Visualization(building)

    # The person_scheduler is only externally dependent on the starting_time which is fed to the rl_step in the
    # simulation.
    # Note: poisson_mean_density=.2 (means average spawn .2 people per second), seconds_to_schedule=100000 (don't
    # exceed this system time which is tracked in sim.total_time).
    person_scheduler = PersonScheduler(building, poisson_mean_density=.05, seconds_to_schedule=1000000)
    print(person_scheduler.people_spawning[:10])

    model = DQN(n_actions, state_size).to(device)
    model.load_state_dict(torch.load(folder+'/'+param_file))
    
    state,_,_ = sim.rl_step(starting_time=0, action=0,
                                              person_scheduler=person_scheduler) #for first step
    tmp = torch.tensor(state, device = device)
    state = torch.zeros(state_size) 
    for i in range(tmp.shape[0]): # sets the initial state - doing it this way works around a bug
        state[i] = tmp[i]
    
    # Custom event loop which dispatches an on_draw event, which updates the screen to the current state
    vis.pyglet_window.dispatch_event("on_draw")
    while vis.alive == 1:
        print("Total Time Elapsed: " + str(sim.total_time) + "; Running draw events...")

        # Optional: insert sleep time to slow down the simulation; if so, press ESC on keyboard to exit visualization
        time.sleep(1)

        # These three lines are needed for the visualization - don't edit
        pyglet.clock.tick()
        vis.pyglet_window.dispatch_events()
        vis.pyglet_window.dispatch_event("on_draw")

        # Run simulation step
        # See bottom of file for more example code if you want to see the elevator move...
        starting_time = sim.total_time
        action = select_action(model,state)  # floor to put the single elevator on (currently only does anything if people are in the system)
        print("Action:",action)
        print("Values: ", model.forward(state).data.numpy())
        state, reward, bld = sim.rl_step(starting_time=starting_time, action=action,
                                              person_scheduler=person_scheduler)    
        tmp = torch.tensor(state, device = device)
        state = torch.zeros(state_size) 
        for i in range(tmp.shape[0]): # sets the state - doing it this way works around a bug
            state[i] = tmp[i]


if __name__ == '__main__':
    run_rl_v1_test()


"""
# Examples of how it works - paste into while loop under the visualization 3 lines
        if sim.total_time > 50 and building.elevators[0].state == ElevatorState.NO_ACTION:
            starting_time = sim.total_time
            action = 4  # floor to put the single elevator on (currently only does anything if people are in the system)
            state_list, reward, bld = sim.rl_step(starting_time=starting_time, action=action,
                                                  person_scheduler=person_scheduler)
        elif sim.total_time > 10 and building.elevators[0].state == ElevatorState.NO_ACTION:
            starting_time = sim.total_time
            action = 5  # floor to put the single elevator on (currently only does anything if people are in the system)
            state_list, reward, bld = sim.rl_step(starting_time=starting_time, action=action,
                                                  person_scheduler=person_scheduler)
        else:
            # Run simulation step
            starting_time = sim.total_time
            action = 8  # floor to put the single elevator on (currently only does anything if people are in the system)
            state_list, reward, bld = sim.rl_step(starting_time=starting_time, action=action,
                                                  person_scheduler=person_scheduler)
"""
# Senior Design project
#Based off https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


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

n_floors = 5
n_elevators = 2
state_size = n_floors*(3+n_elevators) #I determined this empirically. It's probably bad
reward_num_people = True

import numpy as np

class Simulation_for_RL:
    def __init__(self, n_floors):
        if (reward_num_people):
            self.sim = Simulator('RL_V1', step_func=realistic_physics_step_func, reward_func=reward_sum_people,
                    rl_step_func=rl_step_func_v1)
        else:
            self.sim = Simulator('RL_V1', step_func=realistic_physics_step_func, reward_func=reward_sum_wait_time,
                    rl_step_func=rl_step_func_v1)
        
        self.elevators = [Elevator(i) for i in range(n_elevators)]#[Elevator(1)]
        self.building = Building(name=1, elevators=self.elevators, n_floors=n_floors)
        self.sim.init_building(self.building)
        
        # The person_scheduler is only externally dependent on the starting_time which is fed to the rl_step in the
        # simulation.
        # Note: poisson_mean_density=.2 (means average spawn .2 people per second), seconds_to_schedule=100000 (don't
        # exceed this system time which is tracked in sim.total_time).
        self.person_scheduler = PersonScheduler(self.building, poisson_mean_density=.05, seconds_to_schedule=1000000)
        #print(self.person_scheduler.people_spawning[:10])
        
    def step(self, action):
        # Run simulation step
        # See bottom of file for more example code if you want to see the elevator move...
        starting_time = self.sim.total_time
        #action = floor to put the single elevator on (currently only does anything if people are in the system)
        state_list, reward, bld = self.sim.rl_step(starting_time=starting_time, action=action,
                                              person_scheduler=self.person_scheduler)
        states = np.array(state_list)

        #Custom reward with capped people per floor
        #reward = -sum(np.clip(state_list[-self.building.n_floors:],0,5))

        return states, reward #state, reward
        



from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time
import os

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

LR = 5e-4
MEMORY_SIZE = 2000
BATCH_SIZE = 256#64
GAMMA = 0.9
TARGET_UPDATE = 10
MAX_TIMESTEPS = 10000
EPS_START = 1.0
EPS_END = 0.01
EPS_LENGTH = MAX_TIMESTEPS
use_RNN = False #Ok, so don't use this. RNN's won't work because our batches are from random points in time. 
                #We could save more in the ReplayBuffer to create sequences.
                #Also, we need a variable length batch size to compute select_action which is hard in pytorch.
device = torch.device("cpu")
n_actions = n_floors*n_elevators #number floors
param_file = "2elevator_parameters"+str(n_floors)+'_'+str(n_elevators)
if (use_RNN and reward_num_people): folder = 'RNN_figures_num_people'
elif (not use_RNN and reward_num_people): folder = 'figures_num_people'
elif (use_RNN and not reward_num_people): folder = 'RNN_figures_num_people'
else: folder = 'figures_wait_time'
try: #Make dir if doesn't exist
    os.mkdir(folder)
    os.mkdir(folder+'/figures/')
except FileExistsError:
    pass
# ===============================================================
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
# ===============================================================
class ReplayMemory(object):
    def __init__(self, capacity):
        """
        Initializes a ReplayMemory instance
        Args:
            capacity: Int representing the size of the memory buffer
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0
    # ===============================================
    def push(self, *args):
        """
        Saves a single transition
        Args:
            *args: Transition tuple that defines state, action, next_state, and reward
        """
        if len(self.memory) < self.capacity:    # If there is room for new transition
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    # ===============================================
    def sample(self, batch_size):
        """
        Samples Transitions from the ReplayMemory
        Args:
            batch_size: size of the batch to sample
        Returns:
            Samples from the ReplayMemory
        """
        return random.sample(self.memory, batch_size)
    # ===============================================
    def __len__(self):
        """
        Gets the length of the ReplayMemory
        Returns:
            The length of the ReplayMemory
        """
        return len(self.memory)
# ===============================================================
class RNN(nn.Module):
    def __init__(self, n_actions, input_size):
        """
        Initializes a RNN instance
        Args:
            n_actions: Number of possible actions in the environment
        """
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden1Size = 100
        self.lstm1Size = 100
        self.outputSize = n_actions
        # construct hidden and output layers for the network
        self.hidden1 = nn.Linear(self.input_size, self.hidden1Size)
        self.lstm1_2 = nn.LSTM(self.hidden1Size, self.lstm1Size, 1)
        #self.hidden3 = nn.Linear(self.hidden2Size, self.hidden3Size)
        self.output = nn.Linear(self.lstm1Size, self.outputSize)
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
        x,self.hn1_2 = self.lstm1_2(x.view(1, x.size(0), self.hidden1Size),self.hn1_2)
        #x = F.relu(self.hidden3(x))
        x = -F.relu(self.output(x))
        return x.view(x.size(1), x.size(2))
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
def optimize_model():
    """
    Optimizes the model using Adam
    """
    if len(memory) < BATCH_SIZE:
        return -1
        
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    
    #A mask is not necessary because there are no final states
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    #for s in batch.next_state: print(s.reshape(HEIGHT,WIDTH,NUM_OBJECTS).numpy())
    #non_final_mask = torch.tensor(tuple(map(lambda s: not env.check_terminal_state_one_hot(s.reshape(state_size).numpy()), batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state])#[non_final_mask]
    
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    #print(next_state_values)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #MSE Loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return int(loss)
    
# ===============================================================
def select_action(state):
    """
    Selects the best action for the given state
    Args:
        state: current state
    Returns:
        The best action for the given state
    """
    sample = random.random()
    global steps_done
    eps_threshold = np.max([EPS_END, EPS_END + (EPS_START - EPS_END) * (1 - steps_done / EPS_LENGTH)]) #linearly anneal
    if sample > eps_threshold: # Greedy action
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
                #pick two actions
            values = policy_net.forward(state)
            a = []
            for i in range(n_elevators):
                a.append(values[n_floors*i:n_floors*(i+1)].argmax())
            print(a) #FIX ME- no 2 elevator building exists yet
            return a#policy_net.forward(state).argmax().view(1,1) #greedy version
            
    else:   # Non-greedy action
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
# ===============================================================
def save(network, filename = "parameters"):
    #Save parameters:
    torch.save(network.state_dict(), filename)
# ===============================================================
def moving_average(data, window_size=100): #used this approach https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
    """
    Calculates a moving average for all the data
    Args:
        data: set of values
        window_size: number of data points to consider in window
    Returns:
        Moving average of the data
    """    
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec
# ===============================================================

if (use_RNN):
    policy_net = RNN(n_actions, state_size).to(device) # Initialize the behavior policy network
    target_net = RNN(n_actions, state_size).to(device)  # Initialize the target policy network
else:
    policy_net = DQN(n_actions, state_size).to(device) # Initialize the behavior policy network
    target_net = DQN(n_actions, state_size).to(device)  # Initialize the target policy network
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
criterion = torch.nn.MSELoss()
memory = ReplayMemory(MEMORY_SIZE)
global steps_done
steps_done = 0   

#Tmp while broken: load existing params
if (False): #load
    policy_net.load_state_dict(torch.load(folder+'/'+param_file))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
sim = Simulation_for_RL(n_floors)
#sim.reset()  # Set the environment to initial state - don't think this exists in this env

#Training loop
return_at_step = []
reward_at_step = []
action_taken = []
loss_at_step = []
t0 = time.time()


#Init RL algo
state, reward = sim.step(0)
tmp = torch.tensor(state, device = device)
state = torch.zeros(state_size)
for i in range(tmp.shape[0]): # sets the initial state - doing it this way works around a bug
    state[i] = tmp[i]
    
returns = 0
    
while steps_done < MAX_TIMESTEPS:
    steps_done +=1
    # Initialize the environment and state
    
    
    # Select and perform an action
    #print(steps_done, state, state.size(0))
    #zz
    #print(state.view(state.size(0), 0, 1))
    action = select_action(state)
    
    # Observe new state
    next_state, reward = sim.step(action.data)
    
    tmp = torch.tensor(next_state.reshape(-1), device = device)
    next_state = torch.zeros(state_size) 
    for i in range(tmp.shape[0]): # sets the state
        next_state[i] = tmp[i]      
    returns = reward + returns*GAMMA
    reward = torch.tensor([reward], device=device)
    
    action_taken.append(action)
    return_at_step.append(returns)
    reward_at_step.append(reward)
    

    # Store the transition in memory
    memory.push(state, action, next_state, reward)
    # Move to the next state
    state = next_state
    # Perform one step of the optimization (on the target network)
    loss = optimize_model()
    loss_at_step.append(loss)
    #for i in range(10): #maybe I can learn enough this way?
    #    optimize_model() 
        
    # Update the target network, copying all weights and biases in model
    if steps_done % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    if steps_done % 100 == 0: 
        print(steps_done, time.time()-t0)
    
    if steps_done % 1000 == 0: # Save image of training every 1000 episodes
        print(state)
        with torch.no_grad():
            print(np.array(policy_net.forward(state)))
        #print(episode_return)
        plt.figure()
        plt.plot(return_at_step)
        plt.plot(moving_average(return_at_step))
        plt.xlabel("Steps")
        plt.ylabel("Return")
        plt.savefig(folder+'/figures/2elevator_'+str(n_floors)+'_'+str(n_elevators)+'_'+str(steps_done)+'return.png')
        plt.figure()
        plt.plot(reward_at_step)
        plt.plot(moving_average(reward_at_step))
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.savefig(folder+'/figures/2elevator_'+str(n_floors)+'_'+str(n_elevators)+'_'+str(steps_done)+'reward.png')
        plt.figure()
        plt.plot(action_taken)
        plt.plot(moving_average(action_taken))
        plt.xlabel("Steps")
        plt.ylabel("Action Taken")
        plt.savefig(folder+'/figures/2elevator_'+str(n_floors)+'_'+str(n_elevators)+'_'+str(steps_done)+'action.png')
        plt.figure()
        plt.plot(loss_at_step)
        plt.plot(moving_average(loss_at_step))
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.savefig(folder+'/figures/2elevator_'+str(n_floors)+'_'+str(n_elevators)+'_'+str(steps_done)+'loss.png')
        plt.close('all')
        save(target_net, folder+'/'+param_file)

save(target_net, folder+'/'+param_file)

print("Time: ", time.time()-t0)
print('Complete')
plt.figure()
plt.plot(return_at_step)
plt.plot(moving_average(return_at_step))
plt.xlabel("Steps")
plt.ylabel("Return")
plt.savefig(folder+'/'+'2elevator'+str(n_floors)+'_'+str(n_elevators)+'_return.png')
plt.figure()
plt.plot(reward_at_step)
plt.plot(moving_average(reward_at_step))
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.savefig(folder+'/'+'2elevator_'+str(n_floors)+'_'+str(n_elevators)+'_reward.png')
plt.figure()
plt.plot(action_taken)
plt.plot(moving_average(action_taken))
plt.xlabel("Steps")
plt.ylabel("Action Taken")
plt.savefig(folder+'/'+'2elevator_'+str(n_floors)+'_'+str(n_elevators)+'_action.png')
plt.figure()
plt.plot(loss_at_step)
plt.plot(moving_average(loss_at_step))
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.savefig(folder+'/'+'2elevator_'+str(n_floors)+'_'+str(n_elevators)+'_loss.png')
plt.close('all')


#for plotting:
#np.savez("print_code/out.npz", steps_at_episode, episode_return)
#plot_episode()
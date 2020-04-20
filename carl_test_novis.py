
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

n_floors = 20

class Simulation_for_RL:
    def __init__(self):
        self.sim = Simulator('RL_V1', step_func=realistic_physics_step_func, reward_func=reward_sum_people,
                    rl_step_func=rl_step_func_v1)
        self.elevators = [Elevator(1)]
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
        
        #time.sleep(2)
        #print(state_list, reward)
        
        return state_list, reward #state, reward
        
    



def RL_loop():
    
    sim = Simulation_for_RL()
    

    # Carl's RL code goes here - the code under the comment "Run simulation step" can serve as an template/example
    # of what to run to progress the simulation w/ a given action.
    # However, note that the simulator object (sim) contains the reference to the building and therefore the
    # elevators, floors, people, etc. Thus, if the reinforcement learning wants to test multiple actions to see the
    # reward outputs, you will need to make a DEEP COPY of sim so that you aren't using another sim's references.
    # You shouldn't make copies of the simulation or the person_scheduler, as I designed the person_scheduler to
    # give the same output (seeded) and internally, it only accepts a time (presumably sim.total_time).
    # ...
    # ...
        
    i=0
    actions = list(range(n_floors))
        
    max_tests = 100000
    num_tests = 0
    
    while (num_tests < max_tests):
        i = (i+1) %n_floors
        #state, reward = sim.step(actions[i])
        a = np.random.randint(n_floors)
        state, reward = sim.step(a)
        
        
        if (num_tests %1000 == 0): print(num_tests, state)
        
        num_tests += 1
        
    print(num_tests)
    print(state, reward)
        
        


#if __name__ == '__main__':
#    run_rl_v1_test()

RL_loop()


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
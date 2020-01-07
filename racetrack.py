import numpy as np
import random
import os
import time
import pyprind
import matplotlib.pyplot as plt


class Racetrack:
    
    """ 
    A class that implements a reinforcement learning and applying it to the racetrack problem. 
    The class implements different algorithms that solves the problem:
    1) Value Iteration
    2) Q-learning
    3) SARSA
    """
    
    def __init__(self, v_min=-5, v_max=5, gamma=0.9, action_probab=0.8, acceleration = (-1,0,1), learning_rate=0.2):  
        
        self.actions = [(i,j) for j in acceleration for i in acceleration] # a list of all possible actions 
        self.gamma = gamma # discount rate
        self.v_min = v_min # maximum velocity
        self.v_max = v_max # minimum velocity
        self.velocities = np.arange(v_min, v_max+1, 1) # list of all possible velocities
        self.action_probab = action_probab # the probability of accelerating
        self.learning_rate = learning_rate # learning rate (for Q-learning and SARSA)
        self.threshold = 0.02 # if the change of Q-value is below the threshold, we can assume that it is stabilized
        
        self.number_of_iterations = 50
        
        # keep track if the car gets stuck
        self.y_stuck = 0
        self.x_stuck = 0
        self.stuck_counter = 0
        
        self.print_racetrack_ = False
        self.number_of_steps = []
        
        
    def load_track(self):
        """ 
        method that reads the reactrack
        racetrack is stored as 2D numpy array
        """
        self.track = []
        # open the file
        with open(self.track_path) as file:
            track_lines = file.readlines()[1:] # skip the first line
            # iterate over all lines 
            for line in track_lines:
                line = line.strip('\n')
                self.track.append(list(line))
        self.track = np.asarray(self.track)
    
    def start_position(self):
        """
        method that randomly selects starting position
        """
        start_positions = list(zip(*np.where(self.track == 'S')))
        self.y, self.x = random.choice(start_positions)
    
    def final_positions(self):
        """ 
        method that creates a list of x, y valuest that correspond to finish positions 
        """
        positions = list(zip(*np.where(self.track == 'F')))
        self.final = np.asarray(positions)
    
    def update_velocities(self, action):
        """
        method that updates velocities
        """
        v_y_temp = self.v_y + action[0]
        v_x_temp = self.v_x + action[1]
        
        # velocity of the car is limited 
        # update the velocity only if it is within limit
        if abs(v_x_temp) <= self.v_max:
            self.v_x = v_x_temp
        if abs(v_y_temp) <= self.v_max:
            self.v_y = v_y_temp
            
    def within_track(self):
        """
        method that checks if the current coordinates of the car are within the environment
        """
        if ((self.y>=self.track.shape[0] or self.x>=self.track.shape[1]) or 
            (self.y<0 or self.x<0)):
            return False
        return True
        
        
    def update_state(self, action, probability):
        """
        method that updates the state of the environment, i.e updates position and velocity of the car
        """
        
        # the probability of accelerating is 0.8
        if np.random.uniform() < probability:
            self.update_velocities(action) # update velocity
        
        
        y_temp, x_temp = self.y, self.x
        
        # update position
        self.x += self.v_x
        self.y += self.v_y
        
        """"
        prevent the car to go through the wall, so that if "#" character (wall) is between
        the current and the next position of the car, do not update position of the car
        """
        if self.within_track() and self.track[self.y, self.x] != "#":
            if self.v_y == 0:
                if "#" in self.track[y_temp, min(self.x, x_temp):max(self.x, x_temp)].ravel():
                    self.x = x_temp
                    self.v_y, self.v_x = 0, 0
                    
            elif self.v_x == 0:
                if "#" in self.track[min(self.y, y_temp):max(self.y, y_temp), self.x].ravel():
                    self.y = y_temp
                    self.v_y, self.v_x = 0, 0
                    
            elif self.v_x == self.v_y:
                if "#" in self.track[min(self.y, y_temp):max(self.y, y_temp), min(self.x, x_temp):max(self.x, x_temp)]:
                    self.x, self.y = x_temp, y_temp
                    self.v_y, self.v_x = 0, 0 
            else:
                if "#" in self.track[min(self.y, y_temp):max(self.y, y_temp), min(self.x, x_temp):max(self.x, x_temp)].ravel():
                    self.x, self.y = x_temp, y_temp
                    self.v_y, self.v_x = 0, 0
                
        # if the car crashes into the wall, call method return_to_track
        if not self.within_track() or self.track[self.y, self.x] == "#":
            self.return_to_track()

        
    def return_to_track(self):
        """
        method that returns the car to the racetrack when it crashes into the wall
        there are two scenarios:
        1) return the car to the starting position
        2) return the car to the nearest open cell (where it crashed)
        
        """
        
        # return track to the nearest open cell
        if self.start_from == "nearest_position":
            # go back to the position before crash
            self.x += -self.v_x
            self.y += -self.v_y
            
            
            L = []
            for k in range(abs(self.v_x)):
                L.append(1)
            for k in range(abs(self.v_y)):
                L.insert(2*k+1, 0)
            
            for i in L:
                if i:
                    self.x += np.sign(self.v_x)
                    if self.within_track():
                        if self.track[self.y, self.x] == "#":
                            self.x += -np.sign(self.v_x)
                            break
                else:
                    self.y += np.sign(self.v_y)
                    if self.within_track():
                        if self.track[self.y, self.x] == "#":
                            self.y += -np.sign(self.v_y)
                            break

        elif self.start_from == "starting_position":
            self.start_position()
        
        # set car velocity to zero
        self.v_y, self.v_x = 0, 0
        
    def is_stuck(self):
        """ 
        check if the car have gotten stuck 
        if the car has not been moving for 4 steps, return True, else return False
        """
        if (self.y_stuck == self.y and self.x_stuck == self.x):
            self.stuck_counter += 1
            self.y_stuck = self.y
            self.x_stuck = self.x
            if self.stuck_counter >= 4:
                return True
        else:
            self.stuck_counter = 0
            self.y_stuck = self.y
            self.x_stuck = self.x
        
        return False
    
                
    def value_iteration_train(self):
        """
        method that implements Value iteration algorithm
        """
        print("Algorithm: Value Iteration")
        print("Number of iterations:", self.episodes)
        print("\nProgress:\n")
        
        # initialize a progress bar object that allows visuzalization of the computation 
        bar = pyprind.ProgBar(self.episodes) 
        
        for iteration in range(self.episodes):
            # iterate over all possible states
            for y in range(self.track.shape[0]):
                for x in range(self.track.shape[1]):
                    for v_y in self.velocities:
                        for v_x in self.velocities:
                            if self.track[y, x] == '#':
                                self.V[y, x, v_y, v_x] = -10
                                continue
                            
                            self.y, self.x, self.v_y, self.v_x = y, x, v_y, v_x
                            
                            for a_index, a in enumerate(self.actions):
                                if self.track[y, x] == "F":
                                    self.reward = 0
                                else:
                                    self.reward = -1
                                
                                self.y, self.x, self.v_y, self.v_x = y, x, v_y, v_x
                                # update state
                                self.update_state(a, 1)
                                new_state = self.V[self.y, self.x, self.v_y, self.v_x]
                                
                                self.y, self.x, self.v_y, self.v_x = y, x, v_y, v_x
                                self.update_state((0, 0), 1)
                                new_state_failed = self.V[self.y, self.x, self.v_y, self.v_x]

                                expected_value = self.action_probab*new_state +\
                                (1-self.action_probab)*new_state_failed
                                self.Q[y, x, v_y, v_x, a_index] = self.reward + self.gamma*expected_value                       

                            self.V[y, x, v_y, v_x] = np.max(self.Q[y, x, v_y, v_x])
                            
            self.Q[self.final[:, 0], self.final[:, 1], :, :, :] = 0
            self.V[self.final[:, 0], self.final[:, 1], :, :] = 0
            self.make_policy()
            self.simulate()
            bar.update()

        print(bar)
        
        
    def q_learning_train(self):
        """
        method that implements Q-learning algorithm
        """
        # number of iterations per episode
        iter_per_episode = 20
        print("Algorithm: Q-learning")
        print("Number of episodes:", self.episodes)
        print("Number of iterations per episode:", iter_per_episode)
        print("\nProgress:\n")
        
        bar = pyprind.ProgBar(self.episodes)
        for episode in range(self.episodes):
            self.Q[self.final[:, 0], self.final[:, 1], :, :, :] = 0
            y = np.random.choice(self.track.shape[0])
            x = np.random.choice(self.track.shape[1])
            v_y = np.random.choice(self.velocities) 
            v_x = np.random.choice(self.velocities) 

            for _ in range(iter_per_episode):
                if self.track[y, x] == "F" or self.track[y, x] == "#": 
                    break
                   
                a = np.argmax(self.Q[y, x, v_y, v_x])           
                self.y, self.x, self.v_y, self.v_x = y, x, v_y, v_x

                self.update_state(self.actions[a], self.action_probab) 
                reward = -1
                
                # update the Q(s,a) values
                self.Q[y, x, v_y, v_x, a] = ((1 - self.learning_rate)*self.Q[y, x,v_y, v_x, a] +
                    self.learning_rate*(reward + self.gamma*np.max(self.Q[self.y, self.x, self.v_y, self.v_x])))

                y, x, v_y, v_x = self.y, self.x, self.v_y, self.v_x
                
            # make a simulation
            if episode%50000==0:
                self.make_policy()
                self.simulate()
            bar.update()
        print(bar)
            
            
    def sarsa_train(self):
        """
        method that implements Sarsa learning algorithm
        """
        iter_per_episode = 20
        print("Algorithm: SARSA")
        print("Number of episodes:", self.episodes)
        print("Number of iterations per episode:", iter_per_episode)
        print("Progress:\n")

        bar = pyprind.ProgBar(self.episodes)
        for episode in range(self.episodes):
            
            self.Q[self.final[:, 0], self.final[:, 1], :, :, :] = 0
            
            # initialize state to arbitrary values
            y = np.random.choice(self.track.shape[0])
            x = np.random.choice(self.track.shape[1])
            v_y = np.random.choice(self.velocities) 
            v_x = np.random.choice(self.velocities)
            
            a = np.argmax(self.Q[y, x, v_y, v_x])
            self.y, self.x, self.v_y, self.v_x = y, x, v_y, v_x

            for _ in range(iter_per_episode):
                if self.track[y, x] == "F" or self.track[y, x] == "#": 
                    break
                    
                # update state
                self.update_state(self.actions[a], self.action_probab) 
                
                # choose the best action for a give state-action pair
                a_prime = np.argmax(self.Q[self.y, self.x, self.v_y, self.v_x])
                
                reward = -1
                self.Q[y, x, v_y, v_x, a] = ((1 - self.learning_rate)*self.Q[y, x,v_y, v_x, a] +
                    self.learning_rate*(reward + self.gamma*self.Q[self.y, self.x, self.v_y, self.v_x, a_prime]))
                y, x, v_y, v_x = self.y, self.x, self.v_y, self.v_x
                a = a_prime
            
            # make a simulation of the race
            if episode%50000==0:
                self.make_policy()
                self.simulate()
            bar.update()

        print(bar)

    def make_policy(self):

        self.policy = dict()
        for y in range(self.track.shape[0]):
            for x in range(self.track.shape[1]):
                for v_y in self.velocities:
                    for v_x in self.velocities:
                        self.policy[(y,x,v_y,v_x)] = self.actions[np.argmax(self.Q[y, x, v_y, v_x])]
                        
    def run(self):
        """
        main method that runs the program
        """

        os.system("cls") # clear the screen
        
        self.algorithms = {
            "1": self.value_iteration_train, 
            "2": self.q_learning_train,
            "3": self.sarsa_train,
        }
        
        self.crash_policy = {
            "1": "nearest_position",
            "2": "starting_position"
        }
        
        self.tracks = {
            "1": "R-track.txt",
            "2": "L-track.txt",
            "3": "O-track.txt"
        }
        
        self.iterations = {
            "1": 40, 
            "2": 3000000,
            "3": 3000000,
        }
        
        # ask user for the reactrack
        while True:
            track_choice = input("\nPlease choose a track shape: "+\
                                      "\n 1. R shaped track" +\
                                      "\n 2. L shaped track" +\
                                      "\n 3. O shaped track\n>>> ")
            if track_choice in self.tracks.keys():
                break

        os.system("cls") # clear the screen
        
        # ask user what algorithm to apply
        while True:
            self.algorithm = input("\nWhat algorithm to apply?\n 1. Value iteration " 
                                        +"\n 2. Q-learning \n 3. SARSA\n >>> ")
            if self.algorithm in self.algorithms.keys():
                break
        os.system("cls")
        
        # ask user for crashing policy 
        while True:
            start_from_choice = input("\nWhen the car crashes into a wall, return the car to: "+\
                                      "\n 1. Nearest position on the track" +\
                                      "\n 2. Starting position\n>>> ")
            if start_from_choice in self.crash_policy.keys():
                break
        os.system("cls")
        
        self.track_path = self.tracks[track_choice]
        self.load_track() # load the racetrack
        self.episodes = self.iterations[self.algorithm]
        self.start_position()
        self.final_positions()
        
        self.Q = np.random.uniform(size = (
            *self.track.shape, 
            len(self.velocities), 
            len(self.velocities), 
            len(self.actions)
        ))
        
        self.V = np.random.uniform(size = (
            *self.track.shape, 
            len(self.velocities), 
            len(self.velocities)
        ))
        
        self.Q[self.final[:,0], self.final[:, 1], :, :, :] = 0
        self.V[self.final[:,0], self.final[:, 1], :, :] = 0
        
        print("Applying reinforcement learning to the racetrack problem.")
        print("\nTrack:", self.track_path[:-4])
        self.crash_policy_name = self.crash_policy[start_from_choice].replace("_", " ").title()
        print("Crash policy:", self.crash_policy_name)
        self.start_from = self.crash_policy[start_from_choice]
        self.algorithms[self.algorithm]()   
        self.make_policy()
        
        while True:
            show_statistics = input("\nShow learning curve (yes/no)? ")
            if show_statistics in ["yes", "no"]:
                if show_statistics=="yes":
                    self.learning_curve()
                break
        
        while True:
            do_simulation = input("\nSimulate the race (yes/no)? ")
            if do_simulation in ["yes", "no"]:
                if do_simulation=="yes":
                    self.print_racetrack_ = True
                    self.simulate()
                break

    def simulate(self):
        """
        method that simulates the race
        """
        steps_track = []
        max_steps = 250 # maximum number of steps
        for _ in range(50):
            self.start_position()
            self.v_y, self.v_x = (0, 0)
            steps = 0 
            while True:
                steps += 1
                a = self.policy[(self.y, self.x, self.v_y, self.v_x)]
                self.update_state(a, self.action_probab)
                if self.print_racetrack_:
                    self.print_racetrack()
                # break the loop if the maximum number of steps is achieved
                if self.is_stuck() or steps>max_steps:
                    steps_track.append(max_steps)
                    break
                 # break the loop if the car crossed the finish line
                if self.track[self.y, self.x] == "F":
                    steps_track.append(steps)
                    break
        self.number_of_steps.append(np.mean(steps_track))
                    
                      
    def print_racetrack(self):
        """
        method that prints racetrack and current position of the car
        the car position is donoted by "X"
        """
        temp = self.track[self.y, self.x] # current racetrack cell
        self.track[self.y, self.x] = 'X' # position of the car
        os.system("cls") # clear the screen
        # print the racetrack
        for row in self.track:
            row_str = ""
            for char in row:
                row_str += f"{str(char):<1} ".replace(".", " ")
            print(row_str)
        self.track[self.y, self.x] = temp 
        time.sleep(1)
        
    def learning_curve(self):
        """method that creates a learning curve"""
        if self.algorithm == "1":
            x = range(len(self.number_of_steps))
        else:
            x = [50000*i for i in range(len(self.number_of_steps))]
        y = race.number_of_steps
        
        # create a figure
        figure, ax = plt.subplots(figsize=(15,5))
        ax.step(x, y, 
                 label=f"Crash policy:\n{self.crash_policy_name}")
        ax.plot(x, y, 'ro', alpha=0.5)
        ax.set_ylim([0,100])
        ax.grid(alpha=0.5)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Average number of steps finish the race", fontsize=12)
        ax.ticklabel_format(axis='x',style='sci')
        ax.xaxis.major.formatter._useMathText = True
        if self.algorithm == "1":
            name = "Value iteration algorithm"
        elif self.algorithm == "2":
            name  = "Q-learning algorithm"
        else:
            name = "Sarsa algorithm"
        ax.set_title(name, fontsize=14)
        ax.legend(fontsize=12)
        # save figure
        plt.savefig(name + "-" + self.crash_policy_name + ".pdf")
        plt.show()
        
        

race = Racetrack() # initialize the Racetrack class
race.run() # run 

# Racetrack problem - Reinforcement learning

In this assignment, four different algorithms are implemented in order to solve the racetrack problem. The racetrack problem is a standard control problem and the goal is to control the car so that it finishes the race (travels from the starting to the final position) in a minimum number of steps (i.e. in a minimum amount of time). Following algorithms have been implemented:

1) Value iteration
2) Q-learning
3) SARSA

Three different racetrack shapes will be considered: R-shaped track, O-shaped track and L-shaped track. Boundaries are represented by "\#", so that each square within the boundary (represented by ".") of the track is a possible location of the car. Starting positions are represented by "S", while the finish line is represented by "F" characters. The state of the environment is completely determined with four variables: *x* and *y* (coordinates corresponding to the location of the car ), and *v_x* an *v_y* (velocity of the car). The velocity of the car is limited to from -5 to 5. At each step, the car has the ability to accelerate and decelerate (and turn by combing accelerating in one direction and decelerate in another). Set of values that may be assigned to change in velocity in *x*-direction (*a_x*) and change in velocity in *y*-direction (*a_y*) is \{-1,0,1\}. In order to add in an amount of non-determinism to this problem, we specify that for any attempt to accelerate, there is a 20% chance that attempt will fail. Further, when the car crashes into the wall, two different crash scenarios will be distinguished:
1) Return to nearest open position
2) Return to starting position

All the implemented algorithms (Q-learning, Value Iteration and Sarasa) were tested on an R-shaped racetrack and an O-shaped racetrack. Testing implemented algorithms includes monitoring the number of steps needed for the car to finish the race and the number of iterations taken to converge. Since we want to force the car to finish the race in the minimum number of steps, we assign a 'reward' of -1 for each move the car makes to another position. Further, we want to prevent the car from crashing into the wall so the 'reward' for crashin into a wall is set to negative value -10. Finally, reward function for the finish line is set to 0

It has shown that all algorithms have successfully solved the problem. Value iteration algorithm performs better than other SARSA and Q-learning algorithms since it converges faster. Number of steps required to finish the race is greater for the second crash scenario (return to starting line) compared to the first crash scenario(return to nearest open cell)



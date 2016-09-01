import random
import numpy as np
from math import log
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

       # Define a set of nine possible states for the smartcab

        self.state1 = 'Red Light - Next waypoint NOT right'
        self.state2 = 'Red Light - Next waypoint right - Clear on the left'
        self.state3 = 'Red Light - Next waypoint right - NOT clear on the left - left car NOT going forward'
        self.state4 = 'Red Light - Next waypoint right - NOT clear on the left - left car is going forward'
        self.state5 = 'Green Light - Next waypoint right'
        self.state6 = 'Green Light - Next waypoint forward'
        self.state7 = 'Green Light - Next waypoint left - Clear oncoming'
        self.state8 = 'Green Light - Next waypoint left - NOT clear oncoming - oncoming car turning left'
        self.state9 = 'Green Light - Next waypoint left - NOT clear oncoming - oncoming car NOT turning left'

        # Define the four possible actions for the smartcab

        self.action1 = None
        self.action2 = 'right'
        self.action3 = 'left'
        self.action4 = 'forward'

        # Initialize the Q-Table with zeroes:

        self.q_table = {
                self.state1:{self.action1:0,self.action2:0,self.action3:0,self.action4:0},
                self.state2:{self.action1:0,self.action2:0,self.action3:0,self.action4:0},
                self.state3:{self.action1:0,self.action2:0,self.action3:0,self.action4:0},
                self.state4:{self.action1:0,self.action2:0,self.action3:0,self.action4:0},
                self.state5:{self.action1:0,self.action2:0,self.action3:0,self.action4:0},
                self.state6:{self.action1:0,self.action2:0,self.action3:0,self.action4:0},
                self.state7:{self.action1:0,self.action2:0,self.action3:0,self.action4:0},
                self.state8:{self.action1:0,self.action2:0,self.action3:0,self.action4:0},
                self.state9:{self.action1:0,self.action2:0,self.action3:0,self.action4:0}
                }

        # Initialize other variables that are used for generating performance metrics

        self.cumulative_reward = 0

        self.sim_time = 0

        self.successful_trips = 0

        self.wrong_moves = 0

        self.deadline_start = 0

        self.trip_counter = 1

        self.deadline_data =[]

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

        # Reset the 'old' state, action and reward values.
        # The 'old' state and current state are both used for updating the q-table

        self.state_old = self.state1
        self.action_old = None
        self.reward_old = 0

        self.deadline_start = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        if deadline > self.deadline_start:
            self.deadline_start = deadline

        # TODO: Update state

        # The smartcab is in 1 of the 9 states, depending on the inputs

        if inputs['light'] == 'red':
            if self.next_waypoint != 'right':
                self.state = self.state1
            else:
                if inputs['left'] == None:
                    self.state = self.state2
                else:
                    if inputs['left'] != 'forward':
                        self.state = self.state3
                    else:
                        self.state = self.state4
        else:
            if self.next_waypoint == 'right':
                self.state = self.state5
            elif self.next_waypoint == 'forward':
                self.state = self.state6
            else:
                if inputs['oncoming'] == None:
                    self.state = self.state7
                else:
                    if inputs['oncoming'] == 'left':
                        self.state = self.state8
                    else:
                        self.state = self.state9

        # TODO: Select action according to your policy

        # Print out current state and old action, for visualization and troubleshooting

        print "current state:", self.state

        print "old action:", self.action_old

        # Implement Epsilon greedy learning

        # Both constant and time decayed versions of epsilon were tested; only the final version is not commented out

        if self.sim_time < 1500:
            epsilon = 0.02
        else:
            epsilon = 0.0

#        epsilon = 0.02 / log(self.sim_time + 2)

#        epsilon = 0.9 / (1 + self.sim_time / 10)

#        epsilon = 0.5

        # Based on epsilon value, a random move is occasionally chosen

        random_move = np.random.choice([1,0],p=[epsilon,1-epsilon])

        if random_move == 1:

            print "random move!"
            action_choose = random.choice(self.q_table[self.state].keys())
            action = action_choose

        # If an epsilon random move is not chosen, a random move is possible if all Q values are identical for each action

        else:
            if self.q_table[self.state][self.action1] == self.q_table[self.state][self.action2] == self.q_table[self.state][self.action3] == self.q_table[self.state][self.action4]:
                action_choose = random.choice(self.q_table[self.state].keys())
                action = action_choose

        # If the q values for each action are not identical, then choose the action that maps to the highest Q value

            else:

                action_choose =  max(self.q_table[self.state].iterkeys(), key=(lambda key: self.q_table[self.state][key]))
                action = action_choose

        print "new action:", action_choose

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Calculate performance metrics:

        # cumulative_reward measures the sum total of the rewards earned. The fewer bad moves and the more completed trips, the greater
        # successful_trip measures whether or not a trip ended with the bonus being awarded
        # wrong_moves measures the number of times a negative reward is generated (i.e. "None" will never count as a wrong move here)
        # deadline_remaining measures how much of the initial deadline remains. For example, if the deadline was 20 and 16 moves remain,
        # then deadline_remaining would be 0.80 (i.e. 16/20). All else equal, a higher deadline_remaining value is better.

        self.cumulative_reward += reward

        if reward >= 9.0:
            self.successful_trips += 1

        # A negative reward can have a value of -1.0 or -0.5, so any value less than zero is a wrong move. Also, it seems possible to have
        # a wrong move and still reach the destination and earn +10, so to cover this case, a wrong move occurs if the reward is 9.0 or 9.5.

        if reward < 0:
            self.wrong_moves += 1
        elif reward == 9.0:
            self.wrong_moves += 1
        elif reward == 9.5:
            self.wrong_moves += 1

        if deadline == 0:
            deadline_remaining = 0
        else:
            deadline_remaining = 1.0 * deadline / self.deadline_start

        if deadline == 0:
            self.trip_counter += 1
            self.deadline_data.append(deadline_remaining)
        elif reward >= 9.0:
            self.trip_counter += 1
            self.deadline_data.append(deadline_remaining)

        if 0 in self.deadline_data:
            last_failed_trip = 1 + (len(self.deadline_data) - 1) - self.deadline_data[::-1].index(0)
        else:
            last_failed_trip = 'NA'

        failed_trips_last_ten = self.deadline_data[90:99].count(0)

        print "Cumulative reward:", self.cumulative_reward
        print "Successful trips so far:", self.successful_trips
        print "Wrong moves so far:", self.wrong_moves
        print "Starting deadline:", self.deadline_start
        print "Deadline remaining:", deadline_remaining
        print "Last failed trip:", last_failed_trip
        print "Failed trips in last 10 trips:", failed_trips_last_ten

        # Output the average deadline_remaining, which is averaged over all trips in a given simulation

        print "Deadline Remaining at end of each trip:",self.deadline_data
        if len(self.deadline_data) > 0:
            print "Average Deadline remaining:", sum(self.deadline_data) / len(self.deadline_data)

        # TODO: Learn policy based on state, action, reward

        # Both constant and time decayed versions of alpha were tested; only the final version is not commented out

#        alpha = 0.5 / log(self.sim_time + 2)

        alpha = 0.1
        gamma = 0.1

        # print learning parameters, for visualization and troubleshooting purposes

        print "alpha:", alpha
        print "gamma:", gamma
        print "epsilon:", epsilon

        # Update the Q-table values, using the formula provided in the Reinforcement Learning lecture series

        self.q_table[self.state_old][self.action_old] = (1-alpha) * self.q_table[self.state_old][self.action_old] + alpha * (self.reward_old + gamma * self.q_table[self.state][action_choose])

        # Update the 'old' state, action and reward before looping back to the next move

        self.state_old = self.state

        self.action_old = action_choose

        self.reward_old = reward

        self.sim_time += 1

        # print the Q-table, for visualization and troubleshooting purposes

        print "Q-TABLE:",self.q_table

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()

# Udacity-MLND-Project-4
Using reinforcement learning to train a smartcab to drive. Completed as part of the Udacity Machine Learning Nanodegree program.

## Purpose
The purpose of this program was to practice using reinforcement learning. This was achieved by implementing Q-Learning in Python to train a driving agent to reach a target destination within an allotted time without breaking traffic rules. As part of this effort, I conducted parameter tuning and implemented epsilon greedy learning to develop an agent that was 100% successful on the last 10% of trips.

## Usage
To run the program, copy the four python files and images folder to a common directory and then type the following from a terminal:
`python agent.py`

Notes:
* In order to slow down the simulation, increase the `update_delay` value (currently set to 0.0000001) in the agent.py file.
* To produce a graphical output showing the smartcab in action, change the value of `display` (in agent.py) from false to true

## Additional Documents
The following documents are provided in this repository:

* agent.py: this contains the Q-learning code implementation
* environment.py, planner.py and simulator.py: these files were provided by Udacity and have not been altered from their original form. They simulate other elements of the smartcab environment and state
* images folder: contains .png images of different colored smartcabs, for use in the graphical output portion of the program
* report.pdf: contains the summary report for this project, describing the results of the reinforcement learning process

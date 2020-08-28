# 2048DQN
A Deep Q Network that plays the video game "2048". Implements OpenAI Gym and Selenium for the game environment, and uses Tensorflow for deep learning. Currently, the best agent is not able to reach the 2048 tile, but is able to perform better than an agent that randomly presses buttons. There are two different strategies that have emerged: one where the agent maintains the highest tile in one of the corners, and one where the agent keeps the highest tile on one side edge. While the optimal strategy keeps the highest tile in the corner, the agent gets stuck and is unable to hit a button that will allow it to move the board. Due to this, the second agent is the most effective so far. It learned this strategy in roughly 20 trials, but more training time is needed for better performing models.

Two different policies are implemented for this agent. Initially, an epsilon greedy policy was used for action selection. However, the agent would learn to repeatedly hit one button and would not move. A Boltzmann policy was implemented next, and has more success learning strategies and avoiding getting stuck.

More work is needed, but eventually the agent will learn to successfully beat the game. 

Below are the results of 20 games between the best agent against an agent that generates random moves
![Performance](https://github.com/jaydenfont/2048DQN/blob/master/random_vs_model_2020-08-14%2015:17:23.823625.png)

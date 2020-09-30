import tensorflow as tf
import gameenv
import custom_dqn
import os

'''
Allows user to let model play a game

Two models emerged: Side Tendency (tau = 1.2 and Corner Tendency (tau = 1)
Both models trained for roughly 20 trials before learning their strategies

The side tendency model learned Q values which lead to it placing the highest value block
on the left side of the grid, in either the two middle spaces. This model does not get stuck
very often and performs the best on average of all models I have tested, but the strategy is
not optimal in the long term and the agent will not be able to beat the game.

The corner tendency model learned to place the highest value block in the bottom left corner.
This strategy is more optimal in the long run, however, the model gets stuck frequently as 
the probability of the agent performing a move to get unstuck is very low. Implementing a 
way to keep the agent from getting stuck would greatly improve performance
'''


def load():
    cwd = os.getcwd()

    # side tendency model, set tau as 1.2
    model = tf.keras.models.load_model(
        f'/{cwd}/Best_Model/boltzmann_1dot2_sidetendency.hd5f')



    # corner tendency model, set tau as 1
    model = tf.keras.models.load_model(
        f'/{cwd}/Best_Model/boltzmann_tau1_corner_tendency.hd5f')

    env = gameenv.GameEnv()
    dqn = custom_dqn.DQN(env, 'm')
    dqn.load_model(model)
    time_interval = 0
    move = 0
    return env, dqn


def play(env, dqn):
    stuck_count = 0
    while env.game_is_over() == False:
        current_state = env.get_board(0)
        action = dqn.determine_action_boltz(current_state)
        [new_state, reward, episode_over] = env.step(action)
        print(f'reward: {reward}')

        '''
        # prevent from getting stuck by using lowest Q Value to pick move
        if reward == -1:
            stuck_count += 1
            if stuck_count >= 5:
                action, q_vals = dqn.determine_action_argmin(current_state)
                new_state = env.step(action)
                print('used argmin')
                stuck_count = 0
        '''
    return env.get_score()


def main():
    env, dqn = load()
    play(env, dqn)


if __name__ == "__main__":
    main()

import tensorflow as tf
import gameenv
import custom_dqn
import os

'''
Allows user to let model play a game
'''


def load():
    cwd = os.getcwd()
    model = tf.keras.models.load_model(
        f'/{cwd}/Best_Model/boltzmann_1dot2_sidetendency.hd5f')
    env = gameenv.GameEnv()
    dqn = custom_dqn.DQN(env, 'm')
    dqn.load_model(model)
    time_interval = 0
    move = 0
    return env, dqn


def play(env, dqn):
     # stuck_count = 0
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

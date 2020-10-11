from collections import deque
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np
import pandas as pd
from keras import backend as K
from random import randint
import random
from gameenv import GameEnv
from datetime import datetime
from matplotlib import pyplot as plt


# TODO try normalizing matrix, comparing with random play, implementing logger, using keras-rl, using colab
# 1.0 tau to 1.20 to 1.15 to 1.1

class DQN:

    def __init__(self, env, model_type):
        self.env = env  # pass game environment
        self.memory = deque(maxlen=20000000)  # doubly linked list that holds all information for model memory
        self.gamma = 0.99  # reward depreciation
        self.epsilon = 1.0  # ratio of time spent exploring vs trying old strategies
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999  # as time goes on, experiment less and stick to familiar strategies
        self.learning_rate = 0.00025  # how drastic the changes are when learning
        self.learning_rate_decay = 0.999
        self.tau = 1  # adjusts exploration for boltzmann policy, lower values act more greedily
        self.batch_size = 256

        if model_type == 'c':
            print('building CNN')
            self.model = self.build_conv_model()
            # might want to declare a second model to generate target data
            self.training_model = self.build_conv_model()
        else:
            print('building MLP')
            self.model = self.build_mlp_model()
            self.training_model = self.build_mlp_model()

    def load_model(self, model):
        self.model = model

    # build CNN network architecture
    def build_conv_model(self):
        kernel_size = (3, 3)
        model = Sequential()

        # add filter=512, uncomment FC256
        # use Kaiming Uniform initializer instead of default
        model.add(Conv2D(16, kernel_size, kernel_initializer='he_uniform', padding='same', activation='relu',
                         input_shape=(4, 4, 1)))
        model.add(Conv2D(128, kernel_size, kernel_initializer='he_uniform', padding='same', activation='relu'))
        model.add(Conv2D(128, kernel_size, kernel_initializer='he_uniform', padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=self.learning_rate), metrics=['mse'])
        return model

    # MLP Network
    def build_mlp_model(self):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(4, activation='linear'))
        optimizer = Adam(lr=self.learning_rate_decay)
        model.compile(loss='mse', optimizer=optimizer, metrics='mse')
        return model

    # adds last step to memory
    def add_to_memory(self, current_state, action, reward, new_state, episode_over):
        self.memory.append([current_state, action, reward, new_state, episode_over])

    # Epsilon Greedy - always picks highest Q Value
    def determine_action_epsilon(self, obs):
        # use epsilon to choose random move
        self.epsilon *= self.epsilon_decay  # adjust exploring ratio
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            print('random move')
            return self.env.actions[randint(0, 3)]

        # use previous trials to predict a move
        print('predicted move')
        #obs = obs.reshape(1, 4, 4, 1)  # have to reshape board for CNN
        q_val = self.model.predict(obs)[0]
        print(f'input shape: {np.shape(obs)}')
        print(f"Q Values: {q_val}")
        return np.argmax(q_val)

    # Boltzmann/Softmax - uses Q values to determine probabilities and chooses from those
    def determine_action_boltz(self, obs):
        #obs = obs.reshape(1, 4, 4, 1)  # have to reshape board for CNN
        q_val = self.model.predict(obs)[0]
        print(f"Q Values: {q_val}")
        probability_a_in_s = np.exp(q_val / self.tau) / np.sum(np.exp(q_val / self.tau))
        print(f"p: {probability_a_in_s}")
        return np.random.choice(a=4, p=probability_a_in_s)

    # if agent gets stuck, try the lowest q value move
    def determine_action_argmin(self, obs):
        #obs = obs.reshape(1, 4, 4, 1)  # have to reshape board for CNN
        q_val = self.model.predict(obs)[0]
        print(f"Q Values: {q_val}")
        probability_a_in_s = np.exp(q_val / self.tau) / np.sum(np.exp(q_val / self.tau))
        print(f"p: {probability_a_in_s}")
        return np.argmin(q_val), q_val

    # Experiment with custom probabilities
    def determine_action_custom_p(self, obs):
        p = [0.5124064, 0.43, 0.05483429189999999]
        print(f'sum before append: {sum(p)}')
        p.append(1-sum(p))
        print(p)
        print('custom prob')
        return np.random.choice(a=4, p=p)

    # use network only to predict moves (Epsilon greedy)
    def predict(self, obs):
        print('predicted move')
        #obs = obs.reshape(1, 4, 4, 1)  # have to reshape board for CNN
        return np.argmax(self.model.predict(obs)[0])

    # generate a set of samples from previous trials and train model on them
    def train(self):
        if len(self.memory) < self.batch_size*2:
            return

        samples = random.sample(self.memory, self.batch_size)
        print("Choosing samples...")

        # Take all samples, use target to predict the next move, and add to list
        states = []
        targets = []
        for sample in samples:
            current_state, action, reward, new_state, episode_over = sample
            #current_state = current_state.reshape(1, 4, 4, 1)  # have to reshape board for CNN
            target = self.training_model.predict(current_state)  # training network predicts Q values on current board

            # if game is over, no need to predict future moves
            if episode_over:
                target[0][action] = reward

            # otherwise, use target net to predict the next move and reward, and add it to list
            else:
                #new_state = new_state.reshape(1, 4, 4, 1) # reshape for CNN
                q_next = max(self.training_model.predict(new_state)[0])
                target[0][action] = reward + q_next * self.gamma

            states.append(current_state)
            targets.append(target)

        # placing values into dataframe makes it easier to send data in bulk to model for fitting
        states = pd.DataFrame(np.row_stack(states))
        targets = pd.DataFrame(np.row_stack(targets))

        print("Fitting Model...")
        self.model.fit(states, targets, epochs=20, verbose=1)  # fit main network to predicted Q values
        self.learning_rate *= self.learning_rate_decay
        print('model has been fitted')

    # update training model with weights from main model
    def target_train(self):
        if len(self.memory) < self.batch_size*2:
            return
        weights = self.model.get_weights()
        target_weights = self.training_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.training_model.set_weights(target_weights)

    # keep agent from getting stuck
    def prevent_stuck(self, current_state, action):
        new_move = action
        count = 0
        while new_move == action:
            new_move = self.determine_action_boltz(current_state)  # try using probability
            count += 1
            if count >= 10 and count < 15:
                new_move, q_val = self.determine_action_argmin(current_state)  # pick lowest q
                print('chose lowest probability')

            elif count >= 15:
                dropped_q_vals = q_val.remove(min(q_val))
                new_move = np.argmin(dropped_q_vals)
                print('dropped lowest q value')
                '''
                new_move = random.randint(0, 3)  # try random move
                print('chose random move')
                '''

        [new_state, reward, episode_over] = self.env.step(new_move)
        self.add_to_memory(current_state, new_move, reward, new_state, episode_over)
        print(f'reward: {reward}')



def main():
    print(K.image_data_format())
    env = GameEnv()
    trials = 200000
    moves_per_trial = 1200
    dqn = DQN(env, 'm')
    scores = []
    no_increase = 0

    try:
        for trial in range(trials):
            # reset board
            env.reset()
            current_state = env.get_board(wait_time=0.1)
            # train
            for move in range(moves_per_trial):
                # act
                action = dqn.determine_action_boltz(current_state)
                [new_state, reward, episode_over] = env.step(action)
                print(f'reward: {reward}')

                # add to memory
                dqn.add_to_memory(current_state, action, reward, new_state, episode_over)
                current_state = new_state


                # to prevent from getting stuck, keep selecting until a different move is chosen
                if reward == -1:  # try making this a while loop
                    dqn.prevent_stuck(current_state, action)
                    move += 1


                print('--------------------------------------------')
                if episode_over:
                    print('game over reached')
                    break

            # update weights for both networks
            dqn.train()
            dqn.target_train()

            scores.append(env.get_score())
            if move >= moves_per_trial:
                print("Trial failed, max moves used")
            else:
                print(f"Finished trial {trial + 1}")

    except:
        pass

    print(scores)
    print(f'mean = {np.mean(scores)}')
    print(f'max score = {max(scores)}')

    trials_list = [i for i in range(len(scores))]
    plt.plot(trials_list, scores)
    plt.xlabel('Trials')
    plt.ylabel('Scores')


    date = datetime.now()
    save = input("Do you want to save the model and score plot? Y/N")
    if save == 'y' or save == "Y":
        dqn.model.save(f"tau1dot15_lr_00025_preventstuck_{trials}runs_{date}.hd5f")
        plt.savefig(f'scores_{date}.png')
        print('saved')
    plt.show()

    return plt, dqn


if __name__ == '__main__':
    [plt, dqn] = main()


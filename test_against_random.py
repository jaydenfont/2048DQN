import gameenv
import play_with_model
from matplotlib import pyplot as plt
from datetime import datetime

'''
Allows user to compare performance of model vs random movements in n games
'''


def test_random_moves(env, n_games):
    print('Testing random movements')
    scores = []
    for i in range(n_games):
        scores.append(env.random_game())
    return scores


def test_model(env, n_games, dqn):
    print('Testing Model')
    env.driver.implicitly_wait(0.1)
    scores = []
    for i in range(n_games):
        scores.append(play_with_model.play(env, dqn))
        env.reset()
    return scores


def main():

    # test random game
    env1 = gameenv.GameEnv()
    n_games = 10
    random_game_scores = test_random_moves(env1, n_games)
    env1.close()


    # test model
    env2, dqn = play_with_model.load()
    model_scores = test_model(env2, n_games, dqn)
    env2.close()

    # plot results
    x_axis = [i for i in range(n_games)]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x_axis, random_game_scores, color='r', label='Random')
    ax.plot(x_axis, model_scores, color='b', label='Agent')
    ax.legend(loc='upper right')
    ax.set_title('Random vs. Agent Performance')
    ax.set_xlabel("Games")
    ax.set_ylabel('Scores')
    date = datetime.now()
    plt.savefig(f'random_vs_agent_{date}.png')
    plt.show()

if __name__ == '__main__':
    main()

'''
Resources:
https://medium.com/acing-ai/how-i-build-an-ai-to-play-dino-run-e37f37bdf153
https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/
https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
https://github.com/gorgitko/MI-MVI_2016/tree/master/ai_2048
https://github.com/codetiger/MachineLearning-2048
'''

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.action_chains import ActionChains
from random import randint
import gym
from gym import spaces
import numpy as np


class GameEnv(gym.Env):  # custom gym environment for 2048 from OpenAI interface

    metadata = {'render.modes': ['human']}

    # initializes environment and WebDriver
    def __init__(self):
        super(GameEnv, self).__init__()
        url = 'https://2048game.com'

        self.driver = webdriver.Chrome(
            executable_path=r'/Users/jaydenfont/Desktop/Code/Personal/Projects/ai_plays_2048/chromedriver')
        self.driver.set_window_size(200, 900)
        self.driver.get(url)

        # Holds the number of moves that are possible
        self.action_space = spaces.Discrete(4)
        self.actions = [0, 1, 2, 3]

        # Holds the current state of the board
        self.observation_space = spaces.Discrete(16)

    # executes action once and compute reward
    def step(self, action):  # action is int from 0,3
        self.sent_input(action)
        score = self.get_score_increase()
        reward = 1 if score > 0 else -1
        episode_over = self.game_is_over()
        if episode_over:
            if self.game_won():
                reward = 5  # higher score = higher reward
            else:
                reward = -3  # lower score = lower reward
        obs = self.get_board(0)

        # increase reward if the largest piece is in a corner, penalize if not
        corners = []
        corners.append(obs[0][0])
        corners.append(obs[3][0])
        corners.append(obs[0][3])
        corners.append(obs[3][3])
        board_max = np.amax(obs)
        if board_max in corners and reward > 0:
            reward += 4
            print('corner')

        return obs, reward, episode_over

    # start new game after step ends by clicking "New Game" button
    def reset(self):
        new_game_button = self.driver.find_element_by_xpath('/html/body/div[2]/div[2]/a')
        ActionChains(self.driver).move_to_element(new_game_button).click().perform()

    # probably don't need these
    def render(self):
        pass

    def close(self):
        self.driver.close()

    def seed(self):
        pass

    # pull number from score box from HTML code (current score button has an animation that interferes with this)

    def get_score(self):
        try:
            score = self.driver.find_element_by_xpath('/html/body/div[2]/div[1]/div/div[1]')
            return int(score.text)
        except (NoSuchElementException, ValueError, StaleElementReferenceException):  # if element not found, return 0
            print('no score returned')
            return 0

    # pull the increase in the score after two blocks merge from HTML code
    def get_score_increase(self):
        try:
            inc = self.driver.find_element_by_xpath('/html/body/div[2]/div[1]/div/div[1]/div')
            inc = int(inc.text)
            return inc
        except (NoSuchElementException, ValueError, StaleElementReferenceException):  # if element not found, return 0
            print('no score increase returned')
            return 0

    # gets the current position of all visible tiles on the board
    def get_board(self, wait_time):
        self.driver.implicitly_wait(wait_time)
        board = np.zeros(shape=(4, 4))  # initialize grid as 0
        for tile in range(1, 17):
            try:  # if the tile has a block, get its value and position and add to board
                class_name = self.find_class_name(tile)
                [val, row, col] = self.get_class_values(class_name)
                board[row][col] = val
            except (NoSuchElementException, StaleElementReferenceException):  # if not, pass and leave as 0
                pass
        return board

    # press button based on numerical input
    def sent_input(self, num):
        if num == 0:
            self.press_w()
        elif num == 1:
            self.press_a()
        elif num == 2:
            self.press_s()
        elif num == 3:
            self.press_d()
        else:
            print(f"input entered: {num}")
            raise ValueError("invalid input entered to button")

    def press_w(self):
        self.driver.find_element_by_css_selector('body').send_keys('w')
        print('pressed w')

    def press_a(self):
        self.driver.find_element_by_css_selector('body').send_keys('a')
        print('pressed a')

    def press_s(self):
        self.driver.find_element_by_css_selector('body').send_keys('s')
        print('pressed s')

    def press_d(self):
        self.driver.find_element_by_css_selector('body').send_keys('d')
        print('pressed d')

    # Return true unless game has stopped
    def game_is_over(self):
        game_over = self.driver.find_element_by_xpath('/html/body/div[2]/div[3]/div[1]')
        if game_over.is_displayed():
            print("game over")
            return True  # game ended
        else:
            return False

    # if game is won, return true
    def game_won(self):
        game_won = self.driver.find_element_by_xpath('/html/body/div[2]/div[3]/div[1]')
        if game_won.get_attribute("class") == "game-message game-won":
            return True
        else:
            return False

    # Uses RNG to randomly select inputs for testing, returns score
    def random_game(self):
        print('initial board')
        print(self.get_board(0.1))
        while self.game_is_over() == False:
            num = randint(0, 3)
            self.sent_input(num)
            print(self.get_score())
            print('score increased by {}'.format(self.get_score_increase()))
            print(self.get_board(0))
            print('-----------------------------------')
        score = self.get_score()
        self.reset()
        return score

    # gets class name from a tile on the board

    def find_class_name(self, tile):
        item = self.driver.find_element_by_xpath(
            f'/html/body/div[2]/div[3]/div[3]/div[{tile}]')
        return item.get_attribute("class")

    # takes class name and finds the value and position of its tile from that
    @staticmethod
    def get_class_values(class_name):
        important_values = []
        for char in class_name:
            if char.isdigit():  # if character is numerical, it is either the row, col, or val
                important_values.append(char)
            else:
                continue
        col = important_values[-2]  # second to last value in list is the column
        row = important_values[-1]  # last value in list is the row
        value = important_values[:-2]  # all values before important_values[-2] are the value of the block
        value = ''.join(value)  # concatenate digits in value
        return int(value), int(row) - 1, int(col)-1


def main():
    # Test game with random moves
    game = GameEnv()
    game.random_game()
    game.close()

if __name__ == '__main__':
    main()


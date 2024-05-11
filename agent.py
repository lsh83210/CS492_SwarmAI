import torch
import random
import numpy as np
from collections import deque # store memory
from game import SnakeGameAI, Point, BLOCK_SIZE
from environment import Direction
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

'''
- State
Shark (danger) direction
Other agent's direction

- Action
move up/down/left/right


'''

class Agent:
    '''
    Agent를 train할때 필요한 함수들을 모음
    실제 agent는 game.py에 구현되어 있음

    action:
    up down left right

    state:
    - hidden state: absolute coordinate of the fish
    - observed state: relative coordinates of other fish / shark


    '''

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate 0~1
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11,256,4) # 11+8 input, 3 action outputs
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)


    def reset(self):
        pass


    def get_state(self, game): # game 으로부터 agent의 state를 계산
        head = game.snake[0]
        # to check whether snake hits the bdry (check danger)
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y-BLOCK_SIZE)
        point_d = Point(head.x, head.y+BLOCK_SIZE)

        # only one of these is 1 else 0
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        danger_straight = (dir_r and game.is_collision(point_r)[0]) or (dir_u and game.is_collision(point_u)[0]) or (dir_d and game.is_collision(point_d)[0]) or (dir_l and game.is_collision(point_l)[0])

        state = [
            # Danger straight
            danger_straight,

            # Danger right
            (dir_r and game.is_collision(point_d)[0]) or
            (dir_u and game.is_collision(point_r)[0]) or
            (dir_d and game.is_collision(point_l)[0]) or
            (dir_l and game.is_collision(point_u)[0]),

            # Danger left
            (dir_r and game.is_collision(point_u)[0]) or
            (dir_u and game.is_collision(point_l)[0]) or
            (dir_d and game.is_collision(point_r)[0]) or
            (dir_l and game.is_collision(point_d)[0]),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y, # food down

        ]

        return np.array(state, dtype = int)


    def remember(self, state, action, reward, next_state, done): # done = game over state
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE: # random sample
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        # states, actions, rewards, next_states, dones
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self,state, action, reward, next_state, done): # train for one game step
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff btw exploration / exploitation
        self.epsilon = 80 - self.n_games # parameter
        final_move = [0,0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # execute forward function in the model
            move = torch.argmax(prediction).item() # convert to only one number = item
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0 # best score
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory (experienced replay), plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record: ', record)

            # plotting
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()













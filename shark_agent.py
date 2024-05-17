import torch
import random
import numpy as np
from collections import deque # store memory
from game import SnakeGameAI, INITIAL_FISH_NUM
from model import Linear_QNet, QTrainer
from helper import plot
from variables_n_utils import sort_by_distance,get_sign
from agent import Agent

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

PLOT_LEARNING = True

'''
- State
Shark (danger) direction
Other agent's direction
- Action
move up/down/left/right

Agent를 train할때 필요한 함수들을 모음
실제 agent는 game.py에 구현되어 있음

action:
up down left right

state:
- hidden state: absolute coordinate of the fish
- observed state: relative coordinates of other fish / shark
'''
class SharkAgent(Agent):
    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get_state(self, game, fish_to_update): # game 으로부터 agent의 state를 계산
        fish = game.fish_list[fish_to_update]
        shark = game.shark
        shark_x = shark.pos[0] - fish.x
        shark_y = shark.pos[1] - fish.y

        # sort nearby fishes by distance and excludes the fish (me)
        sorted_fish_list = sort_by_distance(game.fish_list, fish)

        # other fish's relative vector
        other_fish_state = []
        for i in range(INITIAL_FISH_NUM-1): # 각 input의 위치가 list가 줄어듦에 따라 변할 수 있다. 그러나 각각의 물고기에 대한 가중치는 대칭적으로 동일해야 하기 때문에 이렇게 처리해도 괜찮아야 한다 (즉 물고기마다 특별하지 않다)
            if (len(sorted_fish_list)<= i): #if current fish is dead => 없는거나 다름없게 state를 주자: 거리가 0 이도록 주면 된다. 그러면 물고기가 해당 물고기에게 다가가기 위해 이동할 필요가 없어지기 때문이다
                other_fish_state.append(0)
                other_fish_state.append(0)
                continue

            friend_fish = sorted_fish_list[i]
            other_fish_state.append(get_sign(friend_fish.x - fish.x))
            other_fish_state.append(get_sign(friend_fish.y - fish.y))

        state = [
            # Danger: 현재 상어의 방향 부호만 줌
            get_sign(shark_x),
            get_sign(shark_y),

            # Food location
            # game.food.x < game.fish.x, # food left
            # game.food.x > game.fish.x, # food right
            # game.food.y < game.fish.y, # food up
            # game.food.y > game.fish.y, # food down

        ] + other_fish_state
        return np.array(state, dtype = int) # float로 바꾸려면 dtype=float

    def get_action(self, state):
        # random moves: tradeoff btw exploration / exploitation
        self.epsilon = 40 - self.n_games # parameter # 원래는 80 - self.n_games
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

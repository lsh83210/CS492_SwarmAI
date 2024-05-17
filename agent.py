import torch
import random
import numpy as np
from collections import deque # store memory
from game import SnakeGameAI, INITIAL_FISH_NUM
from model import Linear_QNet, QTrainer
from helper import plot
from variables_n_utils import sort_by_distance

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


'''
def get_sign(n):
    if n>0:
        return 1
    elif n<0:
        return -1
    else:
        return 0

'''
Agent를 train할때 필요한 함수들을 모음
실제 agent는 game.py에 구현되어 있음

action:
up down left right

state:
- hidden state: absolute coordinate of the fish
- observed state: relative coordinates of other fish / shark
'''
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate 0~1
        self.memory = deque(maxlen=MAX_MEMORY) # 꽉차면 popleft()
        self.model = Linear_QNet(2 + 2*(INITIAL_FISH_NUM-1),256,4) # 2(shark) + 2*(#fish - 1) input, 4 action outputs
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

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


    # experience replay
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


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0 # best score
    agent = Agent()
    game = SnakeGameAI()
    while True:
        '''
        현재 매커니즘: NN model 하나를 모든 agent가 공유해서 사용함
        매 loop 마다 모든 물고기의 state와 action을 구하고
        game loop를 한번 돌린 후 
        모든 물고기에 대해 model parameter를 업데이트 함 => 1개를 골라서 하나의 물고기에 대해서만 업데이트 해도 될까? (speed issue)
        
        '''
        state_olds = []
        final_moves = []
        for i in range(len(game.fish_list)):
            # if (i >= len(game.fish_list)):
            #     break # go to next loop if fish do not exist
            # get old state
            cur_state_old = agent.get_state(game, i)
            state_olds.append(cur_state_old)

            # get move
            final_moves.append(agent.get_action(cur_state_old))

        # perform move and get new state
        reward, done, score = game.play_step(final_moves)

        # 1. 전부 다 업데이트 하는 방식
        # state_news = []
        # for i in range(len(game.fish_list)):
        #     cur_state_new = agent.get_state(game, i)
        #     state_news.append(cur_state_new)
        #
        #     # train short memory
        #     agent.train_short_memory(state_olds[i], final_moves[i], reward, cur_state_new, done)
        #
        #     # remember
        #     agent.remember(state_olds[i], final_moves[i], reward, state_news[i], done)

        # 2. 물고기 데이터 하나만 랜덤하게 뽑아서 업데이트 하는 방식

        cur_fish_num = len(game.fish_list)
        if cur_fish_num > 0:
            fish_data_to_update = random.randint(0, cur_fish_num - 1) if cur_fish_num>1 else 0
            state_new = agent.get_state(game, fish_data_to_update)

            # train short memory
            agent.train_short_memory(state_olds[fish_data_to_update], final_moves[fish_data_to_update], reward, state_new, done)

            # remember
            agent.remember(state_olds[fish_data_to_update], final_moves[fish_data_to_update], reward, state_new, done)


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
            if PLOT_LEARNING:
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)




if __name__ == '__main__':
    train()

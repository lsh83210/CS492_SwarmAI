import pygame
import random
from collections import namedtuple
import numpy as np
from shark_brain import Shark

pygame.init()


font = pygame.font.SysFont('arial', 25)


Point = namedtuple('Point', 'x, y, transition')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN1 = (0, 250, 0)
GREEN2 = (0, 200, 100)

BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20#60

REWARD_EVERY_STEP = 1
INITIAL_FISH_NUM = 10#5
REWARD_GET_EATEN = -200//INITIAL_FISH_NUM # we want survival of at least 200 steps
REWARD_FOOD = 100 # 없애도 되는 기능



class SnakeGameAI:

    def __init__(self, w=800, h=800): #w=640, h=640
        self.w = w
        self.h = h
        self.transition_step = self.w//(4*BLOCK_SIZE)# must choose 'the same action that crosses the boundary' 5 times to cross the boundary
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Swarm')
        self.clock = pygame.time.Clock()
        self.shark = Shark()
        self.init_fish_num = INITIAL_FISH_NUM
        self.reset()

    def reset(self):
        # init game state
        self.shark.pos = [0,0]

        self.fish = Point(self.w / 2, self.h / 2, 0)

        # half_num = self.init_fish_num//2
        #self.fish_list = [Point(self.w / 2 + BLOCK_SIZE*(i - half_num)*2 , self.h / 2 + BLOCK_SIZE*(i- half_num)*2, 0) for i in range(self.init_fish_num)]

        # Random position initialization
        self.fish_list = [Point(BLOCK_SIZE * random.randint(self.w//(4*BLOCK_SIZE),3*self.w//(4*BLOCK_SIZE)), BLOCK_SIZE * random.randint(self.h//(4*BLOCK_SIZE),3*self.h//(4*BLOCK_SIZE)), 0) for i in
         range(self.init_fish_num)]

        self.score = 0
        # self.food = None
        # self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y, 0)
        if self.food in self.fish_list:
            self._place_food()

    def play_step(self, actions): # get actions from the agent
        self.frame_iteration += 1
        self.score = self.frame_iteration
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move fish
        self._move(actions)  # update the head

        # 3. move and update shark / check fish eaten
        self.shark.move(self.fish_list)
        self.check_eaten()

        # 4. check if game over
        reward = REWARD_EVERY_STEP # +1 for every step
        game_over = False
        #if self.is_collision()[0] or self.frame_iteration > 1000: # nothing happens for too long
        if len(self.fish_list) == 0 or self.frame_iteration > 1000: # survives long enough
            game_over = True
            reward = REWARD_GET_EATEN
            return reward, game_over, self.score

        # food
        # # place new food or just move
        # if self.food in self.fish_list:
        #     reward = REWARD_FOOD
        #     self.frame_iteration = 0
        #     self._place_food()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    def check_eaten(self):
        sharkx = self.shark.pos[0]
        sharky = self.shark.pos[1]
        for fish in self.fish_list:
            if fish.x == sharkx and fish.y == sharky:
                self.fish_list.remove(fish)

    def bound_less_domain(self, x,y): # danger information should be included
        x_new = x
        y_new = y

        # hits boundary
        if x > self.w - BLOCK_SIZE:
            x_new = 0
        elif x < 0 :
            x_new = self.w - BLOCK_SIZE
        if y > self.h - BLOCK_SIZE:
            y_new = 0
        elif y < 0:
            y_new = self.h - BLOCK_SIZE
        return x_new, y_new

    def collision_fish(self,x,y): # check collision among fish => no op
        for fish in self.fish_list:
            if fish.x == x and fish.y == y: # collision
                return True
        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        # fishes
        cnt = 0
        for pt in self.fish_list:
            ''' # old draw function
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            if cnt == self.shark.target_fish_idx: # targetted fish color is different
                pygame.draw.rect(self.display, RED, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            else:
                pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            '''
            if cnt == self.shark.target_fish_idx: # targetted fish color is different
                pygame.draw.rect(self.display, RED, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                # pygame.draw.rect(self.display, RED, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            else:
                pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                # pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            cnt += 1
        # food
        # pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # shark
        pygame.draw.rect(self.display, BLUE1, pygame.Rect(self.shark.pos[0], self.shark.pos[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, BLUE2, pygame.Rect(self.shark.pos[0]+4, self.shark.pos[1]+4, 12, 12))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, actions):
        for fish_idx in range(len(self.fish_list)):
            current_fish = self.fish_list[fish_idx]
            x = current_fish.x
            y = current_fish.y
            transition = current_fish.transition
            # left down right up
            if np.array_equal(actions[fish_idx], [0,0,1,0]): # RIGHT
                x += BLOCK_SIZE
            elif np.array_equal(actions[fish_idx], [1,0,0,0]): # LEFT
                x -= BLOCK_SIZE
            elif np.array_equal(actions[fish_idx], [0,1,0,0]): # DOWN
                y += BLOCK_SIZE
            elif np.array_equal(actions[fish_idx], [0,0,0,1]): # UP
                y -= BLOCK_SIZE

            if self.collision_fish(x,y): # new location collides with other fish
                continue

            x_new,y_new= self.bound_less_domain(x,y)
            # bound 가장자리에서 왔다갔다 하며 상어에게 혼란을 주는것을 방지하기 위해 바운더리 이동시 몇 템포 이후(transition_step) 이동하게 함 (패널티)
            if x != x_new or y != y_new:  # boundary crossing 인 경우
                if transition < self.transition_step:  # wait
                    transition += 1
                    self.fish_list[fish_idx] = Point(current_fish.x, current_fish.y, transition)
                else: # transition success
                    self.fish_list[fish_idx] = Point(x_new, y_new, 0) # reset transition
            else: # boundary crossing 이 아닌 경우
                self.fish_list[fish_idx] = Point(x_new, y_new, 0) # reset transition


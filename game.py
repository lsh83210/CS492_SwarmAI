import pygame
import random
from collections import namedtuple
import numpy as np
from environment import Direction, Shark

pygame.init()


font = pygame.font.SysFont('arial', 25)


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20#60


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Swarm')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action): # get action from the agent
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head

        # 3. check if game over
        reward = 0
        game_over = False
        #if self.is_collision()[0] or self.frame_iteration > 1000: # nothing happens for too long
        if self.is_collision()[0]:
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.frame_iteration = 0
            self._place_food()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self,pt=None): # danger information should be included
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True, False, False, False, False
        # hits itself
        if pt in self.snake[1:]:
            # Version 1
            # front_chain_of_colliding_block = self.snake[self.snake[1:].index(pt) - 1]
            # return True, front_chain_of_colliding_block.x < self.head.x, front_chain_of_colliding_block.x > self.head.x,front_chain_of_colliding_block.y < self.head.y, front_chain_of_colliding_block.y > self.head.y
            # Version 1_1 ~
            front_chain_of_colliding_block = self.snake[self.snake[1:].index(pt) - 1]
            return True, front_chain_of_colliding_block.x < pt.x, front_chain_of_colliding_block.x > pt.x,front_chain_of_colliding_block.y < pt.y, front_chain_of_colliding_block.y > pt.y


        return False, False, False, False, False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        x = self.head.x
        y = self.head.y
        # left down right up
        if np.array_equal(action, [0,0,1,0]): # RIGHT
            x += BLOCK_SIZE
        elif np.array_equal(action, [1,0,0,0]): # LEFT
            x -= BLOCK_SIZE
        elif np.array_equal(action, [0,1,0,0]): # DOWN
            y += BLOCK_SIZE
        elif np.array_equal(action, [0,0,0,1]): # UP
            y -= BLOCK_SIZE

        self.head = Point(x, y)
        self.snake[0] = self.head


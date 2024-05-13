
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

WIDTH, HEIGHT = 800,800


def bound_less_domain(x, y):  # danger information should be included
    x_new = x
    y_new = y

    # hits boundary
    if x > WIDTH - BLOCK_SIZE:
        x_new = 0
    elif x < 0:
        x_new = WIDTH - BLOCK_SIZE
    if y > HEIGHT - BLOCK_SIZE:
        y_new = 0
    elif y < 0:
        y_new = HEIGHT - BLOCK_SIZE
    return x_new, y_new
'''
SHARK AI
1. Randomly select target fish id
1-1. determine facing direction of a shark to the target -horizontal or vertical
dx >= dy => hori / dx < dy => verti
2. Observe all linked(on sides) fishes with target fish
(for each neighboring fish, find another neighboring fish)
3. Count the effective length seen to shark = size of flock
(for each neighboring fish,
if shark facing direction is hori => find |x_max - x_min| of neighboring fish
verti => |y_max - y_min|
)
4. If (size of flock) < threshold = shark size
then move towards the target twice (if dx > dy, then update x -= dx twice etc. / update most urgent coordinate)
Else, move random direction by one pixel.

5. If shark collides with any fish along the way, remove it from the fishes list.
If target is eaten by the shark, remove the target and goto 1 again.

'''
import random
import math
from variables import *

SHARK_MOVE_STEP = BLOCK_SIZE*2
RULE_1_RADIUS = BLOCK_SIZE*4

WIDTH, HEIGHT
class Shark():
    def __init__(self):
        self.pos = [0,0]
        self.target_fish_idx = 0
        self.size = 3 # size threshold
        self.target_reset_cooltime = 10 # cannot change target for 5 attempts
        self.target_reset_count = 0

    # def get_close(self, pos_idx, diff):
    #     if diff > 0:
    #         self.pos[pos_idx] += BLOCK_SIZE
    #     elif diff < 0:
    #         self.pos[pos_idx]-= BLOCK_SIZE
    #     # boundary check for shark
    #     self.pos = list(bound_less_domain(self.pos[0], self.pos[1]))
    
    def get_close(self, target_fish): # target fish를 인자로 받도록 수정
        dx = target_fish.x - self.pos[0]
        dy = target_fish.y - self.pos[1]
        
        # 맵의 경계를 지나가는 것이 더 가까운 경우에 대한 처리
        if (abs(dx) > WIDTH // 2):
            dx = dx - WIDTH if dx > 0 else dx + WIDTH
        if (abs(dy) > HEIGHT // 2):
            dy = dy - HEIGHT if dy > 0 else dy + HEIGHT
        
        # 절대값을 보고 x, y의 스칼라 값만 먼저 결정
        move_x = SHARK_MOVE_STEP if abs(dx) > abs(dy) else BLOCK_SIZE if dx != 0 else 0 
        move_y = SHARK_MOVE_STEP if abs(dy) > abs(dx) else BLOCK_SIZE if dy != 0 else 0
        
        # 대각선으로 움직여야 하지만 shark_move_step을 넘어가면 안됨 -> x, y의 값을 1로 조정
        if move_x + move_y > SHARK_MOVE_STEP:
            move_x = BLOCK_SIZE
            move_y = BLOCK_SIZE
            
        # x, y의 부호를 결정
        self.pos[0] += move_x * (1 if dx > 0 else -1)
        self.pos[1] += move_y * (1 if dy > 0 else -1)
        
        self.pos = list(bound_less_domain(self.pos[0], self.pos[1]))
        

    def check_target_alive(self, n):
        return self.target_fish_idx < n

    def move(self, fish_list):
        # move towards the target twice
        if self.check_target_alive(len(fish_list)) and self.measure_fish_size(fish_list) < self.size: # target fish alive and fish size smaller than myself
            target_fish = fish_list[self.target_fish_idx]
            # self.get_close(0, target_fish.x - self.pos[0])
            # self.get_close(1, target_fish.y - self.pos[1])
            self.get_close(target_fish)

        else:
            # self.pos[0] += random.randint(-1,1)*BLOCK_SIZE
            # self.pos[1] += random.randint(-1,1)*BLOCK_SIZE
            if self.target_reset_count == self.target_reset_cooltime:
                self.reset_target(fish_list)
                self.target_reset_count = 0
            else:
                self.target_reset_count += 1

    # RULE 1. A lot of nearby fish
    # RULE 2. Projected view's length (maximum difference of the connected fishes are larger than shark size)

    def measure_fish_size(self, fish_list):
        # RULE 1
        target_fish = fish_list[self.target_fish_idx]
        x,y = target_fish.x, target_fish.y
        # RULE 1
        fish_nearby = 0
        for idx in range(len(fish_list)):
            if idx == self.target_fish_idx:
                continue
            cur_fish = fish_list[idx]
            
            distance = math.sqrt((x - cur_fish.x) ** 2 + (y - cur_fish.y) ** 2)
            if distance <= RULE_1_RADIUS:
                fish_nearby += 1
            # if abs(x - cur_fish.x) <= RULE_1_RADIUS and abs(y - cur_fish.y) <= RULE_1_RADIUS:
            #     fish_nearby += 1

        # RULE 2
        # for idx in range(len(fish_list)):
        #     if idx == self.target_fish_idx:
        #         continue

        return fish_nearby + 1

    def reset_target(self, fish_list):
        self.target_fish_idx = random.randint(0, len(fish_list)-1)
        # print('target reset to ', self.target_fish_idx)


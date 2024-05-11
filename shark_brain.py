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
BLOCK_SIZE = 20
SHARK_MOVE_STEP = BLOCK_SIZE*2

class Shark():
    def __init__(self):
        self.pos = [0,0]
        self.target_fish_idx = 0
        self.size = 3 # size threshold

    def get_close(self, pos_idx, diff):
        if diff > 0:
            self.pos[pos_idx] += BLOCK_SIZE
        elif diff < 0:
            self.pos[pos_idx]-= BLOCK_SIZE
        # else:
        #     pass
    def move(self, fish_list):
        # move towards the target twice
        if self.measure_fish_size(fish_list) < self.size:
            target_fish = fish_list[self.target_fish_idx]
            self.get_close(0, target_fish.x - self.pos[0])
            self.get_close(1, target_fish.y - self.pos[1])

        else:
            self.pos[0] += random.randint(-1,1)*BLOCK_SIZE
            self.pos[1] += random.randint(-1,1)*BLOCK_SIZE
            self.reset_target(fish_list)

    def measure_fish_size(self, fish_list):
        return 1

    def reset_target(self, fish_list):
        self.target_fish_idx = random.randint(0, len(fish_list)-1)


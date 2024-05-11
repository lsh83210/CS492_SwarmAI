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
from enum import Enum
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
class Shark():
    def __init__(self):
        self.position = [1,1]
        self.target_fish = 0
        self.facing_direction = Direction.RIGHT




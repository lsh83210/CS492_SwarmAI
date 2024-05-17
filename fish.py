from variables_n_utils import *

class Fish:
    def __init__(self, x, y, size=1, id=None, color=None):
        self.x = x
        self.y = y
        self.size = size
        self.id = id if id is not None else self.generate_id()
        self.color = color if color is not None else (0, 255, 0)  # 기본 색상은 초록색
        self.alive = True  # 물고기가 살아있는지 여부를 나타내는 플래그

    def generate_id(self):
        # 고유한 ID를 생성하는 로직 (여기서는 간단한 예시로 UUID를 사용)
        import uuid
        return str(uuid.uuid4())

    def move(self, dx, dy):
        if self.alive:
            self.x += dx
            self.y += dy
            self.handle_boundary()

    def handle_boundary(self):
        # 경계를 넘어가면 반대편으로 나타나게 하는 로직
        if self.x < 0:
            self.x += WIDTH
        elif self.x >= WIDTH:
            self.x -= WIDTH

        if self.y < 0:
            self.y += HEIGHT
        elif self.y >= HEIGHT:
            self.y -= HEIGHT

    def update_state(self, is_alive):
        self.alive = is_alive

    def detect_collision(self, other):
        if not self.alive:
            return False
        distance = math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        return distance < (self.size + other.size) / 2

    def render(self, screen):
        if self.alive:
            pygame.draw.circle(screen, self.color, (self.x, self.y), self.size)

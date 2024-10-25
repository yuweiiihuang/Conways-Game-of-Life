import pygame
import numpy as np
from scipy.signal import convolve2d

# 設置窗口參數
WINDOW_SIZE = 600
CELL_SIZE = 3
GRID_SIZE = WINDOW_SIZE // CELL_SIZE
KILL_RADIUS = 4
# 定義字母圖案
LETTER_PATTERNS = {
    'A': [4, 10, 31, 17, 17],
    'B': [30, 17, 30, 17, 30],
    'C': [14, 17, 16, 17, 14],
    'D': [30, 17, 17, 17, 30],
    'E': [31, 16, 30, 16, 31],
    'F': [31, 16, 30, 16, 16],
    'G': [15, 16, 19, 17, 15],
    'H': [17, 17, 31, 17, 17],
    'I': [14, 4, 4, 4, 14],
    'J': [31, 2, 2, 18, 12],
    'K': [17, 18, 28, 18, 17],
    'L': [16, 16, 16, 16, 31],
    'M': [17, 27, 21, 17, 17],
    'N': [17, 25, 21, 19, 17],
    'O': [14, 17, 17, 17, 14],
    'P': [30, 17, 30, 16, 16],
    'Q': [14, 17, 17, 18, 13],
    'R': [30, 17, 30, 17, 17],
    'S': [15, 16, 14, 1, 30],
    'T': [31, 4, 4, 4, 4],
    'U': [17, 17, 17, 17, 14],
    'V': [17, 17, 17, 10, 4],
    'W': [17, 17, 21, 27, 17],
    'X': [17, 10, 4, 10, 17],
    'Y': [17, 10, 4, 4, 4],
    'Z': [31, 2, 4, 8, 31],
    ' ': [0, 0, 0, 0, 0]
}

LETTER_PATTERNS_CACHED = {
    letter: np.array([[1 if char == '1' else 0 for char in bin(row)[2:].zfill(5)] for row in pattern])
    for letter, pattern in LETTER_PATTERNS.items()
}

PROBABILITY_ALIVE = [0.8, 0.2]

def create_pattern(pattern):
    binary_length = 5
    return np.array([[1 if char == '1' else 0 for char in bin(row)[2:].zfill(binary_length)] for row in pattern])

def place_pattern(grid, pattern, top_left_x, top_left_y):
    pattern_height, pattern_width = pattern.shape
    grid[top_left_x:top_left_x + pattern_height, top_left_y:top_left_y + pattern_width] = pattern

def update(grid, kernel):
    neighbor_count = convolve2d(grid, kernel, mode='same', boundary='wrap')
    new_grid = np.where((grid == 1) & ((neighbor_count == 2) | (neighbor_count == 3)), 1, 0)
    new_grid = np.where((grid == 0) & (neighbor_count == 3), 1, new_grid)
    return new_grid
def clear_area(grid, x, y, radius):
    # 計算殺光區域的邊界
    min_x = max(0, x - radius)
    max_x = min(GRID_SIZE, x + radius + 1)
    min_y = max(0, y - radius)
    max_y = min(GRID_SIZE, y + radius + 1)
    
    # 將區域內的細胞設為死亡
    grid[min_x:max_x, min_y:max_y] = 0
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Game of Life - Pygame")

    # 初始化網格
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    
    # 放置文字圖案
    letters_to_display = "HELLO WORLD"
    letter_spacing = 2  # 字母之間的間距
    total_width = sum([LETTER_PATTERNS_CACHED[letter].shape[1] for letter in letters_to_display]) + letter_spacing * (len(letters_to_display) - 1)
    
    # 計算中心起始位置
    start_x = (GRID_SIZE - 5) // 2  # 5 是每個字母的高度
    start_y = (GRID_SIZE - total_width) // 2
    
    current_y = start_y
    for letter in letters_to_display:
        pattern = LETTER_PATTERNS_CACHED.get(letter, LETTER_PATTERNS_CACHED[' '])
        place_pattern(grid, pattern, start_x, current_y)
        current_y += pattern.shape[1] + letter_spacing

    # 設置隨機狀態區域
    top_region = GRID_SIZE // 4
    bottom_region = GRID_SIZE // 4
    random_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    random_grid[:top_region, :] = np.random.choice([0, 1], (top_region, GRID_SIZE), p=PROBABILITY_ALIVE)
    random_grid[-bottom_region:, :] = np.random.choice([0, 1], (bottom_region, GRID_SIZE), p=PROBABILITY_ALIVE)
    grid = np.where(grid == 1, 1, random_grid)

    clock = pygame.time.Clock()
    grid = update(grid, kernel)
        
    screen.fill((0, 0, 0))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[x, y] == 1:
                pygame.draw.rect(screen, (255, 255, 255), (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.display.flip()
    running = True
    pause = True
    create = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            #按p暫停    
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pause = True
            if pygame.mouse.get_pressed()[0]:
                create = True
        while pause:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    pause = False
            #按滑鼠左鍵生成細胞      
            if pygame.mouse.get_pressed()[0]:
                #偵測滑鼠位置
                mouse_x, mouse_y = pygame.mouse.get_pos()
                grid_x, grid_y = mouse_y // CELL_SIZE, mouse_x // CELL_SIZE
                grid[grid_x, grid_y] = 1
                #即時更新在螢幕上
                pygame.draw.rect(screen, (255, 255, 255), (grid_y * CELL_SIZE, grid_x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                pygame.display.update()
            #按滑鼠右鍵摧毀10*10的區域細胞    
            if pygame.mouse.get_pressed()[2]:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                grid_x, grid_y = mouse_y // CELL_SIZE, mouse_x // CELL_SIZE
                clear_area(grid,grid_x,grid_y,KILL_RADIUS)  
                for i in range(grid_x - KILL_RADIUS, grid_x + KILL_RADIUS + 1):
                        for j in range(grid_y - KILL_RADIUS, grid_y + KILL_RADIUS + 1):
                            if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE:
                                pygame.draw.rect(screen, (0,0,0), (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                pygame.display.update()
        """        
        while create:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                if pygame.mouse.get_pressed()[0]:
                    create = False
                if event.type == pygame.KEYDOWN:
                    print(event.unicode.upper())
                    
                    letter = event.unicode.upper()
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    grid_x, grid_y = mouse_y // CELL_SIZE, mouse_x // CELL_SIZE
                    pattern = LETTER_PATTERNS_CACHED.get(letter, LETTER_PATTERNS_CACHED[' '])
                    place_pattern(grid, pattern, start_x, current_y)
        grid = update(grid, kernel)
        """
        screen.fill((0, 0, 0))
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if grid[x, y] == 1:
                    pygame.draw.rect(screen, (255, 255, 255), (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        pygame.display.flip()
        clock.tick(10)  # 控制更新速度

    pygame.quit()

if __name__ == '__main__':
    main()

import pygame
import numpy as np
from scipy.signal import convolve2d
import random

# 設置窗口參數
WINDOW_SIZE = 600
CELL_SIZE = 3
GRID_SIZE = WINDOW_SIZE // CELL_SIZE
KILL_RADIUS = 4
BETTER_ZONE_START = (0,0)
BETTER_ZONE_END = (GRID_SIZE//4, GRID_SIZE//4)
WORSE_ZONE_START = (GRID_SIZE*3 //4, GRID_SIZE*3 // 4)
WORSE_ZONE_END = (GRID_SIZE,GRID_SIZE)

OBSTACLE_SIZE = 4
OBSTACLE_NUM = 10

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
#偵測是否在較好的區域(藍色)
def check_better(x,y):
    return BETTER_ZONE_START[0] <= x < BETTER_ZONE_END[0] and BETTER_ZONE_START[1] <= y < BETTER_ZONE_END[1]
#偵測是否在較差的區域(紅色)
def check_worse(x,y):
    return WORSE_ZONE_START[0] <= x < WORSE_ZONE_END[0] and WORSE_ZONE_START[1] <= y < WORSE_ZONE_END[1]
"""
def generate_obstacles(grid):
    obstacle = []
    for i in range(OBSTACLE_NUM):
        while True:
            x = random.randint(0, GRID_SIZE - OBSTACLE_SIZE)
            y = random.randint(0, GRID_SIZE - OBSTACLE_SIZE)
            overlap = False
            for ox, oy in obstacle:
                if abs(x - ox) < OBSTACLE_SIZE and abs(y - oy) < OBSTACLE_SIZE:
                    is_overlap = True
                    break
            if not overlap:
                grid[x:x + OBSTACLE_SIZE, y:y + OBSTACLE_SIZE] = -1  # -1 代表障礙物
                obstacle.append((x, y))
                print(x,y)
                break    
    return grid
"""
def place_pattern(grid, pattern, top_left_x, top_left_y):
    pattern_height, pattern_width = pattern.shape
    grid[top_left_x:top_left_x + pattern_height, top_left_y:top_left_y + pattern_width] = pattern

#更新地圖
def update(grid, kernel):
    neighbor_count = convolve2d(grid, kernel, mode='same', boundary='wrap')
    new_grid = np.zeros_like(grid)
    for x in range (GRID_SIZE):
        for y in range(GRID_SIZE): 
            #判斷障礙物
            if grid[x,y] ==-1:
                new_grid[x, y] = -1
            #判斷細胞       
            if grid[x,y]==1:
                if check_better(x,y):
                    if neighbor_count[x, y] == 1 or neighbor_count[x, y] == 4:
                        new_grid[x, y] = 1
                elif check_worse(x,y):
                    if neighbor_count[x, y] == 2:
                        new_grid[x, y] = 1       
                else:
                    if neighbor_count[x, y] == 2 or neighbor_count[x, y] == 3:
                        new_grid[x, y] = 1
            #判斷空地            
            elif grid[x,y]==0:
                if check_better(x,y) and neighbor_count[x, y] == 2:
                    new_grid[x, y] = 1
                elif check_worse(x,y) and neighbor_count[x, y] == 4:
                    new_grid[x, y] = 1
                elif not check_better(x,y) and not check_worse(x,y)and neighbor_count[x, y] == 3:
                    new_grid[x, y] = 1     
    return new_grid
#摧毀細胞跟障礙物
def clear_area(grid, x, y, radius):
    # 計算殺光區域的邊界
    min_x = max(0, x - radius)
    max_x = min(GRID_SIZE, x + radius + 1)
    min_y = max(0, y - radius)
    max_y = min(GRID_SIZE, y + radius + 1)
    
    # 將區域內的細胞及障礙物設為空地
    grid[min_x:max_x, min_y:max_y] = 0
def main():
    #初始化
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Game of Life - Pygame")
    # 初始化網格
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    
    # 放置文字圖案
    letters_to_display = "GROUP FOUR"
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
    top_region = GRID_SIZE //4
    bottom_region = GRID_SIZE //4
    
    #地圖歸零
    random_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    
    random_grid[:top_region, :] = np.random.choice([0, 1], (top_region, GRID_SIZE), p=PROBABILITY_ALIVE)
    random_grid[-bottom_region:, :] = np.random.choice([0, 1], (bottom_region, GRID_SIZE), p=PROBABILITY_ALIVE)
    
    #開場隨機放細胞
    """
    random_grid = np.random.choice([0, 1], (GRID_SIZE, GRID_SIZE), p=PROBABILITY_ALIVE)
    """
    grid = np.where(grid == 1, 1, random_grid)
    #grid = generate_obstacles(grid)
    clock = pygame.time.Clock()
    #填顏色(有draw的是上顏色，display就是顯示) 
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (255, 0, 0), (WORSE_ZONE_START[0]*CELL_SIZE,WORSE_ZONE_START[1]*CELL_SIZE, (WORSE_ZONE_END[0]-WORSE_ZONE_START[0])*CELL_SIZE,  (WORSE_ZONE_END[0]-WORSE_ZONE_START[0])*CELL_SIZE))
    pygame.draw.rect(screen, (0, 0, 255), (BETTER_ZONE_START[0]*CELL_SIZE,BETTER_ZONE_START[1]*CELL_SIZE, (BETTER_ZONE_END[0]-BETTER_ZONE_START[0])*CELL_SIZE,  (BETTER_ZONE_END[0]-BETTER_ZONE_START[0])*CELL_SIZE))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[x, y] == 1:
                pygame.draw.rect(screen, (255, 255, 255), (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            if grid[x,y] == -1:
                pygame.draw.rect(screen, (0, 255, 0), (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))    
    pygame.display.flip()
    running = True
    pause = True
    create = False
    while running:
        #讀按鍵
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            #按p暫停    
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pause = True
            """       
            if pygame.mouse.get_pressed()[0]:
                create = True
            """       
        #在p狀態下進行        
        while pause:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    pause = False
            #按滑鼠左鍵生成細胞(不能放障礙物上)      
            if pygame.mouse.get_pressed()[0]:
                #偵測滑鼠位置
                mouse_x, mouse_y = pygame.mouse.get_pos()
                grid_x, grid_y = mouse_y // CELL_SIZE, mouse_x // CELL_SIZE
                if grid[grid_x, grid_y]!=-1:
                    grid[grid_x, grid_y] = 1
                    #即時更新在螢幕上
                    pygame.draw.rect(screen, (255, 255, 255), (grid_y * CELL_SIZE, grid_x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    pygame.display.update()
            #按滑鼠右鍵摧毀10*10的區域細胞和障礙物   
            if pygame.mouse.get_pressed()[2]:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                grid_x, grid_y = mouse_y // CELL_SIZE, mouse_x // CELL_SIZE
                clear_area(grid,grid_x,grid_y,KILL_RADIUS)  
                for i in range(grid_x - KILL_RADIUS, grid_x + KILL_RADIUS + 1):
                        for j in range(grid_y - KILL_RADIUS, grid_y + KILL_RADIUS + 1):
                            if BETTER_ZONE_START[0] <= i < BETTER_ZONE_END[0] and BETTER_ZONE_START[0] <= j < BETTER_ZONE_END[1]:
                                pygame.draw.rect(screen, (0,0,255), (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                            elif WORSE_ZONE_START[0] <= i < WORSE_ZONE_END[0] and WORSE_ZONE_START[0] <= j < WORSE_ZONE_END[1]:
                                pygame.draw.rect(screen, (255,0,0), (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                            elif 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE:
                                pygame.draw.rect(screen, (0,0,0), (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                pygame.display.update()
            #按滑鼠滾輪鍵放障礙物(不能放在細胞上)
            if pygame.mouse.get_pressed()[1]:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                grid_x, grid_y = mouse_y // CELL_SIZE, mouse_x // CELL_SIZE
                if grid[grid_x, grid_y]==0:
                    grid[grid_x, grid_y] = -1
                    #即時更新在螢幕上
                    pygame.draw.rect(screen, (0, 255, 0), (grid_y * CELL_SIZE, grid_x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
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
        """            
        #更新數據並填顏色
        grid = update(grid, kernel)
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 0, 0), (WORSE_ZONE_START[0]*CELL_SIZE,WORSE_ZONE_START[1]*CELL_SIZE, (WORSE_ZONE_END[0]-WORSE_ZONE_START[0])*CELL_SIZE,  (WORSE_ZONE_END[0]-WORSE_ZONE_START[0])*CELL_SIZE))
        pygame.draw.rect(screen, (0, 0, 255), (BETTER_ZONE_START[0]*CELL_SIZE,BETTER_ZONE_START[1]*CELL_SIZE, (BETTER_ZONE_END[0]-BETTER_ZONE_START[0])*CELL_SIZE,  (BETTER_ZONE_END[0]-BETTER_ZONE_START[0])*CELL_SIZE))
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if grid[x, y] == 1:
                    pygame.draw.rect(screen, (255, 255, 255), (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                if grid[x,y] == -1:
                    pygame.draw.rect(screen, (0, 255, 0), (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.flip()
        clock.tick(10)  # 控制更新速度

    pygame.quit()

if __name__ == '__main__':
    main()



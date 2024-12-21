import pygame
import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import os
import random
import time
# 設置窗口參數
WINDOW_SIZE = 600
CELL_SIZE = 3
GRID_SIZE = WINDOW_SIZE // CELL_SIZE
KILL_RADIUS = 4
#改成用陣列存
BETTER_ZONE_START = [0,0]
BETTER_ZONE_END = [GRID_SIZE//4, GRID_SIZE//4]
WORSE_ZONE_START = [GRID_SIZE*3 //4, GRID_SIZE*3 // 4]
WORSE_ZONE_END = [GRID_SIZE,GRID_SIZE]
#INTERVAL可以改變動的速度
UPDATE_INTERVAL = 0.1
OBSTACLE_SIZE = 4
OBSTACLE_NUM = 10

# 預設圖案的存放資料夾
IMAGE_FOLDER = "patterns"

# 載入圖片並轉換為網格
def image_to_grid(image_path, grid_size):
    """
    將圖片轉換為二值化網格。
    :param image_path: 圖片路徑。
    :param grid_size: 網格大小。
    :return: 二值化網格。
    """
    img = Image.open(image_path).convert('L')
    img = img.resize((grid_size, grid_size), Image.Resampling.LANCZOS)
    img_array = np.array(img)
    threshold = 128
    grid = (img_array > threshold).astype(int)
    return grid 

def load_patterns(image_folder, grid_sizes):
    """
    預處理圖片資料夾中的所有圖片，生成對應的二值化網格。
    :param image_folder: 圖片資料夾。
    :param grid_sizes: 每個圖案對應的網格大小陣列。
    :return: 包含所有圖片網格的字典。
    """
    patterns = {}
    for i, filename in enumerate(sorted(os.listdir(image_folder))):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_folder, filename)
            grid_size = grid_sizes[i]
            patterns[i] = image_to_grid(file_path, grid_size)
    return patterns

def place_pattern(grid, pattern, center_x, center_y):
    """
    將圖案放置到指定中心位置。
    :param grid: 主網格。
    :param pattern: 圖案網格。
    :param center_x: 圖案的中心 x 座標。
    :param center_y: 圖案的中心 y 座標。
    """
    pattern_height, pattern_width = pattern.shape
    top_left_x = center_x - pattern_height // 2
    top_left_y = center_y - pattern_width // 2

    if top_left_x < 0 or top_left_y < 0 or top_left_x + pattern_height > grid.shape[0] or top_left_y + pattern_width > grid.shape[1]:
        raise ValueError("圖案超出邊界")

    grid[top_left_x:top_left_x + pattern_height, top_left_y:top_left_y + pattern_width] = pattern

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
#偵測的部分要XY軸顛倒
#偵測是否在較好的區域(藍色)
def check_better(y,x):
    #print(BETTER_ZONE_START[0])
    return BETTER_ZONE_START[0] <= x < BETTER_ZONE_END[0] and BETTER_ZONE_START[1] <= y < BETTER_ZONE_END[1]
#偵測是否在較差的區域(紅色)
def check_worse(y,x):
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
"""
def place_pattern(grid, pattern, top_left_x, top_left_y):
    pattern_height, pattern_width = pattern.shape
    grid[top_left_x:top_left_x + pattern_height, top_left_y:top_left_y + pattern_width] = pattern
"""

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
                    if neighbor_count[x, y] == 2 or neighbor_count[x, y] == 3:
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
    #偵測上次更新的時間
    last_update = time.time()
    #顯示細胞數量字的大小
    font = pygame.font.Font(None, 36)
    # 初始化網格
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    
    # 設定每個圖案的 grid_size
    # [脈衝星, 高斯帕機槍, 太空船, 慨影, 紅綠燈, 人, 工, 智, 慧, 四]
    grid_sizes = [17, 36, 5, 12, 9, 16, 16, 16, 16, 16]

    # 載入圖案
    patterns = load_patterns(IMAGE_FOLDER, grid_sizes)
    
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
     #顯示細胞數量
    def display_cell_count():
        cell_count = np.sum(grid == 1)  # 計算細胞的數量
        cell_count_text = font.render(f"Cell Count: {cell_count}", True, (255,215,0))  # 文字顏色
        screen.blit(cell_count_text, (200, 320))  # 顯示在螢幕的哪個位置
    
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
        # 處理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pause = True
        
        # 暫停狀態處理     
        while pause:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pause = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    pause = False
                if event.type == pygame.KEYDOWN and pygame.K_0 <= event.key <= pygame.K_9:
                    num = event.key - pygame.K_0
                    if num in patterns:
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        grid_x, grid_y = mouse_y // CELL_SIZE, mouse_x // CELL_SIZE
                        try:
                            # 放置圖案
                            place_pattern(grid, patterns[num], grid_x, grid_y)

                            # 計算受影響區域的範圍
                            pattern_height, pattern_width = patterns[num].shape
                            min_x = max(0, grid_x - pattern_height // 2)
                            max_x = min(GRID_SIZE, grid_x + pattern_height // 2)
                            min_y = max(0, grid_y - pattern_width // 2)
                            max_y = min(GRID_SIZE, grid_y + pattern_width // 2)

                            # 重繪受影響區域
                            for x in range(min_x, max_x):
                                for y in range(min_y, max_y):
                                    if grid[x, y] == 1:
                                        color = (255, 255, 255)
                                    elif grid[x, y] == -1:
                                        color = (0, 255, 0)
                                    else:
                                        color = (0, 0, 0)
                                    pygame.draw.rect(screen, color, (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))

                            # 重繪紅藍區域（僅繪製邊框）
                            pygame.draw.rect(screen, (255, 0, 0), 
                                            (WORSE_ZONE_START[0] * CELL_SIZE, WORSE_ZONE_START[1] * CELL_SIZE, 
                                            (WORSE_ZONE_END[0] - WORSE_ZONE_START[0]) * CELL_SIZE, 
                                            (WORSE_ZONE_END[1] - WORSE_ZONE_START[1]) * CELL_SIZE), width=1)  # 設定線框寬度
                            pygame.draw.rect(screen, (0, 0, 255), 
                                            (BETTER_ZONE_START[0] * CELL_SIZE, BETTER_ZONE_START[1] * CELL_SIZE, 
                                            (BETTER_ZONE_END[0] - BETTER_ZONE_START[0]) * CELL_SIZE, 
                                            (BETTER_ZONE_END[1] - BETTER_ZONE_START[1]) * CELL_SIZE), width=1)  # 設定線框寬度

                            pygame.display.update()  # 即時刷新受影響的部分
                        except ValueError:
                            print("無法放置圖案，超出邊界")
                        
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
            #這裡判斷的XY一樣要相反  
            if pygame.mouse.get_pressed()[2]:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                grid_x, grid_y = mouse_y // CELL_SIZE, mouse_x // CELL_SIZE 
                for i in range(grid_x - KILL_RADIUS, grid_x + KILL_RADIUS + 1):
                        for j in range(grid_y - KILL_RADIUS, grid_y + KILL_RADIUS + 1):
                            if BETTER_ZONE_START[0] <= j < BETTER_ZONE_END[0] and BETTER_ZONE_START[1] <= i < BETTER_ZONE_END[1]:
                                pygame.draw.rect(screen, (0,0,255), (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                            elif WORSE_ZONE_START[0] <= j < WORSE_ZONE_END[0] and WORSE_ZONE_START[1] <= i < WORSE_ZONE_END[1]:
                                pygame.draw.rect(screen, (255,0,0), (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                            elif 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE and grid[i,j]==1:
                                pygame.draw.rect(screen, (0,0,0), (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                            elif 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE and grid[i,j]==-1:
                                pygame.draw.rect(screen, (0,0,0), (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))     
                clear_area(grid,grid_x,grid_y,KILL_RADIUS)                 
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
        #新加的
        #偵測執行時間
        current_time = time.time()
        #時間大於INTERVAL才會執行，藉此控制速度
        if current_time - last_update >= UPDATE_INTERVAL:
            last_update = current_time
            #讓好壞區域順時針繞著走，一次動一格
            if BETTER_ZONE_END[0] < GRID_SIZE and BETTER_ZONE_START[1] == 0 :
                BETTER_ZONE_START[0]=BETTER_ZONE_START[0]+1
                BETTER_ZONE_END[0]=BETTER_ZONE_END[0]+1
            elif BETTER_ZONE_END[1] < GRID_SIZE and BETTER_ZONE_END[0] == GRID_SIZE :
                BETTER_ZONE_START[1]=BETTER_ZONE_START[1]+1
                BETTER_ZONE_END[1]=BETTER_ZONE_END[1]+1
            elif BETTER_ZONE_START[0] > 0 and BETTER_ZONE_END[1] == GRID_SIZE :
                BETTER_ZONE_START[0]=BETTER_ZONE_START[0]-1
                BETTER_ZONE_END[0]=BETTER_ZONE_END[0]-1
            elif BETTER_ZONE_END[1] > 0 and BETTER_ZONE_START[0] == 0 :
                BETTER_ZONE_START[1]=BETTER_ZONE_START[1]-1
                BETTER_ZONE_END[1]=BETTER_ZONE_END[1]-1
            if WORSE_ZONE_END[0] < GRID_SIZE and WORSE_ZONE_START[1] == 0 :
                WORSE_ZONE_START[0]=WORSE_ZONE_START[0]+1
                WORSE_ZONE_END[0]=WORSE_ZONE_END[0]+1
            elif WORSE_ZONE_END[1] < GRID_SIZE and WORSE_ZONE_END[0] == GRID_SIZE :
                WORSE_ZONE_START[1]=WORSE_ZONE_START[1]+1
                WORSE_ZONE_END[1]=WORSE_ZONE_END[1]+1
            elif WORSE_ZONE_START[0] > 0 and WORSE_ZONE_END[1] == GRID_SIZE :
                WORSE_ZONE_START[0]=WORSE_ZONE_START[0]-1
                WORSE_ZONE_END[0]=WORSE_ZONE_END[0]-1
            elif WORSE_ZONE_END[1] > 0 and WORSE_ZONE_START[0] == 0 :
                WORSE_ZONE_START[1]=WORSE_ZONE_START[1]-1
                WORSE_ZONE_END[1]=WORSE_ZONE_END[1]-1   
        pygame.draw.rect(screen, (255, 0, 0), (WORSE_ZONE_START[0]*CELL_SIZE,WORSE_ZONE_START[1]*CELL_SIZE, (WORSE_ZONE_END[0]-WORSE_ZONE_START[0])*CELL_SIZE,  (WORSE_ZONE_END[0]-WORSE_ZONE_START[0])*CELL_SIZE))
        pygame.draw.rect(screen, (0, 0, 255), (BETTER_ZONE_START[0]*CELL_SIZE,BETTER_ZONE_START[1]*CELL_SIZE, (BETTER_ZONE_END[0]-BETTER_ZONE_START[0])*CELL_SIZE,  (BETTER_ZONE_END[0]-BETTER_ZONE_START[0])*CELL_SIZE))
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if grid[x, y] == 1:
                    pygame.draw.rect(screen, (255, 255, 255), (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                if grid[x,y] == -1:
                    pygame.draw.rect(screen, (0, 255, 0), (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        display_cell_count()#顯示細胞數量
        pygame.display.flip()
        clock.tick(10)  # 控制更新速度

    pygame.quit()

if __name__ == '__main__':
    main()

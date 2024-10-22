import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

# 定義所有26個英文字母及空格的圖案，使用數字編碼
# 每個數字代表一行，'X' 為1，' ' 為0
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

# 將所有字母圖案轉換為 NumPy 陣列並作為常數快取
LETTER_PATTERNS_CACHED = {letter: np.array([[1 if char == '1' else 0 for char in bin(row)[2:].zfill(5)] for row in pattern]) for letter, pattern in LETTER_PATTERNS.items()}

# 定義隨機狀態的存活概率
PROBABILITY_ALIVE = [0.8, 0.2]

def create_pattern(pattern):
    """
    將字母的數字編碼圖案轉換為二維 NumPy 陣列
    1 表示存活的細胞，0 表示死亡的細胞
    """
    binary_length = 5  # 因為字母圖案是5列寬
    pattern_array = []
    for row in pattern:
        # 將數字轉換為二進位字符串，去掉'0b'，並填充至固定長度
        binary_str = bin(row)[2:].zfill(binary_length)
        # 將每位字符轉換為1或0
        pattern_array.append([1 if char == '1' else 0 for char in binary_str])
    return np.array(pattern_array)

def place_pattern(grid, pattern, top_left_x, top_left_y):
    """
    將 pattern 放置到網格中的指定位置。
    
    :param grid: 主網格（2D NumPy 陣列）
    :param pattern: 字母圖案（2D NumPy 陣列）
    :param top_left_x: 放置 pattern 的左上角 x 坐標
    :param top_left_y: 放置 pattern 的左上角 y 坐標
    """
    pattern_height, pattern_width = pattern.shape
    grid_height, grid_width = grid.shape
    
    # 確保 pattern 不會超出主網格邊界
    if top_left_x + pattern_height > grid_height or top_left_y + pattern_width > grid_width:
        raise ValueError(f"Pattern exceeds grid boundaries at position ({top_left_x}, {top_left_y}) with pattern size ({pattern_height}, {pattern_width}).")
    
    # 將 pattern 嵌入主網格
    grid[top_left_x:top_left_x + pattern_height, top_left_y:top_left_y + pattern_width] = pattern

def update(frame_num, img, grid, kernel):
    """
    更新函數，用於動畫。
    """
    # 計算周圍八個細胞的存活數量
    neighbor_count = convolve2d(grid, kernel, mode='same', boundary='wrap')
    
    # 應用康威生命遊戲的規則
    # 存活細胞維持或死亡
    grid[:] = np.where(
        (grid == 1) & ((neighbor_count == 2) | (neighbor_count == 3)),
        1,
        0
    )
    # 死亡細胞復活
    grid[:] = np.where(
        (grid == 0) & (neighbor_count == 3),
        1,
        grid
    )
    
    # 更新圖像資料
    img.set_data(grid)
    return img,

def main():
    # 定義網格大小
    N = 200
    grid = np.zeros((N, N), dtype=int)
    
    # 定義卷積 kernal ，用於計算每個細胞的鄰居數量
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    # 自定義要顯示的字母序列（包含空格）
    letters_to_display = "hello world".upper()
    
    # 計算所有字母圖案的總寬度和最大高度，考慮間距
    spacing = 2  # 每個字母之間的間距
    letter_widths = [pattern.shape[1] for pattern in LETTER_PATTERNS_CACHED.values()]
    letter_heights = [pattern.shape[0] for pattern in LETTER_PATTERNS_CACHED.values()]
    
    # 假設每個字母的高度相同，取最大值
    max_letter_height = max(letter_heights)
    
    # 計算總寬度
    total_width = sum([LETTER_PATTERNS_CACHED[letter].shape[1] for letter in letters_to_display]) + spacing * (len(letters_to_display) - 1)
    
    # 計算中央起始位置
    start_x = (N - max_letter_height) // 2
    start_y = (N - total_width) // 2
    
    current_y = start_y
    
    # 放置指定的字母和空格到網格中央，保持5單位的間距
    for letter in letters_to_display:
        pattern = LETTER_PATTERNS_CACHED.get(letter, LETTER_PATTERNS_CACHED[' '])  # 默認使用空格
        pattern_height, pattern_width = pattern.shape
        # 計算每個字母的垂直偏移，以便所有字母在垂直方向居中
        offset_x = start_x + (max_letter_height - pattern_height) // 2
        try:
            place_pattern(grid, pattern, offset_x, current_y)
        except ValueError as e:
            print(f"無法放置字母 '{letter}'：{e}")
        # 更新位置以避免重疊
        current_y += pattern_width + spacing
    
    # 初始化其他部分為隨機狀態（上下各25%，中央50%留給字母）
    # 20% 的概率為存活狀態
    random_grid = np.zeros((N, N), dtype=int)
    
    # 上25%
    top_region = N // 4
    random_grid[:top_region, :] = np.random.choice([0, 1], (top_region, N), p=PROBABILITY_ALIVE)
    
    # 下25%
    bottom_region = N // 4
    random_grid[-bottom_region:, :] = np.random.choice([0, 1], (bottom_region, N), p=PROBABILITY_ALIVE)
    
    # 中央50%保留給字母，因此不隨機初始化
    # 將隨機狀態應用到主網格，但保留字母部分
    grid = np.where(grid == 1, 1, random_grid)
    
    # 設定圖像
    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(grid, interpolation='nearest', cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()
    
    # 等待幾秒鐘
    plt.pause(1)  # 等待1秒
    
    # 運行動畫
    ani = animation.FuncAnimation(
        fig, update, fargs=(img, grid, kernel),
        frames=60, interval=50, blit=True
    )
    plt.show()

if __name__ == '__main__':
    main()

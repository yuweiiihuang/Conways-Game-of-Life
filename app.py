import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

# 定義所有26個英文字母及空格的圖案，使用單一整數編碼
# 每個整數的二進位表示為5x5的圖案，從上到下，每行5位，共25位
LETTER_PATTERNS = {
    'A': 0b0010001010111111000110001,
    'B': 0b1111010001111101000111110,
    'C': 0b0111110000100001000001111,
    'D': 0b1111010001100011000111110,
    'E': 0b1111110000111101000011111,
    'F': 0b1111110000111101000010000,
    'G': 0b0111110000100111000101111,
    'H': 0b1000110001111111000110001,
    'I': 0b0111000100001000010001110,
    'J': 0b0011100010000101001001100,
    'K': 0b1000110010111001001010001,
    'L': 0b1000010000100001000011111,
    'M': 0b1000111011101011000110001,
    'N': 0b1000111001101011001110001,
    'O': 0b0111010001100011000101110,
    'P': 0b1111010001111101000010000,
    'Q': 0b0111010001100011001001101,
    'R': 0b1111010001111101001010001,
    'S': 0b0111110000011100000111110,
    'T': 0b1111100100001000010000100,
    'U': 0b1000110001100011000101110,
    'V': 0b1000110001100010101000100,
    'W': 0b1000110001101011101110001,
    'X': 0b1000101010001000101010001,
    'Y': 0b1000101010001000010000100,
    'Z': 0b1111100010001000100011111,
    ' ': 0b0000000000000000000000000
}

# 將所有字母圖案轉換為 NumPy 陣列並作為常數快取
LETTER_PATTERNS_CACHED = {
    letter: np.array([
        [
            1 if (pattern >> (24 - (row * 5 + col))) & 1 else 0
            for col in range(5)
        ]
        for row in range(5)
    ])
    for letter, pattern in LETTER_PATTERNS.items()
}

def int_to_pattern(num):
    """
    將一個整數轉換為5x5的二維網格。
    :param num: 整數值，應在0到2^25-1之間
    :return: 5x5的二維 NumPy 陣列
    """
    if not (0 <= num < 2**25):
        raise ValueError("數字必須在0到2^25-1之間")
    
    # 生成5x5的圖案，從二進位字串中解析每個位來填充圖案
    pattern = np.array([
        [
            1 if (num >> (24 - (row * 5 + col))) & 1 else 0
            for col in range(5)
        ]
        for row in range(5)
    ])
    return pattern

def place_pattern(grid, pattern, top_left_x, top_left_y):
    """
    將 pattern 放置到網格中的指定位置。
    
    :param grid: 主網格（2D NumPy 陣列）
    :param pattern: 字母圖案（2D NumPy 陣列）
    :param top_left_x: 放置 pattern 的左上角 x 座標
    :param top_left_y: 放置 pattern 的左上角 y 座標
    """
    pattern_height, pattern_width = pattern.shape
    grid_height, grid_width = grid.shape
    
    # 確保 pattern 不會超出主網格邊界
    if top_left_x + pattern_height > grid_height or top_left_y + pattern_width > grid_width:
        raise ValueError(f"圖案超出網格邊界，位置 ({top_left_x}, {top_left_y})，圖案大小 ({pattern_height}, {pattern_width})。")
    
    # 將 pattern 嵌入主網格的對應位置
    grid[top_left_x:top_left_x + pattern_height, top_left_y:top_left_y + pattern_width] = pattern

def update(frame_num, img, grid, kernel):
    """
    更新函數，用於動畫。
    :param frame_num: 當前的幀數（未使用，但需要提供給動畫函數）
    :param img: 當前顯示的圖像
    :param grid: 主網格（2D NumPy 陣列）
    :param kernel: 用於計算鄰居數量的卷積kernel
    :return: 更新後的圖像
    """
    # 計算周圍八個細胞的存活數量，使用卷積操作
    neighbor_count = convolve2d(grid, kernel, mode='same', boundary='wrap')
    
    # 應用康威生命遊戲的規則
    # 存活細胞維持或死亡：如果一個細胞是活的，且鄰居數為2或3，則繼續存活，否則死亡
    grid[:] = np.where(
        (grid == 1) & ((neighbor_count == 2) | (neighbor_count == 3)),
        1,
        0
    )
    # 死亡細胞復活：如果一個細胞是死的，且鄰居數為3，則復活
    grid[:] = np.where(
        (grid == 0) & (neighbor_count == 3),
        1,
        grid
    )
    
    # 更新圖像資料並返回
    img.set_data(grid)
    return img,

def main(input_sequence, grid_size=200, pause_duration=3, probability_alive=[0.8, 0.2]):
    """
    主函式，顯示輸入字母或數字序列的網格。
    :param input_sequence: 字母或數字序列
    :param grid_size: 主網格的大小，預設為200x200
    :param pause_duration: 動畫開始前的暫停時間，預設為3秒
    :param probability_alive: 初始化存活細胞的機率，預設為 [0.8, 0.2]
    """
    N = grid_size
    # 初始化主網格為全零的矩陣
    grid = np.zeros((N, N), dtype=int)
    
    # 定義卷積 kernel，用於計算每個細胞的鄰居數量
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    spacing = 2             # 每個圖案之間的間距
    max_letter_height = 5   # 每個圖案固定為5行高
    
    # 計算輸入序列在網格中水平置中的起始位置
    total_width = len(input_sequence) * 5 + (len(input_sequence) - 1) * spacing
    start_x = (N - max_letter_height) // 2  # 垂直置中
    start_y = (N - total_width) // 2  # 水平置中
    
    current_y = start_y
    
    # 將每個輸入字母或數字放入主網格中
    for item in input_sequence:
        if isinstance(item, str) and item.upper() in LETTER_PATTERNS_CACHED:
            pattern = LETTER_PATTERNS_CACHED[item.upper()]
        elif isinstance(item, int):
            pattern = int_to_pattern(item)
        else:
            continue  # 如果不是字母或數字，則跳過
        
        pattern_height, pattern_width = pattern.shape
        offset_x = start_x + (max_letter_height - pattern_height) // 2
        try:
            # 將當前字母圖案放置到網格的指定位置
            place_pattern(grid, pattern, offset_x, current_y)
        except ValueError as e:
            print(f"無法放置項目 '{item}'：{e}")
        current_y += pattern_width + spacing
    
    # 初始化其他部分為隨機狀態（上下各25%，中央50%留給字母）
    random_grid = np.zeros((N, N), dtype=int)
    
    # 上25%的區域隨機初始化
    top_region = N // 4
    random_grid[:top_region, :] = np.random.choice([0, 1], (top_region, N), p=probability_alive)
    
    # 下25%的區域隨機初始化
    bottom_region = N // 4
    random_grid[-bottom_region:, :] = np.random.choice([0, 1], (bottom_region, N), p=probability_alive)
    
    # 將隨機狀態應用到主網格，但保留字母部分的圖案
    grid = np.where(grid == 1, 1, random_grid)
    
    # 設定圖像
    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(grid, interpolation='nearest', cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()
    
    # 動畫開始前的暫停時間
    plt.pause(pause_duration)
    
    # 使用動畫函數進行動畫更新
    ani = animation.FuncAnimation(
        fig, update, fargs=(img, grid, kernel),
        frames=60, interval=50, blit=True
    )
    plt.show()

if __name__ == '__main__':
    # 8:       0b0111010001011101000101110 = 15252014
    # !:       0b0010000100001000000000100 = 4329476
    # 滑翔翼:   0b0000000000000010010100011 = 1187
    #main(input_sequence=[c for c in "Hello World"] + [15252014]) 
    main([15252014], pause_duration=10)
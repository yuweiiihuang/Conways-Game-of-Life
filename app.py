import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

def update(frame_num, img, grid, kernel):
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
    # 初始化網格，隨機設定細胞的存活狀態
    grid = np.random.choice([0, 1], N*N, p=[0.8, 0.2]).reshape(N, N).astype(bool)
    
    # 定義卷積核，用於計算每個細胞的鄰居數量
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    # 設定圖像
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest', cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    
    ani = animation.FuncAnimation(
        fig, update, fargs=(img, grid, kernel),
        frames=200, interval=50, blit=True
    )
    plt.show()

if __name__ == '__main__':
    main()

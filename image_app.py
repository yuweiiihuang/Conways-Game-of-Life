import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d
from PIL import Image

def gaussian_filter(kernel_size=5, sigma=1.0):
    """
    生成高斯濾波器，用於平滑圖像。
    :param kernel_size: 高斯濾波器的大小，必須為奇數。
    :param sigma: 高斯濾波器的標準差，控制平滑程度。
    :return: 高斯濾波器kernel。
    """
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    gauss = np.exp(-0.5 * (ax / sigma) ** 2) # 1D 高斯分佈
    kernel = np.outer(gauss, gauss)  # 生成2D高斯kernel
    return kernel / kernel.sum()  # 正規化

def sobel_edges(image):
    """
    使用 Sobel 算子計算圖像的水平和垂直梯度。
    :param image: 灰階圖像的 NumPy 陣列。
    :return: 梯度強度 (G) 和梯度方向 (theta)。
    """
    # 定義 Sobel 卷積kernel
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[1,  2,  1],
                        [0,  0,  0],
                        [-1, -2, -1]])
    # 計算水平和垂直梯度
    Gx = convolve2d(image, sobel_x, mode='same', boundary='symm')
    Gy = convolve2d(image, sobel_y, mode='same', boundary='symm')
    # 計算梯度強度和方向
    G = np.sqrt(Gx**2 + Gy**2)  # 合成梯度強度
    theta = np.arctan2(Gy, Gx)  # 梯度方向（以弧度表示）
    return G, theta

def non_max_suppression(gradient, theta):
    """
    非極大值抑制，用於細化邊緣。
    :param gradient: 梯度強度。
    :param theta: 梯度方向。
    :return: 經過非極大值抑制的梯度強度。
    """
    H, W = gradient.shape
    suppressed = np.zeros((H, W), dtype=np.float32)  # 初始化抑制結果
    angle = theta * (180.0 / np.pi)  # 將梯度方向從弧度轉為角度
    angle[angle < 0] += 180  # 確保角度範圍為 [0, 180]

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q = 255  # 初始化相鄰像素的梯度強度
            r = 255
            # 根據梯度方向選擇相鄰的像素
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient[i, j + 1]
                r = gradient[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient[i + 1, j - 1]
                r = gradient[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient[i + 1, j]
                r = gradient[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient[i - 1, j - 1]
                r = gradient[i + 1, j + 1]
            # 如果當前像素梯度大於相鄰的像素梯度，保留；否則設為0
            if gradient[i, j] >= q and gradient[i, j] >= r:
                suppressed[i, j] = gradient[i, j]
            else:
                suppressed[i, j] = 0
    return suppressed

def image_to_grid_with_edges(image_path, grid_size):
    """
    讀取圖片並提取邊緣，用於初始化生命遊戲的網格。
    :param image_path: 圖片路徑。
    :param grid_size: 網格大小。
    :return: 邊緣檢測後的二值化網格。
    """
    # 讀取圖片並轉為灰階模式
    img = Image.open(image_path).convert('L')
    img = img.resize((grid_size, grid_size), Image.Resampling.LANCZOS)  # 調整大小
    img_array = np.array(img)
    
    # 高斯平滑
    gaussian_kernel = gaussian_filter(kernel_size=5, sigma=1.0)
    smoothed = convolve2d(img_array, gaussian_kernel, mode='same', boundary='symm')
    
    # Sobel 邊緣檢測
    gradient, theta = sobel_edges(smoothed)
    
    # 非極大值抑制
    suppressed = non_max_suppression(gradient, theta)
    
    # 二值化邊緣檢測結果
    threshold = 10
    binary_edges = (suppressed > threshold).astype(int)
    return binary_edges

def update(frame_num, img, grid, kernel):
    """
    動畫更新函式，用於計算每一幀的細胞狀態。
    :param frame_num: 當前幀編號。
    :param img: 動畫中的圖像對象。
    :param grid: 當前網格狀態。
    :param kernel: 用於計算鄰居數量的卷積kernel。
    """
    # 計算每個細胞周圍的鄰居數量
    neighbor_count = convolve2d(grid, kernel, mode='same', boundary='wrap')
    # 應用生命遊戲的規則
    grid[:] = np.where((grid == 1) & ((neighbor_count == 2) | (neighbor_count == 3)), 1, 0)  # 存活或死亡
    grid[:] = np.where((grid == 0) & (neighbor_count == 3), 1, grid)  # 復活
    img.set_data(grid)  # 更新圖像數據
    return img,

def main_with_image(image_path, grid_size=200, pause_duration=3):
    """
    主函式，用於將圖片作為生命遊戲的初始狀態並顯示動畫。
    :param image_path: 圖片路徑。
    :param grid_size: 網格大小。
    :param pause_duration: 動畫開始前的暫停時間。
    """
    # 使用邊緣檢測初始化網格
    grid = image_to_grid_with_edges(image_path, grid_size)
    # 定義卷積kernel，用於計算鄰居數量
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    # 設置圖像窗口
    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(grid, interpolation='nearest', cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()
    
    # 動畫開始前暫停
    plt.pause(pause_duration)
    
    # 創建動畫
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, kernel), frames=100, interval=100, blit=True)
    plt.show()

if __name__ == '__main__':
    # 使用指定圖片運行生命遊戲
    main_with_image('../為什麼要演奏春日影.jpeg', grid_size=500, pause_duration=3)

# 是又怎樣.jpeg
# 
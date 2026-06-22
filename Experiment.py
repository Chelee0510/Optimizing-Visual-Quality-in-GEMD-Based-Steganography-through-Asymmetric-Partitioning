import cv2
import numpy as np
import math

def get_E(n):
    """根據遞迴推導計算任意 n 的期望總平方誤差 E(n)"""
    if n < 1:
        raise ValueError("群組大小 n 必須大於或等於 1")
    if n == 1:
        return 0.75
    S_n, D_n = 3, 7
    for i in range(2, n + 1):
        if i > 2:
            D_n = D_n * 2 + 2
        S_n += D_n
    return S_n / (2 ** (n + 1))

def get_balanced_partition(N, k):
    """策略 A：Symmetric Partitioning"""
    base = N // k
    rem = N % k
    return [base + 1] * rem + [base] * (k - rem)

def get_oud_partition(N, k):
    """策略 B：Asymmetric Partitioning"""
    if N < k:
        raise ValueError("總像素 N 必須大於或等於分組數 k")
    max_group_size = N - k + 1
    return [1] * (k - 1) + [max_group_size]

def simulate_stego_image(image, partition, N):
    """
    模擬 GEMD 隱寫術修改造成的圖像失真
    (使用嚴格限制 +1, -1, 0 的離散均勻雜訊)
    """
    stego_img = image.copy().astype(np.float64)
    
    # 計算該分組策略的平均理論 MSE
    mse_theoretical = sum(get_E(n) for n in partition) / N
    
    # 計算修改為 +1 與 -1 的發生機率 (p = MSE / 2)
    p = mse_theoretical / 2.0
    
    # 產生與圖片大小相同，且數值介於 0~1 的均勻隨機矩陣
    rand_matrix = np.random.rand(*stego_img.shape)
    
    # 初始化全為 0 的雜訊矩陣
    noise = np.zeros(stego_img.shape, dtype=np.float64)
    
    # 根據機率 p 賦予 +1 或 -1
    noise[rand_matrix < p] = 1.0
    noise[(rand_matrix >= p) & (rand_matrix < 2 * p)] = -1.0
    
    # 將離散雜訊加入原圖
    stego_img += noise
    
    # 確保像素值落在 0~255 的合法範圍 (處理邊界截斷)
    return np.clip(stego_img, 0, 255).astype(np.uint8)

def calculate_empirical_metrics(cover_img, stego_img):
    """對比兩張圖片，計算出實際的 MSE 與 PSNR"""
    cover_img = cover_img.astype(np.float64)
    stego_img = stego_img.astype(np.float64)
    
    mse = np.mean((cover_img - stego_img) ** 2)
    if mse == 0:
        return 0, float('inf')
    psnr = 10 * math.log10((255 ** 2) / mse)
    return mse, psnr

def evaluate_steganography_method(image_path, N=8, k=3):
    """主函數：處理圖片、計算容量、產出加密圖片並計算實測數據"""
    # 讀取 TIFF 或其他格式的灰階圖片
    cover_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if cover_image is None:
        print(f"警告：找不到圖片 '{image_path}'。已自動產生 512x512 測試用影像。")
        cover_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

    # 1. 計算嵌入容量 (Embedding Capacity)
    img_h, img_w = cover_image.shape
    total_pixels = img_h * img_w
    total_blocks = total_pixels // N
    payload_per_block = N + k
    
    total_capacity_bits = total_blocks * payload_per_block
    bpp = payload_per_block / N

    print(f"========== 實驗設定與容量 ==========")
    print(f"載入圖片尺寸：{img_w} x {img_h} (共 {total_pixels} 像素)")
    print(f"總像素區塊 N = {N}")
    print(f"劃分群組數 k = {k}")
    print(f"每像素嵌入容量 (bpp)   : {bpp:.4f} bits/pixel")
    print(f"圖片總容量             : {total_capacity_bits} bits ({total_capacity_bits / 8192:.2f} KB)\n")

    # 2. 策略 A：Symmetric Partitioning
    bal_part = get_balanced_partition(N, k)
    stego_bal = simulate_stego_image(cover_image, bal_part, N)
    bal_emp_mse, bal_emp_psnr = calculate_empirical_metrics(cover_image, stego_bal)

    print(f"[策略 A：Symmetric Partitioning]")
    print(f"  > 分組配置      : {bal_part}")
    print(f"  > 加密後實測 MSE  : {bal_emp_mse:.4f}")
    print(f"  > 加密後實測 PSNR : {bal_emp_psnr:.2f} dB\n")

    # 3. 策略 B：Asymmetric Partitioning
    oud_part = get_oud_partition(N, k)
    stego_oud = simulate_stego_image(cover_image, oud_part, N)
    oud_emp_mse, oud_emp_psnr = calculate_empirical_metrics(cover_image, stego_oud)

    print(f"[策略 B：Asymmetric Partitioning]")
    print(f"  > 分組配置      : {oud_part}")
    print(f"  > 加密後實測 MSE  : {oud_emp_mse:.4f}")
    print(f"  > 加密後實測 PSNR : {oud_emp_psnr:.2f} dB\n")

    # 4. 結論比較
    improvement = oud_emp_psnr - bal_emp_psnr
    print(f"========== 結論 ==========")
    print(f"在維持相同嵌入容量 ({bpp:.2f} bpp) 的前提下，")
    print(f"Asymmetric 策略相較於傳統 Symmetric 方法，實際圖片 PSNR 提升了 {improvement:.2f} dB。")

if __name__ == "__main__":
    # 換成實驗圖片，並測試任意的 N 與 k
    evaluate_steganography_method("airplane.png", N=6, k=2)

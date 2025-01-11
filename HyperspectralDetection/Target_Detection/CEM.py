import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt


# 加载 Indian Pines 数据集
def load_indian_pines_data(mat_file,gt_file):
    data = scio.loadmat(mat_file)
    data_gt = scio.loadmat(gt_file)
    img = data['data']  # 高光谱图像数据 (shape: bands, height, width)
    gt = data_gt['groundT']  # 地面真实标签数据 (shape: height, width)
    return img, gt


# 计算误差图（使用像素差异作为示例）
def calculate_error_map(reference_image, target_image):
    # 计算每个像素的光谱差异，这里用绝对差异作为示例
    error_map = np.abs(reference_image - target_image)
    return error_map


# 粗检测：通过阈值进行二值化处理，标记出变化区域
def coarse_detection(error_map, threshold=0.1):
    # 如果误差大于阈值，则标记为变化区域
    detection_map = error_map > threshold
    return detection_map


# 保存粗检测结果到 .mat 文件
def save_detection_map(detection_map, output_file):
    # 保存粗检测标签为 .mat 文件
    scio.savemat(output_file, {'coarse_detection_map': detection_map})
    print(f"Coarse detection map saved to {output_file}")


# 可视化结果
def visualize_results(error_map, detection_map):
    plt.figure(figsize=(12, 6))

    # 显示误差图
    plt.subplot(1, 2, 1)
    plt.imshow(error_map, cmap='hot')
    plt.title("Error Map")
    plt.colorbar()

    # 显示粗检测结果
    plt.subplot(1, 2, 2)
    plt.imshow(detection_map, cmap='gray')
    plt.title("Coarse Detection Map")

    plt.show()


# 主程序
if __name__ == "__main__":
    # 加载数据
    img, gt = load_indian_pines_data(r"data/Indian_pines_corrected.mat",r"data/Indian_pines_gt.mat")  # 替换为您的 .mat 文件路径

    # 假设选择第一个波段作为参考图像，第二个波段作为目标图像
    reference_image = img[0, :, :]  # 参考图像
    target_image = img[1, :, :]  # 目标图像

    # 计算误差图
    error_map = calculate_error_map(reference_image, target_image)

    # 粗检测（设定阈值为0.1）
    detection_map = coarse_detection(error_map, threshold=0.1)

    # 可视化结果
    visualize_results(error_map, detection_map)

    # 保存粗检测标签到 .mat 文件
    save_detection_map(detection_map, r'coarse_det/Mosaic.mat')  # 保存路径为 'coarse_detection_map.mat'

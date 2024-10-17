import os
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import torch
import torchvision.transforms.functional as TF

# 获取文件夹下的子文件夹，并排除文件夹名包含 'tra_'
def get_subfolders(dir_path):
    return [
        os.path.join(dir_path, subfolder) 
        for subfolder in os.listdir(dir_path) 
        if os.path.isdir(os.path.join(dir_path, subfolder)) and 'tra_' not in subfolder
    ]

def sort_by_timestep(image_files):
    # 根据文件名中的 timestep 提取数字并排序
    return sorted(image_files, key=lambda x: int(x.split('_timestep')[-1].split('.')[0]))

# 计算两张图片的SSIM
def calculate_ssim(image1, image2):
    # 如果输入是 PIL Image，先转换为 numpy 数组
    if isinstance(image1, Image.Image):
        image1 = np.array(image1)
    if isinstance(image2, Image.Image):
        image2 = np.array(image2)
    
    # 转换为 RGB 格式
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    # 计算 SSIM
    return ssim(image1, image2, channel_axis=-1)

def load_npy_as_image(npy_file):
    # 加载 .npy 文件
    data = np.load(npy_file)
    
    # 假设数据的值范围是任意的，先归一化为 [0, 1]
    min_val = data.min()
    max_val = data.max()
    
    if max_val - min_val > 0:
        normalized_data = (data - min_val) / (max_val - min_val)
    else:
        normalized_data = data  # 防止全为0或其他异常情况
    
    # 转换为 [0, 255] 的 uint8 类型，并创建 PIL 图像
    img_data = (normalized_data * 255).astype(np.uint8)
    
    # 如果 .npy 数据是三维的，假设格式是 (C, H, W)，则需要将其转换为 (H, W, C)
    #if img_data.ndim == 3 and img_data.shape[0] in [1, 3]:  # 假设第一维是通道数
    #    img_data = np.transpose(img_data, (1, 2, 0))  # 将通道数移动到最后

    return Image.fromarray(img_data)

# 计算一个文件夹下所有相邻时间步图片的 SSIM
def calculate_ssim_for_folder(image_folder):
    # 获取所有 .npy 文件
    npy_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.npy')]
    
    # 按照时间步排序
    npy_files = sort_by_timestep(npy_files)
    
    ssim_values = []
    
    print(f"开始处理文件夹: {image_folder} ({len(npy_files)} 个文件)")
    
    for i in range(len(npy_files) - 1):
        print(f"计算 {npy_files[i]} 和 {npy_files[i+1]} 的 SSIM...")

        # 将相邻的 .npy 文件加载为图像
        img1 = load_npy_as_image(npy_files[i])
        img2 = load_npy_as_image(npy_files[i + 1])
        
        # 计算 SSIM
        ssim_val = calculate_ssim(img1, img2)
        ssim_values.append(ssim_val)
    
    print(f"完成处理文件夹: {image_folder}\n")
    return ssim_values

# 处理fake和real文件夹下的所有子文件夹
def process_folders(base_dir, output_file):
    fake_dir = os.path.join(base_dir, 'fake')
    real_dir = os.path.join(base_dir, 'real')

    # 获取文件保存路径的目录
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def process_subfolders(base_dir, label):
        data = []
        subfolders = get_subfolders(base_dir)
        
        for subfolder_idx, subfolder in enumerate(subfolders):
            print(f"处理子文件夹 {subfolder_idx + 1}/{len(subfolders)}: {subfolder}")
            ssim_values = calculate_ssim_for_folder(subfolder)
            row = [label] + ssim_values  # 第一列是标签，后面是SSIM数据
            data.append(row)
        
        return data

    # 处理fake和real文件夹
    print("开始处理 fake 文件夹...")
    fake_data = process_subfolders(fake_dir, 0)  # fake文件夹，标签为0

    print("开始处理 real 文件夹...")
    real_data = process_subfolders(real_dir, 1)  # real文件夹，标签为1

    # 合并数据
    all_data = fake_data + real_data

    # 创建DataFrame
    df = pd.DataFrame(all_data)

    # 切块的话，不能打乱打乱数据行
    #df = df.sample(frac=1).reset_index(drop=True)

    # 保存为TSV文件
    df.to_csv(output_file, sep='\t', index=False,header=False)

    print(f"SSIM数据已保存到 {output_file}")

# 执行 Train 数据的处理
# 遍历 segments1 到 segments40 文件夹并处理
for segment_index in range(1, 11):
    # 设置 train_base_dir 和 train_output_file
    train_base_dir = f'/data/usr/lhr/Time_shapelet/SLIC/eps_10/segments{segment_index}/Train'
    train_output_file = f'/data/usr/lhr/Time_shapelet/Time_series/SSIM_eps_10/segments{segment_index}/Deepfacegen_TRAIN.tsv'
    # 调用处理函数处理每个 segments 目录
    process_folders(train_base_dir, train_output_file)


# 执行 Test 数据的处理
# 遍历 segments1 到 segments40 文件夹并处理
for segment_index in range(1, 11):
    # 设置 train_base_dir 和 train_output_file
    test_base_dir = f'/data/usr/lhr/Time_shapelet/SLIC/eps_10/segments{segment_index}/Test'
    test_output_file = f'/data/usr/lhr/Time_shapelet/Time_series/SSIM_eps_10/segments{segment_index}/Deepfacegen_TEST.tsv'
    # 调用处理函数处理每个 segments 目录
    process_folders(test_base_dir, test_output_file)

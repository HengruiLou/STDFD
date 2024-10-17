# 步骤 1：导入必要的库
import os
import numpy as np
from skimage import segmentation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# 步骤 2：定义路径和参数

# 原始数据路径
input_root = '/data/usr/lhr/SD_CODE/DIRE-main/guided-diffusion/eps'

# 输出数据路径
output_root = '/data/usr/lhr/Time_DFactor/SLIC/eps_10'

# 期望的超像素数量
M = 25
M_actual=10
#实际上是达不到M个的，用聚类聚类成统计的40个

# 时间步数
T = 20

# 数据集类型
datasets = ['Train', 'Test']

# 图像类型
types = ['fake', 'real']

# 步骤 3：创建输出目录结构

# 创建输出根目录
if not os.path.exists(output_root):
    os.makedirs(output_root)

# 创建 segments1 到 segments40 文件夹
for i in range(1, M_actual + 1):
    segment_dir = os.path.join(output_root, f'segments{i}')
    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)
    # 创建 Train 和 Test 文件夹
    for dataset in datasets:
        dataset_dir = os.path.join(segment_dir, dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        # 创建 fake 和 real 文件夹
        for t in types:
            type_dir = os.path.join(dataset_dir, t)
            if not os.path.exists(type_dir):
                os.makedirs(type_dir)

# 步骤 4：遍历原始数据并处理

for dataset in datasets:
    for t in types:
        # 输入文件夹路径
        input_folder = os.path.join(input_root, dataset, t)
        # 遍历每个样本文件夹
        for sample_name in os.listdir(input_folder):
            # 如果 sample_name 包含 "tra_"，则跳过
            if "tra_" in sample_name:
                print(f'跳过 {sample_name}，因为包含 "tra_"')
                continue  # 跳过该 sample_name
            sample_input_dir = os.path.join(input_folder, sample_name)
            if not os.path.isdir(sample_input_dir):
                continue
            print(f'正在处理 {dataset}/{t}/{sample_name}')
            
            # 加载 20 个时间步的图像数据
            images = []
            for timestep in range(T):
                npy_file = os.path.join(sample_input_dir, f'eps_timestep{timestep}.npy')
                if not os.path.exists(npy_file):
                    print(f'文件未找到：{npy_file}')
                    continue
                data = np.load(npy_file)
                # 处理数据，转换为图像格式
                # 假设 data 的形状为 (3, H, W)，需要转换为 (H, W, 3)
                if data.ndim == 3:
                    if data.shape[0] == 3:
                        image = np.transpose(data, (1, 2, 0))
                    else:
                        image = data
                else:
                    image = data
                # 将数据归一化到 [0, 1]
                image = image.astype(np.float32)
                data_min = image.min()
                data_max = image.max()
                if data_max > data_min:
                    image = (image - data_min) / (data_max - data_min)
                else:
                    image = np.zeros_like(image)
                images.append(image)
            
            # 检查是否成功加载了所有时间步的图像
            if len(images) != T:
                print(f'警告：加载的图像数量 ({len(images)}) 与预期的时间步数 ({T}) 不符，跳过该样本。')
                continue  # 跳过该样本
            
            # 在第一个时间步的图像上进行 SLIC 超像素分割
            base_image = images[0]
            segments = segmentation.slic(
                base_image,
                n_segments=M,
                compactness=10,
                sigma=1,
                start_label=1
            )
            
            # 获取实际的超像素数量
            labels = np.unique(segments)
            num_superpixels = len(labels)
            print(f'超像素数量：{num_superpixels}')
            
            if num_superpixels != M_actual:
                print(f'超像素数量 ({num_superpixels}) 不等于期望数量 ({M_actual})，执行聚类。')

                # 提取超像素特征：包括颜色和空间位置
                superpixel_features = []
                for label in labels:
                    # 提取每个超像素的掩码
                    mask = segments == label
                    
                    # 计算每个超像素区域的平均颜色
                    mean_color = base_image[mask].mean(axis=0)
                    
                    # 计算每个超像素区域的质心位置
                    coords = np.argwhere(mask)
                    centroid = coords.mean(axis=0)  # 质心的坐标 (y, x)
                    
                    # 将颜色特征和位置信息合并
                    feature = np.concatenate([mean_color, centroid])
                    superpixel_features.append(feature)

                superpixel_features = np.array(superpixel_features)

                # 对颜色和位置进行标准化，使得两者在聚类时有相同的权重
                scaler = StandardScaler()
                superpixel_features_scaled = scaler.fit_transform(superpixel_features)

                # 使用 KMeans 聚类，得到 M 个聚类中心
                kmeans = KMeans(n_clusters=M_actual, random_state=42)
                kmeans.fit(superpixel_features_scaled)

                # 根据聚类结果更新 segments
                new_segments = np.zeros_like(segments)
                for i, label in enumerate(labels):
                    mask = segments == label
                    new_label = kmeans.labels_[i] + 1  # 保证标签从 1 开始
                    new_segments[mask] = new_label

                segments = new_segments
                labels = np.unique(segments)
            else:
                print(f'超像素数量等于期望数量 ({M_actual})。')

            # 提取每个超像素区域的数据并保存
            for segment_label in labels:
                mask = segments == segment_label  # 超像素掩码
                
                # 创建对应的输出文件夹
                segment_index = segment_label  # 标签从 1 开始
                segment_output_dir = os.path.join(
                    output_root,
                    f'segments{segment_index}',
                    dataset,
                    t,
                    sample_name
                )
                if not os.path.exists(segment_output_dir):
                    os.makedirs(segment_output_dir)
                
                # 遍历每个时间步，提取并保存超像素区域
                for timestep, image in enumerate(images):
                    # 提取当前超像素区域
                    
                    # 获取掩码下的坐标位置
                    coords = np.argwhere(mask)
                    y0, x0 = coords.min(axis=0)  # 最小边界
                    y1, x1 = coords.max(axis=0) + 1  # 最大边界
                    
                    # 裁剪出包含超像素区域的最小矩形
                    superpixel_cropped = image[y0:y1, x0:x1]
                    mask_cropped = mask[y0:y1, x0:x1]
                    
                    # 将超像素块外的区域设置为 0
                    superpixel_cropped[~mask_cropped] = 0
                    
                    # 保存裁剪后的超像素块为 .npy 文件
                    output_file = os.path.join(segment_output_dir, f'eps_timestep{timestep}.npy')
                    np.save(output_file, superpixel_cropped)

import os
import pickle
import numpy as np
from archive.load_usr_dataset import load_usr_dataset_by_name
from utils.mp_utils import ParMap, parallel_monitor, NJOBS
import argparse
import warnings
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.autograd import *
from torch import optim
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from utils.base_utils import Queue
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from utils.base_utils import Debugger, syscmd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid, cross_val_score
# 手动遍历参数网格，实时打印训练进度和 ACC 信息
from sklearn.model_selection import KFold
import json
from scipy.stats import skew, kurtosis
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from tslearn.metrics import dtw
from scipy.signal import correlate
# 定义一个自定义的 scoring 函数 (用 accuracy 作为评价指标)
def custom_accuracy_scorer(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def load_and_cluster_DFactors(n_clusters, num_segments,cluster_enable):
    """
    加载多个 .cache 文件中的 DFactor 并对其进行聚类
    :param n_clusters: 聚类的类别数量
    :param num_segments: 要加载的 .cache 文件的数量（从 segments1.cache 到 segmentsN.cache）
    :return: 聚类中心和每个 DFactor 的标签
    """
    # 初始化列表用于存储所有的 DFactor (cand)
    all_DFactors = []

    # Step 1: 加载40个文件并提取DFactor (cand)
    for segment_index in range(1, num_segments+1):  # segments1 到 segments40
        fpath = f'/data/usr/lhr/Time_DFactor/DFactor_global/DFactor_cache_10segfor2/segments{segment_index}.cache'
        
        # 检查文件是否存在
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
                #print(f"文件 {fpath} 已成功加载。")
                
                # 提取第一个元组的第一个元素 (cand)
                if len(data) > 0:
                    cand, local_factor, global_factor, loss = data[0]  # 只提取第一个元组
                    all_DFactors.append(cand)  # 将 cand（DFactor）添加到列表中
                    
                    # 打印 DFactor 的信息
                    #print(f"  文件 {fpath} 最优 DFactor (cand):  {cand}")
                else:
                    print(f"文件 {fpath} 中没有数据")
        else:
            print(f"文件 {fpath} 不存在")
    
    # 检查是否有 DFactor 提取出来
    if len(all_DFactors) == 0:
        print("没有 DFactor 被提取出来。")
        return None, None

    # 将所有 DFactor 转换为 numpy 数组
    all_DFactors_array = np.array(all_DFactors)
    print(f"总共提取到 {len(all_DFactors_array)} 个 DFactor.")
    if cluster_enable:
        # Step 2: 基于提取的 DFactor 进行聚类 (使用KMeans算法聚成 n_clusters 类)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        # 将三维数组重塑为二维数组
        all_DFactors_array_reshaped = all_DFactors_array.reshape(num_segments, -1)
        #print(all_DFactors_array_reshaped)
        # 进行聚类
        kmeans.fit(all_DFactors_array_reshaped)

        # 聚类结果：每个 DFactor 的标签
        labels = kmeans.labels_

        # 聚类中心：n_clusters 个聚类中心 (典型DFactor)
        cluster_centers = kmeans.cluster_centers_
        print("cluster_centers.shape",cluster_centers.shape)
        # Step 3: 返回聚类中心和标签
        print(f"聚类完成，生成了 {n_clusters} 个聚类中心。")
        return cluster_centers, labels
    else:
        #print("all_DFactors_array",all_DFactors_array)
        all_DFactors=all_DFactors_array.reshape(num_segments, -1)
        #print("all_DFactors",all_DFactors)
        return  all_DFactors , 0


def extract_DFactors(x_data, y_data, num_segment, segment_length):
    """
    从输入的时间序列数据中提取所有 DFactor，并保持与标签的关联。
    :param x_data: 时间序列数据，形状为 (N, S, 1)，N 为样本数，S 为时间序列长度
    :param y_data: 样本标签，形状为 (N,)
    :param num_segment: 切分的段数
    :param segment_length: 每段的长度
    :return: DFactor_x_by_sample, DFactor_y_by_sample，每个样本的 DFactor 及其对应的标签
    """
    N, S, _ = x_data.shape  # N: 样本数, S: 时间序列长度
    DFactor_x_by_sample = []  # 存储每条样本的 DFactor
    DFactor_y_by_sample = []  # 存储每条样本对应的标签

    # 遍历每个样本
    for i in range(N):
        sample = x_data[i].squeeze()  # 去掉最后的单通道维度 (S,)
        label = y_data[i]  # 对应的标签
        sample_DFactors = []  # 存储当前样本的所有 DFactor

        # 按照 num_segment 和 segment_length 切分每条样本
        for segment_idx in range(num_segment):
            start = segment_idx * segment_length
            end = start + segment_length
            if end <= S:  # 确保片段不超出边界
                DFactor = sample[start:end]  # 提取 DFactor
                sample_DFactors.append(DFactor)  # 保存 DFactor
        
        # 保存每个样本的所有 DFactor 和标签
        DFactor_x_by_sample.append(sample_DFactors)
        DFactor_y_by_sample.append(label)

    return DFactor_x_by_sample, DFactor_y_by_sample


def compute_feature_vectors(DFactor_x, DFactor_y, cluster_centers, metric='euclidean', smash=1,cluster=False):
    """
    计算每个样本的特征向量，将每个聚类中心的特征拼接成一个整体特征向量。
    :param DFactor_x: 提取出的 DFactor 数据，形状为 (num_DFactors * smash, DFactor_length)
    :param DFactor_y: 对应的样本标签
    :param cluster_centers: 聚类中心，形状为 (n_clusters, DFactor_length)
    :param metric: 距离度量方法
    :param smash: 表示每个样本由多少块合并而成
    :return: feature_vectors (每个样本的特征向量) 和 labels (每个样本的标签)
    """
    num_DFactors_total = len(DFactor_x)  # 总 DFactor 数量
    origin_num = int(len(DFactor_y)/smash)  # 原始样本数量
    print("总 DFactor 数量",num_DFactors_total)
    print("原始样本数量",origin_num)
    n_clusters = cluster_centers.shape[0]  # 聚类中心数目

    # 初始化存储每个样本的特征向量
    feature_vectors = []
    sample_labels = []
    if cluster:
        # 遍历每个原始样本
        for i in range(origin_num):
            DFactors = []  # 用于存储当前样本的 DFactor 块

            # 按跨越 origin_num 的方式提取属于该样本的 DFactor
            for j in range(smash):
                #print("原始标签:{},跨越 origin_num标签是{}".format(DFactor_y[i],DFactor_y[i + j * origin_num]))
                DFactors.append(DFactor_x[i + j * origin_num])  # 提取该样本的 DFactor 块

            label = DFactor_y[i]  # 获取该样本的标签
            sample_feature_vector = []  # 用于存储该样本的特征

            # 对每个聚类中心计算特征
            for j in range(n_clusters):
                cluster_DFactor_distances = []  # 记录与该聚类中心的所有 DFactor 距离
                cluster_DFactor_count = 0  # 记录属于该聚类中心的 DFactor 数量

                # 遍历该样本的所有 DFactor，计算与聚类中心 j 的距离
                for DFactor in DFactors:
                    #DFactor = np.array(DFactor).reshape(1, -1)  # 确保 DFactor 是二维数组
                    dists = pairwise_distances(DFactor, cluster_centers, metric=metric)[0]

                    # 如果当前 DFactor 属于聚类中心 j，则记录该距离
                    if np.argmin(dists) == j:
                        cluster_DFactor_distances.append(dists[j])
                        cluster_DFactor_count += 1

                # 计算聚类中心 j 的特征
                #avg_distance = np.mean(cluster_DFactor_distances) if cluster_DFactor_count > 0 else 0
                # 计算聚类中心 j 的特征：平均距离、最小距离、最大距离
                avg_distance = np.mean(cluster_DFactor_distances) if cluster_DFactor_count > 0 else 0
                min_distance = np.min(cluster_DFactor_distances) if cluster_DFactor_count > 0 else 0
                max_distance = np.max(cluster_DFactor_distances) if cluster_DFactor_count > 0 else 0

                # 将该聚类中心的特征拼接到特征向量中
                cluster_feature_vector = np.hstack((
                    #cluster_DFactor_count,        # 该聚类中心的 DFactor 数量
                    avg_distance,                  # 该聚类中心的平均距离
                    min_distance,                  # 该聚类中心的最小距离
                    max_distance,                  # 该聚类中心的最大距离
                    #cluster_centers[j].flatten()   # 该聚类中心本身的数值
                ))
                sample_feature_vector.append(cluster_feature_vector)

            # 拼接所有聚类中心的特征向量
            feature_vectors.append(np.hstack(sample_feature_vector))
            sample_labels.append(label)

        return np.array(feature_vectors), np.array(sample_labels)
    else:
         # 遍历每个原始样本
        for i in range(origin_num):
            DFactors = []  # 用于存储当前样本的 DFactor 块

            # 按跨越 origin_num 的方式提取属于该样本的 DFactor
            for j in range(smash):
                DFactors.append(DFactor_x[i + j * origin_num])  # 提取该样本的 DFactor 块

            label = DFactor_y[i]  # 获取该样本的标签
            sample_feature_vector = []  # 用于存储该样本的特征

            # 假设聚类中心的数量等于 smash
            for j in range(smash):
                DFactor = DFactors[j]  # 第 j 个块
                cluster_center = cluster_centers[j]  # 第 j 个聚类中心

                # DFactor 和 cluster_center 都是 1D 数组
                DFactor_reshaped = np.array(DFactor).reshape(-1)  # 将 DFactor 转为 1D 数组
                cluster_center_reshaped = np.array(cluster_center).reshape(-1)  # 将聚类中心转为 1D 数组
                n_features = cluster_center_reshaped.shape[0]  # 聚类中心的长度
                # 检查 DFactor_reshaped 的长度
                total_length = DFactor_reshaped.shape[0]
                if total_length % n_features != 0:
                    print(f"第 {j} 个块的 DFactor_reshaped 长度不是聚类中心长度的整数倍，无法拆分")
                    continue  # 或者进行其他处理

                n_DFactors = total_length // n_features  # 子数组的数量


                # 计算距离
                # 如果 cluster_center_reshaped 只有一个元素，可以直接进行向量化计算
                if cluster_center_reshaped.size == 1:
                    # 输出距离信息
                    print(f"第 {j} 个块的 DFactor 是 {DFactor_reshaped}, 聚类中心是 {cluster_center_reshaped}")
                    distances = pairwise_distances(
                        DFactor_reshaped.reshape(-1, 1),
                        cluster_center_reshaped.reshape(1, -1),
                        metric=metric
                    ).flatten()
                    print(f"第 {j} 个块的距离是 {distances}")
                    # 计算平均距离、最小距离和最大距离
                    avg_distance = np.mean(distances)  # 平均距离
                    min_distance = np.min(distances)   # 最小距离
                    max_distance = np.max(distances)   # 最大距离
                    median_distance= np.median(distances)#距离的中位数
                    std_distance= np.std(distances)#距离的标准差。

                    #偏度（Skewness）和峰度（Kurtosis
                    skewness=skew(distances)
                    kurto= kurtosis(distances)

                    #分布拟合参数
                    mu, std = norm.fit(distances)
                    fit_mu=mu
                    fit_std=std


                    #分位数特征
                    quantile_25=np.percentile(distances, 25)
                    quantile_50=np.percentile(distances, 50)
                    quantile_75=np.percentile(distances, 75)

                    # 记录该块的特征向量
                    cluster_feature_vector = np.hstack((
                        avg_distance,  # 平均距离
                        min_distance,  # 最小距离
                        max_distance,   # 最大距离
                        median_distance,
                        std_distance,
                        skewness,
                        kurto,
                        fit_mu,
                        fit_std,
                        quantile_25,
                        quantile_50,
                        quantile_75
                    ))
                    sample_feature_vector.append(cluster_feature_vector)
                else:
                    # 将 DFactor_reshaped 重塑为二维数组
                    DFactor_reshaped = DFactor_reshaped.reshape(n_DFactors, n_features)
                    # 输出拆分结果
                    print(f"第 {j} 个块的 DFactor 是:\n{DFactor_reshaped}, 聚类中心是: {cluster_center_reshaped}")

                    # 如果 cluster_center_reshaped 有多个元素，需要确保维度匹配
                    distances = pairwise_distances(
                        DFactor_reshaped,
                        cluster_center_reshaped.reshape(1, -1),
                        metric=metric
                    ).flatten()
                
                    print(f"第 {j} 个块的距离是 {distances}")
                    # 计算平均距离、最小距离和最大距离
                    avg_distance = np.mean(distances)  # 平均距离
                    min_distance = np.min(distances)   # 最小距离
                    max_distance = np.max(distances)   # 最大距离
                    median_distance= np.median(distances)#距离的中位数
                    std_distance= np.std(distances)#距离的标准差。

                    #偏度（Skewness）和峰度（Kurtosis
                    skewness=skew(distances)
                    kurto= kurtosis(distances)

                    #分布拟合参数
                    mu, std = norm.fit(distances)
                    fit_mu=mu
                    fit_std=std

                    #分位数特征
                    quantile_25=np.percentile(distances, 25)
                    quantile_50=np.percentile(distances, 50)
                    quantile_75=np.percentile(distances, 75)

                    # Step 1: 计算差分序列
                    delta_cluster = np.diff(cluster_center_reshaped)  # 形状为 (n_features - 1,)

                    delta_DFactor = np.diff(DFactor_reshaped, axis=1)  # 形状为 (n_DFactors, n_features - 1)

                    # Step 2a: 计算趋势方向
                    trend_cluster = np.sign(delta_cluster)  # 形状为 (n_features - 1,)
                    trend_DFactor = np.sign(delta_DFactor)  # 形状为 (n_DFactors, n_features - 1)

                    # Step 2b: 计算趋势匹配率
                    trend_cluster_expanded = trend_cluster.reshape(1, -1)  # 扩展维度
                    trend_matches = (trend_DFactor == trend_cluster_expanded)
                    trend_match_ratio = np.mean(trend_matches, axis=1)  # 对每个 DFactor 计算匹配率

                    # Step 2c: 计算趋势相关性
                    trend_correlation = []
                    for delta_s in delta_DFactor:
                        if np.std(delta_s) == 0 or np.std(delta_cluster) == 0:
                            corr_coef = 0  # 如果标准差为零，相关系数设为零
                        else:
                            corr_coef, _ = pearsonr(delta_s, delta_cluster)
                        trend_correlation.append(corr_coef)
                    trend_correlation = np.array(trend_correlation)

                    # Step 2d: 计算趋势距离
                    trend_distances = np.linalg.norm(delta_DFactor - delta_cluster.reshape(1, -1), axis=1)
                    
                    #斯皮尔曼相关系数
                    trend_spearman = []
                    for delta_s in delta_DFactor:
                        if np.std(delta_s) == 0 or np.std(delta_cluster) == 0:
                            corr_coef = 0
                        else:
                            corr_coef, _ = spearmanr(delta_s, delta_cluster)
                        trend_spearman.append(corr_coef)
                    trend_spearman = np.array(trend_spearman)
                    
                    #Kendall’s tau
                    trend_kendall = []
                    for delta_s in delta_DFactor:
                        if np.std(delta_s) == 0 or np.std(delta_cluster) == 0:
                            corr_coef = 0
                        else:
                            corr_coef, _ = kendalltau(delta_s, delta_cluster)
                        trend_kendall.append(corr_coef)
                    trend_kendall = np.array(trend_kendall)
                    
                    #编辑距离（Edit Distance on Real sequence，EDR）
                    trend_dtw = []
                    for delta_s in delta_DFactor:
                        dist = dtw(delta_s, delta_cluster)
                        trend_dtw.append(dist)
                    trend_dtw = np.array(trend_dtw)

                    #模版匹配
                    template_matches = []
                    for s in DFactor_reshaped:
                        # 计算均值和标准差
                        mean_s = np.mean(s)
                        mean_center = np.mean(cluster_center_reshaped)
                        std_s = np.std(s)
                        std_center = np.std(cluster_center_reshaped)
                        
                        # 计算互相关
                        ncc = correlate(s - mean_s, cluster_center_reshaped - mean_center)
                        
                        # 检查标准差是否为零，防止除以零错误
                        denominator = len(s) * std_s * std_center
                        if denominator == 0:
                            print("警告：标准差为零，归一化互相关无法计算，将结果设为零。")
                            ncc_normalized = np.zeros_like(ncc)  # 或者根据需要设置为 np.nan
                        else:
                            ncc_normalized = ncc / denominator
                        
                        # 取最大归一化互相关值
                        max_ncc = np.max(ncc_normalized)
                        template_matches.append(max_ncc)

                    template_matches = np.array(template_matches)
                    

                    

                    # 记录该块的特征向量
                    cluster_feature_vector = np.hstack((
                        avg_distance,  # 平均距离
                        min_distance,  # 最小距离
                        max_distance,   # 最大距离
                        #median_distance,
                        #std_distance,
                        #skewness,
                        #kurto,
                        #fit_mu,
                        #fit_std,
                        quantile_25,
                        quantile_50,
                        quantile_75,
                        # 添加趋势特征的统计量
                        np.mean(trend_match_ratio),  # 趋势匹配率的平均值
                        np.min(trend_match_ratio),   # 趋势匹配率的最小值
                        np.max(trend_match_ratio),   # 趋势匹配率的最大值
                        np.mean(trend_correlation),  # 趋势相关性的平均值
                        np.min(trend_correlation),   # 趋势相关性的最小值
                        np.max(trend_correlation),   # 趋势相关性的最大值
                        np.mean(trend_distances),    # 趋势距离的平均值
                        np.min(trend_distances),     # 趋势距离的最小值
                        np.max(trend_distances),      # 趋势距离的最大值

                        # 添加斯皮尔曼相关系数的统计量
                        np.mean(trend_spearman),
                        np.min(trend_spearman),
                        np.max(trend_spearman),
                        # 添加 Kendall’s tau 的统计量
                        np.mean(trend_kendall),
                        np.min(trend_kendall),
                        np.max(trend_kendall),
                        # 添加 DTW 距离的统计量
                        np.mean(trend_dtw),
                        np.min(trend_dtw),
                        np.max(trend_dtw),
                        # 添加模板匹配的统计量
                        np.mean(template_matches),
                        np.min(template_matches),
                        np.max(template_matches)

                    ))
                    sample_feature_vector.append(cluster_feature_vector)

            # 拼接所有块的特征向量
            feature_vectors.append(np.hstack(sample_feature_vector))
            sample_labels.append(label)

        return np.array(feature_vectors), np.array(sample_labels)



def save_results_to_file(best_params, evaluation_metrics, output_path="model_results.json"):
    # 初始化结果字典
    new_results = {
        "best_params": best_params,
        "evaluation_metrics": evaluation_metrics
    }

    # 检查文件是否存在
    if os.path.exists(output_path):
        # 读取已有的结果
        with open(output_path, 'r') as f:
            try:
                existing_results = json.load(f)
                # 如果现有结果是一个字典而不是列表，将它转变为列表
                if isinstance(existing_results, dict):
                    existing_results = [existing_results]
            except json.JSONDecodeError:
                existing_results = []  # 如果文件为空或格式错误，初始化为空列表
    else:
        existing_results = []

    # 将新的结果添加到已有的结果中
    existing_results.append(new_results)

    # 将更新后的结果写回文件
    with open(output_path, 'w') as f:
        json.dump(existing_results, f, indent=4)

    print(f"评估结果已追加到 {output_path}")



    
def evaluate_model(true_labels, pred_labels, pred_prob):
    acc = accuracy_score(true_labels, pred_labels)
    auc = roc_auc_score(true_labels, pred_prob)
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)

    print("\n测试集评估结果:")
    print(f"Accuracy: {acc}")
    print(f"AUC: {auc}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    # 返回评估指标以便保存
    metrics = {
        "accuracy": acc,
        "AUC": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    return metrics
if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='ucr-Deepfacegen',help='ucr-Earthquakes/WormsTwoClass/Strawberry')#用到了
    parser.add_argument('--num_segment', type=int, default=12, help='number of segment a time series is divided into')#用到了
    parser.add_argument('--seg_length', type=int, default=30, help='segment length')#用到了
    parser.add_argument('--quantitative', type=str, default='Seq_SSIM',help='quantitative value')#定量，用到了
    parser.add_argument('--datapath', type=str, default='/data/usr/lhr/Time_DFactor/Time_series',help='数据路径')#定量，用到了
    parser.add_argument('--smash', type=int, default=40,help='smash number')#定量，用到了
    parser.add_argument('--n_clusters', type=int, default=5,help='cluster number')#定量，用到了
    parser.add_argument('--dis_metric',choices=['euclidean', 'cosine', 'manhattan'],default='euclidean',help="选择距离度量方法，'euclidean', 'cosine', 或 'manhattan'，默认是 'euclidean'")
    parser.add_argument('--cls', choices=['logistic', 'random_forest', 'svm', 'xgboost'], default='xgboost', help="选择分类器：'logistic', 'random_forest', 'svm', 'xgboost'，默认是 'logistic'")
    parser.add_argument('--cluster_enable', action='store_true', default=False, help='bool, whether to use cluster')
    args = parser.parse_args()
    Debugger.info_print('running with {}'.format(args.__dict__))

       
    # 调用
    n_clusters = args.n_clusters  # 聚类的数量
    num = args.smash  # 要加载的文件数量 (segments1 到 segments40)
    cluster_centers, labels = load_and_cluster_DFactors(n_clusters, num,args.cluster_enable)

    # 如果需要，可以打印聚类中心
    if cluster_centers is not None and args.cluster_enable:
        for i, center in enumerate(cluster_centers):
            print(f"聚类中心 {i+1} 的形状: {center.shape}, 数据内容:\n{center}")
    

    # 定义列表用于存储所有的训练集和测试集数据
    x_train_total, y_train_total, x_test_total, y_test_total = [], [], [], []

    if args.dataset.startswith('ucr'):
        dataset = args.dataset.rstrip('\n\r').split('-')[-1]
        quantitative = args.quantitative
        
        # 循环 smash 次，加载对应的 segments 数据
        for i in range(1, args.smash + 1):
            seg = f'segments{i}'
            
            # 每次加载 segments 数据
            x_train, y_train, x_test, y_test = load_usr_dataset_by_name(
                fname=dataset, 
                length=args.seg_length * args.num_segment,
                quantitative=quantitative, 
                seg=seg,
                dir_path=args.datapath
            )

            #print(f"Segment {seg} - y_train.shape:", y_train.shape)
            # 确保 y_train, y_test 是一维的
            y_train = np.ravel(y_train)  # 或者使用 y_train = y_train.flatten()
            y_test = np.ravel(y_test)

            # 检查每个 segment 的数据是否与标签对应
            #print(f"Segment {seg} 标签检查：前5个y_train: {y_train[:5]}")
            #print(f"Segment {seg} 数据检查：前5个x_train: {x_train[:2]}")
            # 将每次加载的数据 append 到总的数据列表中
            x_train_total.append(x_train)
            y_train_total.append(y_train)
            x_test_total.append(x_test)
            y_test_total.append(y_test)


            '''
            # 打印正样本比例
            Debugger.info_print('Segment {} training: {:.2f} positive ratio with {}'.format(
                seg, float(sum(y_train) / len(y_train)), len(y_train)))
            Debugger.info_print('Segment {} test: {:.2f} positive ratio with {}'.format(
                seg, float(sum(y_test) / len(y_test)), len(y_test)))

            '''

    else:
        raise NotImplementedError()

    # 合并数据到 numpy 数组形式
    x_train_total = np.concatenate(x_train_total, axis=0)  # 合并所有训练集的 x 数据
    y_train_total = np.concatenate(y_train_total, axis=0)  # 合并所有训练集的 y 数据
    x_test_total = np.concatenate(x_test_total, axis=0)    # 合并所有测试集的 x 数据
    y_test_total = np.concatenate(y_test_total, axis=0)    # 合并所有测试集的 y 数据
    #print("y_train_total.shape", y_train_total.shape)
    # 标签匹配检查
    #print("合并后 y_train_total 的 shape:", y_train_total.shape)
    #print("合并后 x_train_total 的 shape:", x_train_total.shape)




    #print("x_train_total.shape",x_train_total.shape)
    #print("y_train_total.shape",x_train_total.shape)


    # 打印最终合并后的正样本比例
    Debugger.info_print('Total training: {:.2f} positive ratio with {}'.format(
        float(sum(y_train_total) / len(y_train_total)), len(y_train_total)))
    Debugger.info_print('Total test: {:.2f} positive ratio with {}'.format(
        float(sum(y_test_total) / len(y_test_total)), len(y_test_total)))


     # 对训练集和测试集分别提取 DFactor
    DFactor_train_x, DFactor_train_y = extract_DFactors(x_train_total, y_train_total, int(x_train_total.shape[1] / args.seg_length), args.seg_length)
    DFactor_test_x, DFactor_test_y = extract_DFactors(x_test_total, y_test_total, int(x_test_total.shape[1] / args.seg_length), args.seg_length)

    '''
    dir_path=f'/data/usr/lhr/Time_DFactor/Train/seg/{args.seg}'
    # 如果目录不存在，创建目录
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    train_x=os.path.join(dir_path, 'train_x.npy')
    train_y=os.path.join(dir_path, 'train_y.npy')
    test_x=os.path.join(dir_path, 'test_x.npy')
    test_y=os.path.join(dir_path, 'test_y.npy')

    save_DFactor_to_local(DFactor_train_x, train_x, format='npy')  # 保存为 npy 格式
    save_DFactor_to_local(DFactor_train_y, train_y, format='npy')  # 保存为 npy 格式
    save_DFactor_to_local(DFactor_test_x, test_x, format='npy')  # 保存为 npy 格式
    save_DFactor_to_local(DFactor_test_y, test_y, format='npy')  # 保存为 npy 格式
    '''

    dis_metric=args.dis_metric
    train_feature_vectors, train_labels = compute_feature_vectors(DFactor_train_x, DFactor_train_y,cluster_centers, metric=dis_metric,smash=args.smash,cluster=args.cluster_enable)
    test_feature_vectors, test_labels = compute_feature_vectors(DFactor_test_x, DFactor_test_y, cluster_centers,metric=dis_metric,smash=args.smash,cluster=args.cluster_enable)
    # 输出形状
    print(f"features_x 形状: {train_feature_vectors.shape}")
    print(f"labels_y 形状: {train_labels.shape}")

    # 输出前5行数据
    print("\nfeatures_x 前3行数据:")
    print(train_feature_vectors[:3])

    print("\nlabels_y 前3行数据:")
    print(train_labels[:3])

    # 标准化特征
    scaler = StandardScaler()
    train_feature_vectors = scaler.fit_transform(train_feature_vectors)
    test_feature_vectors = scaler.transform(test_feature_vectors)

    # 根据 args.cls 选择不同的分类器 
    if args.cls == 'logistic':
        model = LogisticRegression(random_state=42)
        param_grid = {
            'penalty': ['l1', 'l2'],  # 增加了 l1, l2 正则化
            'C': [pow(5, i) for i in range(-3, 3)],  # 从 5^-3 到 5^2 的正则化强度
            'intercept_scaling': [pow(5, i) for i in range(-3, 3)],  # 增加 intercept_scaling
            'solver': ['liblinear', 'lbfgs'],  # 增加解算器
            'class_weight': ['balanced', None]  # 增加 class_weight
        }

    elif args.cls == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],  # 树的数量
            'criterion': ['gini', 'entropy'],  # 使用 gini 或 entropy
            'max_features': ['auto', 'log2', None],  # 最大特征数
            'max_depth': [10, 25, 50],  # 树的最大深度
            'min_samples_split': [2, 4, 8],  # 每个节点最小样本分割数
            'min_samples_leaf': [1, 3, 5],  # 叶子节点最小样本数
            'class_weight': ['balanced', None]  # 增加 class_weight
        }

    elif args.cls == 'svm':
        model = SVC(probability=True, random_state=42)
        param_grid = {
            'C': [pow(2, i) for i in range(-2, 2)],  # C 值的范围
            'kernel': ['rbf', 'poly', 'sigmoid'],  # 增加更多的 kernel
            'class_weight': ['balanced', None]  # 增加 class_weight
        }

    elif args.cls == 'xgboost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss',objective='binary:logistic', random_state=42)
        param_grid = {
            'n_estimators': [100, 200,300,400],  # 树的数量
            'max_depth': [1, 2, 4, 8, 12, 16,18],  # 树的深度范围
            'learning_rate': [0.1, 0.2, 0.3,0.4],  # 学习率
            'booster': [ 'gbtree', 'dart'],  # 增加不同的 booster
            #'booster': ['gblinear', 'gbtree', 'dart'],  # 增加不同的 booster
            #'scale_pos_weight': [1, 10, 50, 100],  # class_weight 类似的参数,解决类别失衡的
        }

    
    '''
    # 使用 GridSearchCV 进行超参数调优
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(train_feature_vectors, train_labels)
    '''

    # 创建参数组合列表
    param_list = list(ParameterGrid(param_grid))
    total_params = len(param_list)
    best_score = -np.inf
    best_params = None

    print(f"总共需要评估 {total_params} 个参数组合。")

    for idx, params in enumerate(param_list):
        print(f"\n正在评估参数组合 {idx + 1}/{total_params}: {params}")

        # 设置模型参数
        model.set_params(**params)

        # 执行交叉验证
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, train_feature_vectors, train_labels, cv=cv, scoring='accuracy', n_jobs=-1)

        mean_score = cv_scores.mean()
        std_score = cv_scores.std()

        print(f"交叉验证准确率: {mean_score:.4f} (+/- {std_score:.4f})")

        # 更新最佳参数
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    print(f"\n最佳参数组合为: {best_params}")
    print(f"在交叉验证中的最佳准确率为: {best_score:.4f}")

    # 使用最佳参数在训练集上训练模型
    model.set_params(**best_params)
    model.fit(train_feature_vectors, train_labels)

    # 在测试集上进行预测
    predictions = model.predict(test_feature_vectors)
    probabilities = model.predict_proba(test_feature_vectors)[:, 1]
    # 评估模型
    evaluation_metrics = evaluate_model(test_labels, predictions, probabilities)

    # 保存最佳参数和评估结果到文件

    save_results_to_file(best_params, evaluation_metrics)

    '''
    # 输出最佳参数
    print(f"最佳参数: {grid_search.best_params_}")
    # 预测测试集
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(test_feature_vectors)
    probabilities = best_model.predict_proba(test_feature_vectors)[:, 1]
    
    # 评估结果
    evaluate_model(test_labels, predictions, probabilities)
    # 保存最佳参数和评估结果到文件
    save_results_to_file(best_params, evaluation_metrics, output_path="model_results.json")
    '''


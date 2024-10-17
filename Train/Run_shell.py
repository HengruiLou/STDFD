import subprocess
# 定义基础命令及参数
base_command = [
    "python", "/data/usr/lhr/Time_DFactor/Train/run.py", 
    "--K", "10", 
    "--C", "100", 
    "--num_segment", "19", 
    "--seg_length", "1", 
    "--gpu_enable", 
    "--quantitative", "SSIM_eps",
    "--smash", "20",
    "--n_clusters", "5",
    "--dis_metric", "euclidean",
    "--cls", "xgboost"
]

# 循环执行 --seg 参数为 segments1 到 segments40 的命令
for i in range(1, 41):
    seg_value = f"segments{i}"
    command = base_command + ["--seg", seg_value]
    
    # 打印正在执行的命令
    print(f"正在执行: {' '.join(command)}")
    
    # 使用 subprocess 运行命令
    subprocess.run(command)

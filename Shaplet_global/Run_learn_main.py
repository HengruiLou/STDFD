import subprocess

# 定义基础命令及参数
base_command = [
    "python", "learn_main.py", 
    "--K", "10", 
    "--C", "100", 
    "--num_segment", "9", 
    "--seg_length", "2", 
    "--data_size", "1", 
    "--embed", "concate", 
    "--percentile", "5", 
    "--gpu_enable", 
    "--quantitative", "SSIM_eps_10",
    '--output_path',"/data/usr/lhr/Time_shapelet/Shaplet_global/Shapelet_cache_10segfor2"
]

# 循环执行 --seg 参数为 segments1 到 segments40 的命令
for i in range(1, 11):
    seg_value = f"segments{i}"
    command = base_command + ["--seg", seg_value]
    
    # 打印正在执行的命令
    print(f"正在执行: {' '.join(command)}")
    
    # 使用 subprocess 运行命令
    subprocess.run(command)

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 建议安装: pip install tqdm，用于显示进度条

# ================= 1. 核心工具函数 (你的代码保持不变) =================

def load_bvh_motion(bvh_path):
    """
    解析 BVH 文件 (修复版：支持负数数据，增强鲁棒性)
    """
    joint_order = []
    channel_map = {}
    motion_data = []
    
    try:
        with open(bvh_path, 'r') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # 有些 BVH 可能编码诡异，尝试 fallback
        with open(bvh_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()

    is_motion = False
    channel_index = 0
    current_joint = None

    for line in lines:
        line = line.strip()
        if not line: continue # 跳过空行

        # --- 解析 HIERARCHY ---
        if not is_motion:
            if line.startswith("ROOT") or line.startswith("JOINT"):
                current_joint = line.split()[1]
                joint_order.append(current_joint)
            elif line.startswith("CHANNELS"):
                parts = line.split()
                num_channels = int(parts[1])
                channel_map[current_joint] = list(range(channel_index, channel_index + num_channels))
                channel_index += num_channels
            elif line == "MOTION":
                is_motion = True
                continue

        # --- 解析 MOTION ---
        else:
            # 1. 跳过 Frames 计数行
            if line.startswith("Frames:"):
                continue
            # 2. 跳过 Frame Time 时间行
            if line.startswith("Frame Time:"):
                continue
            
            # 3. 解析数值 (只要能转成 float 就算数，不管是不是负号开头)
            try:
                values = list(map(float, line.split()))
                if len(values) > 0:
                    motion_data.append(values)
            except ValueError:
                # 如果这行既不是数据也不是关键字，就忽略
                continue

    motion_data = np.array(motion_data)

    # 🚨 安全检查：如果没读到数据，抛出异常，防止后面计算报错
    if motion_data.ndim < 2 or motion_data.shape[0] == 0:
        raise ValueError(f"解析失败: 文件中没有有效的 MOTION 数据 (Shape: {motion_data.shape})")

    return joint_order, motion_data, channel_map

def compute_joint_variance(motion_data, channel_indices):
    """计算单个关节在所有帧上的方差"""
    joint_values = motion_data[:, channel_indices]
    variance = np.var(joint_values, axis=0)
    return variance

def compute_variances_for_bvh(bvh_path):
    """加载BVH并计算所有关节的方差"""
    joint_order, motion_data, channel_map = load_bvh_motion(bvh_path)
    joint_variances = {}
    for joint in joint_order:
        var = compute_joint_variance(motion_data, channel_map[joint])
        joint_variances[joint] = var
    return joint_order, joint_variances

# ================= 2. 核心指标计算函数 (你的代码保持不变) =================

def compute_bvh_MSE(gt_bvh, gen_bvh):
    """计算均方误差 (MSE)"""
    try:
        _, gt_motion, _ = load_bvh_motion(gt_bvh)
        _, gen_motion, _ = load_bvh_motion(gen_bvh)

        if gt_motion.shape[1] != gen_motion.shape[1]:
            # print(f"⚠️ 警告: 通道数不匹配 跳过 ({os.path.basename(gt_bvh)})")
            return None

        # 截断到相同长度 (以较短的为准)
        T = min(gt_motion.shape[0], gen_motion.shape[0])
        mse = np.mean((gt_motion[:T] - gen_motion[:T]) ** 2)
        return mse
    except Exception as e:
        print(f"❌ MSE计算出错 {os.path.basename(gt_bvh)}: {e}")
        return None

def compute_AVE(gt_bvh, gen_bvh):
    """计算平均方差误差 (AVE)"""
    try:
        gt_joint_order, gt_vars = compute_variances_for_bvh(gt_bvh)
        gen_joint_order, gen_vars = compute_variances_for_bvh(gen_bvh)

        diffs = []
        for joint in gt_joint_order:
            if joint in gen_vars:
                # 把每个通道的方差差异都加进去
                diff = gt_vars[joint] - gen_vars[joint]
                diffs.append(diff)
        
        if not diffs: return None
        diffs = np.concatenate(diffs)
        # 使用 RMSE (均方根误差) 以匹配 cal_vae.py 的标准
        AVE = np.sqrt(np.mean(diffs ** 2)) 
        return AVE
    except Exception as e:
        print(f"❌ AVE计算出错 {os.path.basename(gt_bvh)}: {e}")
        return None

# ================= 3. 批量处理主逻辑 (新增部分) =================

def evaluate_batch(gt_dir, gen_dir):
    print(f"📊 开始批量评估...")
    print(f"   GT目录: {gt_dir}")
    print(f"   Gen目录: {gen_dir}")

    # 1. 获取所有文件名
    gt_files = set(os.path.basename(f) for f in glob.glob(os.path.join(gt_dir, "*.bvh")))
    gen_files = set(os.path.basename(f) for f in glob.glob(os.path.join(gen_dir, "*.bvh")))

    # 2. 找出共有的文件 (Intersection)
    common_files = sorted(list(gt_files.intersection(gen_files)))
    
    if len(common_files) == 0:
        print("❌ 错误: 两个文件夹中没有找到同名的 .bvh 文件！请检查文件名是否一致。")
        return

    print(f"✅ 找到 {len(common_files)} 对同名文件，开始计算...")

    # 3. 存储结果的列表
    mse_results = []
    ave_results = []
    valid_files = []

    # 4. 循环处理
    for filename in tqdm(common_files, desc="Calculating Metrics"):
        gt_path = os.path.join(gt_dir, filename)
        gen_path = os.path.join(gen_dir, filename)

        # 计算 MSE
        mse = compute_bvh_MSE(gt_path, gen_path)
        # 计算 AVE
        ave = compute_AVE(gt_path, gen_path)

        if mse is not None and ave is not None:
            mse_results.append(mse)
            ave_results.append(ave)
            valid_files.append(filename)

    # 5. 计算统计数据
    if not mse_results:
        print("❌ 没有成功计算出任何有效结果。")
        return

    avg_mse = np.mean(mse_results)
    std_mse = np.std(mse_results)
    avg_ave = np.mean(ave_results)
    std_ave = np.std(ave_results)

    # 6. 打印报告
    print("\n" + "="*40)
    print("📋 最终评估报告 (Evaluation Report)")
    print("="*40)
    print(f"处理文件数: {len(valid_files)} / {len(common_files)}")
    print("-" * 40)
    print(f"MSE (均方误差):")
    print(f"   平均值 (Mean): {avg_mse:.6f}")
    print(f"   标准差 (Std) : {std_mse:.6f}")
    print("-" * 40)
    print(f"AVE (平均方差误差):")
    print(f"   平均值 (Mean): {avg_ave:.6f}")
    print(f"   标准差 (Std) : {std_ave:.6f}")
    print("="*40)





# ================= 4. 运行入口 =================

if __name__ == "__main__":
    # 👇 在这里修改你的文件夹路径
    GT_FOLDER = "/home/hti_2025/hti_2025/dataset/genea2023_dataset/val/main-agent/bvh"
    GEN_FOLDER = "/home/hti_2025/wei/mywork/results/val/mainagent"  # 你的生成结果目录

    # 确保文件夹存在
    if os.path.exists(GT_FOLDER) and os.path.exists(GEN_FOLDER):
        evaluate_batch(GT_FOLDER, GEN_FOLDER)
    else:
        print("❌ 路径不存在，请检查代码底部的 GT_FOLDER 和 GEN_FOLDER 设置。")
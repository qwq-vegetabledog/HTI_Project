import os
import glob
import numpy as np

import matplotlib
# 强制使用 'Agg' 后端，这样不需要显示器也能保存图片
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ================= 1. 核心工具函数 (基于你提供的代码) =================

def load_bvh_motion(bvh_path):
    """
    解析 BVH 文件，返回关节顺序、运动数据和通道映射
    """
    joint_order = []
    channel_map = {}
    motion_data = []
    
    with open(bvh_path, 'r') as f:
        lines = f.readlines()

    is_motion = False
    channel_index = 0
    current_joint = None

    for line in lines:
        line = line.strip()

        if line.startswith("ROOT") or line.startswith("JOINT"):
            current_joint = line.split()[1]
            joint_order.append(current_joint)

        if line.startswith("CHANNELS"):
            parts = line.split()
            num_channels = int(parts[1])
            channel_map[current_joint] = list(range(channel_index, channel_index + num_channels))
            channel_index += num_channels

        if line == "MOTION":
            is_motion = True
            continue

        if is_motion and line and line[0].isdigit():
            # 有些BVH文件行尾可能有空格，使用split()自动处理
            values = list(map(float, line.split()))
            motion_data.append(values)

    motion_data = np.array(motion_data)
    return joint_order, motion_data, channel_map

def compute_joint_variance(motion_data, channel_indices):
    """计算单个关节在所有帧上的方差"""
    joint_values = motion_data[:, channel_indices]  # shape (T, C)
    variance = np.var(joint_values, axis=0)         # variance per channel
    return variance

def compute_variances_for_bvh(bvh_path):
    """加载BVH并计算所有关节的方差"""
    joint_order, motion_data, channel_map = load_bvh_motion(bvh_path)
    joint_variances = {}

    for joint in joint_order:
        var = compute_joint_variance(motion_data, channel_map[joint])
        joint_variances[joint] = var

    return joint_order, joint_variances

# ================= 2. 核心指标计算函数 =================

def compute_bvh_MSE(gt_bvh, gen_bvh):
    """计算均方误差 (MSE)"""
    try:
        gt_joints, gt_motion, gt_channel_map = load_bvh_motion(gt_bvh)
        gen_joints, gen_motion, gen_channel_map = load_bvh_motion(gen_bvh)

        # 确保通道数量一致
        if gt_motion.shape[1] != gen_motion.shape[1]:
            print(f"⚠️ 警告: 通道数不匹配 ({gt_bvh})")
            return None

        # 截断到相同长度
        T = min(gt_motion.shape[0], gen_motion.shape[0])
        gt_motion = gt_motion[:T]
        gen_motion = gen_motion[:T]

        # 计算所有帧和通道的平方误差
        squared_error = (gt_motion - gen_motion) ** 2

        # 计算全局 MSE
        mse = np.mean(squared_error)
        return mse
    except Exception as e:
        print(f"❌ MSE计算出错: {e}")
        return None

def compute_AVE(gt_bvh, gen_bvh):
    """计算平均方差误差 (AVE)"""
    try:
        gt_joint_order, gt_vars = compute_variances_for_bvh(gt_bvh)
        gen_joint_order, gen_vars = compute_variances_for_bvh(gen_bvh)

        # 确保关节顺序一致 (简单校验)
        if len(gt_joint_order) != len(gen_joint_order):
             print(f"⚠️ 警告: 关节数量不匹配 ({gt_bvh})")
             return None

        diffs = []
        for joint in gt_joint_order:
            if joint in gen_vars:
                gt_var = gt_vars[joint]
                gen_var = gen_vars[joint]
                diff = gt_var - gen_var
                diffs.append(diff)
        
        if not diffs:
            return None

        diffs = np.concatenate(diffs)  # flatten to 1D
        
        # AVE = RMSE of variance difference
        AVE = np.sqrt(np.mean(diffs ** 2))
        return AVE
    except Exception as e:
        print(f"❌ AVE计算出错: {e}")
        return None
    
def compute_bvh_MSE_and_AVE(gt_bvh, gen_bvh):
    """同时计算 MSE 和真正的 AVE (物理距离平均误差)"""
    try:
        gt_joints, gt_motion, _ = load_bvh_motion(gt_bvh)
        gen_joints, gen_motion, _ = load_bvh_motion(gen_bvh)

        # 1. 基础校验
        if gt_motion.size == 0 or gen_motion.size == 0:
            return None, None

        # 2. 长度对齐
        T = min(gt_motion.shape[0], gen_motion.shape[0])
        gt_motion = gt_motion[:T]
        gen_motion = gen_motion[:T]

        # 3. 计算 MSE (均方误差)
        # 这对应你表格里的 MSE 栏
        mse = np.mean((gt_motion - gen_motion) ** 2)

        # 4. 计算 AVE (平均绝对误差 / Mean Absolute Error)
        # 这才对应 Baseline 表格里的 AVE 栏
        ave = np.mean(np.abs(gt_motion - gen_motion))

        return mse, ave
    except Exception as e:
        print(f"❌ 计算出错: {gt_bvh} | 错误信息: {e}")
        return None, None

# ================= 3. 批量处理主逻辑 =================

def main():
    # 配置路径
    gen_dir = "/home/hti_2025/hti_2025/src/Co-Speech_Gesture_Generation/output/infer_sample_sound_interloctr"
    gt_dir = "/home/hti_2025/yujia/genea2023_dataset/tst/interloctr/bvh"
    
    # 结果保存列表
    results = [] # 格式: {'name': str, 'mse': float, 'ave': float}
    
    # 获取生成目录下所有的 .bvh 文件
    gen_files = glob.glob(os.path.join(gen_dir, "*_generated.bvh"))
    gen_files.sort() # 排序，保证曲线图顺序一致

    print(f"📂 发现 {len(gen_files)} 个生成文件，开始评估...\n")

    for gen_path in gen_files:
        filename = os.path.basename(gen_path)
        
        # 解析文件名以匹配 GT
        # 假设生成名: val_2023_v0_000_main-agent_generated.bvh
        # 目标 GT 名: val_2023_v0_000_main-agent.bvh
        
        # 去掉结尾的 "_generated.bvh"
        if "_generated.bvh" in filename:
            base_name = filename.replace("_generated.bvh", ".bvh")
        else:
            # 兼容性处理，以防命名规则不同
            base_name = filename 
            
        gt_path = os.path.join(gt_dir, base_name)
        
        # 检查 GT 是否存在
        if not os.path.exists(gt_path):
            print(f"🚫 找不到对应的真实文件: {base_name}，跳过。")
            continue
            
        # 计算指标
        # mse, ave = compute_bvh_MSE_and_AVE(gt_path, gen_path)
        mse = compute_bvh_MSE(gt_path, gen_path)
        ave = compute_AVE(gt_path, gen_path)
        
        if mse is not None and ave is not None:
            results.append({
                'name': base_name.split('.')[0], # 只保留文件名部分作为标签
                'mse': mse,
                'ave': ave
            })
            print(f"✅ 处理: {base_name} -> MSE: {mse:.4f}, AVE: {ave:.4f}")

    if not results:
        print("❌ 没有成功处理任何文件。")
        return

    # ================= 4. 绘图与统计 =================
    
    names = [r['name'] for r in results]
    mses = [r['mse'] for r in results]
    aves = [r['ave'] for r in results]
    
    # 计算全局平均值
    avg_mse = np.mean(mses)
    avg_ave = np.mean(aves)
    
    print("\n" + "="*40)
    print(f"📊 评估完成 (共 {len(results)} 个文件)")
    print(f"Global Average MSE: {avg_mse:.6f}")
    print(f"Global Average AVE: {avg_ave:.6f}")
    print("="*40)

    

if __name__ == "__main__":
    main()
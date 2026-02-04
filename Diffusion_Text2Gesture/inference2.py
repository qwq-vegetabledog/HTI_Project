import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import joblib as jl
import math
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
import sys
from scipy.ndimage import gaussian_filter1d
import os
import glob
from tqdm import tqdm  # 如果没有安装，请 pip install tqdm


# --- 添加这几行代码 ---
# 获取当前脚本所在目录 (/home/hti_2025/wei/mywork)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造 src 目录的路径
src_dir = os.path.join(current_dir, 'src')
# 将 src 加入到 Python 搜索路径中
if src_dir not in sys.path:
    sys.path.append(src_dir)
# --------------------


# 引入配置和模型
from src.config import Config
from src.model.model import Text2GestureModel
from src.model.diffusion import GaussianDiffusion 
from transformers import AutoTokenizer, AutoModel
from src.pymo.parsers import BVHParser
from src.pymo.writers import BVHWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# ⚠️ 必须与你训练时的关节列表完全一致
# ==========================================
TARGET_JOINTS = [
    'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 
    'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 
    'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 
    'b_neck0', 'b_head'
]

# ==========================================
# 1. 辅助函数
# ==========================================
def parse_tsv(tsv_path):
    words = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                words.append([parts[2], float(parts[0]), float(parts[1])])
    return words

def get_words_in_window(all_words, start_time, end_time):
    current_words = []
    for w in all_words:
        word_str, t_s, t_e = w
        word_center = (t_s + t_e) / 2
        if start_time <= word_center < end_time:
            current_words.append(word_str)
    if len(current_words) == 0:
        return "predicting motion"
    return " ".join(current_words)

def load_bert(device):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    text_model = AutoModel.from_pretrained('bert-base-uncased').to(device)
    text_model.eval()
    return tokenizer, text_model

def get_text_embedding(text, tokenizer, text_model, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=30).to(device)
    with torch.no_grad():
        output = text_model(**inputs)
        text_emb = output.last_hidden_state 
    return text_emb

def cleanup_12d_data(poses):
    """
    SVD Orthogonalization (12D -> 6D) with Auto-Reshape support.
    """
    # ==========================================
    # [FIX] Auto-Reshape: Handle 2D input (Frames, 216)
    # ==========================================
    if len(poses.shape) == 2:
        n_frames = poses.shape[0]
        # Calculate joints: Total Dim / 12 (3 Pos + 9 RotMat elements)
        n_joints = poses.shape[1] // 12
        poses = poses.reshape(n_frames, n_joints, 12)
    # ==========================================

    # Original Logic
    n_frames, n_joints, _ = poses.shape
    out_data = np.zeros((n_frames, n_joints, 6)) # 3 Pos + 3 Euler
    
    # Copy Position
    out_data[..., :3] = poses[..., :3]
    
    # Reshape rotation part to 3x3 matrices
    rot_mats = poses[..., 3:].reshape(n_frames, n_joints, 3, 3)
    
    for f in range(n_frames):
        for j in range(n_joints):
            mat = rot_mats[f, j]
            
            # SVD to enforce orthogonality
            u, s, vt = np.linalg.svd(mat)
            clean_mat = np.dot(u, vt)
            
            # Correct reflection
            if np.linalg.det(clean_mat) < 0:
                u[:, -1] *= -1
                clean_mat = np.dot(u, vt)
            
            r = R.from_matrix(clean_mat)
            
            # Output ZXY Euler angles (degrees=True, NO unwrap)
            out_data[f, j, 3:] = r.as_euler('ZXY', degrees=True)
            
    return out_data
# ==========================================
# ⭐ 新增：手动导出 BVH 函数 (替代 Pipeline)
# ==========================================
def export_bvh_manual(motion_data, ref_bvh_path, output_path):
    """
    motion_data: (Frames, 18, 6) 
                 -> [3 Pos, 3 Euler] 
                 -> 其中 Euler 的顺序在 cleanup_12d_data 里必须是 r.as_euler('ZXY', degrees=True)
    ref_bvh_path: 参考 BVH 路径，用于读取骨架结构和 OFFSET
    output_path: 最终生成的 BVH 保存路径
    """
    print(f"Exporting BVH manually using skeleton from: {ref_bvh_path}")
    
    # 1. 解析参考 BVH 获取骨架
    parser = BVHParser()
    ref_data = parser.parse(ref_bvh_path)
    
    # 2. 准备新的 DataFrame (锁定 30 FPS)
    n_frames = motion_data.shape[0]
    frame_time = 1.0 / 30.0 
    new_index = pd.to_timedelta(np.arange(n_frames) * frame_time, unit='s')
    
    # 获取列名并创建空表
    cols = ref_data.values.columns
    new_values = pd.DataFrame(index=new_index, columns=cols)
    
    # 3. 填充初始姿态 (Rest Pose)
    # 这一步很重要：它保证了那些我们没生成的关节（手指、下半身）保持静止，而不是乱飞
    ref_frame_0 = ref_data.values.iloc[0].values
    new_values.iloc[:, :] = np.tile(ref_frame_0, (n_frames, 1))
    
    # ========================================================
    # 🚀 核心修正：通道重映射 (Channel Remapping)
    # ========================================================
    # 根据调试结论：
    # - 模型输出 ch_1 是主力动作 (大幅度数值)
    # - BVH 的 X轴是 Twist (拧毛巾/自转)
    # - BVH 的 Y轴是 Lift (抬胳膊)
    # -> 必须把 ch_1 赋给 Y轴！
    
    for i, joint_name in enumerate(TARGET_JOINTS):
        # 提取模型生成的三个旋转通道
        # 注意：motion_data 的前3位是位置(通常不用)，后3位是旋转
        ch_0 = motion_data[:, i, 3] # 通常对应 Z (Swing/摆动)
        ch_1 = motion_data[:, i, 4] # 通常对应 X (但这里我们要换给 Y)
        ch_2 = motion_data[:, i, 5] # 通常对应 Y (但这里我们要换给 X)
        
        # 1. 设置 Z 轴 (前后摆动) - 保持直通
        if f'{joint_name}_Zrotation' in cols:
            new_values[f'{joint_name}_Zrotation'] = ch_0
            
        # 2. 设置 Y 轴 (抬胳膊) - ✅ 修正点：接收主力数据 ch_1
        if f'{joint_name}_Yrotation' in cols:
            new_values[f'{joint_name}_Yrotation'] = ch_1  
            
        # 3. 设置 X 轴 (拧毛巾/微调) - ✅ 修正点：接收次要数据 ch_2
        if f'{joint_name}_Xrotation' in cols:
            new_values[f'{joint_name}_Xrotation'] = ch_2

    # ========================================================

    # 4. 更新数据
    ref_data.values = new_values

    # 5. 简单检查（确保没有写入全0数据）
    check_col = 'b_r_arm_Yrotation' # 检查 Y 轴是否有数据
    if check_col in new_values.columns:
        col_data = new_values[check_col].values
        range_val = np.max(col_data) - np.min(col_data)
        print(f"📊 数据写入检查 ({check_col}): 变化幅度 = {range_val:.4f}")
        if range_val < 0.1:
            print("⚠️ 警告：动作幅度极小，可能需要检查反归一化或模型效果。")

    # 6. 自动创建目录并保存
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    writer = BVHWriter()
    with open(output_path, 'w') as f:
        writer.write(ref_data, f)
        
    print(f"✅ BVH saved to: {os.path.abspath(output_path)}")
    print(f"Exporting BVH manually using skeleton from: {ref_bvh_path}")
    parser = BVHParser()
    ref_data = parser.parse(ref_bvh_path)
    
    # 准备 DataFrame
    n_frames = motion_data.shape[0]
    frame_time = 1.0 / 30.0 
    new_index = pd.to_timedelta(np.arange(n_frames) * frame_time, unit='s')
    cols = ref_data.values.columns
    new_values = pd.DataFrame(index=new_index, columns=cols)
    ref_frame_0 = ref_data.values.iloc[0].values
    new_values.iloc[:, :] = np.tile(ref_frame_0, (n_frames, 1))
    
    # ========================================================
    # 🚀 最终映射修正 (基于你的观察: X=Twist, Y=Lift, Z=Swing)
    # ========================================================

    # ========================================================
    # 🚀 最终映射修正 (基于你的观察: X=Twist, Y=Lift, Z=Swing)
    # ========================================================
    
    # ========================================================
    # 🛠️ 关节限制配置 (Joint Constraints)
    # ========================================================
    # 定义你要限制的关节及其衰减系数 (0.0=不动, 1.0=原样, 0.5=幅度减半)
    # 0.0 = 完全冻结 (不动)
    # 1.0 = 完全原样 (疯动)
    
    DAMPING_CONFIG = {
        # === 核心躯干 (稳住下盘) ===
        'b_spine0': 0.01,  # 腰别乱扭
        'b_spine1': 0.05,
        'b_spine2': 0.06,
        'b_spine3': 0.07,
        
        # === 头颈 (拒绝摇头晃脑) ===
        'b_neck0': 0.03,
        'b_head': 0.03,    # 头要稳
        
        # === 肩膀 (拒绝紧张耸肩) ===
        'b_l_shoulder': 0.05,
        'b_r_shoulder': 0.05,
        
        # === 大臂 (控制挥手幅度，最关键!) ===
        'b_l_arm': 0.3,   # 稍微收敛一点
        'b_r_arm': 0.3,
        'twist': 0.02,     # ⚠️ 凡是名字带 twist 的，统统按死，防止扭曲
        
        # === 小臂与手 (保留细节) ===
        'b_l_forearm': 0.5, # 让胳膊肘灵活点
        'b_r_forearm': 0.5,
        'b_l_wrist': 0.5,   # 手掌动作可以丰富点
        'b_r_wrist': 0.5,
    }

    print("应用关节限制...")

    
    for i, joint_name in enumerate(TARGET_JOINTS):
        # 提取模型生成的三个通道
        # 根据经验：
        # ch_1 通常是幅度最大的 (上下抬)
        # ch_0 通常是第二大的 (前后摆)
        # ch_2 通常是最小的 (自转)
        ch_0 = motion_data[:, i, 3] 
        ch_1 = motion_data[:, i, 4] 
        ch_2 = motion_data[:, i, 5]
        # 🔍 检查是否需要限制这个关节
        # 我们支持模糊匹配，比如配置写 'arm'，则所有名字带 arm 的都会被限制
        damping_factor = 1.0
        for key, factor in DAMPING_CONFIG.items():
            if key in joint_name:
                damping_factor = factor
                break
        
        # 如果系数小于 1.0，执行【向均值收缩】算法
        if damping_factor < 1.0:
            # 算法公式: 新值 = 均值 + (原值 - 均值) * 系数
            # 这样保证了动作幅度变小，但人不会歪掉
            ch_0 = np.mean(ch_0) + (ch_0 - np.mean(ch_0)) * damping_factor
            ch_1 = np.mean(ch_1) + (ch_1 - np.mean(ch_1)) * damping_factor
            ch_2 = np.mean(ch_2) + (ch_2 - np.mean(ch_2)) * damping_factor

        # --- 下面是之前的映射逻辑 (保持不变) ---

        
        # 1. Z 轴 (摆胳膊) <- 接收 ch_0
        if f'{joint_name}_Zrotation' in cols:
            new_values[f'{joint_name}_Zrotation'] = ch_0
            
        # 2. Y 轴 (抬胳膊 - 核心动作!) <- 接收 ch_1 (之前错给X的数据)
        if f'{joint_name}_Yrotation' in cols:
            new_values[f'{joint_name}_Yrotation'] = ch_1  # ✅ 修正：主力数据给 Y
            
        # 3. X 轴 (拧毛巾 - 辅助动作) <- 接收 ch_2
        if f'{joint_name}_Xrotation' in cols:
            new_values[f'{joint_name}_Xrotation'] = ch_2  # ✅ 修正：次要数据给 X

    # ========================================================

    ref_data.values = new_values
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    writer = BVHWriter()
    with open(output_path, 'w') as f:
        writer.write(ref_data, f)
    print(f"✅ BVH saved to: {output_path}")
# ==========================================
# 3. 长序列生成逻辑
# ==========================================
# def generate_long_sequence(cfg, model, diffusion, tokenizer, text_model, tsv_path):
#     words = parse_tsv(tsv_path)
#     if not words: return None
    
#     total_duration = words[-1][2]
#     FPS = 30 
#     CLIP_FRAMES = 120 
#     CLIP_DURATION = CLIP_FRAMES / FPS 
#     num_clips = math.ceil(total_duration / CLIP_DURATION)
    
#     print(f"Total Duration: {total_duration:.2f}s, Generating {num_clips} clips...")
#     generated_clips = []
    
#     for i in range(num_clips):
#         start_t = i * CLIP_DURATION
#         end_t = start_t + CLIP_DURATION
#         window_text = get_words_in_window(words, start_t, end_t)
#         print(f"Clip {i+1}/{num_clips} [{start_t:.1f}s-{end_t:.1f}s]: '{window_text}'")
        
#         text_emb = get_text_embedding(window_text, tokenizer, text_model, device)
#         sample_shape = (1, CLIP_FRAMES, cfg.INPUT_FEATS)
        
#         # 生成
#         raw_motion = diffusion.sample(sample_shape, text_emb, src_mask=None, guidance_scale=2.5)
#         raw_motion = raw_motion.squeeze(0).cpu().numpy()
#         generated_clips.append(raw_motion)
        
#     full_motion = np.vstack(generated_clips)
#     target_frames = int(total_duration * FPS)
#     full_motion = full_motion[:target_frames]
    
#     return full_motion

def generate_long_sequence(cfg, model, diffusion, tokenizer, text_model, tsv_path):
    # 1. 解析 TSV
    words = parse_tsv(tsv_path)
    if not words: return None
    
    # 🔴 修正点：不要只取最后一个，要找最大值！
    # 遍历所有词，找到最大的 end_time
    total_duration = max([w[2] for w in words])
    
    # 或者手动强制指定时长（如果你确信是 60 秒）
    # total_duration = 60.0 
    
    FPS = 30 
    CLIP_FRAMES = 120 
    CLIP_DURATION = CLIP_FRAMES / FPS 
    num_clips = math.ceil(total_duration / CLIP_DURATION)
    
    print(f"🎯 真实时长: {total_duration:.2f}s | 计划生成: {num_clips} 段")
    
    generated_clips = []
    
    for i in range(num_clips):
        start_t = i * CLIP_DURATION
        end_t = start_t + CLIP_DURATION
        window_text = get_words_in_window(words, start_t, end_t)
        print(f"   - Clip {i+1}/{num_clips} [{start_t:.1f}s-{end_t:.1f}s]: '{window_text}'")
        
        text_emb = get_text_embedding(window_text, tokenizer, text_model, device)
        # 注意：这里大小写要和 Config 类一致，之前是 input_feats
        sample_shape = (1, CLIP_FRAMES, cfg.INPUT_FEATS) 
        
        # 生成
        raw_motion = diffusion.sample(sample_shape, text_emb, src_mask=None, guidance_scale=2.5)
        raw_motion = raw_motion.squeeze(0).cpu().numpy()
        generated_clips.append(raw_motion)
        
    full_motion = np.vstack(generated_clips)
    
    # 计算目标帧数
    target_frames = int(total_duration * FPS)
    
    # 截取
    if full_motion.shape[0] > target_frames:
        print(f"✂️  裁剪多余帧数: {full_motion.shape[0]} -> {target_frames}")
        full_motion = full_motion[:target_frames]
    
    return full_motion


def main():
    # ==============================================================
    # 1. Configuration (Modify paths here)
    # ==============================================================
    input_dir = "/home/hti_2025/hti_2025/dataset/genea2023_dataset/trn/main-agent/tsv"
    output_dir = "/home/hti_2025/wei/mywork/results/trn/main-agent"
    
    # Reference BVH for skeleton structure
    ref_bvh_path = "/home/hti_2025/hti_2025/dataset/genea2023_dataset/val/interloctr/bvh/val_2023_v0_000_interloctr.bvh"
    
    # Model Checkpoint
    checkpoint_path = "/home/hti_2025/wei/mywork/checkpoints/model_epoch_600.pt"

    # Load Config
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==============================================================
    # 2. Initialize Models
    # ==============================================================
    print(f"🚀 Initializing models on device: {device}...")
    
    # Load BERT
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text_model = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
    
    # Load Diffusion Model
    model = Text2GestureModel(
        input_feats=cfg.INPUT_FEATS, latent_dim=cfg.LATENT_DIM, 
        n_heads=cfg.HEADS, n_layers=cfg.LAYERS, text_dim=cfg.TEXT_DIM
    ).to(device)
    
    diffusion = GaussianDiffusion(
        model=model, timesteps=cfg.DIFFUSION_STEPS, 
        loss_type=cfg.LOSS_TYPE, beta_schedule=cfg.BETA_SCHEDULE
    ).to(device)
    
    # Load Weights
    print(f"♻️ Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        diffusion.load_state_dict(checkpoint['model_state_dict'])
        data_mean = checkpoint.get('data_mean', np.array(cfg.DATA_MEAN))
        data_std = checkpoint.get('data_std', np.array(cfg.DATA_STD))
    else:
        diffusion.load_state_dict(checkpoint)
        data_mean = np.array(cfg.DATA_MEAN)
        data_std = np.array(cfg.DATA_STD)
        
    diffusion.eval()

    # Pre-process Mean/Std to CPU Numpy
    if isinstance(data_mean, torch.Tensor): data_mean = data_mean.cpu().numpy()
    if isinstance(data_std, torch.Tensor): data_std = data_std.cpu().numpy()
    
    # Avoid division by zero or extreme scaling
    data_std = np.clip(data_std, a_min=1e-6, a_max=None)

    # ==============================================================
    # 3. Batch Processing Loop
    # ==============================================================
    os.makedirs(output_dir, exist_ok=True)
    tsv_files = sorted(glob.glob(os.path.join(input_dir, "*.tsv")))
    
    print(f"\n📂 Found {len(tsv_files)} TSV files.")
    print(f"   Output Directory: {output_dir}\n")
    
    # Use tqdm for progress bar
    for tsv_path in tqdm(tsv_files, desc="Processing"):
        filename = os.path.basename(tsv_path)
        file_id = os.path.splitext(filename)[0]
        output_bvh_path = os.path.join(output_dir, f"{file_id}.bvh")
        
        try:
            # --- A. Generate Motion ---
            full_raw_motion = generate_long_sequence(cfg, model, diffusion, tokenizer, text_model, tsv_path)
            
            if full_raw_motion is None:
                # print(f"⚠️ Skipped {filename}: Generation failed.")
                continue

            # --- B. Post-Processing ---
            
            # 1. To CPU Numpy (Safe conversion)
            if isinstance(full_raw_motion, torch.Tensor):
                raw_np = full_raw_motion.cpu().numpy()
            else:
                raw_np = full_raw_motion

            # 2. Denormalize
            denorm_motion = raw_np * data_std + data_mean

            # 3. Gaussian Smoothing (Sigma=9.0 as requested)
            denorm_motion = gaussian_filter1d(denorm_motion, sigma=9.0, axis=0)

            # 4. SVD Cleanup (Auto-reshapes 2D -> 3D inside)
            clean_motion = cleanup_12d_data(denorm_motion)
            
            # --- C. Export BVH ---
            export_bvh_manual(clean_motion, ref_bvh_path, output_bvh_path)

        except Exception as e:
            print(f"\n❌ [Error] Failed to process {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n🎉 Batch processing finished!")

if __name__ == '__main__':
    main()

import argparse
import math
import pickle
import pprint
import time
import os
import numpy as np
import torch
import joblib as jl
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

import utils
from utils.data_utils import SubtitleWrapper, normalize_string
from utils.train_utils_sound import set_logger
from data_loader.data_preprocessor_sound import DataPreprocessor
from pymo.writers import BVHWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_gestures(args, pose_decoder, in_audio):
    pose_decoder.eval()

    batch_size = in_audio.size(0)
    n_frames = in_audio.shape[1]
    
    # 初始化全零 poses
    poses = torch.zeros((batch_size, n_frames, pose_decoder.pose_dim)).to(device)
    mean_pose = torch.squeeze(torch.Tensor(args.data_mean)).to(device)
    
    # MODIFIED: 限制初始帧填充的数量，避免覆盖掉生成的动作
    # 如果 args.n_pre_poses 太长（比如正好 40），会导致整段都是静止的
    n_pre = min(args.n_pre_poses, 10) # 建议最多只给 10 帧初始姿态
    poses[:, :n_pre, :] = mean_pose.repeat(batch_size, n_pre, 1)

    print(f"DEBUG: 推理总帧数: {n_frames}, 初始填充帧数: {n_pre}")

    with torch.no_grad():
        # 推理生成
        out_poses = pose_decoder(in_audio=in_audio, in_text=None, poses=poses)

    # MODIFIED: 确保输出维度与输入音频对齐，防止出现多余的 padding
    result = out_poses.squeeze(0).cpu().numpy()
    return result[:n_frames, :]

def make_bvh(save_path, filename_prefix, poses):
    writer = BVHWriter()
    # 路径建议检查是否正确
    pipeline = jl.load('../resource/data_pipe.sav')

    n_poses = poses.shape[0]
    dim_poses = poses.shape[1]
    out_poses = np.zeros((n_poses, dim_poses))

    # MODIFIED: 动态调整平滑窗口。
    # 窗口大小必须是奇数且小于总帧数。原先固定的 15 在 40 帧的短动作里会把动作抹平。
    window_size = 7 if n_poses > 7 else (n_poses // 2 * 2 - 1)
    
    print(f"BVH shape: {poses.shape}, Smoothing window: {window_size}")

    for i in range(dim_poses):
        if window_size >= 3:
            out_poses[:, i] = savgol_filter(poses[:, i], window_size, 2)
        else:
            out_poses[:, i] = poses[:, i]

    # 旋转矩阵转欧拉角逻辑
    out_poses = out_poses.reshape((out_poses.shape[0], -1, 12)) 
    out_data = np.zeros((out_poses.shape[0], out_poses.shape[1], 6))
    for i in range(out_poses.shape[0]):
        for j in range(out_poses.shape[1]):
            out_data[i, j, :3] = out_poses[i, j, :3]
            # 这里的 R 可能需要处理矩阵异常
            try:
                r = R.from_matrix(out_poses[i, j, 3:].reshape(3, 3))
                out_data[i, j, 3:] = r.as_euler('ZXY', degrees=True).flatten()
            except ValueError:
                out_data[i, j, 3:] = 0 # 异常处理

    out_data = out_data.reshape(out_data.shape[0], -1)
    bvh_data = pipeline.inverse_transform([out_data])

    out_bvh_path = os.path.join(save_path, filename_prefix + '_generated.bvh')
    with open(out_bvh_path, 'w') as f:
        writer.write(bvh_data[0], f)

def main():
    # ... 原有的路径配置保持不变 ...
    ckpt_path = '/home/hti_2025/hti_2025/src/Co-Speech_Gesture_Generation/output/checkpoints3/genea_yujia_30fps_v1_checkpoint_020.bin'
    input_dir = '/home/hti_2025/yujia/genea2023_dataset/trn/main-agent/spec'
    save_path = '/home/hti_2025/hti_2025/src/Co-Speech_Gesture_Generation/output/infer_sample_sound_main-agent_20/'
    os.makedirs(save_path, exist_ok=True)

    print(f"正在加载模型: {ckpt_path}")
    args, generator, loss_fn, lang_model, out_dim = utils.train_utils_sound.load_checkpoint_and_model(
        ckpt_path, device)
    
    # 提取标准化参数
    mean = np.array(args.data_mean).squeeze()
    std = np.array(args.data_std).squeeze()
    std = np.clip(std, a_min=0.01, a_max=None)

    audio_files = sorted(list(Path(input_dir).glob("*.npy")) + list(Path(input_dir).glob("*.spec")))
    print(f"共找到 {len(audio_files)} 个文件。开始批量生成...")

    for i, audio_path in enumerate(audio_files):
        try:
            audio_data = np.load(audio_path)
            # MODIFIED: 增加输入维度的打印，便于你观察是否每个文件都只有 40 帧
            print(f"Processing: {audio_path.name}, Input Shape: {audio_data.shape}")
            
            in_audio = torch.from_numpy(audio_data).float().to(device)
            if len(in_audio.shape) == 2: 
                in_audio = in_audio.unsqueeze(0)
            
            # 动态更新当前文件的帧数
            args.n_frames = in_audio.shape[1]
            
            # 推理
            out_poses = generate_gestures(args, generator, in_audio)

            # 反标准化
            out_poses = np.multiply(out_poses, std) + mean

            # 保存 BVH
            filename = audio_path.stem
            make_bvh(save_path, filename, out_poses)
            
        except Exception as e:
            print(f"处理出错 {audio_path.name}: {e}")

    print(f"全部任务完成！")

if __name__ == '__main__':
    main()

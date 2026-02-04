from config import Config
import glob
import os
from pathlib import Path
import pickle  # <--- 修改点 1: 导入 pickle

import librosa
import lmdb
import numpy as np
from sklearn.pipeline import Pipeline
import joblib as jl
from scipy.spatial.transform import Rotation as R

# 请确保你的环境中包含 pymo 和 utils 文件夹
from pymo.preprocessing import *
from pymo.parsers import BVHParser
from utils.data_utils import SubtitleWrapper, normalize_string

# 18 joints (only upper body)
target_joints = ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']

def process_bvh(gesture_filename, dump_pipeline=False):
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline([
        ('jtsel', JointSelector(target_joints, include_root=False)),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    if dump_pipeline:
        # 确保 resource 文件夹存在
        if not os.path.exists('../resource'):
            os.makedirs('../resource')
        jl.dump(data_pipe, os.path.join('../resource', 'data_pipe.sav'))

    # euler -> rotation matrix
    # out_data shape: [Frames, Joints * 6] (3 Pos + 3 Rot Euler)
    # 我们将其 reshape 为 [Frames, Joints, 6]
    out_data = out_data.reshape((out_data.shape[0], out_data.shape[1], -1, 6)) 
    
    # 构造输出矩阵：每个关节 12 维 (3 Pos + 9 RotMatrix)
    # 如果你想改用 6D 旋转 (3 Pos + 6 Rot6D)，请将最后一维 12 改为 9，并只取矩阵前两列
    out_matrix = np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2], 12)) 
    
    for i in range(out_data.shape[0]):  # mirror (这里只有一个 track)
        for j in range(out_data.shape[1]):  # frames
            for k in range(out_data.shape[2]):  # joints
                out_matrix[i, j, k, :3] = out_data[i, j, k, :3]  # positions
                
                # Euler (Degrees) -> Rotation Matrix
                r = R.from_euler('ZXY', out_data[i, j, k, 3:], degrees=True)
                out_matrix[i, j, k, 3:] = r.as_matrix().flatten()  # rotations (9 elements)

    # Flatten joints: [Frames, Joints * 12]
    out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))

    return out_matrix[0]


def make_lmdb_gesture_dataset():
    gesture_path = Config.BVH_DIR
    audio_path = Config.WAV_DIR   # 需在 Config 中添加 WAV_DIR
    text_path = Config.TSV_DIR
    out_path = Config.LMDB_OUTPUT_DIR
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    map_size = 1024 * 1024 * 1024 * 20  # 20 GB
    
    # 创建训练集和测试集环境
    lmdb_train_path = os.path.join(out_path, 'lmdb_train')
    lmdb_test_path = os.path.join(out_path, 'lmdb_test')
    
    db = [lmdb.open(lmdb_train_path, map_size=map_size),
          lmdb.open(lmdb_test_path, map_size=map_size)]

    # delete existing keys (清空数据库)
    for i in range(2):
        with db[i].begin(write=True) as txn:
            txn.drop(db[i].open_db())

    all_poses = []
    bvh_files = sorted(glob.glob(gesture_path + "/*.bvh"))
    save_idx = 0
    
    print(f"Found {len(bvh_files)} files.")

    for bvh_file in bvh_files:
        name = os.path.split(bvh_file)[1][:-4]
        print(f"Processing: {name}")

        # load subtitles
        tsv_path = os.path.join(text_path, name + '.tsv')
        if os.path.isfile(tsv_path):
            subtitle = SubtitleWrapper(tsv_path).get()
        else:
            print(f"Skipping {name}: TSV not found.")
            continue

        # load audio
        wav_path = os.path.join(audio_path, '{}.wav'.format(name))
        if os.path.isfile(wav_path):
            # sr=16000 对应 wav2vec2 等大多数音频模型的输入
            audio_raw, audio_sr = librosa.load(wav_path, mono=True, sr=16000, res_type='kaiser_fast')
        else:
            print(f"Skipping {name}: WAV not found.")
            continue

        # load skeletons
        dump_pipeline = (save_idx == 0) 
        try:
            poses = process_bvh(bvh_file, dump_pipeline)
        except Exception as e:
            print(f"Error processing BVH {name}: {e}")
            continue

        # process structure
        clips = [{'vid': name, 'clips': []},  # train
                 {'vid': name, 'clips': []}]  # validation

        # split strategy (每 100 个由 1 个作为验证集)
        if save_idx % 100 == 0:
            dataset_idx = 1  # validation
        else:
            dataset_idx = 0  # train

        # word preprocessing
        word_list = []
        for wi in range(len(subtitle)):
            word_s = float(subtitle[wi][0])
            word_e = float(subtitle[wi][1])
            word = subtitle[wi][2].strip()

            word_tokens = word.split()

            for t_i, token in enumerate(word_tokens):
                token = normalize_string(token)
                if len(token) > 0:
                    # 简单地将时间均分给每个 token
                    new_s_time = word_s + (word_e - word_s) * t_i / len(word_tokens)
                    new_e_time = word_s + (word_e - word_s) * (t_i + 1) / len(word_tokens)
                    word_list.append([token, new_s_time, new_e_time])

        # save subtitles and skeletons
        # 使用 float32 保证精度，或者 float16 节省空间
        poses = np.asarray(poses, dtype=np.float32)
        
        clips[dataset_idx]['clips'].append(
            {'words': word_list,
             'poses': poses,
             'audio_raw': audio_raw
             })

        # write to db
        for i in range(2):
            with db[i].begin(write=True) as txn:
                if len(clips[i]['clips']) > 0:
                    k = '{:010}'.format(save_idx).encode('ascii')
                    
                    # <--- 修改点 2: 使用 pickle 替代 pyarrow --->
                    # v = pyarrow.serialize(clips[i]).to_buffer() 
                    v = pickle.dumps(clips[i])
                    
                    txn.put(k, v)

        all_poses.append(poses)
        save_idx += 1

    # close db
    for i in range(2):
        db[i].sync()
        db[i].close()

    # calculate data mean
    if len(all_poses) > 0:
        all_poses = np.vstack(all_poses)
        pose_mean = np.mean(all_poses, axis=0, dtype=np.float64)
        pose_std = np.std(all_poses, axis=0, dtype=np.float64)

        print('\nData Statistics (Copy these to your config.yaml):')
        print('data_mean:', str(["{:0.5f}".format(e) for e in pose_mean]).replace("'", ""))
        print('data_std:', str(["{:0.5f}".format(e) for e in pose_std]).replace("'", ""))
    else:
        print("No poses processed.")


if __name__ == '__main__':
    

    make_lmdb_gesture_dataset()
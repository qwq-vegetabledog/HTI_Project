import os
import glob
import joblib as jl
import numpy as np
from sklearn.pipeline import Pipeline
from config import Config

# 引入 pymo
from pymo.preprocessing import *
from pymo.parsers import BVHParser

# 必须与 Config 保持一致
TARGET_JOINTS = [
    'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 
    'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 
    'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 
    'b_neck0', 'b_head'
]

def rebuild_pipeline():
    # 1. 找一个 BVH 文件做样本
    bvh_dir = Config.BVH_DIR # 从 Config 读取 BVH 文件夹路径
    bvh_files = glob.glob(os.path.join(bvh_dir, "*.bvh"))
    
    if not bvh_files:
        print(f"❌ No BVH files found in {bvh_dir}")
        return
    
    sample_bvh = bvh_files[0]
    print(f"📂 Using sample BVH: {sample_bvh}")

    # 2. 解析
    p = BVHParser()
    data_all = [p.parse(sample_bvh)]

    # 3. 定义管道 (必须与训练时一致)
    # 注意：这里我们使用了标准流程
    data_pipe = Pipeline([
        ('jtsel', JointSelector(TARGET_JOINTS, include_root=False)),
        ('np', Numpyfier())
    ])

    # 4. Fit (让管道学习骨骼结构)
    print("⚙️ Fitting pipeline...")
    data_pipe.fit(data_all)

    # 5. 保存
    save_dir = os.path.join(Config.PROJECT_ROOT, 'resource')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'data_pipe.sav')
    
    jl.dump(data_pipe, save_path)
    print(f"✅ Pipeline saved successfully to: {save_path}")
    print("Now you can run inference.py!")

if __name__ == "__main__":
    rebuild_pipeline()
import os
import glob
import subprocess
import sys

def main():
    # ================= 配置部分 =================
    
    # 1. inference.py 的路径 (假设就在当前目录下)
    script_path = "inference.py"
    
    # 2. 模型 Checkpoint 的绝对路径
    ckpt_path = "/home/HTI_project/src/output/train_seq2seq/baseline_icra19_checkpoint_100.bin"
    
    # 3. TSV 文件夹的路径
    tsv_dir = "/home/HTI_project/dataset/genea/genea2023_dataset/val/main-agent/tsv/"
    
    # ===========================================

    # 检查文件是否存在
    if not os.path.exists(tsv_dir):
        print(f"❌ 错误: 找不到 TSV 文件夹: {tsv_dir}")
        return

    # 获取文件夹下所有的 .tsv 文件
    # glob.glob 会返回匹配到的文件完整路径列表
    tsv_files = glob.glob(os.path.join(tsv_dir, "*.tsv"))
    
    # 按文件名排序，保证执行顺序
    tsv_files.sort()

    print(f"📂 发现 {len(tsv_files)} 个 TSV 文件，准备开始处理...\n")

    # 遍历每一个 TSV 文件并执行命令
    for index, tsv_file in enumerate(tsv_files):
        print(f"[{index + 1}/{len(tsv_files)}] 正在处理: {os.path.basename(tsv_file)}")
        
        # 构建命令: python inference.py [CKPT] [TSV]
        # 注意: sys.executable 指向当前环境的 python 解释器
        cmd = [sys.executable, script_path, ckpt_path, tsv_file]
        
        try:
            # 这里的 env=os.environ.copy() 保证了 subprocess 继承当前 shell 的环境变量
            # (比如 PYTHONPATH, LD_LIBRARY_PATH 等)
            result = subprocess.run(cmd, check=True, env=os.environ.copy())
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 处理文件失败: {os.path.basename(tsv_file)}")
            print(f"错误信息: {e}")
            # 如果你想遇到错误继续跑下一个，这里不要 raise，直接 continue 即可
            continue
            
    print("\n✅ 所有任务已完成！")

if __name__ == "__main__":
    main()

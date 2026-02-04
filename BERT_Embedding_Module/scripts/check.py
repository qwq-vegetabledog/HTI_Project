import pandas as pd
import joblib
import os
import sys

# 设定文件路径 (根据你的报错信息，路径是 ../resource/data_pipe.sav)
file_path = '../resource/data_pipe.sav'

print("="*30)
print(f"1. 当前环境检查")
print(f"Python 版本: {sys.version.split()[0]}")
print(f"Pandas 版本: {pd.__version__}")
print(f"Joblib 版本: {joblib.__version__}")
print("="*30)

print(f"2. 文件检查: {file_path}")
if not os.path.exists(file_path):
    print(f"❌ 错误: 找不到文件! 请确认路径是否正确。")
else:
    file_size = os.path.getsize(file_path)
    print(f"文件大小: {file_size} bytes")
    
    # 读取前20个字节，检查是否包含文本换行符或 HTML 标签
    with open(file_path, 'rb') as f:
        header = f.read(20)
        print(f"文件头(Hex): {header.hex()}")
        print(f"文件头(Raw): {header}")
        
        # 检查是否是 Git LFS 指针或 HTML
        if b'version https://git-lfs' in header:
            print("❌ 警告: 这是一个 Git LFS 指针文件，不是真实数据！你需要安装 Git LFS。")
        elif b'<!DOCTYPE html>' in header or b'<html' in header:
            print("❌ 警告: 你下载的是 GitHub 网页 HTML，不是 .sav 文件！请点击 Raw 按钮下载。")
        elif b'\r\n' in header:
             print("❌ 警告: 文件头包含 Windows 换行符 (CRLF)，可能传输时损坏。")

print("="*30)
print(f"3. 尝试加载文件")
try:
    data = joblib.load(file_path)
    print("✅ 成功: 文件加载成功！")
    print(f"数据类型: {type(data)}")
except Exception as e:
    print(f"❌ 失败: 加载报错 -> {e}")
    # 打印更详细的错误类型，帮助判断
    print(f"错误类型: {type(e).__name__}")
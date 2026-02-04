import logging
import os
import pickle
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pyarrow

# 引入预处理器
from src.data.data_preprocessor import DataPreprocessor

class TextMotionDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, data_mean, data_std, max_text_len=30):
        self.lmdb_dir = lmdb_dir

        print(f"📂 [Loader] 正在加载 LMDB 路径: {lmdb_dir}") # <--- 加这行
        self.n_poses = n_poses
        self.mean = np.array(data_mean).squeeze()
        self.std = np.array(data_std).squeeze()
        self.max_text_len = max_text_len
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # ---------------------------------------------------------
        # ⚠️ 关键修改：自动检测并生成 Cache
        # ---------------------------------------------------------
        preloaded_dir = lmdb_dir + '_cache_v2'
        
        if not os.path.exists(preloaded_dir):
            logging.info(f"Cache not found at {preloaded_dir}.")
            logging.info("🚀 Starting automatic data preprocessing (slicing)...")
            logging.info("This may take a few minutes depending on dataset size.")
            
            processor = DataPreprocessor(
                clip_lmdb_dir=lmdb_dir,
                out_lmdb_dir=preloaded_dir,
                n_poses=n_poses,
                subdivision_stride=subdivision_stride, # 默认切片步长
                pose_resampling_fps=pose_resampling_fps
            )
            processor.run()
            logging.info("✅ Preprocessing done.")
        else:
            logging.info(f"Using existing cache: {preloaded_dir}")
        # ---------------------------------------------------------

        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample_bytes = txn.get(key)
            
            # 反序列化
            try:
                sample = pyarrow.deserialize(sample_bytes)
            except:
                sample = pickle.loads(sample_bytes)

            # ==========================================
            # 🛠️ 解包逻辑 (适配 4元素 或 3元素)
            # ==========================================
            if len(sample) == 4:
                # 标准格式: [单词, 动作, 音频, 辅助]
                word_list, pose_seq, audio, aux_info = sample
            
            elif len(sample) == 3:
                # 兼容旧格式: [音频, 动作, 辅助]
                # 这种情况应该很少见，因为你的 DataPreprocessor 应该生成4个元素的
                audio, pose_seq, aux_info = sample
                
                # 尝试补全 word_list
                if isinstance(aux_info, dict) and 'words' in aux_info:
                    word_list = aux_info['words']
                elif isinstance(aux_info, dict) and 'text' in aux_info:
                    word_list = aux_info['text']
                else:
                    word_list = [['<unk>', 0.0, 0.0]] # 找不到就给个空的
            else:
                 # 遇到坏数据返回 None，collate_fn 需要额外处理，或者直接报错
                raise ValueError(f"Unknown data structure: len={len(sample)}")

        # ---------------------------------------------------------
        # 1. 处理文本 (Text Processing)
        # ---------------------------------------------------------
        # 将单词列表拼接成字符串
        # 假设 word_list 里的结构是 [['hello', 0.1, 0.2], ...]
        if word_list and isinstance(word_list[0], (list, tuple)):
             text_str = " ".join([w[0] for w in word_list])
        else:
             text_str = " ".join(word_list) if word_list else ""

        # BERT Tokenizer
        tokenized = self.tokenizer(
            text_str,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        text_ids = tokenized['input_ids'].squeeze(0)
        text_mask = tokenized['attention_mask'].squeeze(0)

        # ---------------------------------------------------------
        # 2. 处理动作与归一化 (Normalization)
        # ---------------------------------------------------------
        # Z-Score Normalization
        epsilon = 1e-6
        std_safe = np.clip(self.std, a_min=epsilon, a_max=None)
        pose_seq = (pose_seq - self.mean) / std_safe
        
        # 转 Tensor
        pose_seq = torch.from_numpy(pose_seq).float()
        audio = torch.from_numpy(audio).float()

        # 返回字典 (适配 collate_fn)
        return {
            "motion": pose_seq,
            "text_ids": text_ids,
            "text_mask": text_mask,
            "audio": audio
        }

def collate_fn(batch):
    batch_motion = [item['motion'] for item in batch]
    batch_text_ids = [item['text_ids'] for item in batch]
    batch_text_mask = [item['text_mask'] for item in batch]
    
    motions = torch.stack(batch_motion)
    text_ids = torch.stack(batch_text_ids)
    text_masks = torch.stack(batch_text_mask)
    
    return {
        "x": motions,
        "cond": text_ids,
        "cond_mask": text_masks
    }

def build_dataloader(lmdb_path, n_poses, mean, std, batch_size, shuffle=True, num_workers=0):
    dataset = TextMotionDataset(
        lmdb_dir=lmdb_path,
        n_poses=n_poses,
        subdivision_stride=10, # 步长，越小生成的数据越多
        pose_resampling_fps=30,
        data_mean=mean,
        data_std=std
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return loader
import logging
import os
import numpy as np
import lmdb
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from data_loader.data_preprocessor_bert import DataPreprocessor
import pyarrow


def word_seq_collate_fn(data):
    """ Collate function for loading BERT embedding sequences of variable lengths """
    # Sort by sequence length (descending) to allow for pack_padded_sequence if needed
    data.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Unpack: emb_seq replaces the original word indices
    emb_seq, poses_seq, audio, aux_info = zip(*data)

    lengths = torch.LongTensor([len(x) for x in emb_seq])

    # --- WARNING FIX: Ensure writable arrays ---
    # PyTorch warns if creating a tensor from a read-only numpy array (which happens with pyarrow/lmdb).
    # Using np.copy() ensures the array is writable and owns its memory.
    emb_seq_list = []
    for x in emb_seq:
        if isinstance(x, np.ndarray):
            emb_seq_list.append(torch.from_numpy(np.copy(x)).float())
        else:
            # Fallback if x is already a tensor or list
            emb_seq_list.append(torch.as_tensor(x).float())

    # Pad sequences (batch_first=True)
    emb_seq = pad_sequence(emb_seq_list, batch_first=True)

    poses_seq = default_collate(poses_seq)
    audio = default_collate(audio)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return emb_seq, lengths, poses_seq, audio, aux_info


class TwhDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, data_mean, data_std):
        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.data_mean = np.array(data_mean).squeeze()
        self.data_std = np.array(data_std).squeeze()

        logging.info(f"Reading data '{lmdb_dir}'...")
        
        # Define a distinct cache folder for BERT to avoid conflicts with original data
        preloaded_dir = lmdb_dir + '_bert_cache'
        
        if not os.path.exists(preloaded_dir):
            logging.info("No cache found. Running DataPreprocessor for BERT...")
            data_sampler = DataPreprocessor(lmdb_dir, preloaded_dir, n_poses,
                                            subdivision_stride, pose_resampling_fps,
                                            data_mean=data_mean, data_std=data_std)
            data_sampler.run()
        else:
            logging.info(f'Found pre-loaded samples from {preloaded_dir}')

        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']
        
        if self.n_samples == 0:
            raise RuntimeError(f"No samples found in {preloaded_dir}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)
            
            if sample is None:
                raise IndexError(f"Sample {idx} not found in LMDB {self.lmdb_dir}")
            
            sample = pyarrow.deserialize(sample)
            # Unpack: Expecting embeddings instead of word indices
            emb_seq, pose_seq, audio, aux_info = sample

        # Normalize poses
        std = np.clip(self.data_std, a_min=0.01, a_max=None)
        pose_seq = (pose_seq - self.data_mean) / std

        # Ensure embeddings are numpy array
        if isinstance(emb_seq, list):
            emb_seq = np.array(emb_seq)
            
        # --- WARNING FIX: Ensure writable arrays ---
        # np.copy() creates a writable copy in memory, satisfying PyTorch requirements
        pose_seq = torch.from_numpy(np.copy(pose_seq)).reshape((pose_seq.shape[0], -1)).float()
        audio = torch.from_numpy(np.copy(audio)).float()
        
        # Note: emb_seq is converted to tensor inside collate_fn to handle padding,
        # so we return it as a numpy array here.

        return emb_seq, pose_seq, audio, aux_info
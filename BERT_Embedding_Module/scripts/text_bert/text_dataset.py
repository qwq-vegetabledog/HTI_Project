import numpy as np
import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(self, files, max_seq_len=300):
        """
        Dataset for embeddings saved in .npz files.
        
        Args:
            files (list): list of paths to .npz files
            max_seq_len (int): maximum sequence size per phrase
        """
        self.data = []
        self.max_seq_len = max_seq_len

        for f in files:
            arr = np.load(f, allow_pickle=True)  # loads .npz file
            for key in arr.files:                # iterates through all the keys (phrases)
                phrase_emb = arr[key]            # real sentence array

                # Limit the maximum size of each sentence.
                if phrase_emb.shape[0] > max_seq_len:
                    phrase_emb = phrase_emb[:max_seq_len]

                # Save as tensor float32
                self.data.append(torch.tensor(phrase_emb, dtype=torch.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns x and y for predicting the next embedding.
        x: all embeddings except the last one
        y: all embeddings except the first one
        """
        phrase = self.data[idx]
        x = phrase[:-1]
        y = phrase[1:]
        return x, y

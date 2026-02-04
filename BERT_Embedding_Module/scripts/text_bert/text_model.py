import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NextWordLSTM(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=256, num_layers=1, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, lengths):
        # x: (batch, seq_len, embed_dim)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

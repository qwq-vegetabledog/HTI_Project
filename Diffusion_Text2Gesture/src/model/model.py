import math
import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pos_emb = self.pe[:x.size(1), :].unsqueeze(0)
        return x + pos_emb

class Text2GestureModel(nn.Module):
    def __init__(
        self, 
        input_feats, 
        latent_dim=512, 
        n_heads=8, 
        n_layers=8,  
        dropout=0.1, 
        text_dim=768
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_proj = nn.Linear(input_feats, latent_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.pos_encoder = PositionalEncoding(latent_dim)
        
        if text_dim != latent_dim:
            self.text_proj = nn.Linear(text_dim, latent_dim)
        else:
            self.text_proj = nn.Identity()

        decoder_layer = TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.final_layer = nn.Linear(latent_dim, input_feats)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _sinusoidal_embedding(self, t, dim):
        device = t.device
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, x, t, context, src_mask=None):
        h = self.input_proj(x)
        t_emb = self._sinusoidal_embedding(t, self.latent_dim)
        t_emb = self.time_mlp(t_emb)
        h = h + t_emb.unsqueeze(1)
        h = self.pos_encoder(h)
        
        context = self.text_proj(context)
        
        if src_mask is not None:
            # 修正 Mask 逻辑：src_mask 为 1 (True) 的地方是有效的，为 0 (False) 是 Pad
            # key_padding_mask 需要：True 是 Pad (被忽略)，False 是有效
            key_padding_mask = (src_mask == 0).bool()
        else:
            key_padding_mask = None

        output = self.transformer(
            tgt=h, 
            memory=context, 
            memory_key_padding_mask=key_padding_mask
        )
        prediction = self.final_layer(output)
        return prediction
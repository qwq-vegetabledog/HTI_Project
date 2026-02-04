import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, loss_type='l2', beta_schedule='cosine'):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type

        # 1. 准备 Betas
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # Pad 这里的 prev 是为了方便计算，t=0 时 prev=1.0
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # 2. 注册为 Buffer (这样它们会自动转到 GPU，并随模型保存)
        # 不需要手动 .to(device)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
        # 采样所需的系数
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    # ============================================================
    #  工具函数：提取对应时间步的系数
    # ============================================================
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # ============================================================
    #  前向加噪过程 (q_sample)
    # ============================================================
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise

    # ============================================================
    #  训练接口 (Forward): 计算 Loss
    # ============================================================
    def forward(self, x_start, context, t=None, src_mask=None, noise=None):
        """
        训练时调用。
        x_start: [Batch, Seq, Dim] 真实数据
        t: [Batch] 时间步
        context: [Batch, Text_Len, Text_Dim] 文本条件
        1. 把 t 移到了后面，并设默认值为 None。
        2. 如果外部没传 t，就在内部随机采样。
        
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        batch_size = x_start.shape[0]
        device = x_start.device

        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

        # 1. 加噪得到 x_t
        x_t = self.q_sample(x_start, t, noise)

        # 2. 模型预测噪声 (Predict Noise)
        pred_noise = self.model(x_t, t, context, src_mask)

        # 3. 计算 Loss (预测噪声 vs 真实噪声)
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, pred_noise, reduction='none')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, pred_noise, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(noise, pred_noise, reduction='none')
        else:
            raise NotImplementedError()

        # 如果有 Mask (处理 Padding)，将 Padding 部分的 Loss 置零
        if src_mask is not None:
            # src_mask: [Batch, Seq] -> [Batch, Seq, 1]
            mask = src_mask.unsqueeze(-1).float()
            loss = loss * mask
            return loss.sum() / mask.sum()
        else:
            return loss.mean()

    # ============================================================
    #  采样单步 (p_sample): 从 x_t 推导 x_{t-1}
    # ============================================================
    @torch.no_grad()
    def p_sample(self, x, t, t_index, context, src_mask=None, guidance_scale=0.0):
        # 1. 模型预测噪声 epsilon
        # Classifier-Free Guidance (CFG)
        if guidance_scale > 0.0:
            # 构造空文本条件 (uncond)
            uncond_context = torch.zeros_like(context) 
            
            # 两次前向
            noise_cond = self.model(x, t, context, src_mask)
            noise_uncond = self.model(x, t, uncond_context, src_mask)
            
            # 混合: uncond + scale * (cond - uncond)
            pred_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            pred_noise = self.model(x, t, context, src_mask)

        # 2. 计算均值 (Mean)
        # 公式: x_{t-1} = 1/sqrt(alpha) * (x_t - beta/sqrt(1-alpha_bar) * epsilon)
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        
        model_mean = sqrt_recip_alphas_t * (x - betas_t / sqrt_one_minus_alpha_cumprod_t * pred_noise)

        # 3. 计算方差 (Variance)
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # ============================================================
    #  完整生成循环 (sample): 从随机噪声 -> 最终结果
    # ============================================================
    @torch.no_grad()
    def sample(self, shape, context, src_mask=None, guidance_scale=2.5):
        """
        生成函数。
        shape: [Batch, Seq, Dim] 生成的数据形状
        context: [Batch, Text_Len, Dim] 文本条件
        guidance_scale: CFG 强度 (通常 2.0 ~ 4.0 效果好)
        """
        device = self.betas.device
        batch_size = shape[0]
        
        # 1. 从纯高斯噪声开始 x_T
        img = torch.randn(shape, device=device)
        
        # 2. 逐步去噪 T -> 0
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # 单步采样
            img = self.p_sample(
                img, t, i, 
                context=context, 
                src_mask=src_mask, 
                guidance_scale=guidance_scale
            )
            
        # 3. 返回最终生成的 x_0
        return img
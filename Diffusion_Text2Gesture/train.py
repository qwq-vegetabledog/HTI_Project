import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import random
import numpy as np

# === 引入模块 ===
from src.config import Config
from src.data.data_loader import build_dataloader
from src.model.model import Text2GestureModel 
from src.model.diffusion import GaussianDiffusion




# print("\n" + "="*50)
# print("🔎 侦探模式：我正在读取的数据路径是：")
# # 这里通常是 args.train_data_path 或者 Config.train_data_path
# # 如果你不确定变量名，可以搜一下代码里 DataPreprocessor 被调用的地方
# try:
#     print(f"PATH: {args.train_data_path}") 
# except:
#     try:
#         print(f"PATH: {Config.train_data_path}") # 或者类似的变量名
#     except:
#         print("无法自动找到变量，请手动检查代码中 Dataset 初始化的位置")
# print("="*50 + "\n")


# exit(0)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(diffusion, val_loader, text_encoder, device):
    """
    验证函数：计算测试集上的平均 Loss
    """
    diffusion.eval() # 切换到评估模式
    total_loss = 0
    
    with torch.no_grad(): # 不计算梯度
        for batch in val_loader:
            # 1. 准备数据
            motions = batch['x'].to(device)
            cond_ids = batch['cond'].to(device)
            cond_mask = batch['cond_mask'].to(device)

            # 2. 文本编码
            text_outputs = text_encoder(input_ids=cond_ids, attention_mask=cond_mask)
            text_embeddings = text_outputs.last_hidden_state

            # 3. 计算 Diffusion Loss (验证时不加 CFG Mask，因为我们要看真实的生成能力)
            # 注意：这里我们计算的是重建误差(MSE)，越低越好
            loss = diffusion(
                x_start=motions, 
                context=text_embeddings, 
                src_mask=None 
            )
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    diffusion.train() # 切换回训练模式
    return avg_loss

def train():
    # 1. 初始化
    set_seed(42)
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    device = torch.device(Config.DEVICE)
    print(f"🚀 Training on device: {device}")
    
    writer = SummaryWriter(log_dir=Config.LOG_DIR)

    # ====================================================
    # 2. 准备数据加载器 (Train & Val)
    # ====================================================
    print("📂 Loading Datasets...")
    
    # 训练集加载器
    train_loader = build_dataloader(
        lmdb_path=Config.LMDB_TRAIN_PATH,
        n_poses=Config.WINDOW_FRAMES,
        mean=Config.DATA_MEAN,
        std=Config.DATA_STD,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,           # 训练集需要打乱
        num_workers=Config.NUM_WORKERS
    )
    
    # 验证集加载器
    val_loader = build_dataloader(
        lmdb_path=Config.LMDB_TEST_PATH,
        n_poses=Config.WINDOW_FRAMES,
        mean=Config.DATA_MEAN,
        std=Config.DATA_STD,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,          # 验证集不需要打乱
        num_workers=Config.NUM_WORKERS
    )
    print(f"✅ Loaded: {len(train_loader)} train steps, {len(val_loader)} val steps per epoch.")

    # 3. 初始化模型
    print("🧠 Initializing Models...")
    text_encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)
    text_encoder.eval() # 冻结 BERT
    for param in text_encoder.parameters():
        param.requires_grad = False

    model = Text2GestureModel(
        input_feats=Config.INPUT_FEATS,
        latent_dim=Config.LATENT_DIM,
        n_layers=Config.LAYERS,
        n_heads=Config.HEADS,
        dropout=Config.DROPOUT,
        text_dim=Config.TEXT_DIM
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        timesteps=Config.DIFFUSION_STEPS,
        loss_type=Config.LOSS_TYPE,
        beta_schedule=Config.BETA_SCHEDULE
    ).to(device)

    # 4. 优化器 & 恢复训练逻辑
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf') # 记录最佳验证损失
    

    if Config.RESUME_CHECKPOINT and os.path.exists(Config.RESUME_CHECKPOINT):
        print(f"♻️ Resuming from: {Config.RESUME_CHECKPOINT}")
        ckpt = torch.load(Config.RESUME_CHECKPOINT, map_location=device)
        diffusion.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        # 如果 checkpoint 里存了 best_loss 就读出来
        if 'best_val_loss' in ckpt:
            best_val_loss = ckpt['best_val_loss']
    else:
        print("✨ Starting from scratch.")

    # ====================================================
    # 5. 训练循环
    # ====================================================
    print("🔥 Start Training Loop...")
    
    for epoch in range(start_epoch, Config.EPOCHS):
        diffusion.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        loss_epoch = 0

        for batch in progress_bar:
            # --- Train Step ---
            motions = batch['x'].to(device)
            cond_ids = batch['cond'].to(device)
            cond_mask = batch['cond_mask'].to(device)

            with torch.no_grad():
                text_emb = text_encoder(input_ids=cond_ids, attention_mask=cond_mask).last_hidden_state

            # CFG Trick (10% unconditioned)
            if random.random() < 0.1:
                text_emb = torch.zeros_like(text_emb)

            loss = diffusion(motions, context=text_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # --- Logging ---
            loss_val = loss.item()
            loss_epoch += loss_val
            global_step += 1
            progress_bar.set_postfix({"Loss": f"{loss_val:.4f}"})
            
            if global_step % 10 == 0:
                writer.add_scalar("Train/Loss", loss_val, global_step)

        avg_train_loss = loss_epoch / len(train_loader)
        writer.add_scalar("Train/Epoch_Loss", avg_train_loss, epoch)

        # ====================================================
        # 6. 验证循环 (Validation Loop)
        # ====================================================
        # 每隔 Config.EVAL_INTERVAL 轮，或者最后一轮，进行验证
        if (epoch + 1) % getattr(Config, 'EVAL_INTERVAL', 5) == 0:
            print(f"\n🔍 Evaluating on Test Set...")
            val_loss = evaluate(diffusion, val_loader, text_encoder, device)
            print(f"    >> Val Loss: {val_loss:.5f} (Best: {best_val_loss:.5f})")
            
            writer.add_scalar("Val/Loss", val_loss, epoch)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(Config.SAVE_DIR, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': diffusion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, best_path)
                print(f"    🏆 New Record! Saved best model to {best_path}")

        # ====================================================
        # 7. 定期保存 (Regular Checkpoint)
        # ====================================================
        if (epoch + 1) % Config.SAVE_INTERVAL == 0:
            save_path = os.path.join(Config.SAVE_DIR, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, save_path)
            print(f"💾 Checkpoint saved: {save_path}")

    print("🎉 Training Finished!")
    writer.close()

if __name__ == '__main__':
    train()
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np

from text_dataset import EmbeddingDataset
from text_model import NextWordLSTM

# -------------------------------------------------------
# Fixed embedding dimension (set manually)
# -------------------------------------------------------
EMBED_DIM = 4*768

# -------------------------------------------------------
# Padding function: pads all batches to the same max_len
# -------------------------------------------------------
def collate_fn_fixed(batch, max_len):
    xs, ys = [], []
    lengths = []

    for x, y in batch:
        L = x.shape[0]
        lengths.append(L)

        if L < max_len:
            pad_x = torch.cat([x, torch.zeros(max_len - L, EMBED_DIM)], dim=0)
            pad_y = torch.cat([y, torch.zeros(max_len - L, EMBED_DIM)], dim=0)

            xs.append(pad_x)
            ys.append(pad_y)

    return torch.stack(xs), torch.stack(ys), torch.tensor(lengths)

# -------------------------------------------------------
# Compute epoch max length
# -------------------------------------------------------
def compute_max_length(dataset):
    max_len = 0
    for x, _ in dataset:
        if x.shape[0] > max_len:
            max_len = x.shape[0]
    return max_len

# -------------------------------------------------------
# Training function
# -------------------------------------------------------
def train_model(train_folders, val_folders):

    # -----------------------------
    # LOAD FILES
    # -----------------------------
    train_files = []
    val_files = []

    for f in train_folders:
        train_files.extend(glob.glob(f"{f}/*.npz"))

    for f in val_folders:
        val_files.extend(glob.glob(f"{f}/*.npz"))

    train_dataset = EmbeddingDataset(train_files)
    val_dataset = EmbeddingDataset(val_files)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # -----------------------------
    # DEVICE & MODEL
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NextWordLSTM(
        embed_dim=EMBED_DIM,
        hidden_dim=512,
        num_layers=2,
        batch_first=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("./next_word", exist_ok=True)

    ckpt_last = "./next_word/last.pth"
    ckpt_best = "./next_word/best.pth"
    monitor_path = "./next_word/monitor_all_epochs.npz"

    # -----------------------------
    # RESUME IF EXISTS
    # -----------------------------
    best_val_loss = float("inf")
    start_epoch = 1
    train_losses = []
    val_losses = []

    # Load last checkpoint
    if os.path.exists(ckpt_last):
        print("Resuming training...")
        checkpoint = torch.load(ckpt_last, map_location=device)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
            start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"Resumed from epoch {start_epoch-1}")

    # Load monitor file if exists
    if os.path.exists(monitor_path):
        data = np.load(monitor_path)
        train_losses = data['train_loss'].tolist()
        val_losses = data['val_loss'].tolist()
        print(f"Loaded monitor with {len(train_losses)} epochs already recorded.")

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    num_epochs = 1000
    
    for epoch in range(start_epoch, num_epochs + 1):

        # recompute max padding length each epoch
        max_len = compute_max_length(train_dataset)

        # dataloaders with fixed padding
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            collate_fn=lambda b: collate_fn_fixed(b, max_len)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=lambda b: collate_fn_fixed(b, max_len)
        )

        # ---------------------
        # Training
        # ---------------------
        model.train()
        total_loss = 0

        for x, y, lengths in tqdm(train_loader, desc=f"Epoch {epoch} (train)"):
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad()
            pred = model(x, lengths)

            # mask padded tokens
            mask = (y.abs().sum(dim=2) != 0)
            loss = ((pred - y)**2 * mask.unsqueeze(2)).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        # ---------------------
        # Validation
        # ---------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y, lengths in tqdm(val_loader, desc=f"Epoch {epoch} (val)"):
                x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                pred = model(x, lengths)

                mask = (y.abs().sum(dim=2) != 0)
                loss = ((pred - y)**2 * mask.unsqueeze(2)).sum() / mask.sum()
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        print(f"\nEpoch {epoch} | Train={avg_train:.4f} | Val={avg_val:.4f}")

        # Append losses to monitor lists
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        # ---------------------
        # Save last checkpoint (with epoch info)
        # ---------------------
        torch.save({
            'model_state': model.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss
        }, ckpt_last)

        # ---------------------
        # Save best model
        # ---------------------
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), ckpt_best)
            print("New best model saved!\n")

        # ---------------------
        # Save monitor (all epochs)
        # ---------------------
        np.savez(
            monitor_path,
            train_loss=np.array(train_losses),
            val_loss=np.array(val_losses)
        )


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    train_folders = [
        "/home/HTI_project/dataset/genea/genea2023_dataset/trn/interloctr/embedding",
        "/home/HTI_project/dataset/genea/genea2023_dataset/trn/main-agent/embedding"
    ]

    val_folders = [
        "/home/HTI_project/dataset/genea/genea2023_dataset/val/interloctr/embedding",
        "/home/HTI_project/dataset/genea/genea2023_dataset/val/main-agent/embedding"
    ]

    train_model(train_folders, val_folders)

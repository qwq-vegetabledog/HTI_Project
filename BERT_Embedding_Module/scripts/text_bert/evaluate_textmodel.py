import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from text_dataset import *
from text_model import *

def collate_fn(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)
    return xs_padded, ys_padded, lengths


def evaluate_model(embedding_folders, model_path):
    # Load files
    files = []
    for folder in embedding_folders:
        files.extend(glob.glob(f"{folder}/*.npz"))

    dataset = EmbeddingDataset(files)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NextWordLSTM(embed_dim=768, hidden_dim=512, num_layers=2, batch_first=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_loss = 0
    total_tokens = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y, lengths in loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            pred = model(x, lengths)

            mask = (y.abs().sum(dim=2) != 0)  # [batch, seq]
            mask_exp = mask.unsqueeze(2)

            loss = ((pred - y)**2 * mask_exp).sum()
            tokens = mask.sum()

            total_loss += loss.item()
            total_tokens += tokens.item()

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / total_tokens

    print("\n================ Evaluation Results ================")
    print(f"Model: {model_path}")
    print(f"MSE: {avg_loss:.6f}")
    print("====================================================\n")

    np.savez("./next_word/evaluation_results.npz",
         preds=np.array(all_preds, dtype=object),
         targets=np.array(all_targets, dtype=object))
    
if __name__ == "__main__":
    folders = [
        "/home/HTI_project/dataset/genea/genea2023_dataset/tst/interloctr/embedding",
        "/home/HTI_project/dataset/genea/genea2023_dataset/tst/main-agent/embedding"
    ]

    model_path = "next_word/next_word_lstm_epoch100.pth"

    evaluate_model(folders, model_path)
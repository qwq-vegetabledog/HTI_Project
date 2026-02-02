import numpy as np
import matplotlib.pyplot as plt

# ==== LOAD DATA ====

data = np.load("./next_word/evaluation_results.npz", allow_pickle=True)
pred_batches = data["preds"]     # array dtype=object: list of batches
target_batches = data["targets"]

# pred_batches[b] -> shape (BATCH, SEQ, 768)
# mas SEQ varia por batch


# ==== FUNÇÃO DE MÁSCARA E MSE ====

def compute_batch_mse(pred, target):
    mask = (np.abs(target).sum(axis=2) != 0)      # shape (batch, seq)
    mask_exp = mask[:, :, None]                  # shape (batch, seq, 1)

    mse = np.sum((pred - target) ** 2 * mask_exp) / np.sum(mask)
    return mse


# ==== CALCULAR MSE GLOBAL E POR BATCH ====

batch_mse = []
total_loss = 0
total_tokens = 0

for pred, target in zip(pred_batches, target_batches):
    mask = (np.abs(target).sum(axis=2) != 0)
    mask_exp = mask[:, :, None]

    loss = np.sum((pred - target) ** 2 * mask_exp)
    tokens = np.sum(mask)

    total_loss += loss
    total_tokens += tokens

    batch_mse.append(loss / tokens)

global_mse = total_loss / total_tokens
print("Global MSE:", global_mse)


# ==== PLOT 1 — PERDA POR BATCH ====

plt.figure(figsize=(12, 6))
plt.plot(batch_mse, linewidth=2)
plt.xlabel("Batch", fontsize=14)
plt.ylabel("MSE", fontsize=14)
plt.title("Evaluation Loss per Batch", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig("./next_word/eval_loss_per_batch.png")
plt.close()


# ==== PLOT 2 — HISTOGRAMA DOS ERROS ====

# coletar todos os erros token a token
all_errors = []

for pred, target in zip(pred_batches, target_batches):
    mask = (np.abs(target).sum(axis=2) != 0)
    mask_exp = mask[:, :, None]

    err = ((pred - target) ** 2 * mask_exp).sum(axis=2)[mask]
    all_errors.append(err)

all_errors = np.concatenate(all_errors)

plt.figure(figsize=(10, 6))
plt.hist(all_errors, bins=80)
plt.xlabel("Token MSE", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Distribution of Token-Level Errors", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig("./next_word/eval_histogram_token_errors.png")
plt.close()

print("Plots saved in ./next_word/")

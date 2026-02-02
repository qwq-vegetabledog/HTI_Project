import numpy as np
import matplotlib.pyplot as plt

monitor_epoch100 = "next_word/monitor_epoch100.npz"
monitor_epoch1000 = "next_word/monitor_epoch1000.npz"

data100 = np.load(monitor_epoch100)
data1000 = np.load(monitor_epoch1000)

def r(x, decimals=4):
    return np.round(x.astype(float), decimals=decimals)

losses = data1000["loss"]
sample_input = data1000["sample_input"]
sample_output = data1000["sample_output"]

print("Input:\n", r(sample_input[1,:3,:5]))
print("Output:\n", r(sample_output[1,:3,:5]))

mask = (np.abs(sample_output).sum(axis=2) != 0)
mask_exp = np.expand_dims(mask, axis=2)

pred100 = data100["sample_pred"]
print("Prediciton 100 epoch:\n", r(pred100[1,:3,:5]))
loss100 = result = np.sum((pred100 - sample_output)**2 * mask_exp) / np.sum(mask)
print("Loss 100 epoch:\n", loss100)

pred1000 = data1000["sample_pred"]
print("Prediciton 1000 epoch:\n", r(pred1000[1,:3,:5]))
loss1000 = result = np.sum((pred1000 - sample_output)**2 * mask_exp) / np.sum(mask)
print("Loss 100 epoch:\n", loss1000)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Training Loss Curve", fontsize=16)
plt.grid(True)

plt.savefig("next_word/monitor_epoch1000.png")

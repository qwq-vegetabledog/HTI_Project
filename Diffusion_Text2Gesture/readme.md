

# Text-to-Gesture Generation via Transformer-based Diffusion

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Introduction

This project implements a **Text-driven Motion Generation System** using a **Transformer-based Gaussian Diffusion Model**. It synthesizes vivid and diverse 3D co-speech gestures from natural language text.

Unlike deterministic regression models that often suffer from **"Mean Collapse"** (resulting in over-smoothed, robotic motion), our diffusion-based approach captures the true distribution of human motion, generating gestures with realistic dynamics and high expressiveness.

### Key Features
* **Architecture:** Transformer Decoder backbone with Cross-Attention for semantic alignment.
* **Diffusion Process:** 1000-step iterative denoising from Gaussian noise.
* **Robust Representation:** Uses **12D feature vectors** (Rotation Matrices + Positions) instead of Euler angles to avoid gimbal lock.
* **Engineering Optimization:** Integrated **SVD Orthogonalization** and **Gaussian Smoothing** for geometric validity and jitter removal.

---

## Project Structure

```text
Diffusion_Project_Release/
├── train.py                   # Main entry for training
├── inference.py               # Main entry for generating motion (sampling)
├── cal_ave_mse.py             # Evaluation script (Calculate MSE & AVE)
├── requirements.txt           # Python dependencies
├── checkpoints/               # Directory for saving model weights
│   └── best_model.pt          # Pre-trained best model
└── src/                       # Source code modules
    ├── config.py              # Configuration settings (hyperparameters)
    ├── twh_dataset_to_lmdb.py # CRITICAL: Data preprocessing script
    ├── model/                 # Model definitions (Diffusion + Transformer)
    ├── data/                  # Data loaders and iterators
    ├── utils/                 # Utility functions (Training, logging)
    └── pymo/                  # BVH processing toolkit (Reference Code)
```

## Installation

1. **Clone the repository**

   Bash

   ```
   git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)
   cd Your-Repo-Name
   ```

2. **Create a virtual environment (Recommended)**

   Bash

   ```
   conda create -n t2g_diffusion python=3.8
   conda activate t2g_diffusion
   ```

3. **Install dependencies**

   Bash

   ```
   # Install project requirements
   pip install -r requirements.txt
   ```

------

## User Manual (Usage)

### 1. Data Preparation (Important)

Before training, you **MUST** convert the raw dataset into LMDB format for efficient I/O.

We use a specific script to preprocess the BVH files into 12D feature vectors and store them.

**Run the following command:**

Bash

```
python src/twh_dataset_to_lmdb.py \
    --input_dir /path/to/your/raw_bvh_dataset \
    --output_dir ./data/lmdb_processed
```

> **Note:** This step calculates the Z-score mean/std and slices the motion into fixed-length windows (120 frames).

### 2. Training

To train the diffusion model from scratch:

Bash

```
python train.py
```

The logs and checkpoints will be saved in the `save/` directory.

### 3. Inference (Generation)

To generate gestures, the model accepts a **TSV (Tab-Separated Values)** file as input. This allows for batch generation of multiple motions at once. 

**1. Prepare Input TSV File**

**2. Change the Path of inference.py**

**3. inference2.py is for batch generation**

## Results

| **Metric** | **Description**        | **Performance**                        |
| ---------- | ---------------------- | -------------------------------------- |
| **MSE**    | Mean Squared Error     | **Higher** (Expected due to diversity) |
| **AVE**    | Average Variance Error | **Lower** (Better dynamics & realism)  |

*Our model achieves significantly lower AVE compared to LSTM baselines, indicating that it successfully captures the high-frequency dynamics of human motion without over-smoothing.*

------

## Acknowledgements & References

We would like to acknowledge that the **data preprocessing pipeline** (specifically the BVH parsing and feature extraction logic) is adapted from the following reference:

- **[Baseline Repository Name/Paper Title]**: *[https://github.com/youngwoo-yoon/Co-Speech_Gesture_Generation.git]*
  - *We utilized their robust BVH-to-Feature conversion methods to ensure data compatibility.*

The core Diffusion and Transformer architecture is implemented by our team.

------

## License

This project is licensed under the MIT License.
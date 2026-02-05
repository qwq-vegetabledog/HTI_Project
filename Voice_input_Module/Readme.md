# HTI_Project: Speech-to-Motion Generation

This project is a deep learning pipeline for generating human motion sequences from speech audio. It utilizes a Seq2Seq architecture to transform acoustic features into motion vectors.

## Project Structure

Based on the local setup, the recommended organization is:

```text
HTI_Project/
├── config/
│   └── seq2seq_sound.yml        # Configuration for hyperparameters & paths
├── data_loader/
│   ├── lmdb_data_loader_sound.py
│   ├── data_preprocessor_sound.py
│   └── twh_dataset_to_lmdb.py   # Script to build the LMDB database
├── model/
│   └── seq2seq_net_sound.py     # Network architecture (Seq2SeqNet)
├── train_eval/
│   └── train_seq2seq_sound.py   # Training logic &    iteration functions
├── utils/
│   ├── wav.py                  # Audio processing (WAV to Spectrogram .npy)
│   └── train_utils_sound.py
├── train_yujia.py               # Main training entry point
├── eval_bhv_sound.py           # Evaluation script
├── inference_sound.py         # Inference/Generation script
├── requirements.txt
└── .gitignore

```
## Step 1: Audio Feature Extraction

Process your raw .wav files into spectrogram features (.npy).
```bash
python utils/wav.py
```

## Step 2: Dataset Preparation

Build the LMDB database using your extracted .npy features and motion data.
```bash
python data_loader/twh_dataset_to_lmdb.py --config config/seq2seq_sound.yml
```

## Step 3: Training
Start the training process using the main script.
```bash
python train_yujia.py --config config/seq2seq_sound.yml
```
## Configuration

Update config/seq2seq_sound.yml to set your local data paths:

audio_feature_path: Path to the folder with .npy files.

train_data_path: Path to your .lmdb folder.

## Evaluation&&Inference

Once training is complete, you can evaluate the model or generate new motions.

To evaluate the model:
```bash
python eval_bhv_sound.py --checkpoint ./path/to/checkpoint.bin
```
To generate motion from audio:
```bash
python inference_sound.py --input_audio demo.wav
```